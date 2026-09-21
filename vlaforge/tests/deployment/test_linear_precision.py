import hashlib
import json
from dataclasses import replace

import pytest
import torch
from vlaforge.analysis.constant_precompute import graph_sha256
from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
)
from vlaforge.deployment.int8_linear import lower_int8_linear
from vlaforge.deployment.linear_precision import (
    lower_exported_all_linear_halves,
    lower_exported_linear,
)
from vlaforge.numerical_context import snapshot


def example(shape=(16, 8)):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(shape[-1], 8)
            with torch.no_grad():
                self.head.weight.copy_(torch.linspace(-1, 1, 8 * shape[-1]).reshape(8, shape[-1]))
                self.head.bias.copy_(torch.linspace(-1, 1, 8))

        def forward(self, value, step_index):
            activation = torch.relu(value)
            result = self.head(activation)
            return result + value[..., :8] * 0.1, step_index + 1

    value = torch.linspace(-260, 260, torch.tensor(shape).prod().item()).reshape(shape)
    model = Model().eval()
    program = torch.export.export(model, (value, torch.tensor([0])), strict=True)
    linear = next(node for node in program.graph.nodes if node.target is torch.ops.aten.linear.default)
    producer = linear.args[0]
    context = snapshot()
    calibration = PrecisionCalibration(
        profile_sha256="0" * 64,
        numerical_context_sha256=hashlib.sha256(json.dumps(context.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        step_keys=("step0", "step1"),
        sites=(CalibrationSite("head", "region", producer.name, "a" * 64, "iteration", True),),
        calibration_samples=(CalibrationSample("cal", "episode0", "2" * 64, "3" * 64),),
        held_out_samples=(CalibrationSample("held", "episode1", "4" * 64, "5" * 64),),
    )
    for step in range(2):
        calibration.observe(sample_id="cal", site="head", step=step, tensor=torch.relu(value / (step + 1)))
    return model, program, calibration, value, linear.name


def lower(program, calibration, name, **options):
    arguments = {"site": "head", "linear_node": name, "step_input": "step_index", "region_artifact_sha256": "a" * 64, "numerical_context": snapshot()}
    arguments.update(options)
    return lower_exported_linear(program, calibration.fit("step"), calibration.report(), **arguments)


def all_linear_half_model(shape=(8, 8)):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(shape[-1], 12)
            self.second = torch.nn.Linear(12, shape[0])

        def forward(self, value):
            return self.second(torch.relu(self.first(value))) + value

    model = Model().eval()
    value = torch.linspace(-3, 3, torch.tensor(shape).prod().item()).reshape(shape)
    return model, torch.export.export(model, (value,), strict=True), value


@pytest.mark.parametrize("precision", ("float16", "bfloat16"))
def test_all_linear_half_rewrite_preserves_abi_and_survives_save_reload(tmp_path, precision):
    model, program, value = all_linear_half_model()
    expected = model(value)
    original = graph_sha256(program)
    result = lower_exported_all_linear_halves(program, precision, snapshot())
    assert graph_sha256(program) == original
    assert result.ledger["linear_count"] == 2
    assert "torch.ops.aten.linear.default" in str(result.program.graph)
    assert result.ledger["precision"] == precision
    path = tmp_path / ("result-" + precision + ".pt2")
    torch.export.save(result.program, path)
    reloaded = torch.export.load(path).module()
    actual = reloaded(value)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("precision", ("int8", "float64", None))
def test_all_linear_half_rejects_unsupported_precision(precision):
    _, program, _ = all_linear_half_model()
    with pytest.raises(ValueError, match="precision_kind"):
        lower_exported_all_linear_halves(program, precision, snapshot())


def half_lowering(program, calibration, name, precision):
    arguments = {"site": "head", "linear_node": name, "step_input": "step_index",
        "region_artifact_sha256": "a" * 64, "numerical_context": snapshot(),
        "precision_kind": precision}
    return lower_exported_linear(program, calibration.fit("step"), calibration.report(), **arguments)


@pytest.mark.parametrize("shape", ((16, 8), (1, 24, 16)))
def test_real_exported_graph_replace_only_selected_linear_and_preserve_abi(tmp_path, shape):
    model, program, calibration, value, name = example(shape)
    original = graph_sha256(program)
    source_output = program.module()(value, torch.tensor([0]))
    result = lower(program, calibration, name)
    assert graph_sha256(program) == original
    assert program.call_spec == result.program.call_spec
    assert program.graph_signature.output_specs == result.program.graph_signature.output_specs
    assert torch.equal(program.module()(value, torch.tensor([0]))[0], source_output[0])
    assert result.ledger["linear_node"] == name
    assert not result.ledger["old_artifact_certificates_reusable"]
    assert not result.ledger["full_model_output_verified"]
    assert str(result.program.graph).count("torch.ops.aten._int_mm.default") == 1
    assert "torch.ops.aten.linear.default" not in str(result.program.graph)
    torch.export.save(result.program, tmp_path / "result.pt2")
    reloaded = torch.export.load(tmp_path / "result.pt2").module()
    for step in (1, 0, 1, 0):
        fixed = lower_int8_linear(calibration.fit("step"), calibration.report(), site="head", step=step, region_artifact_sha256="a" * 64, numerical_context=snapshot(), weight=model.head.weight, bias=model.head.bias)
        expected = fixed.module(torch.relu(value)) + value[..., :8] * 0.1
        actual, next_step = reloaded(value, torch.tensor([step]))
        assert torch.equal(actual, expected)
        assert torch.equal(next_step, torch.tensor([step + 1]))
    invalid, _ = reloaded(value, torch.tensor([-1]))
    assert torch.isnan(invalid).all()


@pytest.mark.parametrize("options,match", (
    ({"linear_node": "relu"}, "exact declared"),
    ({"site": "unknown"}, "exact declared"),
    ({"region_artifact_sha256": "b" * 64}, "exact declared"),
    ({"step_input": "relu"}, "schedule input"),
    ({"step_input": "value"}, "schedule input"),
    ({"step_input": "missing"}, "schedule input"),
))
def test_replacement_rejects_wrong_selection_and_step(options, match):
    _, program, calibration, _, name = example()
    with pytest.raises(ValueError, match=match):
        lower(program, calibration, name, **options)


def test_replacement_rejects_mathematically_similar_but_unobserved_producer():
    _, program, calibration, _, name = example()
    plan = calibration.fit("step")
    changed = replace(plan, sites=(replace(plan.sites[0], node="value"),))
    with pytest.raises(ValueError, match="calibrated producer"):
        lower_exported_linear(program, changed, calibration.report(), site="head", linear_node=name, step_input="step_index", region_artifact_sha256="a" * 64, numerical_context=snapshot())


def test_replacement_rejects_user_weight_not_an_immutable_parameter():
    class Model(torch.nn.Module):
        def forward(self, value, weight, step_index):
            return torch.nn.functional.linear(torch.relu(value), weight), step_index + 1

    _, _, calibration, value, _ = example()
    ep = torch.export.export(Model(), (value, torch.eye(8), torch.tensor([0])))
    linear = next(node.name for node in ep.graph.nodes if node.target is torch.ops.aten.linear.default)
    with pytest.raises(ValueError, match="parameters or persistent"):
        lower(ep, calibration, linear)


def test_replacement_rejects_changed_output_dtype_even_when_input_matches():
    _, program, calibration, _, name = example()
    node = next(node for node in program.graph.nodes if node.name == name)
    node.meta["val"] = node.meta["val"].to(torch.bfloat16)
    with pytest.raises(ValueError, match="output profile"):
        lower(program, calibration, name)


@pytest.mark.parametrize("precision", ("float16", "bfloat16"))
@pytest.mark.parametrize("shape", ((16, 8), (1, 24, 16)))
def test_real_exported_graph_half_replacement_preserves_abi_and_step(tmp_path, shape, precision):
    import torch
    from vlaforge.deployment.half_linear import lower_scheduled_half_linear

    model, program, calibration, value, name = example(shape)
    linear = next(node for node in program.graph.nodes if node.name == name)
    producer = linear.args[0].name
    head = model.head
    result = half_lowering(program, calibration, name, precision)
    assert str(result.program.graph).count("torch.ops.aten.linear.default") == 1
    assert "torch.ops.aten._int_mm.default" not in str(result.program.graph)
    torch.export.save(result.program, tmp_path / "half.pt2")
    reloaded = torch.export.load(tmp_path / "half.pt2").module()
    lowered = lower_scheduled_half_linear(
        calibration.step_keys,
        site="head",
        region_artifact_sha256="a" * 64,
        numerical_context=snapshot(),
        weight=head.weight,
        bias=head.bias,
        input_dtype="float32",
        input_shape=tuple(shape),
        precision=precision,
    )
    activation = torch.relu(value)
    expected = lowered.module(activation, torch.tensor([1])) + value[..., :8] * 0.1
    actual, next_step = reloaded(value, torch.tensor([1]))
    assert torch.equal(actual, expected)
    assert torch.equal(next_step, torch.tensor([2]))
    invalid, _ = reloaded(value, torch.tensor([-1]))
    assert torch.isnan(invalid).all()
    assert not result.ledger["real_low_precision_kernel_verified"]
    assert not result.ledger["lossless_verified"]
    assert result.ledger["lowering"]["precision"] == precision
