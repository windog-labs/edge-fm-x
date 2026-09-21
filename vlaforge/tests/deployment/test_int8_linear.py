import hashlib
import json
from dataclasses import replace

import pytest
import torch
from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
)
from vlaforge.deployment.int8_linear import (
    lower_int8_linear,
    lower_scheduled_int8_linear,
)
from vlaforge.numerical_context import snapshot


def calibrated(dtype=torch.float32, shape=(16, 8), *, context=False, quantize=True):
    collector = PrecisionCalibration(
        profile_sha256="0" * 64,
        numerical_context_sha256=hashlib.sha256(
            json.dumps(
                snapshot().to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        step_keys=("step0:actual-timestep-sha", "step1:actual-timestep-sha"),
        sites=(
            CalibrationSite(
                "linear-input",
                "region",
                "activation",
                "a" * 64,
                "context" if context else "iteration",
                quantize,
            ),
        ),
        calibration_samples=(CalibrationSample("cal", "episode0", "2" * 64, "3" * 64),),
        held_out_samples=(CalibrationSample("held", "episode1", "4" * 64, "5" * 64),),
    )
    for step in (None,) if context else range(2):
        value = torch.zeros(shape, dtype=dtype)
        value.reshape(-1)[0] = 127 * (1 if step is None else step + 1)
        collector.observe(sample_id="cal", site="linear-input", step=step, tensor=value)
    return collector


def scheduled(collector=None, strategy="step"):
    collector = collector or calibrated()
    options = {"step_groups": ((0, 1),)} if strategy == "step-group" else {}
    return lower_scheduled_int8_linear(
        collector.fit(strategy, **options), collector.report(), site="linear-input",
        region_artifact_sha256="a" * 64, numerical_context=snapshot(),
        weight=torch.eye(8, dtype=getattr(torch, collector.report()["observations"][0]["dtype"])) * 127,
    )


@pytest.mark.parametrize("strategy", ("global", "site", "step", "step-group"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_scheduled_integer_gemm_uses_explicit_step_and_matches_static(strategy, dtype):
    collector = calibrated(dtype=dtype)
    result = scheduled(collector, strategy)
    value = torch.linspace(-300, 300, 128).reshape(16, 8).to(dtype)
    for step in (1, 0, 1, 1, 0):
        fixed = lower(collector=collector, strategy=strategy, step=step, weight=torch.eye(8, dtype=dtype) * 127)
        assert torch.equal(result.module(value, torch.tensor([step])), fixed.module(value))
    assert not result.to_data()["implicit_step_counter"]
    assert not result.to_data()["schedule_binding_verified"]


@pytest.mark.parametrize("step", (-1, 2, -(2**63), 2**63 - 1))
def test_scheduled_invalid_step_is_nonfinite_not_silently_clamped(step):
    result = scheduled()
    assert torch.isnan(result.module(torch.ones(16, 8), torch.tensor([step]))).all()


@pytest.mark.parametrize("index", (torch.tensor(0), torch.tensor([0, 1]), torch.tensor([0.0]), torch.tensor([0], dtype=torch.int32)))
def test_scheduled_rejects_wrong_step_metadata(index):
    with pytest.raises(ValueError, match="step index"):
        scheduled().module(torch.ones(16, 8), index)


def test_scheduled_export_reload_preserves_dynamic_step_and_nonfinite_guard(tmp_path):
    result = scheduled()
    value = torch.linspace(-260, 260, 128).reshape(16, 8)
    ep = torch.export.export(result.module, (value, torch.tensor([0])), strict=True)
    torch.export.save(ep, tmp_path / "scheduled.pt2")
    loaded = torch.export.load(tmp_path / "scheduled.pt2").module()
    assert str(ep.graph).count("torch.ops.aten._int_mm.default") == 1
    assert not torch.equal(loaded(value, torch.tensor([0])), loaded(value, torch.tensor([1])))
    for step in (1, 0, 2, -1):
        actual, expected = loaded(value, torch.tensor([step])), result.module(value, torch.tensor([step]))
        assert torch.equal(actual, expected) if step in (0, 1) else torch.isnan(actual).all()


def test_scheduled_rejects_context_site_and_mutated_scale_table():
    with pytest.raises(ValueError, match="iterative site"):
        scheduled(calibrated(context=True))
    result = scheduled()
    result.module.scale_table.data[1] *= 2
    with pytest.raises(ValueError, match="buffers changed"):
        result.to_data()


def test_lowering_report_rejects_same_values_with_different_buffer_layout():
    result = lower()
    before = result.module.weight_int8.clone()
    result.module.weight_int8 = result.module.weight_int8.T
    assert torch.equal(before, result.module.weight_int8)
    with pytest.raises(ValueError, match="buffers changed"):
        result.to_data()


def lower(
    *, collector=None, strategy="step", step=0, weight=None, bias=None, **options
):
    collector = collector or calibrated()
    groups = {"step_groups": ((0, 1),)} if strategy == "step-group" else {}
    plan = collector.fit(strategy, **groups)
    return lower_int8_linear(
        plan,
        collector.report(),
        site="linear-input",
        step=step,
        region_artifact_sha256="a" * 64,
        numerical_context=snapshot(),
        weight=weight if weight is not None else torch.eye(8) * 127,
        bias=bias,
        **options,
    )


def expected(module, value):
    matrix = value.reshape(-1, value.shape[-1]).float()
    activation = torch.round(matrix / module.activation_scale).clamp(-127, 127).long()
    exact = activation @ module.weight_int8.long()
    output = exact.float() * module.activation_scale * module.weight_scale
    if module.bias is not None:
        output = output + module.bias
    return output.to(value.dtype).reshape(
        *value.shape[:-1], module.weight_scale.numel()
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("strategy", ("global", "site", "step", "step-group"))
def test_actual_cpu_integer_gemm_matches_independent_int64_arithmetic(dtype, strategy):
    collector = calibrated(dtype=dtype)
    weight = torch.linspace(-127, 127, 64).reshape(8, 8).to(dtype)
    bias = torch.arange(8, dtype=dtype)
    result = lower(
        collector=collector, strategy=strategy, step=1, weight=weight, bias=bias
    )
    value = torch.linspace(-300, 300, 128).reshape(16, 8).to(dtype)
    assert torch.equal(result.module(value), expected(result.module, value))
    assert result.module.weight_int8.dtype == torch.int8
    assert result.module.activation_scale == 2
    record = result.to_data()
    assert record["accumulator_dtype"] == "int32"
    assert record["gemm"] == "aten._int_mm.default"
    assert not record["real_low_precision_kernel_verified"]
    assert not record["full_model_output_verified"]
    assert not record["held_out_validation_complete"]
    assert not record["lossless_verified"]


def test_rounding_ties_saturation_and_zero_channel_scales_are_explicit():
    weight = torch.eye(8) * 127
    weight[7] = 0
    result = lower(weight=weight)
    values = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -200.0, 200.0, 20.0]).repeat(
        16, 1
    )
    actual = result.module(values)
    row = torch.tensor([0, 2, 2, 0, -2, -127, 127, 0], dtype=torch.float32) * 127
    assert torch.equal(actual, row.repeat(16, 1))
    assert result.module.weight_scale[7] == torch.finfo(torch.float32).tiny


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_activation_cannot_be_hidden_by_integer_conversion(bad):
    result = lower()
    value = torch.ones(16, 8)
    value[0, 0] = bad
    output = result.module(value)
    assert torch.isnan(output[0]).all()
    assert torch.isfinite(output[1:]).all()


def test_context_scale_is_once_only_and_input_storage_is_not_modified():
    collector = calibrated(context=True, shape=(1, 16, 8))
    result = lower(collector=collector, step=None)
    value = torch.ones(1, 16, 8)
    original = value.clone()
    assert result.module(value).shape == (1, 16, 8)
    assert torch.equal(value, original)
    assert result.to_data()["step_key"] is None
    with pytest.raises(ValueError, match="step"):
        lower(collector=collector, step=0)


@pytest.mark.parametrize("step", (None, -1, 2, True, "0"))
def test_invalid_iteration_step_is_rejected(step):
    with pytest.raises(ValueError, match="step"):
        lower(step=step)


@pytest.mark.parametrize("mutation", ("report", "region", "site", "bounds"))
def test_unbound_or_inconsistent_calibration_is_rejected(mutation):
    collector = calibrated()
    plan, report = collector.fit("step"), collector.report()
    site, source = "linear-input", "a" * 64
    if mutation == "report":
        report["observations"][0]["tensor_sha256"] = "9" * 64
    elif mutation == "region":
        source = "b" * 64
    elif mutation == "site":
        site = "unselected"
    else:
        bad_scale = replace(plan.scales[0], maximum=254.0, scale=2.0)
        plan = replace(plan, scales=(bad_scale, plan.scales[1]))
    with pytest.raises(ValueError):
        lower_int8_linear(
            plan,
            report,
            site=site,
            step=0,
            region_artifact_sha256=source,
            numerical_context=snapshot(),
            weight=torch.eye(8),
        )


@pytest.mark.parametrize(
    "kind", ("weight-dtype", "weight-shape", "weight-nan", "bias-shape", "bias-nan")
)
def test_unsupported_weights_or_bias_fail_before_module_construction(kind):
    weight, bias = torch.eye(8), torch.zeros(8)
    if kind == "weight-dtype":
        weight = weight.double()
    elif kind == "weight-shape":
        weight = weight[:, :7]
    elif kind == "weight-nan":
        weight[0, 0] = float("nan")
    elif kind == "bias-shape":
        bias = torch.zeros(7)
    else:
        bias[0] = float("nan")
    with pytest.raises(ValueError, match="weight|bias"):
        lower(weight=weight, bias=bias)


@pytest.mark.parametrize("shape,out", (((15, 8), 8), ((16, 7), 8), ((16, 8), 9)))
def test_cuda_matrix_profile_restrictions_are_not_silent_fallbacks(shape, out):
    with pytest.raises(ValueError, match="integer GEMM profile"):
        lower(collector=calibrated(shape=shape), weight=torch.ones(out, shape[-1]))


def test_owned_weights_and_manifest_cannot_be_changed_through_source_aliases():
    weight, bias = torch.eye(8), torch.ones(8)
    result = lower(weight=weight, bias=bias)
    record = result.to_data()
    weight.add_(100)
    bias.add_(100)
    record["lossless_verified"] = True
    assert not result.to_data()["lossless_verified"]
    assert torch.equal(result.module(torch.ones(16, 8)), torch.full((16, 8), 2.0))
    result.module.weight_int8.data.zero_()
    with pytest.raises(ValueError, match="buffers changed"):
        result.to_data()


def test_export_save_reload_retains_real_integer_operation_and_complete_outputs(
    tmp_path,
):
    result = lower()
    value = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    exported = torch.export.export(result.module, (value,), strict=True)
    assert any(
        node.target == torch.ops.aten._int_mm.default for node in exported.graph.nodes
    )
    torch.export.save(exported, tmp_path / "integer.pt2")
    loaded = torch.export.load(tmp_path / "integer.pt2")
    assert any(
        node.target == torch.ops.aten._int_mm.default for node in loaded.graph.nodes
    )
    assert torch.equal(loaded.module()(value), expected(result.module, value))


def test_changed_runtime_profile_is_rejected():
    result = lower()
    for value in (torch.ones(17, 8), torch.ones(16, 8).half()):
        with pytest.raises(ValueError, match="locked Linear profile"):
            result.module(value)


def test_int32_overflow_bound_is_checked_before_running_gemm():
    inner = 133152
    with pytest.raises(ValueError, match="overflow int32"):
        lower(collector=calibrated(shape=(16, inner)), weight=torch.ones(8, inner))


def test_numeric_policy_mismatch_is_not_implicitly_restored():
    collector = calibrated()
    plan = replace(collector.fit("step"), numerical_context_sha256="9" * 64)
    before = snapshot()
    with pytest.raises(ValueError, match="numerical context"):
        lower_int8_linear(
            plan,
            collector.report(),
            site="linear-input",
            step=0,
            region_artifact_sha256="a" * 64,
            numerical_context=before,
            weight=torch.eye(8),
        )
    assert snapshot() == before


def test_global_scale_uses_other_quantized_sites_but_not_floating_context():
    original = calibrated()
    sites = original.sites + (
        CalibrationSite("other", "region", "other_input", "a" * 64, "iteration", True),
        CalibrationSite("context", "prefix", "norm", "b" * 64, "context", False),
    )
    collector = PrecisionCalibration(
        profile_sha256=original.profile_sha256,
        numerical_context_sha256=original.numerical_context_sha256,
        step_keys=original.step_keys,
        sites=sites,
        calibration_samples=original.calibration_samples,
        held_out_samples=original.held_out_samples,
    )
    for step in range(2):
        collector.observe(
            sample_id="cal",
            site="linear-input",
            step=step,
            tensor=torch.full((16, 8), 127.0),
        )
        collector.observe(
            sample_id="cal", site="other", step=step, tensor=torch.full((16, 16), 381.0)
        )
    collector.observe(
        sample_id="cal", site="context", step=None, tensor=torch.tensor([127000.0])
    )
    assert lower(collector=collector, strategy="global").module.activation_scale == 3
    assert lower(collector=collector, strategy="site").module.activation_scale == 1
