import hashlib
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
)
from vlaforge.analysis.precision_probe import (
    PrecisionProbe,
    ProbeRegion,
    metadata_digest,
    owned_tensor_snapshot,
    tensor_bundle_digest,
    tensor_identity,
)
from vlaforge.numerical_context import snapshot


class Prefix(torch.nn.Module):
    def forward(self, value):
        return value * 2


class Step(torch.nn.Module):
    def forward(self, value, noise):
        owned = value.clone()
        owned.add_(noise)
        return owned, owned.square()


def saved(tmp_path, name, module, args, stage):
    path = tmp_path / (name + ".pt2")
    torch.export.save(torch.export.export(module, args, strict=True), path)
    return ProbeRegion(name, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), stage)


def case(tmp_path, shape=(2, 3)):
    value = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    noise = torch.ones_like(value) / 2
    regions = (saved(tmp_path, "prefix", Prefix(), (value,), "context"),
               saved(tmp_path, "step", Step(), (value, noise), "iteration"))
    sites = (CalibrationSite("context", "prefix", "mul", regions[0].artifact_sha256, "context", False),
             CalibrationSite("before-inplace", "step", "clone", regions[1].artifact_sha256, "iteration", True))
    profile = {"schedule": {"N": 3, "scale": .5}, "input_shape": list(shape)}
    probe = PrecisionProbe(regions=regions, sites=sites, step_keys=("index0", "index1", "index2"),
                           profile=profile, numerical_context=snapshot())
    inputs = {"observation": value, "noise": noise}
    sample = CalibrationSample("calibration", "episode0", tensor_bundle_digest({"observation": value}),
                               tensor_bundle_digest({"noise": noise}))
    held_inputs = {"observation": value + 5, "noise": noise}
    held = CalibrationSample("held", "episode1", tensor_bundle_digest({"observation": value + 5}),
                             tensor_bundle_digest({"noise": noise}))
    collector = PrecisionCalibration(profile_sha256=metadata_digest(profile),
                                      numerical_context_sha256=probe.numerical_context_sha256,
                                      step_keys=("index0", "index1", "index2"), sites=sites,
                                      calibration_samples=(sample,), held_out_samples=(held,))
    return probe, collector, sample, inputs, held, held_inputs


def execute(regions, inputs, count=3):
    value = regions["prefix"](inputs["observation"])
    for _ in range(count):
        value, auxiliary = regions["step"](value, inputs["noise"])
    return value, auxiliary


def reference(inputs):
    return execute({"prefix": Prefix(), "step": Step()}, inputs)


def observe(probe, sample, inputs, **overrides):
    kwargs = {"sample": sample, "inputs": inputs, "noise_names": ("noise",),
              "expected_outputs": reference(inputs), "invoke": lambda regions: execute(regions, inputs)}
    kwargs.update(overrides)
    return probe.observe_sample(**kwargs)


@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 2)])
def test_real_export_loop_retains_full_outputs_and_owned_pre_inplace_snapshots(tmp_path, shape):
    probe, collector, sample, inputs, _, _ = case(tmp_path, shape)
    original = {name: value.clone() for name, value in inputs.items()}
    rng = torch.get_rng_state().clone()
    result = observe(probe, sample, inputs)
    report = result.report()
    assert report["complete_output_bitwise_equal"] and report["implicit_rng_unchanged"]
    assert [item["region"] for item in report["calls"]] == ["prefix", "step", "step", "step"]
    observed = [item for item in report["observations"] if item["site"] == "before-inplace"]
    assert [item["statistics"]["minimum"] for item in observed] == [2., 2.5, 3.]
    assert [item["step"] for item in observed] == [0, 1, 2]
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, original[name]) for name, value in inputs.items())
    updated = result.publish(collector)
    assert len(updated.report()["observations"]) == 4
    with pytest.raises(ValueError, match="incomplete"): collector.report()
    assert not updated.fit("step").to_data()["real_low_precision_kernel_verified"]


def test_held_out_runs_can_be_validated_but_never_published(tmp_path):
    probe, collector, _, _, held, held_inputs = case(tmp_path)
    result = observe(probe, held, held_inputs)
    assert result.report()["complete_output_bitwise_equal"]
    with pytest.raises(ValueError, match="held-out"): result.publish(collector)
    with pytest.raises(ValueError, match="incomplete"): collector.report()


@pytest.mark.parametrize("mutation", ["few", "many", "output", "structure", "input", "noise", "profile", "exception", "rng"])
def test_failed_sample_cannot_publish_or_poison_next_run(tmp_path, mutation):
    probe, _, sample, inputs, _, _ = case(tmp_path)
    options = {}
    if mutation in ("few", "many"):
        options["invoke"] = lambda regions: execute(regions, inputs, 2 if mutation == "few" else 4)
    if mutation == "output": options["expected_outputs"] = tuple(item + 1 for item in reference(inputs))
    if mutation == "structure": options["expected_outputs"] = list(reference(inputs))
    if mutation == "input": sample = replace(sample, input_sha256="0" * 64)
    if mutation == "noise": sample = replace(sample, noise_sha256="0" * 64)
    if mutation == "profile":
        options["invoke"] = lambda regions: regions["prefix"](inputs["observation"].flatten())
    if mutation == "exception":
        def fail(regions):
            regions["prefix"](inputs["observation"])
            raise RuntimeError("actual invocation failed")
        options["invoke"] = fail
    if mutation == "rng":
        def random(regions):
            value = execute(regions, inputs)
            torch.rand(1)
            return value
        options["invoke"] = random
    rng = torch.get_rng_state().clone()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            observe(probe, sample, inputs, **options)
    finally:
        torch.set_rng_state(rng)
    sample = CalibrationSample("calibration", "episode0", tensor_bundle_digest({"observation": inputs["observation"]}),
                               tensor_bundle_digest({"noise": inputs["noise"]}))
    assert observe(probe, sample, inputs).report()["complete_output_bitwise_equal"]


def test_callback_input_mutation_is_detected_even_with_matching_final_outputs(tmp_path):
    probe, _, sample, inputs, _, _ = case(tmp_path)
    expected = reference(inputs)
    def mutate(regions):
        result = execute(regions, inputs)
        inputs["observation"].add_(1)
        return result
    with pytest.raises(ValueError, match="invocation inputs changed"):
        observe(probe, sample, inputs, invoke=mutate, expected_outputs=expected)


@pytest.mark.parametrize("kind", ["rng", "input", "buffer"])
def test_real_export_effect_audit_rejects_hidden_effects(tmp_path, kind):
    class Effect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.ones(2))
        def forward(self, value):
            if kind == "rng": return value + torch.rand_like(value)
            if kind == "input": return value.add_(1)
            self.state.add_(1)
            return value + self.state
    region = saved(tmp_path, "effect", Effect(), (torch.ones(2),), "context")
    site = CalibrationSite("selected", "effect", "add", region.artifact_sha256, "context", True)
    with pytest.raises(ValueError, match="hidden effects"):
        PrecisionProbe(regions=(region,), sites=(site,), step_keys=("step0",),
                       profile={"N": 1}, numerical_context=snapshot())


@pytest.mark.parametrize("mutation", ["file", "sitehash", "missing", "nested", "duplicate", "stepkeys", "legacy_context"])
def test_invalid_artifact_selection_and_context_rejected_before_execution(tmp_path, mutation):
    probe, _, _, _, _, _ = case(tmp_path)
    regions, sites, steps = probe._regions, probe._sites, probe._step_keys
    context = snapshot()
    if mutation == "file": regions = (replace(regions[0], artifact_sha256="0" * 64), regions[1])
    if mutation == "sitehash": sites = (replace(sites[0], artifact_sha256="0" * 64), sites[1])
    if mutation == "missing": sites = (replace(sites[0], node="missing"), sites[1])
    if mutation == "nested": sites = (replace(sites[0], node="subgraph.mul"), sites[1])
    if mutation == "duplicate": sites = (*sites, sites[0])
    if mutation == "stepkeys": steps = ("index0", "index0")
    if mutation == "legacy_context": context = None
    with pytest.raises(ValueError):
        PrecisionProbe(regions=regions, sites=sites, step_keys=steps,
                       profile={"N": 3}, numerical_context=context)


def test_publish_is_atomic_on_duplicate_or_changed_declarations(tmp_path):
    probe, collector, sample, inputs, _, _ = case(tmp_path)
    result = observe(probe, sample, inputs)
    updated = result.publish(collector)
    before = updated.report()
    with pytest.raises(ValueError, match="duplicate"): result.publish(updated)
    assert updated.report() == before
    collector.profile_sha256 = "0" * 64
    with pytest.raises(ValueError, match="declarations differ"): result.publish(collector)
    report = result.report()
    report["observations"][0]["statistics"]["maximum"] = 0
    assert result.report()["observations"][0]["statistics"]["maximum"] != 0
    retained = result.owned_snapshot("before-inplace", 0)
    retained.add_(100)
    assert result.owned_snapshot("before-inplace", 0).min().item() == 2
    with pytest.raises(ValueError, match="unknown"): result.owned_snapshot("before-inplace", 9)


def test_numerical_policy_change_rejected_without_implicit_setter(tmp_path):
    probe, _, sample, inputs, _, _ = case(tmp_path)
    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high" if before != "high" else "highest")
        changed = torch.get_float32_matmul_precision()
        with pytest.raises(ValueError, match="numerical context mismatch"):
            observe(probe, sample, inputs)
        assert torch.get_float32_matmul_precision() == changed
    finally:
        torch.set_float32_matmul_precision(before)


@pytest.mark.parametrize("mutation", ["version", "data-add", "data-replace"])
def test_parameterized_norm_gemm_keeps_all_outputs_and_rejects_final_state_write(tmp_path, mutation):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.eye(3))
        def forward(self, value, noise):
            norm = torch.nn.functional.layer_norm(value, (3,))
            return norm @ self.weight + noise, value.sum(dim=-1)
    layer = Layer()
    inputs = {"observation": torch.arange(6, dtype=torch.float32).reshape(2, 3),
              "noise": torch.ones((2, 3))}
    region = saved(tmp_path, "layer", layer, tuple(inputs.values()), "iteration")
    site = CalibrationSite("norm", "layer", "layer_norm", region.artifact_sha256, "iteration", True)
    probe = PrecisionProbe(regions=(region,), sites=(site,), step_keys=("actual-step0",),
                           profile={"N": 1}, numerical_context=snapshot())
    sample = CalibrationSample("cal", "episode0", tensor_bundle_digest({"observation": inputs["observation"]}),
                               tensor_bundle_digest({"noise": inputs["noise"]}))
    expected = layer(*inputs.values())
    run = lambda regions: regions["layer"](*inputs.values())
    result = probe.observe_sample(sample=sample, inputs=inputs, noise_names=("noise",),
                                  expected_outputs=expected, invoke=run)
    assert len(result.report()["outputs"]) == 2
    def mutate_after(regions):
        output = run(regions)
        value = next(probe._modules["layer"].buffers())
        if mutation == "version": value.add_(1)
        if mutation == "data-add": value.data.add_(1)
        if mutation == "data-replace": value.data = torch.zeros_like(value)
        return output
    with pytest.raises(ValueError, match="(state changed after|contents changed after)"):
        probe.observe_sample(sample=sample, inputs=inputs, noise_names=("noise",),
                             expected_outputs=expected, invoke=mutate_after)


def test_input_metadata_change_is_not_hidden_by_symmetric_values(tmp_path):
    probe, _, sample, inputs, _, _ = case(tmp_path, (2, 2))
    inputs["observation"].fill_(1)
    sample = replace(sample, input_sha256=tensor_bundle_digest({"observation": inputs["observation"]}))
    expected = reference(inputs)
    def transpose(regions):
        outputs = execute(regions, inputs)
        inputs["observation"].transpose_(0, 1)
        return outputs
    with pytest.raises(ValueError, match="invocation inputs changed"):
        observe(probe, sample, inputs, expected_outputs=expected, invoke=transpose)


def test_reference_alias_cannot_change_official_baseline_during_callback(tmp_path):
    probe, _, sample, inputs, _, _ = case(tmp_path)
    expected = reference(inputs)
    def replace_reference(regions):
        execute(regions, inputs)
        expected[0].add_(1)
        return expected
    with pytest.raises(ValueError, match="references changed"):
        observe(probe, sample, inputs, expected_outputs=expected, invoke=replace_reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64])
def test_singleton_nonunit_stride_has_owned_canonical_bytes(dtype):
    value = torch.arange(3200, dtype=dtype)[::1600][:1]
    assert value.is_contiguous() and value.stride() == (1600,)
    owned = owned_tensor_snapshot(value)
    assert owned.stride() == (1,) and torch.equal(owned, value)
    assert owned.data_ptr() != value.data_ptr()
    assert tensor_identity(value) == tensor_identity(owned)
