"""Typed compile/bootstrap fixture tests, not native pretrained-model evidence."""

import json
from copy import deepcopy

import pytest

from vlaforge.adapters.openpi.openpi_numerical import (
    binding_from_legacy_compile,
    binding_from_observed_compile,
    render_explicit_numerical_worker,
)
from vlaforge.deployment.numerical import RegionNumericalBinding
from vlaforge.numerical_context import snapshot


@pytest.fixture
def observed():
    torch = pytest.importorskip("torch")
    if not torch.__version__.startswith("2.10.0"):
        pytest.skip("complete v2 LibTorch provider domain requires actual Torch2.10")
    context = snapshot()
    archive = {"sha256": "1" * 64, "size_bytes": 3, "path": "/fixture/graph.pt2"}
    region = {"archive": archive, "capture_evidence": {"graph_digest": "2" * 64}}
    manifest = {
        "artifact": {"sha256": "3" * 64, "size_bytes": 4},
        "exported_program": {"sha256": archive["sha256"]},
        "target": "sm_90",
        "torch_version": str(torch.__version__),
        "inductor_profile": "fixture",
        "inductor_configs": {"fixture": True},
        "backend_program_audit": {"passes": [], "rewrites": []},
        "backend_graph_passes": [],
        "backend_graph_rewrites": [],
        "backend_package_audit": {"passes": []},
    }
    entry = {
        "region": "fixture",
        "status": "compiled-unvalidated",
        "export": archive,
        "observed_numerical_context_before": context.to_dict(),
        "observed_numerical_context_after": context.to_dict(),
    }
    build = {
        "profile": "fixture",
        "target": "sm_90",
        "configs": {"fixture": True},
        "source_files": {"fixture.py": {"sha256": "4" * 64}},
        "context_implementation": {"sha256": "5" * 64},
        "numerical_context": context.to_dict(),
    }
    return context, region, manifest, entry, build


def test_binding_retains_complete_actual_context_and_dual_pass_provenance(observed):
    binding = binding_from_observed_compile("fixture", *observed)
    assert len(binding.requirement.policy.values) == 24
    assert RegionNumericalBinding.from_dict(binding.to_dict()) == binding
    binding.require_runtime_deployable()
    assert "complete_python_context_before" in binding.compile_record.configuration_json
    assert "backend_program_audit" in binding.compile_record.configuration_json
    assert "backend_package_audit" in binding.compile_record.configuration_json


@pytest.mark.parametrize(
    "field", ["observed_numerical_context_before", "observed_numerical_context_after"]
)
def test_missing_compile_observation_never_defaults(observed, field):
    values = deepcopy(observed)
    values[3].pop(field)
    with pytest.raises(KeyError):
        binding_from_observed_compile("fixture", *values)


@pytest.mark.parametrize(
    "field",
    [
        "torch_version",
        "target",
        "exported_program",
        "inductor_profile",
        "inductor_configs",
    ],
)
def test_binding_rejects_mismatched_compiler_provenance(observed, field):
    values = deepcopy(observed)
    values[2][field] = {"sha256": "9" * 64} if field == "exported_program" else "wrong"
    with pytest.raises(ValueError, match="provenance"):
        binding_from_observed_compile("fixture", *values)


def test_complete_compile_context_drift_is_rejected(observed):
    values = deepcopy(observed)
    before = values[3]["observed_numerical_context_before"]
    before["cudnn_benchmark"] = not before["cudnn_benchmark"]
    with pytest.raises(ValueError, match="contexts"):
        binding_from_observed_compile("fixture", *values)


def test_partial_legacy_split_k_observation_is_rejected(observed):
    values = deepcopy(observed)
    values[3]["observed_numerical_context_after"].pop(
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k"
    )
    with pytest.raises(ValueError):
        binding_from_observed_compile("fixture", *values)


def test_legacy_build_report_context_binding_keeps_actual_policy(observed):
    context, region, manifest, entry, build = deepcopy(observed)
    for key in ("observed_numerical_context_before", "observed_numerical_context_after"):
        entry.pop(key)
    build["caller_policy_restoration_verified"] = True
    build["caller_context_after_restore"] = context.to_dict()
    binding = binding_from_legacy_compile("fixture", context, region, manifest, entry, build)
    assert len(binding.requirement.policy.values) == 24
    assert RegionNumericalBinding.from_dict(binding.to_dict()) == binding
    binding.require_runtime_deployable()
    assert "complete_python_context" in binding.compile_record.configuration_json


def test_legacy_binding_requires_restoration_evidence(observed):
    context, region, manifest, entry, build = deepcopy(observed)
    for key in ("observed_numerical_context_before", "observed_numerical_context_after"):
        entry.pop(key)
    with pytest.raises(ValueError, match="context|restoration"):
        binding_from_legacy_compile("fixture", context, region, manifest, entry, build)


def test_legacy_restored_caller_context_may_differ_from_compile_context(observed):
    context, region, manifest, entry, build = deepcopy(observed)
    for key in ("observed_numerical_context_before", "observed_numerical_context_after"):
        entry.pop(key)
    build["caller_policy_restoration_verified"] = True
    caller = deepcopy(context.to_dict())
    caller["cudnn_benchmark"] = True
    build["caller_context_after_restore"] = caller
    binding = binding_from_legacy_compile("fixture", context, region, manifest, entry, build)
    configuration = json.loads(binding.compile_record.configuration_json)
    assert configuration["complete_python_context"] == context.to_dict()


def test_explicit_worker_uses_public_initializer_not_adapter_setters(observed):
    binding = binding_from_observed_compile("fixture", *observed)
    rendered = render_explicit_numerical_worker(
        "int main(int argc, char** argv) { return 0; }",
        [binding],
        acknowledge_exclusive_process=True,
        acknowledge_calling_thread=True,
    )
    assert "vlaforge_libtorch_numerical_initialize_worker" in rendered
    assert (
        "const auto initialized = vlaforge_initialize_numerical_worker();" in rendered
    )
    assert "return vlaforge_explicit_worker_body(argc, argv);" in rendered
    assert "setFloat32MatmulPrecision" not in rendered
    assert "setAllowTF32" not in rendered


@pytest.mark.parametrize(
    "ack_process,ack_thread", [(False, True), (True, False), (1, True), (True, 1)]
)
def test_worker_wrapper_requires_two_actual_boolean_acknowledgements(
    observed, ack_process, ack_thread
):
    binding = binding_from_observed_compile("fixture", *observed)
    with pytest.raises(ValueError):
        render_explicit_numerical_worker(
            "int main(int argc, char** argv) { return 0; }",
            [binding],
            acknowledge_exclusive_process=ack_process,
            acknowledge_calling_thread=ack_thread,
        )


def test_worker_wrapper_rejects_an_ambiguous_entrypoint(observed):
    binding = binding_from_observed_compile("fixture", *observed)
    with pytest.raises(ValueError, match="signature"):
        render_explicit_numerical_worker(
            "int main() { return 0; }",
            [binding],
            acknowledge_exclusive_process=True,
            acknowledge_calling_thread=True,
        )
