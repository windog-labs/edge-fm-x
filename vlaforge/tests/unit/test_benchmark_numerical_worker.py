"""Explicit numerical worker preparation; native provider tests are separate."""

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.deployment.libtorch_numerical import NAMESPACE, REDUCTION_API
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.numerical_context import NumericalContext
from vlaforge.validation.session_benchmark import numerical_worker_bootstrap


def tool():
    spec = importlib.util.spec_from_file_location(
        "numerical_benchmark_tool", Path(__file__).resolve().parents[2] / "tools/benchmark_session.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def option():
    return {"mode": "libtorch-explicit-exclusive-process/1",
            "acknowledge_exclusive_process": True, "acknowledge_calling_thread": True}


def binding(name="region", precision="highest", backend="torchscript", namespace=NAMESPACE):
    excluded = {"schema", "torch_version", "reduction_api", "float32_matmul_precision",
                "autocast_cpu_dtype", "autocast_cuda_dtype"}
    values = {key: False for key in NumericalContext.__dataclass_fields__ if key not in excluded}
    values.update(autocast_cpu_dtype="bfloat16", autocast_cuda_dtype="float16",
                  float32_matmul_precision=precision, cuda_matmul_allow_tf32=precision != "highest",
                  torch_release="2.10.0", reduction_api=REDUCTION_API)
    policy = NumericalPolicy(namespace, tuple(sorted(values.items())))
    record = NumericalCompileRecord(backend, "cpu", "test/1", "1" * 64, "2" * 64, "3" * 64, 1,
                                     policy, policy, policy, '{"fixture":true}')
    return RegionNumericalBinding(name, NumericalRequirement(policy, "same-precision", policy.digest(), record.digest()),
                                  record, PROVIDER_REQUIRED)


def test_legacy_default_has_no_setter_or_source():
    assert numerical_worker_bootstrap({}) is None
    source, call, metadata = tool().numerical_worker_initialization({}, (None, None))
    assert source == call == ""
    assert metadata == {"mode": "none", "configured": False}


@pytest.mark.parametrize("value", [None, False, True, "auto", {}, {"mode": "off"}])
def test_present_invalid_option_is_never_a_default(value):
    with pytest.raises(ValueError):
        numerical_worker_bootstrap({"numerical_worker_bootstrap": value})


@pytest.mark.parametrize("key", ["acknowledge_exclusive_process", "acknowledge_calling_thread"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_exact_explicit_acknowledgements(key, value):
    selected = option()
    selected[key] = value
    with pytest.raises(ValueError):
        numerical_worker_bootstrap({"numerical_worker_bootstrap": selected})


def test_unknown_option_field_rejected():
    selected = dict(option(), implicit_setter=True)
    with pytest.raises(ValueError):
        numerical_worker_bootstrap({"numerical_worker_bootstrap": selected})


def test_required_provider_never_prepares_without_opt_in():
    with pytest.raises(ValueError, match="explicit"):
        tool().numerical_worker_initialization({}, (binding(),))


@pytest.mark.parametrize("values", [(), (None,), (None, "binding")])
def test_opt_in_requires_complete_typed_bindings(values):
    with pytest.raises(ValueError):
        tool().numerical_worker_initialization({"numerical_worker_bootstrap": option()}, values)


def test_partial_policy_coverage_rejected():
    with pytest.raises(ValueError, match="every Region"):
        tool().numerical_worker_initialization({"numerical_worker_bootstrap": option()}, (binding(), None))


def test_conflicting_policy_and_unknown_provider_rejected():
    selected = {"numerical_worker_bootstrap": option()}
    for values in ((binding(), binding("other", "medium")),
                   (binding(backend="shared_plugin"),),
                   (binding(namespace="other.namespace/1"),)):
        with pytest.raises(ValueError):
            tool().numerical_worker_initialization(selected, values)


def test_unimplemented_provider_cannot_be_promoted():
    source = replace(binding(), runtime_enforcement="unimplemented")
    with pytest.raises(ValueError, match="actual provider"):
        tool().numerical_worker_initialization({"numerical_worker_bootstrap": option()}, (source,))


def test_unsupported_native_policy_version_rejected():
    original = binding()
    values = dict(original.requirement.policy.values)
    values["torch_release"] = "2.9.0"
    policy = NumericalPolicy(NAMESPACE, tuple(sorted(values.items())))
    record = replace(original.compile_record, reference_policy=policy, requested_compile_policy=policy,
                     observed_compile_policy=policy)
    requirement = NumericalRequirement(policy, "same-precision", policy.digest(), record.digest())
    changed = RegionNumericalBinding("region", requirement, record, PROVIDER_REQUIRED)
    with pytest.raises(ValueError):
        tool().numerical_worker_initialization({"numerical_worker_bootstrap": option()}, (changed,))


def test_prepare_rejects_missing_opt_in_before_writing_or_building(tmp_path, monkeypatch):
    from vlaforge.validation.session_benchmark import BOUNDARY, SCHEMA

    api = tool()
    protocol = {"schema": SCHEMA, "boundary": BOUNDARY, "warmup": 128, "measured": 1024,
                "processes": 5, "policies": ["off"], "bundles": {"off": "fixture"},
                "quality_gate": "failed", "samples": [{}] * 16, "evidence": ["fixture.json"]}
    source = tmp_path / "protocol.json"
    api.write(source, protocol)
    manifest = SimpleNamespace(verify_files=lambda path: None,
                               region_artifacts=[SimpleNamespace(numerical_binding=binding())])
    monkeypatch.setattr(api, "load_bundle_manifest", lambda path: manifest)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="explicit"):
        api.prepare(SimpleNamespace(protocol=source, output=destination))
    assert not destination.exists()


def test_public_initializer_same_thread_after_handshake_before_allocations():
    api = tool()
    selected = {"numerical_worker_bootstrap": option()}
    source, call, metadata = api.numerical_worker_initialization(selected, (binding(), binding("other", backend="aoti")))
    assert source.count("constexpr VLAForgeNumericalRequirementView") == 1
    assert call.count("vlaforge_initialize_numerical_worker()") == 1
    assert "NUMERICAL_WORKER_BOOTSTRAP_OK," in call
    assert metadata["mode"] == option()["mode"]
    assert metadata["configured"] and metadata["boundary"] == "after-owner-handshake-before-tensor-and-session"
    assert len(metadata["binding_digests"]) == 2
    template = Path(api.__file__).with_name("session_benchmark_runner.cpp.in").read_text()
    assert template.count("// @NUMERICAL_WORKER_BOOTSTRAP@") == 1
    rendered = source + template.replace("// @NUMERICAL_WORKER_BOOTSTRAP@", call)
    assert rendered.index("cudaDeviceReset()") < rendered.index("NUMERICAL_WORKER_BOOTSTRAP_OK,")
    assert rendered.index("NUMERICAL_WORKER_BOOTSTRAP_OK,") < rendered.index("std::vector<Buffer> buffers")
    assert rendered.index("NUMERICAL_WORKER_BOOTSTRAP_OK,") < rendered.index("vlaforge_model_session_create_from_bundle")
    assert option() == selected["numerical_worker_bootstrap"]


def test_actual_success_marker_is_required_and_not_duplicated(tmp_path):
    api = tool()
    _, _, metadata = api.numerical_worker_initialization({"numerical_worker_bootstrap": option()}, (binding(),))
    api.write(tmp_path / "prepared.json", {"numerical_worker_initialization": {"off": metadata}})
    api.write(tmp_path / "source/numerical-workers.json", {"off": metadata})
    folder = tmp_path / "process"
    folder.mkdir()
    marker = "NUMERICAL_WORKER_BOOTSTRAP_OK," + metadata["policy_sha256"]
    (folder / "stderr.log").write_text(marker + "\n")
    result = api.verify_numerical_worker_execution(tmp_path, folder, "off")
    assert result["initialization_success_marker_observed"]
    for text in ("", marker + "\n" + marker + "\n", marker[:-1] + "0\n"):
        (folder / "stderr.log").write_text(text)
        with pytest.raises(ValueError):
            api.verify_numerical_worker_execution(tmp_path, folder, "off")


def test_prepared_metadata_cannot_disable_frozen_initialization(tmp_path):
    api = tool()
    _, _, metadata = api.numerical_worker_initialization({"numerical_worker_bootstrap": option()}, (binding(),))
    api.write(tmp_path / "prepared.json", {})
    api.write(tmp_path / "source/numerical-workers.json", {"off": metadata})
    with pytest.raises(ValueError, match="frozen source"):
        api.verify_numerical_worker_execution(tmp_path, tmp_path, "off")


def test_old_prepared_metadata_has_no_implicit_initialization(tmp_path):
    api = tool()
    api.write(tmp_path / "prepared.json", {})
    assert api.verify_numerical_worker_execution(tmp_path, tmp_path, "off") == {"mode": "none", "configured": False}


def test_option_returns_owned_metadata():
    protocol = {"numerical_worker_bootstrap": option()}
    before = copy.deepcopy(protocol)
    normalized = numerical_worker_bootstrap(protocol)
    normalized["mode"] = "changed"
    assert protocol == before
