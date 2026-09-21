"""CPU evidence and ABI rejection gates, not real CogACT output evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools/build_real_cogact_torchscript.py"
_SPEC = importlib.util.spec_from_file_location("cogact_torchscript_tool", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_storage_equality_distinguishes_signed_zero():
    assert not tool.exact_metrics(np.array([0.0], np.float32), np.array([-0.0], np.float32))["bitwise_equal"]


def test_nonfinite_is_not_hidden_by_matching_bytes():
    result = tool.exact_metrics(np.array([np.nan], np.float32), np.array([np.nan], np.float32))
    assert result["bitwise_equal"] and not result["all_finite"]


def test_output_dtype_is_not_silently_cast():
    assert not tool.exact_metrics(np.ones((16, 7), np.float32), np.ones((16, 7), np.float64))["bitwise_equal"]


def archive(tmp_path):
    from vlaforge.deployment import torchscript_export

    folder = tmp_path / "artifacts"
    folder.mkdir()
    artifact = folder / "prefix.pt"
    artifact.write_bytes(b"cpu-evidence-validation-only")
    captured = {"export_files": {"prefix.pt2e": {"sha256": "a" * 64, "size": 123}}}
    record = {"status": "region_cases_passed", "exported_program": {"sha256": "a" * 64, "size_bytes": 123},
              "artifact": tool.identity(artifact), "validation": {
                  "status": "region_cases_passed", "backend_variant": "torchscript-aten/1",
                  "jit_optimization": False, "archive_device_policy": "preserve",
                  "numerical_provider_enforcement": False, "implementation_sha256": tool.digest(torchscript_export.__file__),
                  "artifact_sha256": tool.digest(artifact), "validation_cases": [{"bitwise_equal": True}],
                  "effect_audits": [{"passed": True}]}}
    return captured, record


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("backend_variant", "different"), ("jit_optimization", True),
    ("archive_device_policy", "cpu"), ("numerical_provider_enforcement", True),
    ("implementation_sha256", "b" * 64), ("artifact_sha256", "b" * 64),
    ("validation_cases", []), ("validation_cases", [{"bitwise_equal": False}]),
    ("effect_audits", []), ("effect_audits", [{"passed": False}]),
])
def test_archive_cannot_outlive_its_policy_or_evidence(tmp_path, field, value):
    captured, record = archive(tmp_path)
    record["validation"][field] = value
    tool.write(tmp_path / "artifacts/prefix.compile.json", record)
    with pytest.raises(ValueError, match="identity mismatch"):
        tool.verify_archive(tmp_path, "prefix", captured)


def test_archive_bytes_are_bound(tmp_path):
    captured, record = archive(tmp_path)
    tool.write(tmp_path / "artifacts/prefix.compile.json", record)
    tool.verify_archive(tmp_path, "prefix", captured)
    (tmp_path / "artifacts/prefix.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="identity mismatch"):
        tool.verify_archive(tmp_path, "prefix", captured)


def test_runner_preserves_all_output_dtypes_and_sizes():
    def port(name, dtype, shape):
        return SimpleNamespace(name=name, device="cuda:0", payload=SimpleNamespace(dtype=dtype, shape=shape))
    module = SimpleNamespace(inputs=[port("state", "uint8", (16,))],
        outputs=[port("normalized_action_chunk", "float32", (16, 7)),
                 port("native_action_chunk", "float64", (16, 7)), port("draws_consumed", "int64", (1,))])
    source = tool.runner_source(module)
    assert "@INPUTS@" not in source and "@OUTPUTS@" not in source
    assert '{"native_action_chunk",VLAFORGE_DTYPE_F64,{16,7},896u}' in source
    assert '{"normalized_action_chunk",VLAFORGE_DTYPE_F32,{16,7},448u}' in source
    assert "stamp.revision = run + 1u" in source
    assert "api->destroy(session)" in source


def test_capture_report_downgrade_fails_before_deserialization(tmp_path):
    (tmp_path / "report.json").write_text("{}")
    args = SimpleNamespace(capture=tmp_path, capture_report_sha256="a" * 64)
    with pytest.raises(ValueError, match="report bytes changed"):
        tool.verify_capture(args)


def test_owner_reset_is_before_any_session_or_tensor():
    source = _PATH.with_name("cogact_fresh_runner.cpp.in").read_text()
    assert source.index("!Cuda(cudaDeviceReset())") < source.index("std::vector<Buffer> buffers")
    assert source.index("!Cuda(cudaDeviceReset())") < source.index("vlaforge_model_session_create_from_bundle(")


def test_owner_reset_rejects_already_imported_torch(monkeypatch, tmp_path):
    import sys

    path = _PATH.with_name("cogact_gpu_monitor.py")
    spec = importlib.util.spec_from_file_location("cogact_gpu_monitor_test", path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setenv("COGACT_GPU_OWNER_FOLDER", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="precede Torch import"):
        monitor.child_handshake()


def test_autonomous_runner_declares_explicit_cpp_rng_contract():
    source = _PATH.with_name("cogact_fresh_runner.cpp.in").read_text()
    assert "vlaforge/backends/libtorch_rng.h" in source
    assert "VLAFORGE_RNG_SEEDS" in source
    assert "GenerateCogactRng" in source
    assert "provider.State()" in source
    assert "provider.Normal({1, 16, 7})" in source
    assert "provider.NormalLike(template_tensor)" in source
    assert "at::manual_seed" not in source
    assert "getDefaultCUDAGenerator" not in source
    assert "external RNG" not in source
    start = source.index("const auto start =")
    assert source.index("api->bind_tensor") < start
    assert source.index("!GenerateCogactRng(*rng, seed, buffers)", start) < source.index(
        "api->run(session)", start)
    assert source.index("const auto timed_end =", start) > source.index(
        "api->run(session)", start)


def monitor_module():
    path = _PATH.with_name("cogact_gpu_monitor.py")
    spec = importlib.util.spec_from_file_location("cogact_gpu_monitor_negative_test", path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    return monitor


def test_existing_owner_is_never_launched_over_or_signalled(monkeypatch, tmp_path):
    monitor = monitor_module()
    monkeypatch.setattr(monitor, "owners", lambda gpu: [{"pid": 777}])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    monkeypatch.setattr(monitor.os, "killpg", lambda *a: pytest.fail("must not signal existing owner"))
    with pytest.raises(RuntimeError, match="refusing worker launch"):
        monitor.run_monitored(["not-executed"], tmp_path / "monitor")


def test_bad_owner_registration_only_signals_our_process_group(monkeypatch, tmp_path):
    monitor = monitor_module()
    calls = iter([[], [{"pid": 777}]])
    monkeypatch.setattr(monitor, "owners", lambda gpu: next(calls))
    process = SimpleNamespace(pid=12, returncode=None)
    process.poll = lambda: process.returncode

    def wait(timeout=None):
        process.returncode = -15
        return process.returncode

    process.wait = wait

    def launch(*args, **kwargs):
        folder = Path(kwargs["env"]["COGACT_GPU_OWNER_FOLDER"])
        monitor.write(folder / "owner-registered-1.json", {"pid": 13, "ordinal": 0})
        return process

    signals = []
    monkeypatch.setattr(monitor.subprocess, "Popen", launch)
    monkeypatch.setattr(monitor.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="does not identify our child"):
        monitor.run_monitored(["fake-child"], tmp_path / "monitor")
    assert signals == [(12, monitor.signal.SIGTERM)]


@pytest.mark.parametrize("dtype", ("float32", "float64", "int64", "uint8", "bool"))
def test_full_contract_preserves_captured_dtype_spelling(tmp_path, dtype):
    from vlaforge.frontend.tensor_types import canonical_tensor_dtype
    from vlaforge.ir.program import Module

    artifact = tmp_path / "test.pt"
    artifact.write_bytes(b"cpu-contract-validation-only")
    value = {"name": "value", "type": {"kind": "tensor", "shape": [1], "dtype": dtype, "layout": "contiguous"},
             "device": "cuda:0", "dimensions": [{"static": 1}], "alignment": 1}
    evidence = {"region_name": "step", "inputs": [value], "outputs": [value], "graph_digest": "a" * 64,
                "effect_audit": {"hidden_mutation": False, "hidden_rng": False, "external_io": False}}
    ledger = []
    contract = tool.artifact_contract(Module("contract-only", (), (), (), (), ()), 0, evidence, artifact, "b" * 64, ledger=ledger)
    assert contract.inputs[0].type.dtype == canonical_tensor_dtype(dtype)
    assert bool(ledger) == (dtype != "bool")
    assert evidence["inputs"][0]["type"]["dtype"] == dtype


def test_reuse_requires_explicit_original_reports(tmp_path):
    args = SimpleNamespace(reuse_root=None)
    with pytest.raises(ValueError, match="explicit existing"):
        tool.adopt_existing(args, None, None)


def test_explicit_type_conversion_changes_only_ir_types_not_metadata():
    from vlaforge.ir.program import InputPort, Module
    from vlaforge.ir.types import TensorType

    original = Module("aliases", (InputPort("x", TensorType((2,), "float32"), input_id=0),), (), (), (), (),
                      metadata={"dtype": "float32", "literal": {"kind": "tensor", "dtype": "float64"}})
    ledger = []
    converted = tool.normalize_ir_tensor_types(original, ledger=ledger)
    assert converted.inputs[0].payload.dtype == "f32"
    assert original.inputs[0].payload.dtype == "float32"
    assert converted.metadata == original.metadata
    assert len(ledger) == 1 and ledger[0]["path"] == "module.inputs[0].payload"
