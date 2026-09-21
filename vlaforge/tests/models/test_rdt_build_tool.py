import importlib.util
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

from vlaforge.ir.program import InputPort, Module, OutputPort
from vlaforge.ir.types import TensorType

path = Path(__file__).resolve().parents[2] / "tools/build_real_rdt_fresh.py"
spec = importlib.util.spec_from_file_location("rdt_build_tool", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.mark.parametrize("mutation", [None, "missing_pre", "stale_pre", "missing_ledger"])
def test_aten_artifact_requires_matching_pre_aot_source_and_ledger(tmp_path, mutation):
    from vlaforge.deployment.aoti_export import backend_pass_records, backend_program_pass_records
    from vlaforge.deployment.aoti_package import package_pass_records
    from vlaforge.deployment.aoti_profile import aoti_configs

    package = tmp_path / "test.pt2"
    package.write_bytes(b"CPU metadata fixture, not an executable")
    configs = aoti_configs("aten-preserving")
    record = {
        "status": "passed", "target": "sm_90", "inductor_profile": "aten-preserving",
        "inductor_configs": configs, "backend_graph_passes": backend_pass_records(configs),
        "backend_package_audit": {"passes": package_pass_records(configs)},
        "backend_program_audit": {"passes": backend_program_pass_records(configs), "rewrites": []},
        "exported_program": {"sha256": "captured"},
        "artifact": {"sha256": tool.digest(package), "size_bytes": package.stat().st_size},
    }
    if mutation == "missing_pre":
        del record["backend_program_audit"]
    elif mutation == "stale_pre":
        record["backend_program_audit"]["passes"][0]["source_sha256"] = "previous-helper"
    elif mutation == "missing_ledger":
        del record["backend_program_audit"]["rewrites"]
    tool.write(tmp_path / "test.compile.json", record)
    captured = {"export": {"sha256": "captured"}}
    if mutation is None:
        assert tool.verify_artifact(tmp_path, "test", captured, "aten-preserving") == record
    else:
        with pytest.raises(ValueError, match="compiled artifact/profile/source mismatch"):
            tool.verify_artifact(tmp_path, "test", captured, "aten-preserving")


def test_metrics_do_not_merge_technical_and_paper_gates():
    reference = np.ones((1, 2, 3), dtype=np.float32)
    report = tool.fidelity(reference, reference + .005)
    assert report["gates"]["technical_tolerance"]
    assert not report["gates"]["paper_numeric_threshold"]
    assert not report["gates"]["full_paper_acceptance"]
    assert not report["gates"]["exact_values"]


def test_metrics_bind_the_actual_series_observation_identifier():
    values = np.ones((1, 2, 3), dtype=np.float32)
    report = tool.action_fidelity(values, values, np.array([1, 0, 1]), sample_id="episode5-step145-seed7")
    assert report["unified"]["sample_id"] == "episode5-step145-seed7"
    assert report["active"]["sample_id"] == "episode5-step145-seed7"


def test_undefined_zero_vector_cosine_not_paper_pass():
    values = np.zeros((1, 2, 3), dtype=np.float32)
    report = tool.fidelity(values, values)
    assert report["gates"]["exact_values"]
    assert report["metrics"]["cosine_similarity"] is None
    assert not report["gates"]["paper_numeric_threshold"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_cannot_be_tolerance_pass(value):
    with pytest.raises(ValueError):
        tool.fidelity(np.ones((1, 2, 3)), np.full((1, 2, 3), value))


def test_runner_uses_declared_bf16_size_and_input_metadata():
    module = Module(name="test", inputs=(InputPort("noise", TensorType((1, 64, 128), "bf16"), input_id=0, device="cuda:0"),),
                    outputs=(OutputPort("action_chunk", TensorType((1, 64, 128), "bf16"), output_id=0, group="actions", device="cuda:0"),),
                    states=(), regions=(), invocations=())
    source = tool.runner_source(module)
    assert '"noise", VLAFORGE_DTYPE_BF16, {1,64,128}, 16384u' in source
    assert "kOutputCount = 8192u" in source
    assert "@INPUTS@" not in source and "@OUTPUT_COUNT@" not in source
    assert "PYTHON" not in source


def test_runner_rejects_undeclared_output_precision():
    module = Module(name="test", inputs=(), states=(), regions=(), invocations=(),
                    outputs=(OutputPort("action_chunk", TensorType((1, 2, 3), "f32"), output_id=0, group="actions", device="cuda:0"),))
    with pytest.raises(ValueError, match="BF16"):
        tool.runner_source(module)


def test_extraction_retains_and_verifies_every_package_dependency(tmp_path):
    package = tmp_path / "model.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/data/aotinductor/model/wrapper.so", b"weights-and-machine-code")
        archive.writestr("model/data/aotinductor/model/kernel.cubin", b"cuda-code")
    destination = tmp_path / "unpacked"
    result = tool.unpack_package(package, destination)
    assert result["library"].endswith("wrapper.so")
    assert len(result["files"]) == 2
    assert tool.unpack_package(package, destination) == result
    sidecar = destination / "model/data/aotinductor/model/kernel.cubin"
    sidecar.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="dependency changed"):
        tool.unpack_package(package, destination)
    result["files"]["model/data/aotinductor/model/kernel.cubin"] = tool.file_identity(sidecar)
    tool.write(destination / "extraction.json", result)
    with pytest.raises(ValueError, match="original package"):
        tool.unpack_package(package, destination)


@pytest.mark.parametrize("name,mode", [("../escape", stat.S_IFREG), ("/absolute", stat.S_IFREG), ("link", stat.S_IFLNK)])
def test_extraction_rejects_unsafe_archive_members(tmp_path, name, mode):
    package = tmp_path / "model.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/wrapper.so", b"machine-code")
        member = zipfile.ZipInfo(name)
        member.external_attr = (mode | 0o600) << 16
        archive.writestr(member, b"outside")
    with pytest.raises(ValueError, match="unsafe archive"):
        tool.unpack_package(package, tmp_path / "unpacked")
    assert not (tmp_path / "escape").exists()


def test_extraction_rejects_unimplemented_external_constant_relocation(tmp_path):
    package = tmp_path / "model.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/data/aotinductor/model/wrapper.so", b"machine-code")
        archive.writestr("model/data/constants/weights.bin", b"weights")
    with pytest.raises(ValueError, match="constant-blob relocation"):
        tool.unpack_package(package, tmp_path / "unpacked")


@pytest.mark.parametrize("count", [1, 2])
def test_raw_callable_restores_region_output_structure(count):
    class Runner:
        def run(self, inputs):
            assert inputs == ["input"]
            return list(range(count))
    result = tool.raw_callable(Runner(), count)("input")
    assert result == (0 if count == 1 else (0, 1))


def test_raw_callable_rejects_changed_output_arity():
    class Runner:
        def run(self, inputs):
            return []
    with pytest.raises(ValueError, match="output arity"):
        tool.raw_callable(Runner(), 1)("input")


def test_paper_gate_cannot_be_diluted_by_inactive_action_dimensions():
    reference = np.zeros((1, 2, 128), np.float32)
    reference[..., :14] = 1
    candidate = reference.copy()
    candidate[..., :14] += .005
    mask = np.zeros((1, 1, 128), np.float32)
    mask[..., :14] = 1
    report = tool.action_fidelity(reference, candidate, mask)
    assert report["unified"]["gates"]["paper_numeric_threshold"]
    assert not report["active"]["gates"]["paper_numeric_threshold"]
    assert not report["gates"]["paper_numeric_threshold"]
    assert report["active"]["metrics"]["count"] == 28


@pytest.mark.parametrize("mask", [np.zeros(3), np.ones(2), np.array([0, .5, 1])])
def test_action_fidelity_rejects_invalid_embodiment_masks(mask):
    with pytest.raises(ValueError):
        tool.action_fidelity(np.ones((1, 2, 3)), np.ones((1, 2, 3)), mask)
