from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_real_minddrive_physical_abi.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_real_minddrive_physical_abi",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    name = "vision_stem"
    export = tmp_path / f"{name}.pt2e"
    export.write_bytes(b"captured-export")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / f"{name}.so"
    artifact.write_bytes(b"compiled-artifact")
    compile_report_root = tmp_path / "reports"
    compile_report_root.mkdir()
    capture_contract = {
        "shape": [1, 2, 3],
        "dtype": "f32",
        "device": "cuda:0",
        "strides": [6, 3, 1],
        "storage_offset": 0,
        "contiguous": True,
        "layout": "contiguous",
    }
    compile_contract = {
        **capture_contract,
        "dtype": "torch.float32",
        "device": "cuda:0",
    }
    compile_report = {
        "passed": True,
        "export": {"sha256": _sha256(export)},
        "artifact": {"sha256": _sha256(artifact)},
        "physical_tensor_abi": {
            "required": True,
            "inputs": [compile_contract],
            "reference_outputs": [compile_contract],
        },
    }
    (compile_report_root / f"{name}.json").write_text(
        json.dumps(compile_report),
        encoding="utf-8",
    )
    capture = {
        "name": name,
        "artifact": str(export),
        "artifact_sha256": _sha256(export),
        "strict_export": True,
        "effect_audit_passed": True,
        "inputs": [capture_contract],
        "outputs": [capture_contract],
    }
    return capture, compile_report_root, artifact_root


def test_physical_abi_audit_binds_exact_export_and_artifact(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    capture, reports, artifacts = _fixture(tmp_path)

    evidence = module._audit_region(
        capture,
        compile_report_root=reports,
        artifact_root=artifacts,
    )

    assert evidence["passed"]
    assert all(evidence["checks"].values())


def test_physical_abi_audit_rejects_reused_artifact(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    capture, reports, artifacts = _fixture(tmp_path)
    (artifacts / "vision_stem.so").write_bytes(b"artifact-from-old-capture")

    evidence = module._audit_region(
        capture,
        compile_report_root=reports,
        artifact_root=artifacts,
    )

    assert not evidence["passed"]
    assert not evidence["checks"]["artifact_hash_matches_compile_report"]


def test_physical_abi_audit_rejects_stride_contract_drift(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    capture, reports, artifacts = _fixture(tmp_path)
    capture["inputs"][0]["strides"] = [1, 1, 1]
    capture["inputs"][0]["contiguous"] = False
    capture["inputs"][0]["layout"] = "strided"

    evidence = module._audit_region(
        capture,
        compile_report_root=reports,
        artifact_root=artifacts,
    )

    assert not evidence["passed"]
    assert not evidence["checks"]["input_contract_matches"]
    assert not evidence["checks"]["capture_inputs_contiguous"]
