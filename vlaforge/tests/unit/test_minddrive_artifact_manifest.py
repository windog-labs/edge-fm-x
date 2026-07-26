from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_real_minddrive_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_real_minddrive_artifacts",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    module: ModuleType,
) -> tuple[tuple[Path, ...], Path, Path]:
    captures = tmp_path / "captures"
    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    (reports / "vision").mkdir(parents=True)
    captures.mkdir()
    artifacts.mkdir()
    contract = {
        "shape": [1, 2],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "strides": [2, 1],
        "storage_offset": 0,
        "contiguous": True,
        "layout": "contiguous",
    }
    direct_contract = {**contract, "dtype": "f32"}
    vision_records = []
    map_records = []
    logical_records = {}
    for name in sorted(module._EXPECTED_REGIONS):
        export = captures / f"{name}.pt2e"
        export.write_bytes(f"export:{name}".encode())
        artifact = artifacts / f"{name}.so"
        artifact.write_bytes(f"artifact:{name}".encode())
        report_path = module._compile_report_path(reports, name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema": "vlaforge.minddrive_aoti24_compile/1",
                    "passed": True,
                    "region": name,
                    "inductor_profile": "default",
                    "export": {"sha256": _sha256(export)},
                    "artifact": {"sha256": _sha256(artifact)},
                    "physical_tensor_abi": {
                        "required": True,
                        "inputs": [contract],
                        "reference_outputs": [contract],
                    },
                }
            ),
            encoding="utf-8",
        )
        if name in module._VISION_REGIONS:
            vision_records.append(
                {
                    "name": name,
                    "artifact": str(export),
                    "artifact_sha256": _sha256(export),
                    "inputs": [direct_contract],
                    "outputs": [direct_contract],
                }
            )
        elif name in module._MAP_REGIONS:
            map_records.append(
                {
                    "name": name,
                    "artifact": str(export),
                    "artifact_sha256": _sha256(export),
                    "inputs": [direct_contract],
                    "outputs": [direct_contract],
                }
            )
        else:
            logical_records[name] = {
                "artifact": {
                    "path": str(export),
                    "sha256": _sha256(export),
                }
            }
    capture_reports = []
    for filename, regions in (
        ("vision.json", vision_records),
        ("map.json", map_records),
        ("logical.json", logical_records),
    ):
        path = captures / filename
        path.write_text(
            json.dumps(
                {
                    "schema": f"fixture.{filename}/1",
                    "passed": True,
                    "regions": regions,
                }
            ),
            encoding="utf-8",
        )
        capture_reports.append(path)
    return tuple(capture_reports), reports, artifacts


def test_complete_manifest_binds_exact_64_region_set(
    tmp_path: Path,
) -> None:
    module = _load_module()
    capture_reports, reports, artifacts = _fixture(tmp_path, module)

    manifest = module.build_manifest(
        capture_reports=capture_reports,
        compile_report_root=reports,
        artifact_root=artifacts,
    )

    assert manifest["passed"]
    assert len(manifest["regions"]) == 64
    assert not manifest["unexpected_artifacts"]
    assert all(item["passed"] for item in manifest["regions"])


def test_complete_manifest_rejects_stale_unreferenced_artifact(
    tmp_path: Path,
) -> None:
    module = _load_module()
    capture_reports, reports, artifacts = _fixture(tmp_path, module)
    (artifacts / "map_encoder.so").write_bytes(b"stale logical artifact")

    manifest = module.build_manifest(
        capture_reports=capture_reports,
        compile_report_root=reports,
        artifact_root=artifacts,
    )

    assert not manifest["passed"]
    assert manifest["unexpected_artifacts"] == ["map_encoder"]
    assert not manifest["checks"]["artifact_region_set_exact"]
