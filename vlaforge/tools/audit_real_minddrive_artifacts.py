#!/usr/bin/env python3
"""Build an immutable, complete MindDrive AOTI artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_LOGICAL_REGIONS = {
    "position_encoder",
    "detection_encoder",
    "detection_decoder",
    "decision_expert",
    "action_expert",
    "trajectory_decoder",
}
_VISION_REGIONS = {
    "vision_stem",
    "vision_finish",
    *(
        f"vision_block_{index:02d}_{part}"
        for index in range(24)
        for part in ("pre", "post")
    ),
}
_MAP_REGIONS = {
    "map_front",
    "map_finish",
    *(f"map_decoder_layer_{index:02d}" for index in range(6)),
}
_EXPECTED_REGIONS = _LOGICAL_REGIONS | _VISION_REGIONS | _MAP_REGIONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _capture_artifacts(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = _load_json(path)
    if not bool(report.get("passed")):
        raise ValueError(f"capture report did not pass: {path}")
    schema = str(report.get("schema", ""))
    records: dict[str, dict[str, Any]] = {}
    regions = report.get("regions")
    if isinstance(regions, list):
        for item in regions:
            name = str(item["name"])
            records[name] = {
                "path": str(Path(str(item["artifact"])).resolve()),
                "sha256": str(item["artifact_sha256"]),
                "physical_inputs": item.get("inputs"),
                "physical_outputs": item.get("outputs"),
            }
    elif isinstance(regions, dict):
        for name, item in regions.items():
            artifact = item["artifact"]
            records[str(name)] = {
                "path": str(Path(str(artifact["path"])).resolve()),
                "sha256": str(artifact["sha256"]),
                "physical_inputs": None,
                "physical_outputs": None,
            }
    elif isinstance(report.get("artifact"), dict):
        strict_export = report.get("strict_export")
        if not isinstance(strict_export, dict):
            raise ValueError(f"capture report has no Region identity: {path}")
        artifact = report["artifact"]
        name = str(strict_export["region_name"])
        records[name] = {
            "path": str(Path(str(artifact["path"])).resolve()),
            "sha256": str(artifact["sha256"]),
            "physical_inputs": None,
            "physical_outputs": None,
        }
    else:
        raise ValueError(f"unsupported capture report structure: {path}")
    return records, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema": schema,
        "passed": True,
    }


def _all_contiguous(values: Any) -> bool:
    return (
        isinstance(values, list)
        and all(
            isinstance(value, dict)
            and bool(value.get("contiguous"))
            and value.get("layout") == "contiguous"
            for value in values
        )
    )


def _compile_report_path(root: Path, name: str) -> Path:
    if name in _VISION_REGIONS:
        return root / "vision" / f"{name}.json"
    return root / f"{name}.json"


def _audit_region(
    name: str,
    *,
    capture: dict[str, Any],
    compile_report_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    export_path = Path(str(capture["path"])).resolve()
    compile_path = _compile_report_path(compile_report_root, name).resolve()
    artifact_path = (artifact_root / f"{name}.so").resolve()
    if not export_path.is_file():
        raise FileNotFoundError(f"{name}: capture export is missing")
    if not compile_path.is_file():
        raise FileNotFoundError(f"{name}: compile report is missing")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{name}: artifact is missing")
    compile_report = _load_json(compile_path)
    physical = compile_report.get("physical_tensor_abi")
    if not isinstance(physical, dict):
        physical = {}
    checks = {
        "capture_export_hash_matches": (
            _sha256(export_path) == capture["sha256"]
        ),
        "compile_passed": bool(compile_report.get("passed")),
        "compile_region_matches": compile_report.get("region") == name,
        "compile_export_hash_matches_capture": (
            compile_report.get("export", {}).get("sha256")
            == capture["sha256"]
        ),
        "physical_abi_required": physical.get("required") is True,
        "compiled_inputs_contiguous": _all_contiguous(
            physical.get("inputs")
        ),
        "compiled_outputs_contiguous": _all_contiguous(
            physical.get("reference_outputs")
        ),
        "artifact_hash_matches_compile_report": (
            _sha256(artifact_path)
            == compile_report.get("artifact", {}).get("sha256")
        ),
        "artifact_is_not_hardlinked": artifact_path.stat().st_nlink == 1,
        "compile_report_is_not_hardlinked": compile_path.stat().st_nlink == 1,
    }
    if capture["physical_inputs"] is not None:
        checks["capture_inputs_contiguous"] = _all_contiguous(
            capture["physical_inputs"]
        )
        checks["capture_outputs_contiguous"] = _all_contiguous(
            capture["physical_outputs"]
        )
    return {
        "name": name,
        "logical_region": (
            "vision_encoder"
            if name in _VISION_REGIONS
            else "map_encoder"
            if name in _MAP_REGIONS
            else name
        ),
        "physical_partition": (
            name in _VISION_REGIONS or name in _MAP_REGIONS
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "capture_export": {
            "path": str(export_path),
            "sha256": capture["sha256"],
            "size_bytes": export_path.stat().st_size,
        },
        "compile_report": {
            "path": str(compile_path),
            "sha256": _sha256(compile_path),
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "link_count": artifact_path.stat().st_nlink,
        },
        "compile_profile": compile_report.get("inductor_profile"),
        "physical_tensor_abi": physical,
    }


def build_manifest(
    *,
    capture_reports: tuple[Path, ...],
    compile_report_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    captures: dict[str, dict[str, Any]] = {}
    capture_provenance = []
    for path in capture_reports:
        records, provenance = _capture_artifacts(path.resolve())
        overlap = set(records) & set(captures)
        if overlap:
            raise ValueError(
                f"capture reports contain duplicate Regions: {sorted(overlap)}"
            )
        captures.update(records)
        capture_provenance.append(provenance)
    selected = {
        name: captures[name]
        for name in _EXPECTED_REGIONS
        if name in captures
    }
    missing_captures = sorted(_EXPECTED_REGIONS - set(selected))
    artifacts_on_disk = {
        path.stem for path in artifact_root.glob("*.so") if path.is_file()
    }
    regions = (
        [
            _audit_region(
                name,
                capture=selected[name],
                compile_report_root=compile_report_root,
                artifact_root=artifact_root,
            )
            for name in sorted(_EXPECTED_REGIONS)
        ]
        if not missing_captures
        else []
    )
    checks = {
        "capture_reports_passed": all(
            bool(item["passed"]) for item in capture_provenance
        ),
        "capture_region_set_complete": not missing_captures,
        "artifact_region_set_exact": artifacts_on_disk == _EXPECTED_REGIONS,
        "region_count_is_64": len(regions) == 64,
        "all_regions_bound_to_exact_artifacts": bool(regions)
        and all(bool(item["passed"]) for item in regions),
    }
    artifact_identities = {
        item["name"]: item["artifact"]["sha256"] for item in regions
    }
    identity_payload = json.dumps(
        artifact_identities, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "vlaforge.minddrive_aoti_artifact_manifest/1",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_region_count": 64,
        "core_op_delta": 0,
        "logical_regions": [
            "vision_encoder",
            "position_encoder",
            "map_encoder",
            "detection_encoder",
            "decision_expert",
            "action_expert",
            "trajectory_decoder",
            "detection_decoder",
        ],
        "physical_tensor_abi": {
            "layout": "contiguous",
            "arbitrary_strides_in_semantic_ir": False,
        },
        "capture_reports": capture_provenance,
        "compile_report_root": str(compile_report_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "artifact_set_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "missing_capture_regions": missing_captures,
        "unexpected_artifacts": sorted(artifacts_on_disk - _EXPECTED_REGIONS),
        "missing_artifacts": sorted(_EXPECTED_REGIONS - artifacts_on_disk),
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-report", type=Path, action="append", required=True
    )
    parser.add_argument("--compile-report-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_manifest(
        capture_reports=tuple(args.capture_report),
        compile_report_root=args.compile_report_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise ValueError(
            f"MindDrive artifact audit failed: {report['checks']}"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "regions": len(report["regions"]),
                "artifact_set_sha256": report["artifact_set_sha256"],
                "manifest": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
