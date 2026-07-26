#!/usr/bin/env python3
"""Bind MindDrive physical captures to their exact compiled artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_TORCH_DTYPES = {
    "f16": "torch.float16",
    "f32": "torch.float32",
    "f64": "torch.float64",
    "i32": "torch.int32",
    "i64": "torch.int64",
    "bool": "torch.bool",
}


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


def _normalize_capture_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": value["shape"],
        "dtype": _TORCH_DTYPES[value["dtype"]],
        "strides": value["strides"],
        "storage_offset": value["storage_offset"],
        "contiguous": value["contiguous"],
        "layout": value["layout"],
    }


def _normalize_compile_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "shape",
            "dtype",
            "strides",
            "storage_offset",
            "contiguous",
            "layout",
        )
    }


def _audit_region(
    capture: dict[str, Any],
    *,
    compile_report_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    name = str(capture["name"])
    export_path = Path(str(capture["artifact"])).resolve()
    compile_report_path = (compile_report_root / f"{name}.json").resolve()
    artifact_path = (artifact_root / f"{name}.so").resolve()
    if not export_path.is_file():
        raise FileNotFoundError(f"{name}: capture export is missing")
    if not compile_report_path.is_file():
        raise FileNotFoundError(f"{name}: compile report is missing")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{name}: compiled artifact is missing")

    compile_report = _load_json(compile_report_path)
    physical = compile_report.get("physical_tensor_abi")
    if not isinstance(physical, dict) or not physical.get("required"):
        raise ValueError(
            f"{name}: compile report did not enforce the physical tensor ABI"
        )
    capture_inputs = [
        _normalize_capture_contract(value) for value in capture["inputs"]
    ]
    capture_outputs = [
        _normalize_capture_contract(value) for value in capture["outputs"]
    ]
    compiled_inputs = [
        _normalize_compile_contract(value)
        for value in physical["inputs"]
    ]
    compiled_reference_outputs = [
        _normalize_compile_contract(value)
        for value in physical["reference_outputs"]
    ]
    checks = {
        "capture_passed": bool(capture["strict_export"])
        and bool(capture["effect_audit_passed"]),
        "compile_passed": bool(compile_report["passed"]),
        "capture_export_hash_matches": (
            _sha256(export_path) == capture["artifact_sha256"]
        ),
        "compile_export_hash_matches_capture": (
            compile_report["export"]["sha256"]
            == capture["artifact_sha256"]
        ),
        "artifact_hash_matches_compile_report": (
            _sha256(artifact_path)
            == compile_report["artifact"]["sha256"]
        ),
        "input_contract_matches": capture_inputs == compiled_inputs,
        "reference_output_contract_matches": (
            capture_outputs == compiled_reference_outputs
        ),
        "capture_inputs_contiguous": all(
            bool(value["contiguous"]) for value in capture["inputs"]
        ),
        "capture_outputs_contiguous": all(
            bool(value["contiguous"]) for value in capture["outputs"]
        ),
    }
    return {
        "name": name,
        "passed": all(checks.values()),
        "checks": checks,
        "capture_export": {
            "path": str(export_path),
            "sha256": capture["artifact_sha256"],
        },
        "compile_report": {
            "path": str(compile_report_path),
            "sha256": _sha256(compile_report_path),
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "inputs": capture_inputs,
        "outputs": capture_outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--compile-report-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture_report_path = args.capture_report.resolve()
    compile_report_root = args.compile_report_root.resolve()
    artifact_root = args.artifact_root.resolve()
    capture_report = _load_json(capture_report_path)
    if (
        capture_report.get("schema")
        != "vlaforge.minddrive_partitioned_vision_capture/2"
    ):
        raise ValueError(
            "MindDrive physical ABI audit requires capture schema version 2"
        )
    abi = capture_report.get("physical_tensor_abi")
    if not isinstance(abi, dict) or abi.get("layout") != "contiguous":
        raise ValueError("capture report does not declare a contiguous ABI")

    regions = [
        _audit_region(
            capture,
            compile_report_root=compile_report_root,
            artifact_root=artifact_root,
        )
        for capture in capture_report["regions"]
    ]
    names = [str(item["name"]) for item in regions]
    checks = {
        "capture_report_passed": bool(capture_report["passed"]),
        "region_count_is_50": len(regions) == 50,
        "region_names_unique": len(names) == len(set(names)),
        "all_regions_bound_to_exact_artifacts": all(
            bool(item["passed"]) for item in regions
        ),
    }
    report = {
        "schema": "vlaforge.minddrive_physical_abi_manifest/1",
        "passed": all(checks.values()),
        "checks": checks,
        "logical_region": "vision_encoder",
        "physical_tensor_abi": {
            "layout": "contiguous",
            "arbitrary_strides_in_semantic_ir": False,
        },
        "capture_report": {
            "path": str(capture_report_path),
            "sha256": _sha256(capture_report_path),
        },
        "compile_report_root": str(compile_report_root),
        "artifact_root": str(artifact_root),
        "regions": regions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise ValueError(f"MindDrive physical ABI audit failed: {checks}")
    print(
        json.dumps(
            {
                "passed": True,
                "regions": len(regions),
                "manifest": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
