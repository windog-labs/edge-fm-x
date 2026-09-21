"""Hash-bound CPU comparison of complete recorded Region boundary dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest


DTYPES = {
    1: ("torch.bool", "?", 1),
    2: ("torch.int32", "<i4", 4),
    3: ("torch.int64", "<i8", 8),
    4: ("torch.float16", "<f2", 2),
    5: ("torch.bfloat16", "<u2", 2),
    6: ("torch.float32", "<f4", 4),
    7: ("torch.float64", "<f8", 8),
}


def _file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("trace file escapes its recorded directory or is missing")
    return path


def runtime_region_ids(bundle, expected_manifest):
    manifest_path = bundle / "bundle.json"
    if file_digest(manifest_path) != expected_manifest:
        raise ValueError("native bundle manifest changed")
    manifest = json.loads(manifest_path.read_text())
    item = manifest["semantic_ir"]
    ir_path = _file(bundle, item["path"])
    digest = file_digest(ir_path)
    if any(digest[key] != item[key] for key in ("sha256", "size_bytes")):
        raise ValueError("compiled semantic IR checksum mismatch")
    names = [region["name"] for region in json.loads(ir_path.read_text())["regions"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("compiled Region ordering is invalid")
    return {str(index): name for index, name in enumerate(names)}, {
        "bundle_manifest": expected_manifest,
        "compiled_semantic_ir": {"path": str(ir_path), **digest},
        "mapping_rule": "C++ codegen enumerates compiled semantic IR Regions, not source declaration order",
    }


def read_trace(path, sha256):
    path = path.resolve()
    if file_digest(path)["sha256"] != sha256:
        raise ValueError("trace report SHA mismatch")
    report = json.loads(path.read_text())
    rows = {}
    provenance = {}
    native = report.get("schema") == "vlaforge.openpi_native_session_audit/1"
    if native:
        if (
            report.get("native_exit_code") != 0
            or report.get("trace_native") is not True
        ):
            raise ValueError("native trace needs complete actual execution")
        capture = report["selection"]["capture"]["sha256"]
        region_ids, provenance = runtime_region_ids(
            path.parent / "bundle", report["bundle"]
        )
        provenance.update(
            declared_region_ids=report.get("trace_region_ids"),
            verified_runtime_region_ids=region_ids,
            declared_region_ids_match_runtime=report.get("trace_region_ids")
            == region_ids,
        )
        root = path.parent / "runs/region-trace"
        for name, digest in report["trace_files"].items():
            if file_digest(_file(root, name)) != digest:
                raise ValueError("native trace bytes changed after execution")
        records = [
            json.loads(line)
            for line in (root / "tensors.jsonl").read_text().splitlines()
        ]
        policies = [
            json.loads(line)
            for line in (root / "numerical.jsonl").read_text().splitlines()
        ]
        for record in records:
            record["region"] = region_ids[str(record["region_id"])]
            record["dtype"] = DTYPES[record["dtype_enum"]][0]
    else:
        if (
            report.get("schema") != "vlaforge.openpi_same_artifact_runtime_probe/1"
            or report.get("status") != "diagnostic-complete"
        ):
            raise ValueError("requires a complete real Python artifact trace")
        capture = report["capture"]["sha256"]
        root = path.parent / "region-trace"
        records = []
        policies = [report["execution_context"]]
        for call in report["calls"]:
            for direction in ("input", "output"):
                for record in call[direction + "s"]:
                    records.append(
                        {
                            **record,
                            "call": call["call"],
                            "region": call["region"],
                            "direction": direction,
                        }
                    )
    for record in records:
        key = (record["call"], record["direction"], record["index"])
        if key in rows:
            raise ValueError("duplicate Region trace boundary")
        payload = _file(root, record["file"])
        digest = file_digest(payload)
        if digest["size_bytes"] != record["size_bytes"] or (
            not native and digest["sha256"] != record["sha256"]
        ):
            raise ValueError("trace tensor checksum or size mismatch")
        dtype = next(
            (item for item in DTYPES.values() if item[0] == record["dtype"]), None
        )
        if (
            dtype is None
            or math.prod(record["shape"]) * dtype[2] != digest["size_bytes"]
        ):
            raise ValueError(
                "trace tensor shape/dtype does not describe its full bytes"
            )
        rows[key] = {**record, "path": payload, "sha256": digest["sha256"]}
    return capture, rows, policies, provenance


def compare_tensor_bytes(left, right, dtype):
    import numpy as np

    descriptor = next(item for item in DTYPES.values() if item[0] == dtype)
    a, b = (np.frombuffer(value, dtype=descriptor[1]) for value in (left, right))
    if a.shape != b.shape:
        raise ValueError("full tensor byte lengths differ")
    if dtype == "torch.bfloat16":
        a, b = ((value.astype(np.uint32) << 16).view(np.float32) for value in (a, b))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "count": a.size,
        "exact_bytes": left == right,
        "exact_values": bool(np.array_equal(a, b)),
        "finite": finite,
        "mismatch_count": int(np.count_nonzero(a != b)),
        "mean_squared_error": float(np.mean(delta * delta)) if finite else None,
        "maximum_absolute_error": float(np.max(np.abs(delta), initial=0))
        if finite
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-sha256", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output must be new")
    left_capture, left, left_policy, left_provenance = read_trace(
        args.left, args.left_sha256
    )
    right_capture, right, right_policy, right_provenance = read_trace(
        args.right, args.right_sha256
    )
    if left_capture != right_capture or set(left) != set(right):
        raise ValueError(
            "compared traces must cover the same capture and complete calls"
        )
    result = {
        "schema": "vlaforge.openpi_runtime_boundary_comparison/1",
        "status": "diagnostic-complete",
        "tool": file_digest(Path(__file__)),
        "left": {"path": str(args.left), **file_digest(args.left)},
        "right": {"path": str(args.right), **file_digest(args.right)},
        "capture_sha256": left_capture,
        "left_observed_policies": left_policy,
        "right_observed_policies": right_policy,
        "left_trace_provenance": left_provenance,
        "right_trace_provenance": right_provenance,
        "performance_measured": False,
        "numeric_acceptance_claimed": False,
        "boundaries": [],
    }
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a[field] != b[field] for field in ("region", "shape", "dtype")):
            raise ValueError("Region boundary name/shape/dtype mismatch")
        row = {
            "call": key[0],
            "direction": key[1],
            "index": key[2],
            "region": a["region"],
            "shape": a["shape"],
            "dtype": a["dtype"],
            "left_sha256": a["sha256"],
            "right_sha256": b["sha256"],
            "left_stride": a["stride"],
            "right_stride": b["stride"],
            "left_storage_offset": a["storage_offset"],
            "right_storage_offset": b["storage_offset"],
            "left_address_mod_256": a["address_mod_256"],
            "right_address_mod_256": b["address_mod_256"],
            "metrics": compare_tensor_bytes(
                a["path"].read_bytes(), b["path"].read_bytes(), a["dtype"]
            ),
        }
        result["boundaries"].append(row)
    mismatches = [
        row for row in result["boundaries"] if not row["metrics"]["exact_bytes"]
    ]
    result.update(
        all_boundary_bytes_exact=not mismatches,
        first_mismatch=mismatches[0] if mismatches else None,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
