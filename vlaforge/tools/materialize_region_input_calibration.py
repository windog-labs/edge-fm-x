#!/usr/bin/env python3
"""Materialize calibrated and held-out input packs from a verified handoff.

The input order and semantic names come from a captured-region JSON file. The
tool is model-agnostic: it pairs those names with a board input handoff,
verifies every source file, and writes separate calibration and held-out NPZ
packs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_NUMPY_DTYPES = {
    "bool": np.bool_,
    "f32": np.float32,
    "float32": np.float32,
    "f64": np.float64,
    "float64": np.float64,
    "i32": np.int32,
    "int32": np.int32,
    "i64": np.int64,
    "int64": np.int64,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"handoff path escapes its root: {relative}")
    return path


def verify_handoff(root: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema") != "vlaforge.board_input_handoff/1"
        or manifest.get("scope") != "data-and-reference-only"
    ):
        raise ValueError("expected a verified data/reference handoff")
    files = manifest.get("files")
    samples = manifest.get("samples")
    if not isinstance(files, list) or not files:
        raise ValueError("handoff must declare its files")
    if not isinstance(samples, list) or not samples:
        raise ValueError("handoff must declare its samples")
    for entry in files:
        path = safe_member(root, entry["path"])
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"handoff size changed: {entry['path']}")
        if digest(path) != entry["sha256"]:
            raise ValueError(f"handoff digest changed: {entry['path']}")


def load_binary(path: Path, specification: dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(_NUMPY_DTYPES[specification["dtype"]])
    except KeyError as error:
        raise ValueError(
            f"unsupported handoff dtype: {specification['dtype']}"
        ) from error
    value = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(specification["shape"], dtype=np.int64))
    if value.size != expected:
        raise ValueError(
            f"{path.name} contains {value.size} values, expected {expected}"
        )
    return value.reshape(specification["shape"])


def region_input_names(capture: dict[str, Any]) -> list[str]:
    inputs = capture.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("capture JSON must declare region inputs")
    names = [entry.get("name") for entry in inputs]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("capture inputs must have non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("capture input names must be unique")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--capture-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, required=True)
    args = parser.parse_args()

    handoff = args.handoff.resolve(strict=True)
    capture_path = args.capture_json.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("calibration output must be new")
    if args.calibration_count < 1:
        raise ValueError("calibration count must be positive")

    handoff_manifest = load_json(handoff / "manifest.json")
    capture = load_json(capture_path)
    verify_handoff(handoff, handoff_manifest)
    names = region_input_names(capture)
    input_specs = {
        item["name"]: item for item in handoff_manifest["inputs"]
    }
    missing = [name for name in names if name not in input_specs]
    if missing:
        raise ValueError(f"handoff is missing region inputs: {missing}")
    samples = handoff_manifest["samples"]
    if not 1 <= args.calibration_count < len(samples):
        raise ValueError("calibration count must leave at least one held-out sample")

    split_values: dict[str, dict[str, list[np.ndarray]]] = {
        "calibration": {},
        "held-out": {},
    }
    split_samples: dict[str, list[int]] = {
        "calibration": [],
        "held-out": [],
    }
    for index, sample in enumerate(samples):
        split = "calibration" if index < args.calibration_count else "held-out"
        split_samples[split].append(int(sample["sample_id"]))
        for name in names:
            relative = sample["inputs"].get(name)
            if relative is None:
                raise ValueError(
                    f"sample {sample['sample_id']} is missing input {name}"
                )
            value = load_binary(safe_member(handoff, relative), input_specs[name])
            split_values[split].setdefault(name, []).append(value)

    output.mkdir(parents=True)
    split_evidence: dict[str, Any] = {}
    for split in ("calibration", "held-out"):
        split_root = output / split
        split_root.mkdir()
        path = split_root / "inputs.npz"
        arrays = {
            name: np.stack(values, axis=0)
            for name, values in split_values[split].items()
        }
        np.savez(path, **arrays)
        split_evidence[split] = {
            "path": str(path.relative_to(output)),
            "sha256": digest(path),
            "sample_ids": split_samples[split],
            "inputs": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in sorted(arrays.items())
            },
        }

    result = {
        "schema": "vlaforge.region_input_calibration/1",
        "status": "materialized",
        "handoff": str(handoff),
        "handoff_manifest_sha256": digest(handoff / "manifest.json"),
        "capture_json": str(capture_path),
        "capture_json_sha256": digest(capture_path),
        "input_names": names,
        "calibration_count": args.calibration_count,
        "held_out_count": len(samples) - args.calibration_count,
        "splits": split_evidence,
    }
    write_json(output / "manifest.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
