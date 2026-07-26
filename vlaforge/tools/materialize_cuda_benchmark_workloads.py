#!/usr/bin/env python3
"""Create five deterministic real-model benchmark input profiles.

All profiles preserve the compiled static shape/dtype/device contract.  They
vary only tensor contents so eager, direct AOTI, and generated Session paths
consume byte-identical inputs for each workload.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


_SCHEMA = "vlaforge.cuda_benchmark_workloads/1"
_MODELS = ("smolvla", "diffusiondrive")
_PROFILES = (
    ("baseline", None, 1.0, 0.0),
    ("observation_minus_1pct", "observation", 0.99, 0.0),
    ("observation_plus_1pct", "observation", 1.01, 0.0),
    ("context_plus_0p01", "context", 1.0, 0.01),
    ("noise_plus_1pct", "noise", 1.01, 0.0),
)
_GROUPS = {
    "smolvla": {
        "observation": ("image",),
        "context": ("state",),
        "noise": ("noise",),
    },
    "diffusiondrive": {
        "observation": ("camera_feature", "lidar_feature"),
        "context": ("status_feature",),
        "noise": ("noise",),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_f32(payload: bytes, *, scale: float, offset: float) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("benchmark workload materializer requires little endian")
    values = array.array("f")
    values.frombytes(payload)
    for index, value in enumerate(values):
        values[index] = value * scale + offset
    return values.tobytes()


def materialize(model: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    if model not in _MODELS:
        raise ValueError(f"unknown model: {model}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be absent or empty: {output_root}")
    source_manifest_path = source_root / "inputs.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    inputs = source_manifest.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("source input manifest has no inputs")
    source_records = {}
    for name, metadata in sorted(inputs.items()):
        source = source_root / f"{name}.bin"
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_sha256 = _sha256(source)
        if (
            int(metadata["size_bytes"]) != source.stat().st_size
            or str(metadata["sha256"]) != actual_sha256
        ):
            raise ValueError(f"source input identity mismatch: {name}")
        source_records[name] = {
            "path": str(source.resolve()),
            "sha256": actual_sha256,
            "size_bytes": source.stat().st_size,
            "dtype": str(metadata["dtype"]),
            "shape": [int(item) for item in metadata["shape"]],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    profiles = []
    for profile_index, (name, group, scale, offset) in enumerate(_PROFILES):
        profile_root = output_root / f"{profile_index:02d}_{name}"
        profile_root.mkdir()
        changed = set() if group is None else set(_GROUPS[model][group])
        output_inputs = {}
        for input_name, metadata in sorted(inputs.items()):
            source = source_root / f"{input_name}.bin"
            destination = profile_root / source.name
            if input_name in changed:
                if metadata["dtype"] != "float32":
                    raise ValueError(
                        f"profile transform requires f32 input: {input_name}"
                    )
                destination.write_bytes(
                    _transform_f32(
                        source.read_bytes(),
                        scale=scale,
                        offset=offset,
                    )
                )
            else:
                shutil.copyfile(source, destination)
            output_inputs[input_name] = {
                **metadata,
                "path": str(destination.resolve()),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        profile_manifest = {
            "schema": "vlaforge.cuda_benchmark_workload/1",
            "model": model,
            "profile_id": profile_index,
            "name": name,
            "transform": {
                "group": group,
                "inputs": sorted(changed),
                "scale": scale,
                "offset": offset,
            },
            "inputs": output_inputs,
        }
        (profile_root / "inputs.json").write_text(
            json.dumps(profile_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        profiles.append(
            {
                "profile_id": profile_index,
                "name": name,
                "root": str(profile_root.resolve()),
                "manifest_sha256": _sha256(profile_root / "inputs.json"),
                "transform": profile_manifest["transform"],
                "input_sha256": {
                    key: value["sha256"]
                    for key, value in output_inputs.items()
                },
            }
        )

    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "model": model,
        "source": {
            "root": str(source_root.resolve()),
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": _sha256(source_manifest_path),
            "inputs": source_records,
        },
        "profiles": profiles,
        "invariants": {
            "profile_count": len(profiles),
            "static_shape_dtype_unchanged": True,
            "deterministic": True,
            "same_bytes_across_execution_paths": True,
        },
        "claim_boundary": (
            "content-only deterministic workload profiles; not additional "
            "model coverage or data-distribution evidence"
        ),
    }
    (output_root / "workloads.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=_MODELS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = materialize(
        args.model,
        args.source_root.resolve(),
        args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["model"],
                "profiles": len(report["profiles"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
