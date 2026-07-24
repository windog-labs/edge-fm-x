#!/usr/bin/env python3
"""Materialize the pinned OpenVLA exported example inputs as raw C++ ABI data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    program = torch.export.load(args.prefill_export)
    values = tuple(program.example_inputs[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, value in enumerate(values):
        path = args.output_dir / f"input_{index}.bin"
        raw = (
            value.detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        path.write_bytes(raw)
        records.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype).removeprefix("torch."),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    report = {
        "schema": "vlaforge.real_openvla_inputs/1",
        "export_path": str(args.prefill_export.resolve()),
        "export_sha256": _sha256(args.prefill_export),
        "inputs": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"materialized inputs={len(records)} output={args.output_dir}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
