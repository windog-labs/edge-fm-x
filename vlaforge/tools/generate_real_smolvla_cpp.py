#!/usr/bin/env python3
"""Generate a no-Python C++ runner from real SmolVLA exported regions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from vlaforge.codegen import (
    generate_real_smolvla_aoti_runner,
    smolvla_spec_from_exported_programs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("off", "conservative", "verified", "auto", "force-on"),
        default="verified",
    )
    parser.add_argument("--allow-test-profile", action="store_true")
    parser.add_argument("--optimization-benchmark", action="store_true")
    args = parser.parse_args()

    prefix = torch.export.load(args.export_dir / "prepare_prefix.pt2e")
    solver = torch.export.load(args.export_dir / "solver_step.pt2e")
    trim = torch.export.load(args.export_dir / "trim_action_chunk.pt2e")
    spec = smolvla_spec_from_exported_programs(prefix, solver, trim)
    sources = generate_real_smolvla_aoti_runner(
        spec,
        compiler_profile=args.profile,
        allow_test_profile=args.allow_test_profile,
        optimization_benchmark=args.optimization_benchmark,
    )
    sources.write(args.output_dir)
    manifest = {
        "schema": "vlaforge.real_smolvla_codegen/2",
        "source_digest": sources.digest(),
        "spec": asdict(spec),
        "files": [name for name, _ in sources.files],
        "optimization_benchmark": args.optimization_benchmark,
        "compiler_profile": args.profile,
        "compilation_certificate": json.loads(
            sources.as_dict()["compilation_certificate.json"]
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"generated SmolVLA C++ digest={sources.digest()} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
