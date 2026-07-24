#!/usr/bin/env python3
"""Generate the no-Python OpenVLA C++ TorchScript runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vlaforge.codegen import (
    generate_real_openvla_torchscript_runner,
    openvla_spec_from_capture_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--optimization-benchmark", action="store_true")
    args = parser.parse_args()
    reports = {
        name: json.loads(
            (args.capture_dir / f"{name}.capture.json").read_text(
                encoding="utf-8"
            )
        )
        for name in (
            "generate_action_tokens_prefill",
            "generate_action_tokens_decode_step",
            "detokenize_action",
        )
    }
    spec = openvla_spec_from_capture_reports(
        reports["generate_action_tokens_prefill"],
        reports["generate_action_tokens_decode_step"],
        reports["detokenize_action"],
    )
    sources = generate_real_openvla_torchscript_runner(
        spec,
        optimization_benchmark=args.optimization_benchmark,
    )
    sources.write(args.output_dir)
    manifest = {
        "schema": "vlaforge.real_openvla_codegen/1",
        "source_digest": sources.digest(),
        "spec": asdict(spec),
        "files": [name for name, _ in sources.files],
        "optimization_benchmark": args.optimization_benchmark,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"generated OpenVLA C++ digest={sources.digest()} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
