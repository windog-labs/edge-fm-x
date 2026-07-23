#!/usr/bin/env python3
"""Run the opt-in real OpenVLA evidence gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vlaforge.adapters.openvla_real import (
    RealOpenVLAConfig,
    run_real_openvla,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--instruction", default="pick up the block")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    evidence = run_real_openvla(
        RealOpenVLAConfig(
            checkpoint_path=args.checkpoint_path,
            revision=args.revision,
            device=args.device,
            unnorm_key=args.unnorm_key,
            instruction=args.instruction,
            tolerance=args.tolerance,
            load_in_4bit=not args.no_4bit,
        ),
        trace_path=args.trace,
    )
    evidence.write(args.report)
    print(json.dumps(asdict(evidence), indent=2, sort_keys=True))
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
