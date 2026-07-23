#!/usr/bin/env python3
"""Run the opt-in real SmolVLA evidence gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vlaforge.adapters.smolvla_real import (
    RealSmolVLAConfig,
    run_real_smolvla,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--lerobot-revision", default="unknown")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    evidence = run_real_smolvla(
        RealSmolVLAConfig(
            policy_path=args.policy_path,
            vlm_path=args.vlm_path,
            device=args.device,
            num_steps=args.num_steps,
            tolerance=args.tolerance,
            lerobot_revision=args.lerobot_revision,
        ),
        trace_path=args.trace,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    evidence.write(args.report)
    print(json.dumps(asdict(evidence), indent=2, sort_keys=True))
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
