#!/usr/bin/env python3
"""Capture and audit real SmolVLA TensorRegion boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlaforge.adapters.smolvla_frontend import audit_real_smolvla_frontend
from vlaforge.adapters.smolvla_real import RealSmolVLAConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = audit_real_smolvla_frontend(
        RealSmolVLAConfig(
            policy_path=args.policy_path,
            vlm_path=args.vlm_path,
            device=args.device,
            num_steps=args.num_steps,
            tolerance=args.tolerance,
            lerobot_revision=args.revision,
        ),
        report_path=args.report,
    )
    print(
        f"model={report.model} regions={len(report.regions)} "
        f"passed={report.passed} report={args.report}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
