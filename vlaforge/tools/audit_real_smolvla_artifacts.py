#!/usr/bin/env python3
"""Audit real SmolVLA exported Regions against packaged CUDA artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlaforge.adapters.smolvla_artifact import (
    audit_real_smolvla_artifacts,
)
from vlaforge.adapters.smolvla_real import RealSmolVLAConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--final-max-abs-tolerance", type=float, default=5e-2)
    parser.add_argument("--final-mean-abs-tolerance", type=float, default=1e-2)
    parser.add_argument("--region-nrmse-tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    report = audit_real_smolvla_artifacts(
        RealSmolVLAConfig(
            policy_path=args.policy_path,
            vlm_path=args.vlm_path,
            device=args.device,
            num_steps=args.num_steps,
            tolerance=args.final_max_abs_tolerance,
            lerobot_revision=args.revision,
        ),
        export_dir=args.export_dir,
        artifact_dir=args.artifact_dir,
        frontend_report=args.frontend_report,
        report_path=args.report,
        final_max_abs_tolerance=args.final_max_abs_tolerance,
        final_mean_abs_tolerance=args.final_mean_abs_tolerance,
        region_nrmse_tolerance=args.region_nrmse_tolerance,
    )
    print(
        f"model=SmolVLA level={report.evidence_level} "
        f"passed={report.passed} "
        f"final_max_abs={report.artifact_final_vs_eager.maximum_absolute_error} "
        f"report={args.report}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
