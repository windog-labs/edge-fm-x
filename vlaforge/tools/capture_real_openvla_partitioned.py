#!/usr/bin/env python3
"""Capture memory-bounded real OpenVLA CUDA artifact partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlaforge.adapters.openvla_partitioned import (
    OPENVLA_UPSTREAM_REVISION,
    OpenVLAPartitionCaptureConfig,
    capture_real_openvla_partitioned,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default=OPENVLA_UPSTREAM_REVISION,
    )
    parser.add_argument("--reference-frontend-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--instruction", default="pick up the block")
    args = parser.parse_args()

    report = capture_real_openvla_partitioned(
        OpenVLAPartitionCaptureConfig(
            checkpoint_path=args.checkpoint,
            revision=args.revision,
            device=args.device,
            unnorm_key=args.unnorm_key,
            instruction=args.instruction,
            reference_frontend_report=args.reference_frontend_report,
        ),
        output_root=args.output_root,
        report_path=args.report,
    )
    print(
        f"model={report['model']} status={report['status']} "
        f"regions={len(report['regions'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
