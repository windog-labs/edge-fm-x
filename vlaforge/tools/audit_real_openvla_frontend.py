#!/usr/bin/env python3
"""Capture and audit real OpenVLA BF16 prefill/decode TensorRegions."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlaforge.adapters.openvla_frontend import (
    OpenVLAFrontendConfig,
    audit_real_openvla_frontend,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--instruction", default="pick up the block")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = audit_real_openvla_frontend(
        OpenVLAFrontendConfig(
            checkpoint_path=args.checkpoint,
            revision=args.revision,
            unnorm_key=args.unnorm_key,
            instruction=args.instruction,
            cpu_threads=args.cpu_threads,
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
