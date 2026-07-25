#!/usr/bin/env python3
"""Audit real DiffusionDrive exports against packaged CUDA artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.diffusiondrive_artifact import (  # noqa: E402
    audit_real_diffusiondrive_artifacts,
)
from vlaforge.adapters.diffusiondrive_real import (  # noqa: E402
    DIFFUSIONDRIVE_HF_REVISION,
    DIFFUSIONDRIVE_UPSTREAM_REVISION,
    RealDiffusionDriveConfig,
)


def _repository_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "source_dirty": dirty}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--upstream-revision",
        default=DIFFUSIONDRIVE_UPSTREAM_REVISION,
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=DIFFUSIONDRIVE_HF_REVISION,
    )
    parser.add_argument("--region-nrmse-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--trajectory-max-abs-tolerance",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--trajectory-mean-abs-tolerance",
        type=float,
        default=5e-4,
    )
    args = parser.parse_args()

    report = audit_real_diffusiondrive_artifacts(
        RealDiffusionDriveConfig(
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            device=args.device,
            upstream_revision=args.upstream_revision,
            checkpoint_revision=args.checkpoint_revision,
        ),
        export_dir=args.export_dir,
        artifact_dir=args.artifact_dir,
        frontend_report=args.frontend_report,
        region_nrmse_tolerance=args.region_nrmse_tolerance,
        trajectory_max_abs_tolerance=args.trajectory_max_abs_tolerance,
        trajectory_mean_abs_tolerance=args.trajectory_mean_abs_tolerance,
    )
    report["repository"] = _repository_state()
    report["reproduction"] = {
        "command": [
            sys.executable,
            str(Path(__file__).relative_to(_REPOSITORY_ROOT)),
            "--source-root",
            str(args.source_root.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--export-dir",
            str(args.export_dir.resolve()),
            "--artifact-dir",
            str(args.artifact_dir.resolve()),
            "--frontend-report",
            str(args.frontend_report.resolve()),
            "--report",
            "<report.json>",
            "--device",
            args.device,
        ],
        "environment": {
            "PYTHONPATH": "vlaforge/python",
            "CUDA_VISIBLE_DEVICES": os.getenv(
                "CUDA_VISIBLE_DEVICES", "<unset>"
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"model=DiffusionDrive level={report['evidence_level']} "
        f"passed={report['passed']} report={args.report}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
