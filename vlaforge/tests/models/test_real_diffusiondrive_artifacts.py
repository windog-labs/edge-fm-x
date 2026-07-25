from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cuda_aoti
@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_REAL_DIFFUSIONDRIVE_L3") != "1",
    reason="set VLAFORGE_RUN_REAL_DIFFUSIONDRIVE_L3=1 for real L3 audit",
)
def test_real_diffusiondrive_artifact_parity(tmp_path: Path) -> None:
    source_root = os.getenv("VLAFORGE_DIFFUSIONDRIVE_SOURCE_ROOT")
    checkpoint = os.getenv("VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT")
    l3_root = os.getenv("VLAFORGE_DIFFUSIONDRIVE_L3_ROOT")
    if not all((source_root, checkpoint, l3_root)):
        pytest.skip(
            "set VLAFORGE_DIFFUSIONDRIVE_SOURCE_ROOT, "
            "VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT, and "
            "VLAFORGE_DIFFUSIONDRIVE_L3_ROOT"
        )

    project_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "diffusiondrive-real-l3.json"
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "tools"
                / "audit_real_diffusiondrive_artifacts.py"
            ),
            "--source-root",
            source_root,
            "--checkpoint",
            checkpoint,
            "--export-dir",
            str(Path(l3_root) / "exports"),
            "--artifact-dir",
            str(Path(l3_root) / "artifacts"),
            "--frontend-report",
            str(Path(l3_root) / "frontend.json"),
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L3"
    assert payload["correctness"]["exported_vs_eager"]["all_exact"]
    assert payload["correctness"]["artifact_vs_eager"][
        "all_regions_within_nrmse"
    ]
    assert payload["correctness"]["artifact_vs_eager"][
        "trajectory_within_absolute_tolerance"
    ]
    assert payload["correctness"]["artifact_repeatability"]["all_exact"]
    assert len(payload["artifacts"]) == 5
    assert all(item["target"] == "sm_86" for item in payload["artifacts"])
