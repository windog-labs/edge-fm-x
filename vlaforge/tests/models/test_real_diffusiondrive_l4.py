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
    os.environ.get("VLAFORGE_RUN_REAL_DIFFUSIONDRIVE_L4") != "1",
    reason="set VLAFORGE_RUN_REAL_DIFFUSIONDRIVE_L4=1 for real L4 audit",
)
def test_real_diffusiondrive_generated_cuda_session(
    tmp_path: Path,
) -> None:
    l3_root = os.getenv("VLAFORGE_DIFFUSIONDRIVE_L3_ROOT")
    checkpoint = os.getenv("VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT")
    if not l3_root or not checkpoint:
        pytest.skip(
            "set VLAFORGE_DIFFUSIONDRIVE_L3_ROOT and "
            "VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT"
        )

    project_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "diffusiondrive-real-l4.json"
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "tools"
                / "build_real_diffusiondrive_l4.py"
            ),
            "--l3-root",
            str(l3_root),
            "--support-root",
            str(tmp_path / "support"),
            "--bundle-root",
            str(tmp_path / "bundle"),
            "--checkpoint",
            str(checkpoint),
            "--target",
            "sm_86",
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L4"
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["semantic_ir"]["persistent_state_slots"] == 0
    assert payload["correctness"]["all_named_outputs_exact"]
    assert payload["correctness"]["typed_generic_equal"]
    assert payload["cache"]["same_revision_hit"]
    assert payload["cache"]["new_revision_miss"]
    assert payload["cache"]["missing_revision_miss"]
    assert payload["transaction"][
        "failure_retry_transaction_aborts"
    ] == 1
    assert payload["bundle"]["invalid_python_environment"]
    assert payload["bundle"]["links_libpython"] is False
