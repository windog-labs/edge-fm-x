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
    os.environ.get("VLAFORGE_RUN_REAL_DIFFUSIONDRIVE") != "1",
    reason=(
        "set VLAFORGE_RUN_REAL_DIFFUSIONDRIVE=1 for the real checkpoint audit"
    ),
)
def test_real_diffusiondrive_frontend_capture(tmp_path: Path) -> None:
    source_root = os.getenv("VLAFORGE_DIFFUSIONDRIVE_SOURCE_ROOT")
    checkpoint = os.getenv("VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT")
    if not source_root or not checkpoint:
        pytest.skip(
            "set VLAFORGE_DIFFUSIONDRIVE_SOURCE_ROOT and "
            "VLAFORGE_DIFFUSIONDRIVE_CHECKPOINT"
        )

    project_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "diffusiondrive-real-l2.json"
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "tools"
                / "audit_real_diffusiondrive_frontend.py"
            ),
            "--source-root",
            source_root,
            "--checkpoint",
            checkpoint,
            "--export-dir",
            str(tmp_path / "exports"),
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L2"
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["semantic_ir"]["cache_input_ids"] == [0, 1, 2]
    assert payload["semantic_ir"]["bounded_denoise_steps"] == 2
    assert payload["checkpoint"]["missing_keys"] == []
    assert payload["checkpoint"]["unexpected_keys"] == []
    for metrics in payload["correctness"][
        "upstream_forward_vs_region_chain"
    ].values():
        assert metrics["maximum_absolute_error"] <= 1e-5
    assert set(
        payload["correctness"]["upstream_forward_vs_region_chain"]
    ) == {
        "candidate_trajectories",
        "candidate_scores",
        "trajectory",
        "bev_semantic_map",
        "agent_states",
        "agent_labels",
    }
    assert all(
        record["effect_audit"]["passed"]
        for record in payload["captures"]
    )
