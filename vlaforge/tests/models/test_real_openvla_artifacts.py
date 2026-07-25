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
    os.environ.get("VLAFORGE_RUN_REAL_OPENVLA_L3") != "1",
    reason="set VLAFORGE_RUN_REAL_OPENVLA_L3=1 for real L3 audit",
)
def test_real_openvla_partitioned_artifact_parity(
    tmp_path: Path,
) -> None:
    capture_root = os.getenv("VLAFORGE_OPENVLA_CAPTURE_ROOT")
    artifact_root = os.getenv("VLAFORGE_OPENVLA_L3_ROOT")
    if not capture_root or not artifact_root:
        pytest.skip(
            "set VLAFORGE_OPENVLA_CAPTURE_ROOT and "
            "VLAFORGE_OPENVLA_L3_ROOT"
        )

    project_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "openvla-real-l3.json"
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "tools"
                / "audit_real_openvla_artifacts.py"
            ),
            "--capture-root",
            capture_root,
            "--output-root",
            artifact_root,
            "--audit-only",
            "--pipeline-repeats",
            "2",
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L3"
    assert payload["partition"]["core_op_delta"] == 0
    assert len(payload["partition"]["artifact_regions"]) == 36
    assert payload["compile"][
        "all_active_version_normalizations_exact"
    ]
    assert payload["correctness"][
        "all_region_outputs_within_tolerance"
    ]
    assert payload["correctness"]["action_tokens_equal"]
    assert payload["correctness"]["final_action_equal"]
    assert payload["correctness"]["repeated_pipeline_exact"]
    assert payload["correctness"]["pipeline_repeats"] == 2
    assert payload["memory"]["authoritative_state_bytes"] == 0
    assert payload["memory"]["derived_fixed_kv_bytes"] > 0
