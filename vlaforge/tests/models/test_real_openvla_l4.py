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
    os.environ.get("VLAFORGE_RUN_REAL_OPENVLA_L4") != "1",
    reason="set VLAFORGE_RUN_REAL_OPENVLA_L4=1 for real L4 audit",
)
def test_real_openvla_generated_cuda_session(tmp_path: Path) -> None:
    capture_root = os.getenv("VLAFORGE_OPENVLA_CAPTURE_ROOT")
    l3_root = os.getenv("VLAFORGE_OPENVLA_L3_ROOT")
    support_root = os.getenv("VLAFORGE_OPENVLA_L4_SUPPORT_ROOT")
    if not capture_root or not l3_root:
        pytest.skip(
            "set VLAFORGE_OPENVLA_CAPTURE_ROOT and "
            "VLAFORGE_OPENVLA_L3_ROOT"
        )

    project_root = Path(__file__).resolve().parents[2]
    repository_root = project_root.parent
    l3_report = Path(
        os.getenv(
            "VLAFORGE_OPENVLA_L3_REPORT",
            str(
                repository_root
                / "doc"
                / "reports"
                / "vlaforge_real_v03"
                / "openvla_artifact_l3.json"
            ),
        )
    )
    report = tmp_path / "openvla-real-l4.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "tools" / "assemble_real_openvla_l4.py"),
            "--capture-root",
            capture_root,
            "--l3-root",
            l3_root,
            "--l3-report",
            str(l3_report),
            "--support-root",
            support_root or str(tmp_path / "support"),
            "--bundle-root",
            str(tmp_path / "bundle"),
            "--input-root",
            str(tmp_path / "inputs"),
            "--report",
            str(report),
            "--python",
            sys.executable,
            "--target",
            "sm_86",
            "--runs",
            "1",
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L4"
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["semantic_ir"]["persistent_state_slots"] == 0
    assert payload["artifacts"]["invocation_resident"] == 36
    assert payload["artifacts"]["session_resident"] == 2
    assert payload["correctness"]["typed_generic_equal"]
    assert payload["correctness"]["action_maximum_absolute_error"] <= 1e-12
    assert payload["transaction"]["previous_output_preserved"]
    assert payload["transaction"]["transaction_abort_delta"] == 1
    assert payload["transaction"]["recovered"]
    assert payload["bundle"]["invalid_python_environment"]
    assert payload["bundle"]["links_libpython"] is False
