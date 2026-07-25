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
    os.environ.get("VLAFORGE_RUN_REAL_SMOLVLA_L4") != "1",
    reason="set VLAFORGE_RUN_REAL_SMOLVLA_L4=1 for the real L4 audit",
)
def test_real_smolvla_generated_cuda_session(tmp_path: Path) -> None:
    l3_root = os.getenv("VLAFORGE_SMOLVLA_L3_ROOT")
    policy_path = os.getenv("VLAFORGE_SMOLVLA_POLICY_PATH")
    vlm_path = os.getenv("VLAFORGE_SMOLVLA_VLM_PATH")
    revision = os.getenv("VLAFORGE_LEROBOT_REVISION")
    if not all((l3_root, policy_path, vlm_path, revision)):
        pytest.skip(
            "set VLAFORGE_SMOLVLA_L3_ROOT, VLAFORGE_SMOLVLA_POLICY_PATH, "
            "VLAFORGE_SMOLVLA_VLM_PATH, and VLAFORGE_LEROBOT_REVISION"
        )

    source_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "smolvla-real-l4.json"
    subprocess.run(
        [
            sys.executable,
            str(source_root / "tools" / "build_real_smolvla_l4.py"),
            "--l3-root",
            str(l3_root),
            "--support-root",
            str(tmp_path / "support"),
            "--bundle-root",
            str(tmp_path / "bundle"),
            "--checkpoint",
            str(Path(policy_path) / "model.safetensors"),
            "--vlm-path",
            str(vlm_path),
            "--upstream-revision",
            str(revision),
            "--target",
            "sm_86",
            "--python",
            sys.executable,
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L4"
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["correctness"][
        "direct_artifact_vs_generated_chunk"
    ]["exact"]
    assert payload["cache"]["same_revision_hit"]
    assert payload["cache"]["new_revision_miss"]
    assert payload["cache"]["missing_revision_miss"]
    assert payload["transaction"]["state_version_sequence"] == "passed"
    assert payload["transaction"]["failure_retry_transaction_aborts"] == 1
    assert payload["bundle"]["invalid_python_environment"]
    assert payload["bundle"]["links_libpython"] is False
