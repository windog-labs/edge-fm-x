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
    os.environ.get("VLAFORGE_RUN_REAL_AUTOVLA") != "1",
    reason="set VLAFORGE_RUN_REAL_AUTOVLA=1 for the real checkpoint audit",
)
def test_real_autovla_partitioned_frontend_capture(
    tmp_path: Path,
) -> None:
    source_root = os.getenv("VLAFORGE_AUTOVLA_SOURCE_ROOT")
    checkpoint = os.getenv("VLAFORGE_AUTOVLA_CHECKPOINT")
    codebook = os.getenv("VLAFORGE_AUTOVLA_CODEBOOK")
    qwen_config = os.getenv("VLAFORGE_AUTOVLA_QWEN_CONFIG")
    if not all((source_root, checkpoint, codebook, qwen_config)):
        pytest.skip(
            "set AutoVLA source, checkpoint, codebook, and Qwen config paths"
        )

    project_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "autovla-real-l2.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "tools" / "audit_real_autovla_frontend.py"),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--codebook",
            str(codebook),
            "--qwen-config",
            str(qwen_config),
            "--export-dir",
            str(tmp_path / "exports"),
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == (
        "L2-partitioned-real-checkpoint-frontend"
    )
    assert payload["semantic_ir"]["core_op_delta"] == 0
    assert payload["exact_reuse"]["semantic"] == {
        "events": 3,
        "hits": 1,
        "misses": 2,
    }
    assert payload["correctness"]["semantic_plan_trace_exact"]
    assert payload["correctness"]["action_tokens_in_range"]
    assert all(
        capture["effect_audit"]["passed"]
        for capture in payload["captures"]
    )
