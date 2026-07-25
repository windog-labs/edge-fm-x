from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cuda_aoti
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_CUDA_AOTI") != "1",
    reason="set VLAFORGE_RUN_CUDA_AOTI=1 for the CUDA AOTI audit",
)
def test_cuda_aoti_region_backend(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(source_root / "tools" / "audit_cuda_aoti_region.py"),
            "--work-dir",
            str(tmp_path / "audit"),
            "--report",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["python_linked"] is False
    assert set(payload["backend_negative_cases"]) == {
        "load-failure",
        "missing-input-binding",
        "wrong-output-shape",
        "wrong-target",
    }
    generated = payload["generated_session"]
    assert generated["status"] == "passed"
    assert generated["bundle_verified"] is True
    assert generated["python_linked"] is False
    assert set(generated["negative_cases"]) == {
        "corrupt-artifact",
        "missing-artifact",
        "wrong-device",
        "wrong-dtype",
        "wrong-layout",
        "wrong-shape",
    }
