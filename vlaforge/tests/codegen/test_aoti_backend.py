from __future__ import annotations

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
    text = report.read_text(encoding="utf-8")
    assert '"status": "passed"' in text
    assert '"python_linked": false' in text
