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
    os.environ.get("VLAFORGE_RUN_REAL_MINDDRIVE_L4") != "1",
    reason="set VLAFORGE_RUN_REAL_MINDDRIVE_L4=1 for the real L4 audit",
)
def test_real_minddrive_generated_cuda_session(tmp_path: Path) -> None:
    archive_text = os.getenv("VLAFORGE_MINDDRIVE_ARCHIVE_ROOT")
    if not archive_text:
        pytest.skip("set VLAFORGE_MINDDRIVE_ARCHIVE_ROOT")
    archive = Path(archive_text).resolve()
    reports = archive / "artifacts" / "reports"
    frontend = archive / "frontend"
    bundle = archive / "l4" / "bundle"
    frame_inputs = (
        frontend / "real_invocation_inputs.pt",
        frontend / "sequence_00400_00401" / "real_invocation_inputs.pt",
        frontend
        / "heldout_sequence_00400_00401_00402"
        / "real_invocation_inputs.pt",
        frontend
        / "heldout_v2_sequence_00400_00401_00402_00403"
        / "real_invocation_inputs.pt",
        frontend
        / "heldout_v3_sequence_00400_00401_00402_00403_00404"
        / "real_invocation_inputs.pt",
    )
    project_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(project_root / "tools" / "build_real_minddrive_l4.py"),
        "--artifact-manifest",
        str(reports / "aoti24_all_contiguous_v6" / "artifact_manifest.json"),
        "--sequence-report",
        str(
            reports
            / "aoti24_all_contiguous_v6"
            / "sequences"
            / "sequence_manifest.json"
        ),
        "--bundle-root",
        str(bundle),
        "--support-root",
        str(tmp_path / "evidence"),
        "--reference-tensors",
        str(
            reports
            / (
                "development_l3_aoti24_v6_sdpa_contract_v3_"
                "00400_00401_00402_00403_00404.pt"
            )
        ),
        "--target",
        "sm_86",
        "--reuse-bundle",
    ]
    for frame_input in frame_inputs:
        command.extend(("--frame-input", str(frame_input)))
    subprocess.run(command, check=True)

    payload = json.loads(
        (tmp_path / "evidence" / "minddrive-real-l4.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "L4"
    assert payload["semantic_ir"] == {
        "authoritative_states": 16,
        "core_op_delta": 0,
        "io_schema_digest": (
            "62e566cde5ca9cec1bbb7949b7eacf42393d1316e35a16e9c1b839f6220ded2a"
        ),
        "logical_regions": 8,
        "named_outputs": 10,
    }
    assert payload["bundle"]["physical_artifacts"] == 66
    assert payload["bundle"]["python_linked"] is False
    assert all(
        value["typed_vs_l3_exact"] and value["typed_vs_generic_exact"]
        for value in payload["output_parity"].values()
    )
    assert payload["transactional_checks"] == {
        "episode_reset": True,
        "new_revision_cache_miss": True,
        "retry_commits": True,
        "same_revision_cache_hit": True,
        "state_commit_count": 128,
        "state_count": 16,
        "validation_failure_aborts": True,
    }
