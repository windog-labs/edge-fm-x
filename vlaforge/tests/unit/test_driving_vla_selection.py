from __future__ import annotations

import json
from pathlib import Path

from vlaforge.adapters.model_contracts import model_contract


ROOT = Path(__file__).resolve().parents[3]
REPORT = (
    ROOT
    / "doc"
    / "reports"
    / "vlaforge_driving_vla_selection_v01"
    / "candidate_selection.json"
)


def test_minddrive_selection_matches_pinned_model_contract() -> None:
    payload = json.loads(REPORT.read_text())
    selected = next(
        item
        for item in payload["candidates"]
        if item["name"] == payload["selected"]
    )
    contract = model_contract("MindDrive 0.5B")

    assert payload["schema"] == "vlaforge.driving_vla_candidate_selection/1"
    assert selected["upstream_revision"] == contract.revision
    assert selected["repository"] == contract.repository
    assert contract.current_evidence == selected["current_evidence"]
    assert contract.fixture_factory is None
    assert "L0 provenance only" in payload["claim"]

    hashes = {item["sha256"] for item in selected["primary_files"]}
    assert len(hashes) == 3
    assert all(len(value) == 64 for value in hashes)
    assert sum(item["size_bytes"] for item in selected["primary_files"]) < (
        selected["required_download_bytes"]
    )


def test_selection_records_access_and_real_input_boundaries() -> None:
    payload = json.loads(REPORT.read_text())
    candidates = {item["name"]: item for item in payload["candidates"]}

    assert candidates["OpenDriveVLA 0.5B"]["gated"] is True
    assert candidates["OpenDriveVLA 0.5B"]["access_on_host"] is False
    assert candidates["ReCogDrive 2B"]["planner_checkpoint_size_bytes"] > 0

    real_input = payload["real_input"]
    assert real_input["camera_count"] == 6
    assert real_input["raw_camera_shape"] == [900, 1600, 3]
    assert real_input["license"] == "CC BY-NC-ND 4.0"
