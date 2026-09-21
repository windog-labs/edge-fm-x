"""Observation-series integrity fixtures, not real-model execution evidence."""

import hashlib
import json
import sys

import pytest
from vlaforge.adapters.openpi.openpi_series import load_series_index


def make_index(root):
    samples = []
    for frame in (0, 10):
        source = root / f"frame-{frame}.json"
        source.write_text(json.dumps({"schema": "vlaforge.openpi_input_pack/1", "dataset": "fixture", "revision": "pinned",
            "episode_index": 0, "frame_index": frame, "noise": {"seed": frame + 1}}))
        samples.append({"sample_id": f"frame-{frame}", "manifest": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "episode_index": 0, "frame_index": frame, "noise_seed": frame + 1})
    index = {"schema": "vlaforge.openpi_observation_series/1", "dataset": "fixture", "revision": "pinned", "samples": samples}
    path = root / "index.json"
    path.write_text(json.dumps(index))
    return path, index


def test_loads_distinct_chronological_inputs(tmp_path):
    path, index = make_index(tmp_path)
    restored, samples = load_series_index(path)
    assert restored == index and len(samples) == 2
    assert [entry[1]["frame_index"] for entry in samples] == [0, 10]


@pytest.mark.parametrize("change", ["schema", "single", "duplicate-id", "reorder", "sha", "frame", "seed", "revision", "escape", "duplicate-observation"])
def test_bad_series_rejected_before_model_import(tmp_path, change):
    path, value = make_index(tmp_path)
    first = value["samples"][0]
    if change == "schema":
        value["schema"] += "future"
    elif change == "single":
        value["samples"] = value["samples"][:1]
    elif change == "duplicate-id":
        value["samples"][1]["sample_id"] = first["sample_id"]
    elif change == "reorder":
        value["samples"].reverse()
    elif change == "sha":
        first["sha256"] = "0" * 64
    elif change == "frame":
        first["frame_index"] = 5
    elif change == "seed":
        first["noise_seed"] = 2
    elif change == "revision":
        value["revision"] = "changed"
    elif change == "escape":
        outside = tmp_path.parent / (tmp_path.name + "-outside.json")
        outside.write_bytes((tmp_path / first["manifest"]).read_bytes())
        first["manifest"] = "../" + outside.name
    else:
        value["samples"][1] = {**first, "sample_id": "different-noise-is-not-another-observation"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_series_index(path)


def test_reference_rejects_extraction_option_before_cuda_import(monkeypatch, capsys):
    from vlaforge.adapters.openpi.openpi_series import main

    monkeypatch.setattr(sys, "argv", ["openpi_series", "--phase", "reference",
        "--capture-report", "capture.json", "--capture-sha256", "0" * 64,
        "--input-series", "inputs.json", "--input-series-sha256", "1" * 64,
        "--output", "out", "--package-extraction-root", "/tmp"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "only used by direct validation" in capsys.readouterr().err
