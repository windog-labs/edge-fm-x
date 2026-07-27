from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "probe_real_autovla_full_eager.py"
    )
    specification = importlib.util.spec_from_file_location(
        "probe_real_autovla_full_eager",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path) -> Path:
    images: dict[str, list[str]] = {}
    for camera in (
        "front_camera",
        "front_left_camera",
        "front_right_camera",
    ):
        frames = []
        for index in range(4):
            frame = tmp_path / camera / f"{index}.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f"{camera}-{index}".encode())
            frames.append(str(frame.relative_to(tmp_path)))
        images[camera] = frames
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vlaforge.autovla_full_input/1",
                "sample_id": "unit-sample",
                "revision": 7,
                "images": images,
                "vehicle_velocity": [1.0, 0.0],
                "vehicle_acceleration": [0.1, 0.0],
                "driving_command": "turn left",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_input_manifest_resolves_externally_assembled_history(
    tmp_path: Path,
) -> None:
    probe = _module()
    features, evidence = probe.load_input_manifest(_manifest(tmp_path))
    assert evidence["revision"] == 7
    assert evidence["history_assembled_externally"] is True
    assert evidence["sensor_sync_performed_by_vlaforge"] is False
    assert len(evidence["images"]) == 12
    assert features["sensor_data_path"] == ""
    assert all(
        Path(frame).is_absolute()
        for frames in features["images"].values()
        for frame in frames
    )


def test_input_manifest_rejects_unbounded_or_missing_history(
    tmp_path: Path,
) -> None:
    probe = _module()
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["images"]["front_camera"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly four"):
        probe.load_input_manifest(path)
