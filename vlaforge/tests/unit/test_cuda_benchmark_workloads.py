from __future__ import annotations

import array
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "materialize_cuda_benchmark_workloads.py"
    )
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_f32(path: Path, values: list[float]) -> None:
    payload = array.array("f", values).tobytes()
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializes_five_deterministic_static_profiles(
    tmp_path: Path,
) -> None:
    workloads = _module()
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "image": [1.0, 2.0],
        "state": [3.0, 4.0],
        "noise": [5.0, 6.0],
    }
    inputs = {}
    for name, values in tensors.items():
        path = source / f"{name}.bin"
        _write_f32(path, values)
        inputs[name] = {
            "dtype": "float32",
            "shape": [2],
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (source / "inputs.json").write_text(
        json.dumps({"inputs": inputs}),
        encoding="utf-8",
    )

    first = workloads.materialize("smolvla", source, tmp_path / "first")
    second = workloads.materialize("smolvla", source, tmp_path / "second")
    assert len(first["profiles"]) == 5
    assert [item["name"] for item in first["profiles"]] == [
        "baseline",
        "observation_minus_1pct",
        "observation_plus_1pct",
        "context_plus_0p01",
        "noise_plus_1pct",
    ]
    assert [
        item["input_sha256"] for item in first["profiles"]
    ] == [
        item["input_sha256"] for item in second["profiles"]
    ]
    assert (
        first["profiles"][0]["input_sha256"]["state"]
        == first["profiles"][1]["input_sha256"]["state"]
    )
    assert (
        first["profiles"][0]["input_sha256"]["image"]
        != first["profiles"][1]["input_sha256"]["image"]
    )
    assert first["invariants"]["static_shape_dtype_unchanged"] is True
