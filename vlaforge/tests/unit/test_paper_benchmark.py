from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_paper_artifact.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_paper_artifact",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_summary_is_deterministic_and_retains_samples() -> None:
    benchmark = _module()
    values = [float(index) for index in range(30)]
    first = benchmark._summarize(values, 200, seed=7)
    second = benchmark._summarize(values, 200, seed=7)
    assert first == second
    assert first["samples"] == values
    assert first["p50"]["estimate"] == 14.5
    assert first["p95"]["ci95"][0] <= first["p95"]["estimate"]
    assert first["p95"]["ci95"][1] >= first["p95"]["estimate"]


def test_declared_backend_tensor_bytes_are_separate_and_recursive() -> None:
    benchmark = _module()
    specification = {
        "inputs": [
            {"shape": [1, 3, 4, 4], "dtype": "f32"},
            {"shape": [1, 8], "dtype": "i64"},
        ],
        "outputs": ({"shape": [2, 5], "dtype": "bf16"},),
    }
    assert benchmark._declared_tensor_bytes(specification) == (
        1 * 3 * 4 * 4 * 4 + 1 * 8 * 8 + 2 * 5 * 2
    )
