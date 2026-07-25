from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_real_model_paths.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_real_model_paths",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_latency_summary_uses_nearest_rank_percentiles() -> None:
    benchmark = _module()
    result = benchmark._latency_summary(list(range(1, 101)))

    assert result["count"] == 100
    assert result["mean_ns"] == pytest.approx(50.5)
    assert result["p50_ns"] == 50
    assert result["p90_ns"] == 90
    assert result["p99_ns"] == 99
