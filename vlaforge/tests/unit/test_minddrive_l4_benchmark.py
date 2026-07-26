from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_minddrive_l4.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_minddrive_l4",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cluster_summary_resamples_independent_processes() -> None:
    benchmark = _module()
    summary = benchmark._cluster_summary(
        [[100, 110], [200, 210], [300, 310]],
        bootstrap_resamples=1000,
        seed=7,
    )

    assert summary["processes"] == 3
    assert summary["samples_per_process"] == [2, 2, 2]
    assert summary["mean_ns"]["estimate"] == pytest.approx(205)
    assert summary["p50_ns"]["estimate"] == 200
    assert summary["process_mean_stddev_ns"] == pytest.approx(100)


def test_output_validation_requires_revision_paths_to_match() -> None:
    benchmark = _module()

    def records(values: list[float]) -> list[dict[str, object]]:
        return [
            {
                "raw_samples": [
                    {"output_probe": value} for value in values
                ]
            },
            {
                "raw_samples": [
                    {"output_probe": value} for value in values
                ]
            },
        ]

    reports = {
        "full": records([1.0, 2.0]),
        "same": records([3.0, 4.0]),
        "new": records([3.0, 4.0]),
        "missing": records([3.0, 4.0]),
    }
    validation = benchmark._validate_outputs(reports)
    assert validation["fresh_process_deterministic"]
    assert validation["same_new_missing_exact"]

    reports["missing"][0]["raw_samples"][1]["output_probe"] = 5.0
    with pytest.raises(RuntimeError, match="missing outputs changed"):
        benchmark._validate_outputs(reports)
