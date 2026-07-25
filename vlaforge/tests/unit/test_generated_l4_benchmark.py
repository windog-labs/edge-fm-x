from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_generated_l4.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_generated_l4",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_latency_summary_uses_nearest_rank_percentiles() -> None:
    benchmark = _module()
    result = benchmark._summary(list(range(1, 101)))

    assert result["count"] == 100
    assert result["mean_ns"] == pytest.approx(50.5)
    assert result["p50_ns"] == 50
    assert result["p90_ns"] == 90
    assert result["p99_ns"] == 99


def test_runner_output_parser_preserves_trace_and_samples() -> None:
    benchmark = _module()
    samples, summary = benchmark._parse_output(
        "\n".join(
            (
                "SAMPLE,0,100,7,1,0.25",
                "SAMPLE,1,90,8,1,0.5",
                "SUMMARY,10,19,20,21,22,29,30,31,32,0.75,"
                "2,3,4,5,6,0,6,1,7,8",
            )
        )
    )

    assert [item["latency_ns"] for item in samples] == [100, 90]
    assert summary["cache_hits"] == 3
    assert summary["cache_misses"] == 4
    assert summary["transaction_aborts"] == 0
    assert summary["state_1_version"] == 8


def test_diffusiondrive_same_revision_requires_exact_cache_hits() -> None:
    benchmark = _module()
    samples = [
        {
            "index": 0,
            "latency_ns": 100,
            "revision": 7,
            "revision_present": True,
            "output_probe": 0.25,
        }
    ]
    runtime = {
        "transaction_commits": 1,
        "transaction_aborts": 0,
        "output_commits": 1,
        "cache_hits": 1,
        "cache_misses": 0,
    }

    benchmark._validate_runtime(
        model="diffusiondrive",
        mode="same",
        sample_count=1,
        samples=samples,
        runtime=runtime,
    )

    runtime["cache_hits"] = 0
    with pytest.raises(RuntimeError, match="revision/cache"):
        benchmark._validate_runtime(
            model="diffusiondrive",
            mode="same",
            sample_count=1,
            samples=samples,
            runtime=runtime,
        )
