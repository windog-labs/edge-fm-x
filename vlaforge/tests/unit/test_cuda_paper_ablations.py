from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_cuda_paper_ablations.py"
    )
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_schedule_has_four_modes_and_five_processes() -> None:
    benchmark = _module()
    first = benchmark._schedule(5, seed=17)
    second = benchmark._schedule(5, seed=17)
    assert first == second
    assert len(first) == 2 * 4 * 5
    assert len({item["key"] for item in first}) == len(first)
    for model in benchmark._MODELS:
        for mode in benchmark._MODES:
            assert {
                item["repeat"]
                for item in first
                if item["model"] == model and item["mode"] == mode
            } == set(range(5))


def test_static_arena_uses_certificate_control_and_soak() -> None:
    benchmark = _module()
    reports = {
        model: {
            "memory": {
                "arena_baseline_bytes": 120,
                "arena_compiled_bytes": 100,
                "arena_saved_bytes": 20,
                "authoritative_state_bytes": 4,
                "derived_cache_bytes": 80,
            }
        }
        for model in benchmark._MODELS
    }
    real_cuda = {
        "soak": {
            model: {
                "transaction_commits": 10_000,
                "transaction_aborts": 0,
                "cuda_drift_bytes": 0,
                "rss_drift_kib": 8,
            }
            for model in benchmark._MODELS
        }
    }
    rows = benchmark._static_arena(
        l4_reports=reports,
        real_cuda=real_cuda,
    )
    assert len(rows) == 2
    assert rows[0]["saved_percent"] == 100.0 / 6.0
    assert rows[0]["soak_runs"] == 10_000


def test_transactions_require_abort_preservation_and_retry() -> None:
    benchmark = _module()
    reports = {
        "smolvla": {
            "transaction": {
                "failure_preserved_uncommitted_output": True,
                "failure_retry_transaction_aborts": 1,
                "failure_retry_transaction_commits": 1,
                "failure_retry_state_commits": 2,
                "validation_failure_status_code": 7,
                "state_version_sequence": "passed",
                "failure_retry_cache_hits": 1,
                "failure_retry_cache_misses": 1,
            }
        },
        "diffusiondrive": {
            "transaction": {
                "failure_exposed_no_uncommitted_output": True,
                "failure_retry_transaction_aborts": 1,
                "failure_retry_transaction_commits": 1,
                "failure_retry_state_commits": 0,
                "validation_failure_status_code": 7,
                "failure_retry_cache_hits": 1,
                "failure_retry_cache_misses": 1,
            }
        },
    }
    rows = benchmark._transactions(reports)
    assert len(rows) == 2
    assert all(item["passed"] for item in rows)
    assert rows[0]["authoritative_state_version_sequence"] == "passed"


def test_cluster_bootstrap_is_deterministic() -> None:
    benchmark = _module()
    samples = [[100, 110], [105, 115], [110, 120], [115, 125], [120, 130]]
    first = benchmark._cluster_summary(
        samples,
        bootstrap_resamples=200,
        seed=9,
    )
    second = benchmark._cluster_summary(
        samples,
        bootstrap_resamples=200,
        seed=9,
    )
    assert first == second
    assert first["processes"] == 5
    assert first["mean_ns"]["ci95"][0] <= first["mean_ns"]["estimate"]
    assert first["mean_ns"]["ci95"][1] >= first["mean_ns"]["estimate"]
