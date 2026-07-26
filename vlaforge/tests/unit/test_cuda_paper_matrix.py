from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_cuda_paper_matrix.py"
    )
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cluster_bootstrap_is_process_aware_and_deterministic() -> None:
    benchmark = _module()
    samples = [
        [100, 110, 120],
        [105, 115, 125],
        [110, 120, 130],
        [115, 125, 135],
        [120, 130, 140],
    ]
    first = benchmark._cluster_summary(
        samples,
        bootstrap_resamples=200,
        seed=7,
    )
    second = benchmark._cluster_summary(
        samples,
        bootstrap_resamples=200,
        seed=7,
    )
    assert first == second
    assert first["processes"] == 5
    assert first["samples_per_process"] == [3] * 5
    assert first["mean_ns"]["ci95"][0] <= first["mean_ns"]["estimate"]
    assert first["mean_ns"]["ci95"][1] >= first["mean_ns"]["estimate"]
    assert first["process_mean_stddev_ns"] > 0


def test_schedule_has_five_processes_for_every_path_and_workload() -> None:
    benchmark = _module()
    workloads = {
        model: {
            "profiles": [
                {
                    "profile_id": index,
                    "name": f"workload_{index}",
                    "root": f"/tmp/{model}/{index}",
                }
                for index in range(5)
            ]
        }
        for model in benchmark._MODELS
    }
    first = benchmark._schedule(workloads, 5, seed=19)
    second = benchmark._schedule(workloads, 5, seed=19)
    assert first == second
    assert len(first) == 2 * 5 * 5 * 3
    assert len({item["key"] for item in first}) == len(first)
    for model in benchmark._MODELS:
        for workload_index in range(5):
            for path in benchmark._PATHS:
                matches = [
                    item
                    for item in first
                    if item["model"] == model
                    and item["workload"] == f"workload_{workload_index}"
                    and item["path"] == path
                ]
                assert {item["repeat"] for item in matches} == set(range(5))


def test_config_records_explicit_pythonpath(
    tmp_path: Path,
) -> None:
    benchmark = _module()
    config = {
        "schema": benchmark._CONFIG_SCHEMA,
        "python": sys.executable,
        "pythonpath": [str(tmp_path)],
        "models": {
            "smolvla": {
                key: "value"
                for key in (
                    "workloads",
                    "bundle_root",
                    "l3_root",
                    "support_root",
                    "checkpoint",
                    "vlm_path",
                    "upstream_revision",
                )
            },
            "diffusiondrive": {
                key: "value"
                for key in (
                    "workloads",
                    "bundle_root",
                    "l3_root",
                    "source_root",
                    "checkpoint",
                    "upstream_revision",
                )
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert benchmark._load_config(path)["pythonpath"] == [str(tmp_path)]
