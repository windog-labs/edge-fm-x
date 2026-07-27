from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_minddrive_path_matrix.py"
    )
    specification = importlib.util.spec_from_file_location(
        "benchmark_minddrive_path_matrix",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_schedule_has_every_path_and_repeat_once() -> None:
    benchmark = _module()

    tasks = benchmark._schedule(5, seed=7)

    assert len(tasks) == 15
    assert {
        (task["path"], task["repeat"]) for task in tasks
    } == {
        (path, repeat)
        for path in benchmark._PATHS
        for repeat in range(5)
    }
    assert [task["schedule_index"] for task in tasks] == list(range(15))


def test_probe_validation_uses_predeclared_path_tolerances() -> None:
    benchmark = _module()
    direct = [0.1, 0.2, 0.3]
    generated = [0.1 + 1.0e-8, 0.2, 0.3]
    eager = [0.102, 0.199, 0.301]
    sequences = {
        "eager": [eager, eager],
        "direct_artifact": [direct, direct],
        "generated_session": [generated, generated],
    }

    result = benchmark._validate_probe_sequences(sequences)

    assert result["all_passed"] is True
    assert result["comparisons"][0][
        "direct_generated_maximum_absolute_error"
    ] == pytest.approx(1.0e-8)
    assert result["comparisons"][0][
        "eager_direct_maximum_absolute_error"
    ] == pytest.approx(0.002)


def test_probe_validation_rejects_post_hoc_tolerance_expansion() -> None:
    benchmark = _module()
    direct = [0.1, 0.2]
    generated = [0.1, 0.2]
    eager = [0.1, 0.204]

    with pytest.raises(ValueError, match="eager/direct"):
        benchmark._validate_probe_sequences(
            {
                "eager": [eager, eager],
                "direct_artifact": [direct, direct],
                "generated_session": [generated, generated],
            }
        )


def test_cluster_summary_bootstraps_independent_processes() -> None:
    benchmark = _module()

    summary = benchmark._cluster_summary(
        [[10, 20], [30, 40]],
        bootstrap_resamples=100,
        seed=11,
    )

    assert summary["processes"] == 2
    assert summary["samples_per_process"] == [2, 2]
    assert summary["mean_ns"]["estimate"] == pytest.approx(25.0)
    assert summary["process_mean_stddev_ns"] == pytest.approx(
        14.142135623730951
    )


def test_resume_archives_incomplete_task_without_deleting_evidence(
    tmp_path: Path,
) -> None:
    benchmark = _module()
    task_root = tmp_path / "repeat_01"
    task_root.mkdir()
    failure_log = task_root / "process.log"
    failure_log.write_text("failed\n", encoding="utf-8")

    first = benchmark._archive_incomplete_task(task_root)
    task_root.mkdir()
    (task_root / "process.log").write_text(
        "failed again\n", encoding="utf-8"
    )
    second = benchmark._archive_incomplete_task(task_root)

    assert first.name == "repeat_01.failed_attempt_00"
    assert first.joinpath("process.log").read_text(
        encoding="utf-8"
    ) == "failed\n"
    assert second.name == "repeat_01.failed_attempt_01"
    assert second.joinpath("process.log").read_text(
        encoding="utf-8"
    ) == "failed again\n"
