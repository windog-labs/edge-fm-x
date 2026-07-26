#!/usr/bin/env python3
"""Run and aggregate the four paper ablations on real Host-CUDA bundles.

The executable part launches fresh generated no-Python C++ Session processes
for exact-reuse modes.  The final report then combines those measurements with
already verified real-model L4 failure injection, static-memory certificates,
10k-Run residency evidence, the formal direct-artifact control matrix, and the
clean installed-wheel artifact evaluation.

This script does not compile or benchmark legacy EdgeFM CUDA kernels.  Model
kernels are the existing upstream-generated AOTI artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
_SCHEMA = "vlaforge.cuda_paper_ablations/1"
_CONFIG_SCHEMA = "vlaforge.cuda_paper_matrix_config/1"
_MODELS = ("smolvla", "diffusiondrive")
_MODES = ("full", "same", "new", "missing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("samples must be non-empty")
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _metric(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p99": _nearest_rank(values, 0.99),
        "minimum": min(values),
        "maximum": max(values),
    }


def _cluster_summary(
    process_samples: list[list[int]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(process_samples) < 2 or any(not item for item in process_samples):
        raise ValueError("cluster summary requires independent processes")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    flat = [float(value) for group in process_samples for value in group]
    estimates = _metric(flat)
    process_means = [statistics.fmean(group) for group in process_samples]
    generator = random.Random(seed)
    bootstraps = {name: [] for name in ("mean", "p50", "p90", "p99")}
    for _ in range(bootstrap_resamples):
        selected = [
            process_samples[generator.randrange(len(process_samples))]
            for _ in process_samples
        ]
        metrics = _metric(
            [float(value) for group in selected for value in group]
        )
        for name in bootstraps:
            bootstraps[name].append(float(metrics[name]))
    result: dict[str, Any] = {
        "processes": len(process_samples),
        "samples_per_process": [len(item) for item in process_samples],
        "process_mean_ns": process_means,
        "process_mean_stddev_ns": statistics.stdev(process_means),
        "throughput_runs_per_second": 1e9 / float(estimates["mean"]),
        "minimum_ns": int(estimates["minimum"]),
        "maximum_ns": int(estimates["maximum"]),
    }
    for name in ("mean", "p50", "p90", "p99"):
        bootstrap = bootstraps[name]
        result[f"{name}_ns"] = {
            "estimate": float(estimates[name]),
            "ci95": [
                _nearest_rank(bootstrap, 0.025),
                _nearest_rank(bootstrap, 0.975),
            ],
        }
    return result


def _scalar_summary(
    values: list[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    clustered = _cluster_summary(
        [[int(round(value))] for value in values],
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "samples": values,
        "mean": clustered["mean_ns"]["estimate"],
        "ci95": clustered["mean_ns"]["ci95"],
        "stddev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = _json(path)
    if config.get("schema") != _CONFIG_SCHEMA:
        raise ValueError(f"invalid config schema: {path}")
    if set(config.get("models", {})) != set(_MODELS):
        raise ValueError("config must define SmolVLA and DiffusionDrive")
    python = Path(config["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    for model in _MODELS:
        model_config = config["models"][model]
        for required in ("workloads", "bundle_root"):
            if required not in model_config:
                raise ValueError(f"{model} config missing {required}")
        if not Path(model_config["bundle_root"]).joinpath("bundle.json").is_file():
            raise FileNotFoundError(model_config["bundle_root"])
        workload = _json(Path(model_config["workloads"]))
        profiles = workload.get("profiles", ())
        baseline = [
            item for item in profiles if item.get("name") == "baseline"
        ]
        if len(baseline) != 1 or not Path(baseline[0]["root"]).is_dir():
            raise ValueError(f"{model} has no usable baseline workload")
        model_config["ablation_input_root"] = baseline[0]["root"]
    return config


def _schedule(process_repeats: int, *, seed: int) -> list[dict[str, Any]]:
    tasks = [
        {"model": model, "mode": mode, "repeat": repeat}
        for model in _MODELS
        for mode in _MODES
        for repeat in range(process_repeats)
    ]
    random.Random(seed).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["schedule_index"] = index
        task["key"] = (
            f"{task['model']}/{task['mode']}/"
            f"repeat_{task['repeat']:02d}"
        )
    return tasks


def _task_root(output_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "raw"
        / str(task["model"])
        / str(task["mode"])
        / f"repeat_{int(task['repeat']):02d}"
    )


def _report_path(
    output_root: Path,
    task: Mapping[str, Any],
    samples: int,
) -> Path:
    return _task_root(output_root, task) / (
        f"{task['model']}_{task['mode']}_{samples}.json"
    )


def _read_samples(report_path: Path, samples: int) -> list[int]:
    report = _json(report_path)
    if report.get("status") != "passed" or int(report["samples"]) != samples:
        raise ValueError(f"incomplete report: {report_path}")
    csv_path = report_path.with_suffix(".csv")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [int(item["latency_ns"]) for item in rows]
    probes = [float(item["output_probe"]) for item in rows]
    if (
        len(values) != samples
        or any(not math.isfinite(item) for item in probes)
    ):
        raise ValueError(f"invalid samples: {csv_path}")
    return values


def _command(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    task: Mapping[str, Any],
    warmup: int,
    samples: int,
) -> list[str]:
    task_root = _task_root(output_root, task)
    binary_root = output_root / "binaries" / str(task["model"])
    model_config = config["models"][task["model"]]
    command = [
        str(config["python"]),
        str(_SOURCE_ROOT / "tools" / "benchmark_generated_l4.py"),
        "--model",
        str(task["model"]),
        "--bundle-root",
        str(model_config["bundle_root"]),
        "--input-root",
        str(model_config["ablation_input_root"]),
        "--output-root",
        str(task_root),
        "--binary-root",
        str(binary_root),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--mode",
        str(task["mode"]),
    ]
    if (binary_root / "bin/vlaforge_generated_benchmark").is_file():
        command.append("--reuse-binary")
    return command


def _run_tasks(
    *,
    tasks: list[dict[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
    warmup: int,
    samples: int,
) -> None:
    environment = {
        **dict(os.environ),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
    }
    progress_path = output_root / "progress.json"
    completed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_root = _task_root(output_root, task)
        task_root.mkdir(parents=True, exist_ok=True)
        report_path = _report_path(output_root, task, samples)
        if report_path.is_file():
            _read_samples(report_path, samples)
            completed_tasks.append({**task, "status": "reused"})
            continue
        command = _command(
            config=config,
            output_root=output_root,
            task=task,
            warmup=warmup,
            samples=samples,
        )
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        (task_root / "process.log").write_text(
            json.dumps(
                {
                    "command": command,
                    "elapsed_seconds": elapsed,
                    "returncode": completed.returncode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n\n[stdout]\n"
            + completed.stdout
            + "\n[stderr]\n"
            + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ablation task failed: {task['key']}; "
                f"see {task_root / 'process.log'}"
            )
        _read_samples(report_path, samples)
        completed_tasks.append(
            {**task, "status": "executed", "wall_seconds": elapsed}
        )
        progress_path.write_text(
            json.dumps(
                {
                    "schema": "vlaforge.cuda_paper_ablation_progress/1",
                    "completed": len(completed_tasks),
                    "total": len(tasks),
                    "tasks": completed_tasks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _copy_raw(
    *,
    output_root: Path,
    report_dir: Path,
    task: Mapping[str, Any],
    samples: int,
) -> tuple[Path, list[dict[str, Any]]]:
    source = _report_path(output_root, task, samples)
    destination_root = (
        report_dir
        / "raw"
        / str(task["model"])
        / str(task["mode"])
        / f"repeat_{int(task['repeat']):02d}"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "report.json"
    shutil.copyfile(source, destination)
    records = []
    for source_path, destination_path in (
        (source, destination),
        (source.with_suffix(".csv"), destination.with_suffix(".csv")),
    ):
        shutil.copyfile(source_path, destination_path)
        records.append(
            {
                "path": str(destination_path.relative_to(report_dir)),
                "sha256": _sha256(destination_path),
                "size_bytes": destination_path.stat().st_size,
            }
        )
    return destination, records


def _exact_reuse(
    *,
    tasks: list[dict[str, Any]],
    output_root: Path,
    report_dir: Path,
    process_repeats: int,
    warmup: int,
    samples: int,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    inventory = []
    for task in tasks:
        source = _report_path(output_root, task, samples)
        report = _json(source)
        latencies = _read_samples(source, samples)
        copied, records = _copy_raw(
            output_root=output_root,
            report_dir=report_dir,
            task=task,
            samples=samples,
        )
        inventory.extend(records)
        grouped.setdefault((task["model"], task["mode"]), []).append(
            {
                "repeat": int(task["repeat"]),
                "latencies": latencies,
                "report": report,
                "raw_report": str(copied.relative_to(report_dir)),
            }
        )

    cells = []
    for (model, mode), records in sorted(grouped.items()):
        records.sort(key=lambda item: item["repeat"])
        if len(records) != process_repeats:
            raise ValueError(f"{model}/{mode} process count mismatch")
        seed = int.from_bytes(
            hashlib.sha256(f"{model}/{mode}".encode()).digest()[:8],
            "little",
        )
        runtimes = [item["report"]["runtime"] for item in records]
        if any(int(item["transaction_aborts"]) != 0 for item in runtimes):
            raise ValueError(f"{model}/{mode} observed transaction abort")
        if model == "diffusiondrive":
            expected_hits = samples if mode == "same" else 0
            expected_misses = 0 if mode == "same" else samples
            if any(
                int(item["cache_hits"]) != expected_hits
                or int(item["cache_misses"]) != expected_misses
                for item in runtimes
            ):
                raise ValueError(
                    f"{model}/{mode} exact-cache accounting mismatch"
                )
        cell = {
            "model": model,
            "mode": mode,
            "processes": process_repeats,
            "warmup_per_process": warmup,
            "samples_per_process": samples,
            "steady_latency": _cluster_summary(
                [item["latencies"] for item in records],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            ),
            "fresh_process_initialization": _scalar_summary(
                [float(item["initialization_ns"]) for item in runtimes],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 1,
            ),
            "first_run": _scalar_summary(
                [float(item["first_run_ns"]) for item in runtimes],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 2,
            ),
            "trace_totals": {
                key: sum(int(item[key]) for item in runtimes)
                for key in (
                    "regions",
                    "cache_hits",
                    "cache_misses",
                    "state_commits",
                    "transaction_commits",
                    "transaction_aborts",
                    "output_commits",
                    "resets",
                )
            },
            "memory": {
                "maximum_absolute_rss_drift_kib": max(
                    abs(
                        int(item["report"]["memory"]["rss_drift_kib"])
                    )
                    for item in records
                ),
                "maximum_absolute_cuda_drift_bytes": max(
                    abs(
                        int(
                            item["report"]["memory"][
                                "cuda_used_drift_bytes"
                            ]
                        )
                    )
                    for item in records
                ),
            },
            "raw_reports": [item["raw_report"] for item in records],
        }
        cells.append(cell)
    index = {(item["model"], item["mode"]): item for item in cells}
    for cell in cells:
        full = index[(cell["model"], "full")]["steady_latency"]["mean_ns"][
            "estimate"
        ]
        current = cell["steady_latency"]["mean_ns"]["estimate"]
        cell["full_over_mode_speedup"] = full / current
        cell["interpretation"] = (
            "exact condition-cache control"
            if cell["model"] == "diffusiondrive"
            else (
                "full recompute control"
                if cell["mode"] == "full"
                else "ChunkedAction Adapter queue/cursor path; not a core cache-only result"
            )
        )
    return cells, inventory


def _static_arena(
    *,
    l4_reports: Mapping[str, Mapping[str, Any]],
    real_cuda: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for model in _MODELS:
        memory = l4_reports[model]["memory"]
        soak = real_cuda["soak"][model]
        baseline = int(memory["arena_baseline_bytes"])
        compiled = int(memory["arena_compiled_bytes"])
        saved = int(memory["arena_saved_bytes"])
        if baseline - compiled != saved or compiled <= 0 or saved < 0:
            raise ValueError(f"{model} static-arena certificate mismatch")
        if (
            int(soak["cuda_drift_bytes"]) != 0
            or int(soak["transaction_aborts"]) != 0
            or int(soak["transaction_commits"]) != 10_000
        ):
            raise ValueError(f"{model} residency/soak evidence mismatch")
        rows.append(
            {
                "model": model,
                "control": "unpacked logical-lifetime memory-plan baseline",
                "treatment": "verified packed static arena",
                "baseline_bytes": baseline,
                "compiled_bytes": compiled,
                "saved_bytes": saved,
                "saved_percent": 100.0 * saved / baseline,
                "authoritative_state_bytes": int(
                    memory["authoritative_state_bytes"]
                ),
                "derived_cache_bytes": int(memory["derived_cache_bytes"]),
                "soak_runs": int(soak["transaction_commits"]),
                "soak_cuda_drift_bytes": int(soak["cuda_drift_bytes"]),
                "soak_rss_drift_kib": int(soak["rss_drift_kib"]),
                "claim_boundary": (
                    "plan-level lifetime-packing ablation plus generated "
                    "Session residency check; not a dynamic allocator "
                    "microbenchmark"
                ),
            }
        )
    return rows


def _transactions(
    l4_reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    preservation_keys = {
        "smolvla": "failure_preserved_uncommitted_output",
        "diffusiondrive": "failure_exposed_no_uncommitted_output",
    }
    for model in _MODELS:
        transaction = l4_reports[model]["transaction"]
        preserved = bool(transaction[preservation_keys[model]])
        if (
            not preserved
            or int(transaction["failure_retry_transaction_aborts"]) != 1
            or int(transaction["failure_retry_transaction_commits"]) != 1
            or int(transaction["validation_failure_status_code"]) == 0
        ):
            raise ValueError(f"{model} transaction failure/retry mismatch")
        if model == "smolvla" and (
            transaction["state_version_sequence"] != "passed"
            or int(transaction["failure_retry_state_commits"]) != 2
        ):
            raise ValueError("SmolVLA authoritative-state rollback mismatch")
        rows.append(
            {
                "model": model,
                "injected_failure": "non-finite output validation",
                "validation_failure_status_code": int(
                    transaction["validation_failure_status_code"]
                ),
                "transaction_abort_delta": int(
                    transaction["failure_retry_transaction_aborts"]
                ),
                "retry_commit_delta": int(
                    transaction["failure_retry_transaction_commits"]
                ),
                "retry_state_commit_delta": int(
                    transaction["failure_retry_state_commits"]
                ),
                "committed_output_preserved": preserved,
                "authoritative_state_version_sequence": transaction.get(
                    "state_version_sequence",
                    "not_applicable_stateless_model",
                ),
                "retry_cache_hits": int(
                    transaction["failure_retry_cache_hits"]
                ),
                "retry_cache_misses": int(
                    transaction["failure_retry_cache_misses"]
                ),
                "passed": True,
            }
        )
    return rows


def _deployment_boundary(
    *,
    matrix: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        matrix.get("status") != "passed"
        or not matrix["summary"]["all_output_parity_passed"]
    ):
        raise ValueError("formal CUDA matrix did not pass")
    comparisons = {}
    for model in _MODELS:
        rows = [
            item for item in matrix["comparisons"] if item["model"] == model
        ]
        if len(rows) != 5:
            raise ValueError(f"{model} deployment control count mismatch")
        overhead = [
            float(item["generated_over_direct_percent"]) for item in rows
        ]
        comparisons[model] = {
            "control": "direct AOTI artifact invocation",
            "treatment": "generated no-Python C++ ModelSession::Run",
            "workloads": len(rows),
            "generated_over_direct_percent": {
                "mean": statistics.fmean(overhead),
                "minimum": min(overhead),
                "maximum": max(overhead),
            },
            "direct_generated_output_exact": all(
                item["passed"]
                for item in matrix["output_parity"]
                if item["model"] == model
            ),
        }
    wheel = reproducibility["installed_wheel_artifact_evaluation"][
        "artifact_evaluation"
    ]
    bundles = [
        wheel["session_resident_bundle"],
        wheel["invocation_resident_bundle"],
    ]
    expected_cases = {
        "abi-mismatch",
        "corrupt-artifact",
        "missing-artifact",
        "schema-mismatch",
        "wrong-device",
        "wrong-dtype",
        "wrong-layout",
        "wrong-shape",
    }
    if any(
        bundle["status"] != "passed"
        or not bundle["invalid_python_environment_run"]
        or bundle["python_linked"]
        or set(bundle["negative_cases"]) != expected_cases
        or set(bundle["negative_cases"].values()) != {"rejected"}
        for bundle in bundles
    ):
        raise ValueError("clean-wheel deployment boundary mismatch")
    return {
        "real_model_control": comparisons,
        "clean_wheel_artifact_evaluation": {
            "installed_from_wheel_only": True,
            "non_git_cwd": True,
            "residency_variants": [
                bundle["artifact_residency"] for bundle in bundles
            ],
            "invalid_python_environment_run": True,
            "links_libpython": False,
            "negative_contract_cases_per_variant": len(expected_cases),
            "all_negative_contract_cases_rejected": True,
            "maximum_absolute_error": max(
                float(bundle["max_abs_error"]) for bundle in bundles
            ),
        },
        "claim_boundary": (
            "direct AOTI is the orchestration-overhead control; the wheel "
            "evaluation is synthetic packaging evidence and does not add "
            "real-model coverage"
        ),
    }


def _write_exact_csv(path: Path, cells: list[Mapping[str, Any]]) -> None:
    fields = (
        "model",
        "mode",
        "processes",
        "samples_per_process",
        "mean_ms",
        "mean_ci95_low_ms",
        "mean_ci95_high_ms",
        "process_mean_stddev_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "full_over_mode_speedup",
        "cache_hits",
        "cache_misses",
        "transaction_commits",
        "transaction_aborts",
        "maximum_absolute_cuda_drift_bytes",
        "maximum_absolute_rss_drift_kib",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            latency = cell["steady_latency"]
            trace = cell["trace_totals"]
            memory = cell["memory"]
            writer.writerow(
                {
                    "model": cell["model"],
                    "mode": cell["mode"],
                    "processes": cell["processes"],
                    "samples_per_process": cell["samples_per_process"],
                    "mean_ms": latency["mean_ns"]["estimate"] / 1e6,
                    "mean_ci95_low_ms": latency["mean_ns"]["ci95"][0] / 1e6,
                    "mean_ci95_high_ms": latency["mean_ns"]["ci95"][1] / 1e6,
                    "process_mean_stddev_ms": (
                        latency["process_mean_stddev_ns"] / 1e6
                    ),
                    "p50_ms": latency["p50_ns"]["estimate"] / 1e6,
                    "p90_ms": latency["p90_ns"]["estimate"] / 1e6,
                    "p99_ms": latency["p99_ns"]["estimate"] / 1e6,
                    "full_over_mode_speedup": cell[
                        "full_over_mode_speedup"
                    ],
                    "cache_hits": trace["cache_hits"],
                    "cache_misses": trace["cache_misses"],
                    "transaction_commits": trace["transaction_commits"],
                    "transaction_aborts": trace["transaction_aborts"],
                    "maximum_absolute_cuda_drift_bytes": memory[
                        "maximum_absolute_cuda_drift_bytes"
                    ],
                    "maximum_absolute_rss_drift_kib": memory[
                        "maximum_absolute_rss_drift_kib"
                    ],
                }
            )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VLAForge paper ablations",
        "",
        f"Status: **{report['status']}**.",
        "",
        "## 1. InputRevision exact reuse",
        "",
        (
            "Five fresh processes per cell; confidence intervals use "
            "process-cluster bootstrap. DiffusionDrive is the cache-only "
            "control. SmolVLA non-full modes also exercise the Adapter-owned "
            "action queue and therefore are not presented as cache-only."
        ),
        "",
        "| Model | Mode | Mean ms [95% CI] | p50 | p90 | p99 | Full/mode | Hit/miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["exact_reuse"]["cells"]:
        latency = cell["steady_latency"]
        trace = cell["trace_totals"]
        lines.append(
            f"| {cell['model']} | {cell['mode']} | "
            f"{latency['mean_ns']['estimate'] / 1e6:.3f} "
            f"[{latency['mean_ns']['ci95'][0] / 1e6:.3f}, "
            f"{latency['mean_ns']['ci95'][1] / 1e6:.3f}] | "
            f"{latency['p50_ns']['estimate'] / 1e6:.3f} | "
            f"{latency['p90_ns']['estimate'] / 1e6:.3f} | "
            f"{latency['p99_ns']['estimate'] / 1e6:.3f} | "
            f"{cell['full_over_mode_speedup']:.3f}x | "
            f"{trace['cache_hits']}/{trace['cache_misses']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Static arena lifetime packing",
            "",
            "| Model | Baseline | Packed | Saved | 10k CUDA drift | RSS drift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["static_arena"]:
        lines.append(
            f"| {row['model']} | {row['baseline_bytes']:,} B | "
            f"{row['compiled_bytes']:,} B | {row['saved_bytes']:,} B "
            f"({row['saved_percent']:.3f}%) | "
            f"{row['soak_cuda_drift_bytes']} B | "
            f"{row['soak_rss_drift_kib']} KiB |"
        )
    lines.extend(
        [
            "",
            (
                "The control is the compiler certificate's unpacked logical-"
                "lifetime memory plan. This is a plan-level packing ablation, "
                "not a dynamic allocator microbenchmark."
            ),
            "",
            "## 3. Transaction failure and retry",
            "",
            "| Model | Abort | Retry commit | Retry state commit | Output preserved | State version |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in report["transaction_failure_retry"]:
        lines.append(
            f"| {row['model']} | {row['transaction_abort_delta']} | "
            f"{row['retry_commit_delta']} | "
            f"{row['retry_state_commit_delta']} | "
            f"{str(row['committed_output_preserved']).lower()} | "
            f"{row['authoritative_state_version_sequence']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Deployment boundary",
            "",
            "| Model | Generated C++ overhead vs direct AOTI | Exact direct/C++ output |",
            "|---|---:|---|",
        ]
    )
    for model, item in report["deployment_boundary"][
        "real_model_control"
    ].items():
        overhead = item["generated_over_direct_percent"]
        lines.append(
            f"| {model} | mean {overhead['mean']:+.3f}% "
            f"[{overhead['minimum']:+.3f}%, "
            f"{overhead['maximum']:+.3f}%] | "
            f"{str(item['direct_generated_output_exact']).lower()} |"
        )
    clean = report["deployment_boundary"][
        "clean_wheel_artifact_evaluation"
    ]
    lines.extend(
        [
            "",
            (
                f"The clean installed-wheel evaluation passed both "
                f"{'/'.join(clean['residency_variants'])} residency modes, "
                f"ran under an invalid Python environment without libpython, "
                f"and rejected all "
                f"{clean['negative_contract_cases_per_variant']} negative "
                f"schema/ABI/artifact/input cases per variant."
            ),
            "",
            "## Claim boundary",
            "",
            (
                "All timing is RTX 3060 (sm_86)/CUDA 12.8 Host-CUDA. "
                "AOTI/cuDNN/CUTLASS/Triton model kernels are upstream; "
                "VLAForge claims the state/cache/transaction/memory and "
                "verified artifact orchestration semantics. Orin, real-vehicle "
                "loops, sensor synchronization, middleware and legacy EdgeFM "
                "kernel optimization are out of scope."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--matrix-report", type=Path, required=True)
    parser.add_argument("--smolvla-l4-report", type=Path, required=True)
    parser.add_argument("--diffusiondrive-l4-report", type=Path, required=True)
    parser.add_argument("--real-cuda-report", type=Path, required=True)
    parser.add_argument("--reproducibility-report", type=Path, required=True)
    parser.add_argument("--process-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--schedule-seed", type=int, default=20260726)
    args = parser.parse_args(argv)
    if (
        args.process_repeats < 2
        or args.warmup < 0
        or args.samples < 1
        or args.bootstrap_resamples < 100
    ):
        parser.error("invalid process/warmup/sample/bootstrap counts")
    config = _load_config(args.config)
    tasks = _schedule(args.process_repeats, seed=args.schedule_seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _run_tasks(
        tasks=tasks,
        config=config,
        output_root=args.output_root,
        warmup=args.warmup,
        samples=args.samples,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    exact, raw_inventory = _exact_reuse(
        tasks=tasks,
        output_root=args.output_root,
        report_dir=args.report_dir,
        process_repeats=args.process_repeats,
        warmup=args.warmup,
        samples=args.samples,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    l4_reports = {
        "smolvla": _json(args.smolvla_l4_report),
        "diffusiondrive": _json(args.diffusiondrive_l4_report),
    }
    real_cuda = _json(args.real_cuda_report)
    matrix = _json(args.matrix_report)
    reproducibility = _json(args.reproducibility_report)
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": {
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "source_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=_REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "environment": {
            "host": platform.platform(),
            "gpu": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version,"
                    "memory.total,power.limit",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": str(config["python"]),
        },
        "protocol": {
            "exact_reuse_processes_per_cell": args.process_repeats,
            "exact_reuse_warmup_per_process": args.warmup,
            "exact_reuse_samples_per_process": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_unit": "independent process cluster",
            "schedule_seed": args.schedule_seed,
            "timing_boundary": (
                "generated no-Python C++ ModelSession::Run including backend "
                "synchronization"
            ),
        },
        "exact_reuse": {
            "cells": exact,
            "primary_cache_only_model": "diffusiondrive",
            "smolvla_caveat": (
                "non-full modes include ChunkedAction Adapter queue/cursor "
                "behavior and are not a core cache-only ablation"
            ),
        },
        "static_arena": _static_arena(
            l4_reports=l4_reports,
            real_cuda=real_cuda,
        ),
        "transaction_failure_retry": _transactions(l4_reports),
        "deployment_boundary": _deployment_boundary(
            matrix=matrix,
            reproducibility=reproducibility,
        ),
        "source_evidence": {
            "formal_cuda_matrix": str(args.matrix_report.resolve()),
            "smolvla_l4": str(args.smolvla_l4_report.resolve()),
            "diffusiondrive_l4": str(
                args.diffusiondrive_l4_report.resolve()
            ),
            "real_cuda_soak": str(args.real_cuda_report.resolve()),
            "installed_wheel": str(args.reproducibility_report.resolve()),
        },
        "raw_evidence": sorted(
            raw_inventory,
            key=lambda item: item["path"],
        ),
        "summary": {
            "ablation_count": 4,
            "exact_reuse_task_count": len(tasks),
            "exact_reuse_cell_count": len(exact),
            "minimum_processes_per_exact_reuse_cell": min(
                item["processes"] for item in exact
            ),
            "all_transaction_failure_retry_passed": all(
                item["passed"]
                for item in _transactions(l4_reports)
            ),
            "clean_wheel_boundary_passed": True,
        },
        "claim_boundary": {
            "host_cuda": True,
            "gpu": "RTX 3060 sm_86",
            "orin": False,
            "real_vehicle_or_sensor_loop": False,
            "legacy_edgefm_cuda_kernels": False,
            "model_kernel_optimization_attributed_to_vlaforge": False,
        },
    }
    json_path = args.report_dir / "paper_ablations.json"
    csv_path = args.report_dir / "exact_reuse.csv"
    markdown_path = args.report_dir / "README.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_exact_csv(csv_path, exact)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
