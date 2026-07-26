#!/usr/bin/env python3
"""Run and aggregate the paper-grade real Host-CUDA benchmark matrix.

The harness launches one fresh process per model/path/workload/repetition,
retains raw samples, and uses process-cluster bootstrap confidence intervals.
It never treats the SmolVLA queue fast path or DiffusionDrive exact-cache hit
path as a full model invocation: every matrix cell uses ``full`` generated
Session mode, which resets/recomputes as required.
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
from typing import Any, Iterable, Mapping


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
_SCHEMA = "vlaforge.cuda_paper_matrix/1"
_CONFIG_SCHEMA = "vlaforge.cuda_paper_matrix_config/1"
_PATHS = ("eager", "direct_artifact", "generated_session")
_MODELS = ("smolvla", "diffusiondrive")
_OUTPUT_TOLERANCE = {"smolvla": 0.05, "diffusiondrive": 0.005}


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
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean": mean,
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
    if len(process_samples) < 2 or any(not values for values in process_samples):
        raise ValueError("cluster summary requires at least two non-empty processes")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    flat = [float(value) for values in process_samples for value in values]
    estimates = _metric(flat)
    process_means = [statistics.fmean(values) for values in process_samples]
    generator = random.Random(seed)
    bootstraps = {name: [] for name in ("mean", "p50", "p90", "p99")}
    for _ in range(bootstrap_resamples):
        selected = [
            process_samples[generator.randrange(len(process_samples))]
            for _ in process_samples
        ]
        metrics = _metric(
            [float(value) for values in selected for value in values]
        )
        for name in bootstraps:
            bootstraps[name].append(float(metrics[name]))
    result: dict[str, Any] = {
        "processes": len(process_samples),
        "samples_per_process": [len(values) for values in process_samples],
        "process_mean_ns": process_means,
        "process_mean_stddev_ns": statistics.stdev(process_means),
        "throughput_runs_per_second": 1e9 / float(estimates["mean"]),
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
    result.update(
        {
            "minimum_ns": int(estimates["minimum"]),
            "maximum_ns": int(estimates["maximum"]),
        }
    )
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


def _load_workloads(path: Path, model: str) -> dict[str, Any]:
    report = _json(path)
    if (
        report.get("schema") != "vlaforge.cuda_benchmark_workloads/1"
        or report.get("status") != "passed"
        or report.get("model") != model
        or len(report.get("profiles", ())) < 5
    ):
        raise ValueError(f"invalid workload manifest: {path}")
    for profile in report["profiles"]:
        if not Path(profile["root"]).is_dir():
            raise FileNotFoundError(profile["root"])
    return report


def _load_config(path: Path) -> dict[str, Any]:
    config = _json(path)
    if config.get("schema") != _CONFIG_SCHEMA:
        raise ValueError(f"invalid matrix config schema: {path}")
    if set(config.get("models", {})) != set(_MODELS):
        raise ValueError("matrix config must define SmolVLA and DiffusionDrive")
    python = Path(config["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    pythonpath = config.get("pythonpath", [])
    if (
        not isinstance(pythonpath, list)
        or not all(isinstance(item, str) for item in pythonpath)
    ):
        raise ValueError("matrix config pythonpath must be a string list")
    for item in pythonpath:
        if not Path(item).is_dir():
            raise FileNotFoundError(item)
    required = {
        "smolvla": (
            "workloads",
            "bundle_root",
            "l3_root",
            "support_root",
            "checkpoint",
            "vlm_path",
            "upstream_revision",
        ),
        "diffusiondrive": (
            "workloads",
            "bundle_root",
            "l3_root",
            "source_root",
            "checkpoint",
            "upstream_revision",
        ),
    }
    for model, keys in required.items():
        missing = [key for key in keys if key not in config["models"][model]]
        if missing:
            raise ValueError(f"{model} config missing: {missing}")
    return config


def _schedule(
    workloads: Mapping[str, Mapping[str, Any]],
    process_repeats: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    tasks = []
    for model in _MODELS:
        for profile in workloads[model]["profiles"]:
            for repeat in range(process_repeats):
                for path in _PATHS:
                    tasks.append(
                        {
                            "model": model,
                            "profile_id": int(profile["profile_id"]),
                            "workload": str(profile["name"]),
                            "input_root": str(profile["root"]),
                            "repeat": repeat,
                            "path": path,
                        }
                    )
    random.Random(seed).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["schedule_index"] = index
        task["key"] = (
            f"{task['model']}/{task['workload']}/"
            f"repeat_{task['repeat']:02d}/{task['path']}"
        )
    return tasks


def _telemetry() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,"
            "clocks.mem,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip()}
    names = (
        "temperature_c",
        "power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "memory_used_mib",
    )
    fields = [item.strip() for item in completed.stdout.split(",")]
    return {
        "available": len(fields) == len(names),
        **{
            name: float(value)
            for name, value in zip(names, fields, strict=False)
        },
    }


def _task_directory(output_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "raw"
        / str(task["model"])
        / str(task["workload"])
        / f"repeat_{int(task['repeat']):02d}"
        / str(task["path"])
    )


def _result_path(
    output_root: Path,
    task: Mapping[str, Any],
    samples: int,
) -> Path:
    task_root = _task_directory(output_root, task)
    if task["path"] == "generated_session":
        return task_root / f"{task['model']}_full_{samples}.json"
    return task_root / "report.json"


def _result_samples(
    report_path: Path,
    task: Mapping[str, Any],
    samples: int,
) -> tuple[list[int], list[float]]:
    report = _json(report_path)
    if report.get("status") != "passed" or int(report["samples"]) != samples:
        raise ValueError(f"incomplete benchmark report: {report_path}")
    if task["path"] != "generated_session":
        records = report["samples_raw"]
    else:
        csv_path = report_path.with_suffix(".csv")
        with csv_path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    latency = [int(item["latency_ns"]) for item in records]
    probes = [float(item["output_probe"]) for item in records]
    if len(latency) != samples or any(not math.isfinite(item) for item in probes):
        raise ValueError(f"invalid raw benchmark samples: {report_path}")
    return latency, probes


def _model_command(
    *,
    python: str,
    model: str,
    path: str,
    input_root: str,
    output: Path,
    warmup: int,
    samples: int,
    model_config: Mapping[str, Any],
) -> list[str]:
    command = [
        python,
        str(_SOURCE_ROOT / "tools" / "benchmark_real_model_paths.py"),
        "--model",
        model,
        "--path",
        path,
        "--input-root",
        input_root,
        "--output",
        str(output),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
    ]
    if path == "direct_artifact":
        command.extend(["--l3-root", str(model_config["l3_root"])])
        if model == "smolvla":
            command.extend(
                ["--support-root", str(model_config["support_root"])]
            )
    elif model == "smolvla":
        command.extend(
            [
                "--checkpoint",
                str(model_config["checkpoint"]),
                "--vlm-path",
                str(model_config["vlm_path"]),
                "--upstream-revision",
                str(model_config["upstream_revision"]),
            ]
        )
    else:
        command.extend(
            [
                "--source-root",
                str(model_config["source_root"]),
                "--checkpoint",
                str(model_config["checkpoint"]),
                "--upstream-revision",
                str(model_config["upstream_revision"]),
            ]
        )
    return command


def _generated_command(
    *,
    python: str,
    task: Mapping[str, Any],
    task_root: Path,
    binary_root: Path,
    warmup: int,
    samples: int,
    bundle_root: str,
) -> list[str]:
    command = [
        python,
        str(_SOURCE_ROOT / "tools" / "benchmark_generated_l4.py"),
        "--model",
        str(task["model"]),
        "--bundle-root",
        bundle_root,
        "--input-root",
        str(task["input_root"]),
        "--output-root",
        str(task_root),
        "--binary-root",
        str(binary_root),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--mode",
        "full",
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
) -> list[dict[str, Any]]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    if config.get("pythonpath"):
        environment["PYTHONPATH"] = os.pathsep.join(config["pythonpath"])
    progress_path = output_root / "progress.json"
    completed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_root = _task_directory(output_root, task)
        task_root.mkdir(parents=True, exist_ok=True)
        result_path = _result_path(output_root, task, samples)
        if result_path.is_file():
            _result_samples(result_path, task, samples)
            completed_tasks.append(
                {
                    **task,
                    "status": "reused",
                    "report": str(result_path),
                }
            )
            continue
        model_config = config["models"][task["model"]]
        if task["path"] == "generated_session":
            command = _generated_command(
                python=str(config["python"]),
                task=task,
                task_root=task_root,
                binary_root=output_root
                / "binaries"
                / str(task["model"]),
                warmup=warmup,
                samples=samples,
                bundle_root=str(model_config["bundle_root"]),
            )
        else:
            command = _model_command(
                python=str(config["python"]),
                model=str(task["model"]),
                path=str(task["path"]),
                input_root=str(task["input_root"]),
                output=result_path,
                warmup=warmup,
                samples=samples,
                model_config=model_config,
            )
        before = _telemetry()
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
                f"benchmark task failed: {task['key']}; "
                f"see {task_root / 'process.log'}"
            )
        _result_samples(result_path, task, samples)
        completed_tasks.append(
            {
                **task,
                "status": "executed",
                "report": str(result_path),
                "command": command,
                "wall_seconds": elapsed,
                "telemetry_before": before,
                "telemetry_after": _telemetry(),
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "schema": "vlaforge.cuda_paper_matrix_progress/1",
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
    return completed_tasks


def _report_runtime(report: Mapping[str, Any], path: str) -> dict[str, Any]:
    runtime = report["runtime"]
    if path == "generated_session":
        initialization_ns = int(runtime["initialization_ns"])
    else:
        initialization_ns = int(
            round(float(report["initialization_seconds"]) * 1e9)
        )
    return {
        "initialization_ns": initialization_ns,
        "first_run_ns": int(runtime["first_run_ns"]),
        "maximum_rss_kib": int(runtime["maximum_rss_kib"]),
        "cuda_peak_bytes": int(
            runtime.get(
                "cuda_used_peak_sampled_bytes",
                runtime.get("torch_reserved_peak_bytes", 0),
            )
        ),
        "rss_drift_kib": int(report["memory"]["rss_drift_kib"]),
        "cuda_drift_bytes": int(
            report["memory"].get(
                "cuda_used_drift_bytes",
                report["memory"].get("torch_reserved_drift_bytes", 0),
            )
        ),
    }


def _copy_raw(
    report_path: Path,
    task: Mapping[str, Any],
    report_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    destination_root = (
        report_dir
        / "raw"
        / str(task["model"])
        / str(task["workload"])
        / f"repeat_{int(task['repeat']):02d}"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{task['path']}.json"
    shutil.copyfile(report_path, destination)
    copied = [
        {
            "path": str(destination.relative_to(report_dir)),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
    ]
    csv_path = report_path.with_suffix(".csv")
    if csv_path.is_file():
        csv_destination = destination.with_suffix(".csv")
        shutil.copyfile(csv_path, csv_destination)
        copied.append(
            {
                "path": str(csv_destination.relative_to(report_dir)),
                "sha256": _sha256(csv_destination),
                "size_bytes": csv_destination.stat().st_size,
            }
        )
    return destination, copied


def _aggregate(
    *,
    tasks: list[dict[str, Any]],
    output_root: Path,
    report_dir: Path,
    workloads: Mapping[str, Mapping[str, Any]],
    process_repeats: int,
    samples: int,
    warmup: int,
    bootstrap_resamples: int,
    schedule_seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    raw_inventory = []
    parity: dict[tuple[str, str, int], dict[str, float]] = {}
    for task in tasks:
        report_path = _result_path(output_root, task, samples)
        latencies, probes = _result_samples(report_path, task, samples)
        report = _json(report_path)
        destination, copied = _copy_raw(report_path, task, report_dir)
        raw_inventory.extend(copied)
        cell = (task["model"], task["workload"], task["path"])
        records.setdefault(cell, []).append(
            {
                "repeat": int(task["repeat"]),
                "latencies": latencies,
                "probes": probes,
                "runtime": _report_runtime(report, str(task["path"])),
                "raw_report": str(destination.relative_to(report_dir)),
                "report": report,
            }
        )
        probe_values = set(probes)
        if len(probe_values) != 1:
            raise ValueError(f"output changed within process: {task['key']}")
        parity.setdefault(
            (task["model"], task["workload"], int(task["repeat"])),
            {},
        )[str(task["path"])] = probes[0]

    parity_records = []
    for key, values in sorted(parity.items()):
        if set(values) != set(_PATHS):
            raise ValueError(f"missing path for parity cell: {key}")
        direct_generated = abs(
            values["direct_artifact"] - values["generated_session"]
        )
        eager_direct = abs(values["eager"] - values["direct_artifact"])
        tolerance = _OUTPUT_TOLERANCE[key[0]]
        if direct_generated > 1e-7 or eager_direct > tolerance:
            raise ValueError(
                f"output parity failed for {key}: "
                f"{direct_generated=}, {eager_direct=}, {tolerance=}"
            )
        parity_records.append(
            {
                "model": key[0],
                "workload": key[1],
                "repeat": key[2],
                "eager_probe": values["eager"],
                "direct_artifact_probe": values["direct_artifact"],
                "generated_session_probe": values["generated_session"],
                "direct_generated_absolute_error": direct_generated,
                "eager_direct_absolute_error": eager_direct,
                "eager_direct_tolerance": tolerance,
                "passed": True,
            }
        )

    cells = []
    for (model, workload, path), process_records in sorted(records.items()):
        process_records.sort(key=lambda item: item["repeat"])
        if len(process_records) != process_repeats:
            raise ValueError(
                f"{model}/{workload}/{path} has "
                f"{len(process_records)} processes"
            )
        seed = int.from_bytes(
            hashlib.sha256(
                f"{model}/{workload}/{path}".encode()
            ).digest()[:8],
            "little",
        )
        runtimes = [item["runtime"] for item in process_records]
        cell = {
            "model": model,
            "workload": workload,
            "path": path,
            "processes": process_repeats,
            "warmup_per_process": warmup,
            "steady_samples_per_process": samples,
            "steady_latency": _cluster_summary(
                [item["latencies"] for item in process_records],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            ),
            "fresh_process_initialization": _scalar_summary(
                [item["initialization_ns"] for item in runtimes],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 1,
            ),
            "first_run": _scalar_summary(
                [item["first_run_ns"] for item in runtimes],
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 2,
            ),
            "memory": {
                "maximum_rss_kib": max(
                    item["maximum_rss_kib"] for item in runtimes
                ),
                "maximum_cuda_bytes": max(
                    item["cuda_peak_bytes"] for item in runtimes
                ),
                "maximum_absolute_rss_drift_kib": max(
                    abs(item["rss_drift_kib"]) for item in runtimes
                ),
                "maximum_absolute_cuda_drift_bytes": max(
                    abs(item["cuda_drift_bytes"]) for item in runtimes
                ),
            },
            "raw_reports": [
                item["raw_report"] for item in process_records
            ],
        }
        if path == "generated_session":
            bundle = process_records[0]["report"]["bundle"]
            cell["generated_bundle"] = {
                "digest": bundle["digest"],
                "io_schema_digest": bundle["io_schema_digest"],
                "artifact_bytes": bundle["artifact_bytes"],
                "static_arena": process_records[0]["report"]["memory"][
                    "static_arena"
                ],
                "invalid_python_environment": True,
                "links_libpython": False,
            }
        cells.append(cell)

    indexed = {
        (cell["model"], cell["workload"], cell["path"]): cell
        for cell in cells
    }
    comparisons = []
    for model in _MODELS:
        for profile in workloads[model]["profiles"]:
            workload = profile["name"]
            eager = indexed[(model, workload, "eager")]["steady_latency"][
                "mean_ns"
            ]["estimate"]
            direct = indexed[
                (model, workload, "direct_artifact")
            ]["steady_latency"]["mean_ns"]["estimate"]
            generated = indexed[
                (model, workload, "generated_session")
            ]["steady_latency"]["mean_ns"]["estimate"]
            comparisons.append(
                {
                    "model": model,
                    "workload": workload,
                    "eager_mean_ns": eager,
                    "direct_artifact_mean_ns": direct,
                    "generated_session_mean_ns": generated,
                    "generated_over_direct_percent": (
                        (generated / direct - 1.0) * 100.0
                    ),
                    "eager_over_generated_speedup": eager / generated,
                }
            )

    config_path = report_dir / "matrix_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    workload_records = {}
    for model, workload in workloads.items():
        path = report_dir / f"{model}_workloads.json"
        path.write_text(
            json.dumps(workload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        workload_records[model] = {
            "path": str(path.relative_to(report_dir)),
            "sha256": _sha256(path),
        }
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
        "protocol": {
            "models": list(_MODELS),
            "paths": list(_PATHS),
            "workloads_per_model": 5,
            "independent_processes_per_cell": process_repeats,
            "warmup_per_process": warmup,
            "steady_samples_per_process": samples,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_unit": "independent process cluster",
            "schedule_seed": schedule_seed,
            "timing_boundary": (
                "one full eager/action-chunk or planning invocation, direct "
                "AOTI invocation, or generated ModelSession::Run including "
                "backend synchronization"
            ),
            "cold_start_definition": (
                "fresh process initialization and first Run; OS page cache "
                "is not forcibly dropped"
            ),
            "steady_definition": (
                "post-first-Run, post-warmup full invocation; cache-hit and "
                "SmolVLA queue fast paths are excluded"
            ),
        },
        "environment": {
            "host": platform.platform(),
            "python": str(config["python"]),
            "pythonpath": list(config.get("pythonpath", ())),
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
        },
        "workloads": workload_records,
        "cells": cells,
        "comparisons": comparisons,
        "output_parity": parity_records,
        "raw_evidence": sorted(raw_inventory, key=lambda item: item["path"]),
        "claim_boundary": {
            "host_cuda": True,
            "orin": False,
            "model_kernel_optimization_attributed_to_vlaforge": False,
            "cache_hit_reported_as_full_invocation": False,
            "queue_fast_path_reported_as_full_invocation": False,
            "os_page_cache_controlled": False,
        },
        "summary": {
            "task_count": len(tasks),
            "cell_count": len(cells),
            "parity_cell_count": len(parity_records),
            "all_output_parity_passed": all(
                item["passed"] for item in parity_records
            ),
            "minimum_processes_per_cell": min(
                cell["processes"] for cell in cells
            ),
            "minimum_workloads_per_model": min(
                sum(cell["model"] == model and cell["path"] == "eager" for cell in cells)
                for model in _MODELS
            ),
        },
    }
    return report


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
        "model",
        "workload",
        "path",
        "processes",
        "samples_per_process",
        "mean_ms",
        "mean_ci95_low_ms",
        "mean_ci95_high_ms",
        "process_mean_stddev_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "initialization_mean_ms",
        "first_run_mean_ms",
        "maximum_rss_mib",
        "maximum_cuda_mib",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in report["cells"]:
            steady = cell["steady_latency"]
            memory = cell["memory"]
            writer.writerow(
                {
                    "model": cell["model"],
                    "workload": cell["workload"],
                    "path": cell["path"],
                    "processes": cell["processes"],
                    "samples_per_process": cell[
                        "steady_samples_per_process"
                    ],
                    "mean_ms": steady["mean_ns"]["estimate"] / 1e6,
                    "mean_ci95_low_ms": steady["mean_ns"]["ci95"][0] / 1e6,
                    "mean_ci95_high_ms": steady["mean_ns"]["ci95"][1] / 1e6,
                    "process_mean_stddev_ms": (
                        steady["process_mean_stddev_ns"] / 1e6
                    ),
                    "p50_ms": steady["p50_ns"]["estimate"] / 1e6,
                    "p90_ms": steady["p90_ns"]["estimate"] / 1e6,
                    "p99_ms": steady["p99_ns"]["estimate"] / 1e6,
                    "initialization_mean_ms": (
                        cell["fresh_process_initialization"]["mean"] / 1e6
                    ),
                    "first_run_mean_ms": cell["first_run"]["mean"] / 1e6,
                    "maximum_rss_mib": memory["maximum_rss_kib"] / 1024,
                    "maximum_cuda_mib": memory["maximum_cuda_bytes"] / 2**20,
                }
            )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VLAForge paper-grade Host-CUDA matrix",
        "",
        f"Status: **{report['status']}**.",
        "",
        (
            "Each cell uses five independent processes and five deterministic "
            "content profiles per model. Confidence intervals use process-"
            "cluster bootstrap resampling."
        ),
        "",
        "| Model | Workload | Path | Mean ms [95% CI] | p50 | p90 | p99 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        steady = cell["steady_latency"]
        mean = steady["mean_ns"]
        lines.append(
            f"| {cell['model']} | {cell['workload']} | {cell['path']} | "
            f"{mean['estimate'] / 1e6:.3f} "
            f"[{mean['ci95'][0] / 1e6:.3f}, "
            f"{mean['ci95'][1] / 1e6:.3f}] | "
            f"{steady['p50_ns']['estimate'] / 1e6:.3f} | "
            f"{steady['p90_ns']['estimate'] / 1e6:.3f} | "
            f"{steady['p99_ns']['estimate'] / 1e6:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Measurement boundary",
            "",
            "- Fresh-process initialization and first Run are reported "
            "separately from steady-state latency.",
            "- OS page cache is not forcibly dropped; initialization is a "
            "fresh-process measurement, not a physical cold-storage claim.",
            "- Generated cells use full recomputation. DiffusionDrive cache "
            "hits and SmolVLA queue-consumption fast paths are excluded.",
            "- AOTI/cuDNN/CUTLASS/Triton model kernels are upstream work. "
            "VLAForge claims only measured generated-Session orchestration.",
            "- This is RTX 3060 Host-CUDA evidence, not Orin evidence.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--process-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--schedule-seed", type=int, default=20260726)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.process_repeats < 5
        or args.warmup < 1
        or args.samples < 30
        or args.bootstrap_resamples < 1000
    ):
        parser.error(
            "paper matrix requires repeats>=5, warmup>=1, samples>=30, "
            "bootstrap-resamples>=1000"
        )
    config = _load_config(args.config.resolve())
    workloads = {
        model: _load_workloads(
            Path(config["models"][model]["workloads"]),
            model,
        )
        for model in _MODELS
    }
    tasks = _schedule(
        workloads,
        args.process_repeats,
        seed=args.schedule_seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "schedule.json").write_text(
        json.dumps(
            {
                "schema": "vlaforge.cuda_paper_matrix_schedule/1",
                "seed": args.schedule_seed,
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.aggregate_only:
        _run_tasks(
            tasks=tasks,
            config=config,
            output_root=args.output_root.resolve(),
            warmup=args.warmup,
            samples=args.samples,
        )
    report = _aggregate(
        tasks=tasks,
        output_root=args.output_root.resolve(),
        report_dir=args.report_dir.resolve(),
        workloads=workloads,
        process_repeats=args.process_repeats,
        samples=args.samples,
        warmup=args.warmup,
        bootstrap_resamples=args.bootstrap_resamples,
        schedule_seed=args.schedule_seed,
        config=config,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "cuda_paper_matrix.json"
    csv_path = args.report_dir / "cuda_paper_matrix.csv"
    markdown_path = args.report_dir / "README.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
