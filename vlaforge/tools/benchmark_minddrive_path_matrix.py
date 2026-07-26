#!/usr/bin/env python3
"""Run the formal MindDrive eager/AOTI/generated Host-CUDA path matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PATH_TOOL = Path(__file__).with_name("benchmark_real_model_paths.py")
_GENERATED_TOOL = Path(__file__).with_name("benchmark_generated_l4.py")
_SCHEMA = "vlaforge.minddrive_path_matrix/1"
_PATHS = ("eager", "direct_artifact", "generated_session")
_FRAME_COUNT = 5
_DIRECT_GENERATED_TOLERANCE = 1.0e-7
_EAGER_DIRECT_TOLERANCE = 3.0e-3


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


def _metrics(values: list[float]) -> dict[str, float]:
    return {
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
    if len(process_samples) < 2 or any(
        not samples for samples in process_samples
    ):
        raise ValueError("cluster summary requires multiple processes")
    flat = [float(item) for process in process_samples for item in process]
    estimates = _metrics(flat)
    process_means = [
        statistics.fmean(process) for process in process_samples
    ]
    generator = random.Random(seed)
    distributions = {
        name: [] for name in ("mean", "p50", "p90", "p99")
    }
    for _ in range(bootstrap_resamples):
        selected = [
            process_samples[generator.randrange(len(process_samples))]
            for _ in process_samples
        ]
        metrics = _metrics(
            [float(item) for process in selected for item in process]
        )
        for name in distributions:
            distributions[name].append(metrics[name])
    result: dict[str, Any] = {
        "processes": len(process_samples),
        "samples_per_process": [len(item) for item in process_samples],
        "process_mean_ns": process_means,
        "process_mean_stddev_ns": statistics.stdev(process_means),
        "minimum_ns": int(estimates["minimum"]),
        "maximum_ns": int(estimates["maximum"]),
        "throughput_runs_per_second": 1e9 / estimates["mean"],
    }
    for name, values in distributions.items():
        result[f"{name}_ns"] = {
            "estimate": estimates[name],
            "ci95": [
                _nearest_rank(values, 0.025),
                _nearest_rank(values, 0.975),
            ],
        }
    return result


def _scalar_summary(
    values: list[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    summary = _cluster_summary(
        [[int(round(value))] for value in values],
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "samples": values,
        "mean": summary["mean_ns"]["estimate"],
        "ci95": summary["mean_ns"]["ci95"],
        "stddev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _schedule(process_repeats: int, *, seed: int) -> list[dict[str, Any]]:
    tasks = [
        {"path": path, "repeat": repeat}
        for repeat in range(process_repeats)
        for path in _PATHS
    ]
    random.Random(seed).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["schedule_index"] = index
        task["key"] = f"{task['path']}/repeat_{task['repeat']:02d}"
    return tasks


def _task_root(output_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "raw"
        / str(task["path"])
        / f"repeat_{int(task['repeat']):02d}"
    )


def _report_path(
    output_root: Path,
    task: Mapping[str, Any],
    *,
    samples: int,
) -> Path:
    root = _task_root(output_root, task)
    if task["path"] == "generated_session":
        return root / f"minddrive_full_{samples}.json"
    return root / "report.json"


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


def _path_command(
    args: argparse.Namespace,
    *,
    path: str,
    report_path: Path,
) -> list[str]:
    return [
        str(args.torch_python),
        str(_PATH_TOOL),
        "--model",
        "minddrive",
        "--path",
        path,
        "--input-root",
        str(args.input_root),
        "--bundle-root",
        str(args.bundle_root),
        "--source-root",
        str(args.source_root),
        "--release-root",
        str(args.release_root),
        "--frame-root",
        str(args.frame_root),
        "--output",
        str(report_path),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--first-run-counts-as-warmup",
    ]


def _generated_command(
    args: argparse.Namespace,
    *,
    task_root: Path,
) -> list[str]:
    command = [
        str(args.torch_python),
        str(_GENERATED_TOOL),
        "--model",
        "minddrive",
        "--bundle-root",
        str(args.bundle_root),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(task_root),
        "--binary-root",
        str(args.binary_root),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--mode",
        "full",
    ]
    if (
        args.binary_root / "bin" / "vlaforge_generated_benchmark"
    ).is_file():
        command.append("--reuse-binary")
    return command


def _extract_samples(
    report_path: Path,
    *,
    path: str,
    samples: int,
) -> tuple[list[int], list[float]]:
    report = _json(report_path)
    if report.get("status") != "passed" or int(report["samples"]) != samples:
        raise ValueError(f"incomplete benchmark report: {report_path}")
    if path == "generated_session":
        with report_path.with_suffix(".csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            records = list(csv.DictReader(handle))
    else:
        records = report["samples_raw"]
    latencies = [int(item["latency_ns"]) for item in records]
    probes = [float(item["output_probe"]) for item in records]
    if (
        len(latencies) != samples
        or len(probes) != samples
        or any(not math.isfinite(item) for item in probes)
    ):
        raise ValueError(f"invalid raw benchmark samples: {report_path}")
    return latencies, probes


def _run_tasks(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
) -> None:
    environment = {
        **dict(os.environ),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(_REPOSITORY_ROOT / "vlaforge" / "python"),
    }
    progress_path = args.output_root / "progress.json"
    completed_tasks = []
    for task in tasks:
        task_root = _task_root(args.output_root, task)
        report_path = _report_path(
            args.output_root, task, samples=args.samples
        )
        if report_path.is_file():
            _extract_samples(
                report_path,
                path=str(task["path"]),
                samples=args.samples,
            )
            completed_tasks.append(
                {
                    **task,
                    "status": "reused",
                    "report": str(report_path),
                }
            )
            continue
        if task_root.exists() and any(task_root.iterdir()):
            raise RuntimeError(
                f"incomplete task directory requires inspection: {task_root}"
            )
        task_root.mkdir(parents=True, exist_ok=True)
        if task["path"] == "generated_session":
            command = _generated_command(args, task_root=task_root)
        else:
            command = _path_command(
                args,
                path=str(task["path"]),
                report_path=report_path,
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
        wall_seconds = time.perf_counter() - started
        (task_root / "process.log").write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "wall_seconds": wall_seconds,
                    "telemetry_before": before,
                    "telemetry_after": _telemetry(),
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
        _extract_samples(
            report_path,
            path=str(task["path"]),
            samples=args.samples,
        )
        completed_tasks.append(
            {
                **task,
                "status": "passed",
                "report": str(report_path),
                "wall_seconds": wall_seconds,
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "schema": "vlaforge.minddrive_path_matrix_progress/1",
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


def _runtime_record(
    report: Mapping[str, Any], *, path: str
) -> dict[str, int]:
    runtime = report["runtime"]
    initialization_ns = (
        int(runtime["initialization_ns"])
        if path == "generated_session"
        else int(round(float(report["initialization_seconds"]) * 1e9))
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
        "torch_allocated_peak_bytes": int(
            runtime.get("torch_allocated_peak_bytes", 0)
        ),
        "torch_reserved_peak_bytes": int(
            runtime.get("torch_reserved_peak_bytes", 0)
        ),
    }


def _maximum_error(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("probe sequences have different lengths")
    return max(
        (abs(left - right) for left, right in zip(lhs, rhs, strict=True)),
        default=0.0,
    )


def _validate_probe_sequences(
    sequences: Mapping[str, list[list[float]]],
) -> dict[str, Any]:
    if set(sequences) != set(_PATHS):
        raise ValueError("probe validation requires all three paths")
    process_count = len(sequences["eager"])
    if process_count < 2 or any(
        len(records) != process_count for records in sequences.values()
    ):
        raise ValueError("probe validation requires aligned processes")
    within_path = {}
    for path, records in sequences.items():
        reference = records[0]
        maximum = max(
            _maximum_error(reference, candidate)
            for candidate in records[1:]
        )
        tolerance = (
            _EAGER_DIRECT_TOLERANCE
            if path == "eager"
            else _DIRECT_GENERATED_TOLERANCE
        )
        if maximum > tolerance:
            raise ValueError(
                f"{path} output probes changed across processes: {maximum}"
            )
        within_path[path] = {
            "maximum_absolute_error": maximum,
            "tolerance": tolerance,
            "passed": True,
        }
    comparisons = []
    for repeat in range(process_count):
        direct_generated = _maximum_error(
            sequences["direct_artifact"][repeat],
            sequences["generated_session"][repeat],
        )
        eager_direct = _maximum_error(
            sequences["eager"][repeat],
            sequences["direct_artifact"][repeat],
        )
        if direct_generated > _DIRECT_GENERATED_TOLERANCE:
            raise ValueError(
                "direct/generated output probe parity failed: "
                f"{direct_generated}"
            )
        if eager_direct > _EAGER_DIRECT_TOLERANCE:
            raise ValueError(
                f"eager/direct output probe parity failed: {eager_direct}"
            )
        comparisons.append(
            {
                "repeat": repeat,
                "direct_generated_maximum_absolute_error": (
                    direct_generated
                ),
                "direct_generated_tolerance": (
                    _DIRECT_GENERATED_TOLERANCE
                ),
                "eager_direct_maximum_absolute_error": eager_direct,
                "eager_direct_tolerance": _EAGER_DIRECT_TOLERANCE,
                "passed": True,
            }
        )
    return {
        "probe_only": True,
        "full_output_parity_evidence": (
            "doc/reports/vlaforge_minddrive_v01/minddrive_l4.json"
        ),
        "within_path": within_path,
        "comparisons": comparisons,
        "all_passed": True,
    }


def _aggregate(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {
        path: [] for path in _PATHS
    }
    raw_inventory = []
    sequences: dict[str, list[list[float]]] = {
        path: [] for path in _PATHS
    }
    for task in sorted(tasks, key=lambda item: (item["path"], item["repeat"])):
        path = str(task["path"])
        report_path = _report_path(
            args.output_root, task, samples=args.samples
        )
        latencies, probes = _extract_samples(
            report_path,
            path=path,
            samples=args.samples,
        )
        report = _json(report_path)
        records[path].append(
            {
                "repeat": int(task["repeat"]),
                "latencies": latencies,
                "probes": probes,
                "runtime": _runtime_record(report, path=path),
                "report": report,
                "report_path": str(report_path),
            }
        )
        sequences[path].append(probes)
        for artifact in (
            report_path,
            report_path.with_suffix(".csv"),
            _task_root(args.output_root, task) / "process.log",
        ):
            if artifact.is_file():
                raw_inventory.append(
                    {
                        "path": str(
                            artifact.relative_to(args.output_root)
                        ),
                        "sha256": _sha256(artifact),
                        "size_bytes": artifact.stat().st_size,
                    }
                )
    probe_validation = _validate_probe_sequences(sequences)
    cells = {}
    for path_index, path in enumerate(_PATHS):
        path_records = records[path]
        if len(path_records) != args.process_repeats:
            raise ValueError(f"{path} process count is incomplete")
        runtimes = [item["runtime"] for item in path_records]
        cells[path] = {
            "processes": args.process_repeats,
            "warmup_runs_per_process": args.warmup,
            "samples_per_process": args.samples,
            "warm_latency": _cluster_summary(
                [item["latencies"] for item in path_records],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + path_index,
            ),
            "initialization_ns": _scalar_summary(
                [item["initialization_ns"] for item in runtimes],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 10 + path_index,
            ),
            "first_run_ns": _scalar_summary(
                [item["first_run_ns"] for item in runtimes],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 20 + path_index,
            ),
            "memory": {
                "maximum_rss_kib": max(
                    item["maximum_rss_kib"] for item in runtimes
                ),
                "maximum_cuda_bytes": max(
                    item["cuda_peak_bytes"] for item in runtimes
                ),
                "maximum_torch_allocated_bytes": max(
                    item["torch_allocated_peak_bytes"]
                    for item in runtimes
                ),
                "maximum_torch_reserved_bytes": max(
                    item["torch_reserved_peak_bytes"]
                    for item in runtimes
                ),
                "rss_drift_kib": [
                    item["rss_drift_kib"] for item in runtimes
                ],
                "cuda_drift_bytes": [
                    item["cuda_drift_bytes"] for item in runtimes
                ],
            },
            "raw_reports": [
                str(
                    Path(item["report_path"]).relative_to(
                        args.output_root
                    )
                )
                for item in path_records
            ],
        }
    eager = cells["eager"]["warm_latency"]["mean_ns"]["estimate"]
    direct = cells["direct_artifact"]["warm_latency"]["mean_ns"][
        "estimate"
    ]
    generated = cells["generated_session"]["warm_latency"]["mean_ns"][
        "estimate"
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first_generated = records["generated_session"][0]["report"]
    return {
        "schema": _SCHEMA,
        "status": "passed",
        "passed": True,
        "model": "MindDrive 0.5B",
        "evidence": "real-eager-direct-AOTI-generated-C++-path-matrix",
        "repository": {
            "revision": revision,
            "source_dirty": False,
        },
        "protocol": {
            "paths": list(_PATHS),
            "processes_per_path": args.process_repeats,
            "warmup_runs_per_process": args.warmup,
            "samples_per_process": args.samples,
            "stateful_frame_cycle": [
                "frame_00400",
                "frame_00401",
                "frame_00402",
                "frame_00403",
                "frame_00404",
            ],
            "warmup_alignment": (
                "first Run is included in warmup on every path; warmup is "
                "a multiple of five, so measured samples begin on frame_00400"
            ),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_unit": "independent fresh process",
            "schedule_seed": args.seed,
            "timed_boundary": (
                "one full stateful MindDrive invocation plus backend CUDA "
                "synchronization"
            ),
            "initialization_boundary": {
                "eager": (
                    "official model/checkpoint/frontend and five prepared "
                    "inputs"
                ),
                "direct_artifact": (
                    "66 persistent AOTI runners and five prepared inputs"
                ),
                "generated_session": (
                    "ModelSession construction after C++ input files load"
                ),
                "cross_path_initialization_directly_comparable": False,
            },
        },
        "cells": cells,
        "comparison": {
            "eager_mean_ns": eager,
            "direct_artifact_mean_ns": direct,
            "generated_session_mean_ns": generated,
            "eager_over_generated_speedup": eager / generated,
            "generated_over_direct_percent": (
                (generated / direct - 1.0) * 100.0
            ),
            "orchestration_boundary": (
                "direct and generated execute the same 66 physical AOTI "
                "artifacts, five-frame inputs, explicit state carry, and "
                "trajectory probe; the ratio includes their Python versus "
                "generated C++ composition/state-management boundaries"
            ),
        },
        "output_validation": probe_validation,
        "bundle": first_generated["bundle"],
        "memory_contract": first_generated["memory"]["declared_contract"],
        "environment": {
            **first_generated["environment"],
            "host": platform.platform(),
            "torch_python": str(args.torch_python),
        },
        "raw_evidence": sorted(
            raw_inventory, key=lambda item: item["path"]
        ),
        "classification": (
            "real MindDrive Host-CUDA path evidence on RTX 3060 sm_86; "
            "no Orin, cross-GPU, closed-loop, power, thermal, sensor, "
            "middleware, or model-kernel-optimization claim"
        ),
        "reproduction": {
            "command": [sys.executable, str(Path(__file__).resolve())]
            + sys.argv[1:]
        },
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
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
        for name in _PATHS:
            cell = report["cells"][name]
            latency = cell["warm_latency"]
            memory = cell["memory"]
            writer.writerow(
                {
                    "path": name,
                    "processes": cell["processes"],
                    "samples_per_process": cell["samples_per_process"],
                    "mean_ms": latency["mean_ns"]["estimate"] / 1e6,
                    "mean_ci95_low_ms": (
                        latency["mean_ns"]["ci95"][0] / 1e6
                    ),
                    "mean_ci95_high_ms": (
                        latency["mean_ns"]["ci95"][1] / 1e6
                    ),
                    "process_mean_stddev_ms": (
                        latency["process_mean_stddev_ns"] / 1e6
                    ),
                    "p50_ms": latency["p50_ns"]["estimate"] / 1e6,
                    "p90_ms": latency["p90_ns"]["estimate"] / 1e6,
                    "p99_ms": latency["p99_ns"]["estimate"] / 1e6,
                    "initialization_mean_ms": (
                        cell["initialization_ns"]["mean"] / 1e6
                    ),
                    "first_run_mean_ms": (
                        cell["first_run_ns"]["mean"] / 1e6
                    ),
                    "maximum_rss_mib": (
                        memory["maximum_rss_kib"] / 1024
                    ),
                    "maximum_cuda_mib": (
                        memory["maximum_cuda_bytes"] / 2**20
                    ),
                }
            )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MindDrive eager/direct-AOTI/generated-C++ matrix",
        "",
        f"Status: **{report['status']}**.",
        "",
        "| Path | Init mean | First Run | Warm mean [95% CI] | "
        "p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _PATHS:
        cell = report["cells"][name]
        latency = cell["warm_latency"]
        mean = latency["mean_ns"]
        lines.append(
            f"| {name} | "
            f"{cell['initialization_ns']['mean'] / 1e6:.2f} ms | "
            f"{cell['first_run_ns']['mean'] / 1e6:.2f} ms | "
            f"{mean['estimate'] / 1e6:.2f} "
            f"[{mean['ci95'][0] / 1e6:.2f}, "
            f"{mean['ci95'][1] / 1e6:.2f}] ms | "
            f"{latency['p50_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['p90_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['p99_ns']['estimate'] / 1e6:.2f} ms |"
        )
    comparison = report["comparison"]
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Eager/generated warm speedup: "
            f"{comparison['eager_over_generated_speedup']:.3f}x.",
            f"- Generated/direct warm delta: "
            f"{comparison['generated_over_direct_percent']:.3f}%.",
            "- Direct and generated execute the same 66 physical AOTI "
            "artifacts. Their delta includes Python versus generated C++ "
            "composition and state-management boundaries.",
            "- Initialization values are reported but are not directly "
            "compared: generated Session initialization begins after C++ "
            "input loading, while Python path initialization includes input "
            "preparation.",
            "- Output validation here uses aligned scalar probes. Full named-"
            "output parity remains backed by the real L4 correctness report.",
            "- This is RTX 3060 Host-CUDA evidence, not Orin evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--process-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.process_repeats < 5
        or args.warmup < _FRAME_COUNT
        or args.warmup % _FRAME_COUNT
        or args.samples < 10
        or args.samples % _FRAME_COUNT
        or args.bootstrap_resamples < 1000
    ):
        parser.error(
            "formal path matrix requires repeats>=5, warmup>=5 and "
            "divisible by 5, samples>=10 and divisible by 5, and "
            "bootstrap-resamples>=1000"
        )
    for path in (
        args.torch_python,
        args.bundle_root / "bundle.json",
        args.input_root / "frame_00400",
        args.source_root / "mmcv",
        args.release_root / "minddrive_rltrain.pth",
        args.frame_root / "camera" / "rgb_front" / "00400.jpg",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.torch_python = args.torch_python.resolve()
    args.bundle_root = args.bundle_root.resolve()
    args.input_root = args.input_root.resolve()
    args.source_root = args.source_root.resolve()
    args.release_root = args.release_root.resolve()
    args.frame_root = args.frame_root.resolve()
    args.output_root = args.output_root.resolve()
    args.binary_root = args.binary_root.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.aggregate_only and not args.resume:
            raise ValueError("output root must be absent or empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.binary_root.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked:
        raise RuntimeError("formal benchmark requires a clean worktree")
    tasks = _schedule(args.process_repeats, seed=args.seed)
    schedule_payload = {
        "schema": "vlaforge.minddrive_path_matrix_schedule/1",
        "seed": args.seed,
        "tasks": tasks,
    }
    schedule_path = args.output_root / "schedule.json"
    if schedule_path.is_file():
        if _json(schedule_path) != schedule_payload:
            raise ValueError("existing schedule does not match arguments")
    else:
        schedule_path.write_text(
            json.dumps(schedule_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.aggregate_only:
        _run_tasks(args, tasks)
    report = _aggregate(args, tasks)
    (args.output_root / "minddrive_path_matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "minddrive_path_matrix.csv", report)
    (args.output_root / "README.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "revision": report["repository"]["revision"],
                "comparison": report["comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
