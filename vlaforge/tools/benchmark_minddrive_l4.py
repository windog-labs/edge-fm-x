#!/usr/bin/env python3
"""Run the formal fresh-process MindDrive generated-L4 benchmark matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_TOOL = Path(__file__).with_name("benchmark_generated_l4.py")
_SCHEMA = "vlaforge.minddrive_l4_benchmark/1"
_MODES = ("full", "same", "new", "missing")


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
            [
                float(item)
                for process in selected
                for item in process
            ]
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


def _run_one(
    *,
    torch_python: Path,
    bundle_root: Path,
    input_root: Path,
    output_root: Path,
    binary_root: Path,
    mode: str,
    repeat: int,
    warmup: int,
    samples: int,
) -> tuple[Path, dict[str, Any]]:
    task_root = output_root / "raw" / mode / f"repeat_{repeat:02d}"
    task_root.mkdir(parents=True, exist_ok=False)
    binary = binary_root / "bin" / "vlaforge_generated_benchmark"
    command = [
        str(torch_python),
        str(_BENCHMARK_TOOL),
        "--model",
        "minddrive",
        "--bundle-root",
        str(bundle_root),
        "--input-root",
        str(input_root),
        "--output-root",
        str(task_root),
        "--binary-root",
        str(binary_root),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--mode",
        mode,
    ]
    if binary.is_file():
        command.append("--reuse-binary")
    environment = {
        **dict(os.environ),
        "PYTHONPATH": str(
            _REPOSITORY_ROOT / "vlaforge" / "python"
        ),
    }
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    (task_root / "orchestrator.stdout").write_text(
        completed.stdout, encoding="utf-8"
    )
    (task_root / "orchestrator.stderr").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MindDrive benchmark failed for {mode}/repeat_{repeat:02d}: "
            f"{completed.stderr[-2000:]}"
        )
    report_path = task_root / f"minddrive_{mode}_{samples}.json"
    report = _json(report_path)
    if report.get("status") != "passed":
        raise RuntimeError(f"benchmark report did not pass: {report_path}")
    return report_path, report


def _validate_outputs(
    reports: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sequences: dict[str, list[list[float]]] = {}
    for mode, records in reports.items():
        sequences[mode] = [
            [
                float(item["output_probe"])
                for item in record["raw_samples"]
            ]
            for record in records
        ]
        if any(
            sequence != sequences[mode][0]
            for sequence in sequences[mode][1:]
        ):
            raise RuntimeError(
                f"{mode} outputs changed across fresh processes"
            )
    exact_modes = ("same", "new", "missing")
    reference = sequences["same"][0]
    if any(sequences[mode][0] != reference for mode in exact_modes):
        raise RuntimeError(
            "same/new/missing revision changed MindDrive outputs"
        )
    return {
        "fresh_process_deterministic": True,
        "same_new_missing_exact": True,
        "full_sequence_probe": sequences["full"][0],
        "exact_revision_probe": reference,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MindDrive real L4 Host-CUDA benchmark",
        "",
        f"Status: **{report['status']}**.",
        "",
        "| Revision mode | Init mean | First Run mean | Warm mean | "
        "p50 | p90 | p99 | Process std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in _MODES:
        cell = report["cells"][mode]
        latency = cell["warm_latency"]
        lines.append(
            "| "
            f"{mode} | "
            f"{cell['initialization_ns']['mean'] / 1e6:.2f} ms | "
            f"{cell['first_run_ns']['mean'] / 1e6:.2f} ms | "
            f"{latency['mean_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['p50_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['p90_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['p99_ns']['estimate'] / 1e6:.2f} ms | "
            f"{latency['process_mean_stddev_ns'] / 1e6:.2f} ms |"
        )
    memory = report["memory_contract"]
    lines.extend(
        [
            "",
            "## Declared memory classes",
            "",
            f"- External input per invocation: "
            f"{memory['external_input_bytes_per_invocation']} bytes",
            f"- External output per invocation: "
            f"{memory['external_output_bytes_per_invocation']} bytes",
            f"- Per-Run static arena: "
            f"{memory['per_run_static_arena_bytes']} bytes",
            f"- Authoritative state arena: "
            f"{memory['authoritative_state_arena_capacity_bytes']} bytes",
            f"- Derived-cache physical capacity: "
            f"{memory['derived_cache_physical_capacity_bytes']} bytes",
            "",
            "All cells are five independent processes running the real "
            "MindDrive 0.5B generated no-Python C++ Session on RTX 3060 "
            "sm_86. `full` cycles the five held-out real frames; `same`, "
            "`new`, and `missing` isolate InputRevision behavior on the "
            "same frame while authoritative state continues to commit.",
            "",
            "This is Host-CUDA evidence, not Orin, cross-GPU, closed-loop, "
            "power, thermal, or model-kernel-optimization evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--process-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args(argv)
    if (
        args.process_repeats < 5
        or args.warmup < 1
        or args.samples < 2
        or args.bootstrap_resamples < 1000
    ):
        parser.error(
            "formal matrix requires repeats>=5, warmup>=1, samples>=2, "
            "bootstrap-resamples>=1000"
        )
    for path in (
        args.torch_python,
        args.bundle_root / "bundle.json",
        args.input_root / "frame_00400",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
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

    reports: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in _MODES
    }
    report_paths: list[str] = []
    for mode in _MODES:
        for repeat in range(args.process_repeats):
            report_path, report = _run_one(
                torch_python=args.torch_python.resolve(),
                bundle_root=args.bundle_root.resolve(),
                input_root=args.input_root.resolve(),
                output_root=args.output_root.resolve(),
                binary_root=args.binary_root.resolve(),
                mode=mode,
                repeat=repeat,
                warmup=args.warmup,
                samples=args.samples,
            )
            samples_path = report_path.with_suffix(".csv")
            raw_samples = []
            for line in samples_path.read_text(encoding="utf-8").splitlines()[
                1:
            ]:
                fields = line.split(",")
                raw_samples.append(
                    {
                        "latency_ns": int(fields[1]),
                        "output_probe": float(fields[4]),
                    }
                )
            report["raw_samples"] = raw_samples
            reports[mode].append(report)
            report_paths.append(str(report_path))

    cells: dict[str, Any] = {}
    for mode_index, mode in enumerate(_MODES):
        records = reports[mode]
        cells[mode] = {
            "processes": args.process_repeats,
            "samples_per_process": args.samples,
            "warm_latency": _cluster_summary(
                [
                    [
                        int(item["latency_ns"])
                        for item in record["raw_samples"]
                    ]
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + mode_index,
            ),
            "initialization_ns": _scalar_summary(
                [
                    float(record["runtime"]["initialization_ns"])
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 10 + mode_index,
            ),
            "first_run_ns": _scalar_summary(
                [
                    float(record["runtime"]["first_run_ns"])
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 20 + mode_index,
            ),
            "rss_initialized_kib": _scalar_summary(
                [
                    float(record["runtime"]["rss_initialized_kib"])
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 30 + mode_index,
            ),
            "maximum_rss_kib": _scalar_summary(
                [
                    float(record["runtime"]["maximum_rss_kib"])
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 40 + mode_index,
            ),
            "cuda_used_initialized_bytes": _scalar_summary(
                [
                    float(
                        record["runtime"]["cuda_used_initialized_bytes"]
                    )
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 50 + mode_index,
            ),
            "cuda_used_peak_sampled_bytes": _scalar_summary(
                [
                    float(
                        record["runtime"][
                            "cuda_used_peak_sampled_bytes"
                        ]
                    )
                    for record in records
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + 60 + mode_index,
            ),
            "rss_drift_kib": [
                int(record["memory"]["rss_drift_kib"])
                for record in records
            ],
            "cuda_used_drift_bytes": [
                int(record["memory"]["cuda_used_drift_bytes"])
                for record in records
            ],
            "cache_hits": [
                int(record["runtime"]["cache_hits"])
                for record in records
            ],
            "cache_misses": [
                int(record["runtime"]["cache_misses"])
                for record in records
            ],
            "state_commits": [
                int(record["runtime"]["state_commits"])
                for record in records
            ],
            "transaction_commits": [
                int(record["runtime"]["transaction_commits"])
                for record in records
            ],
            "output_commits": [
                int(record["runtime"]["output_commits"])
                for record in records
            ],
        }

    output_validation = _validate_outputs(reports)
    first = reports["full"][0]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "passed": True,
        "model": "MindDrive 0.5B",
        "evidence": "real-L4-generated-no-Python-C++",
        "repository": {
            "revision": revision,
            "source_dirty": False,
        },
        "protocol": {
            "processes_per_cell": args.process_repeats,
            "warmup_runs": args.warmup,
            "samples_per_process": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_unit": "independent fresh process",
            "modes": list(_MODES),
        },
        "cells": cells,
        "output_validation": output_validation,
        "memory_contract": first["memory"]["declared_contract"],
        "bundle": first["bundle"],
        "runner": first["runner"],
        "raw_reports": report_paths,
        "environment": {
            **first["environment"],
            "host": platform.platform(),
        },
        "classification": (
            "real MindDrive Host-CUDA generated C++ Session performance; "
            "no Orin, cross-GPU, closed-loop, power, thermal, or "
            "model-kernel-optimization claim"
        ),
        "reproduction": {
            "command": [sys.executable, str(Path(__file__).resolve())]
            + sys.argv[1:]
        },
    }
    (args.output_root / "minddrive_l4_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "README.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "revision": revision,
                "cells": {
                    mode: {
                        "mean_ns": cells[mode]["warm_latency"]["mean_ns"],
                        "initialization_ns": cells[mode][
                            "initialization_ns"
                        ]["mean"],
                        "first_run_ns": cells[mode]["first_run_ns"]["mean"],
                    }
                    for mode in _MODES
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
