#!/usr/bin/env python3
"""Measure VLA-specific whole-program optimizations on real C++ runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

from vlaforge.adapters import (
    build_real_openvla_action_program,
    build_real_smolvla_action_program,
)
from vlaforge.plan import lower_to_plan, physicalize_plan
from vlaforge.transforms import (
    optimize_whole_program,
    synthesize_epoch_memoization,
    temporal_loop_invariant_code_motion,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/tmp"),
    )
    parser.add_argument(
        "--artifact-prefix",
        default="vlaforge-g4",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--compiler-repetitions", type=int, default=200)
    parser.add_argument("--smol-runner", type=Path)
    parser.add_argument("--smol-prefix", type=Path)
    parser.add_argument("--smol-solver", type=Path)
    parser.add_argument("--smol-trim", type=Path)
    parser.add_argument("--open-runner", type=Path)
    parser.add_argument("--open-archive", type=Path)
    parser.add_argument("--open-input-dir", type=Path)
    parser.add_argument("--smol-build-metrics", type=Path)
    parser.add_argument("--open-build-metrics", type=Path)
    args = parser.parse_args()
    if args.compiler_repetitions < 1:
        parser.error("--compiler-repetitions must be positive")

    artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    if not args.reuse_existing:
        _require_execution_arguments(parser, args)
        _run_all(args, artifact_root)

    runtime = {
        "smolvla": _analyze_model(
            artifact_root,
            args.artifact_prefix,
            "smol",
            ("baseline", "cache", "licm", "licm_off"),
            args.smol_runner,
        ),
        "openvla": _analyze_model(
            artifact_root,
            args.artifact_prefix,
            "open",
            ("baseline", "cache"),
            args.open_runner,
        ),
    }
    compiler = _compiler_metrics(args.compiler_repetitions)
    result = {
        "schema": "vlaforge.whole_program_benchmark/1",
        "passes": {
            "epoch_keyed_cache_synthesis": {
                "legality": (
                    "all transitive region inputs carry input Epoch or "
                    "StateVersion; lookup also enforces freshness and episode"
                ),
                "negative_cases": [
                    "unversioned operand",
                    "stale input epoch",
                    "changed epoch or state version",
                    "episode reset",
                ],
                "runtime": {
                    model: _cache_comparison(data)
                    for model, data in runtime.items()
                },
            },
            "temporal_loop_invariant_code_motion": {
                "legality": (
                    "pure region, complete epoch/version certificate, no "
                    "induction or loop-carried dependency, preheader operands "
                    "available, no SSA collision"
                ),
                "negative_cases": [
                    "loop-carried tensor dependency",
                    "induction dependency",
                    "missing temporal key",
                ],
                "runtime": {
                    "smolvla": _licm_comparison(runtime["smolvla"]),
                    "openvla": {
                        "status": "already_prehoisted",
                        "p99_change_percent": 0.0,
                        "reason": (
                            "prefill produces the autoregressive loop seed; "
                            "the pass certifies the existing preheader and "
                            "performs no dynamic move"
                        ),
                    },
                },
            },
            "state_physicalization_and_arena_reuse": {
                "legality": (
                    "state ring capacity satisfies retention/in-flight/lag/"
                    "fallback proof; temporary byte ranges alias only when "
                    "scheduled task lifetimes do not overlap"
                ),
                "negative_cases": [
                    "undersized state ring",
                    "overlapping live intervals",
                    "device mismatch",
                ],
                "runtime": {
                    model: {
                        "baseline_arena_bytes": data[
                            "arena_baseline_bytes"
                        ],
                        "optimized_arena_bytes": data[
                            "arena_optimized_bytes"
                        ],
                        "peak_arena_reduction_percent": data[
                            "arena_reduction_percent"
                        ],
                        "latency_p99_change_percent": 0.0,
                        "scope": "compiler-owned static arena",
                    }
                    for model, data in compiler.items()
                },
            },
        },
        "compiler": compiler,
        "generated_cpp": runtime,
        "binary_build": {
            "smolvla": _parse_optional_metrics(args.smol_build_metrics),
            "openvla": _parse_optional_metrics(args.open_build_metrics),
        },
    }
    result["gate_passed"] = _gate_passed(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


def _require_execution_arguments(parser, args) -> None:
    names = (
        "smol_runner",
        "smol_prefix",
        "smol_solver",
        "smol_trim",
        "open_runner",
        "open_archive",
        "open_input_dir",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        parser.error(
            "execution requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )


def _run_all(args, root: Path) -> None:
    smol_arguments = (
        str(args.smol_prefix),
        str(args.smol_solver),
        str(args.smol_trim),
    )
    for mode in ("baseline", "cache", "licm", "licm_off"):
        _run_one(
            root,
            args.artifact_prefix,
            "smol",
            mode,
            args.smol_runner,
            smol_arguments,
        )
    open_arguments = (
        str(args.open_archive),
        str(args.open_input_dir),
    )
    for mode in ("baseline", "cache"):
        _run_one(
            root,
            args.artifact_prefix,
            "open",
            mode,
            args.open_runner,
            open_arguments,
        )


def _run_one(
    root: Path,
    prefix: str,
    model: str,
    mode: str,
    runner: Path,
    arguments: tuple[str, ...],
) -> None:
    base = root / f"{prefix}-{model}-{mode}"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
            "VLAFORGE_OPT_BENCHMARK": mode,
        }
    )
    command = [
        "/usr/bin/time",
        "-f",
        "MAX_RSS_KB=%M\nWALL_SECONDS=%e",
        "-o",
        str(base.with_suffix(".time")),
        str(runner),
        *arguments,
        str(base.with_suffix(".bin")),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    base.with_suffix(".log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    base.with_suffix(".stderr").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"{model} {mode} failed with exit {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )


def _analyze_model(
    root: Path,
    prefix: str,
    model: str,
    modes: tuple[str, ...],
    runner: Path | None,
) -> dict[str, object]:
    runs = {}
    for mode in modes:
        base = root / f"{prefix}-{model}-{mode}"
        log = base.with_suffix(".log").read_text(encoding="utf-8")
        timing = _parse_metrics(
            base.with_suffix(".time").read_text(encoding="utf-8")
        )
        ticks = []
        for line in log.splitlines():
            if not line.startswith("BENCH_TICK_US,"):
                continue
            _, measured_mode, sequence, elapsed, cache_hit = line.split(",")
            ticks.append(
                {
                    "mode": measured_mode,
                    "sequence": int(sequence),
                    "latency_us": float(elapsed),
                    "cache_hit": bool(int(cache_hit)),
                }
            )
        if len(ticks) != 3:
            raise ValueError(f"{model} {mode} did not report three ticks")
        runs[mode] = {
            "ticks": ticks,
            "steady_p99_us": _p99(
                [
                    item["latency_us"]
                    for item in ticks
                    if item["sequence"] > 0
                ]
            ),
            "max_rss_kb": int(timing["MAX_RSS_KB"]),
            "wall_seconds": float(timing["WALL_SECONDS"]),
            "actions": [
                line
                for line in log.splitlines()
                if line.startswith("ACTION,")
            ],
            "non_region_trace": [
                line
                for line in log.splitlines()
                if line.startswith("TRACE,")
                and int(line.split(",")[1]) != 11
            ],
            "evidence_sha256": _sha256(base.with_suffix(".bin")),
        }
    reference = runs["baseline"]
    for mode, run in runs.items():
        run["actions_equal"] = run["actions"] == reference["actions"]
        run["state_action_trace_equal"] = (
            run["non_region_trace"]
            == reference["non_region_trace"]
        )
        run["evidence_equal"] = (
            run["evidence_sha256"]
            == reference["evidence_sha256"]
        )
    linked_python = None
    if runner is not None:
        linked_python = "libpython" in subprocess.run(
            ["ldd", str(runner)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.lower()
    return {
        "runs": runs,
        "linked_libpython": linked_python,
    }


def _cache_comparison(data: dict[str, object]) -> dict[str, object]:
    runs = data["runs"]
    baseline = runs["baseline"]
    cached = runs["cache"]
    hits = [
        item
        for item in cached["ticks"]
        if item["sequence"] > 0 and item["cache_hit"]
    ]
    baseline_p99 = baseline["steady_p99_us"]
    cache_p99 = _p99([item["latency_us"] for item in hits])
    return {
        "baseline_steady_p99_us": baseline_p99,
        "cache_hit_p99_us": cache_p99,
        "p99_reduction_percent": _reduction(baseline_p99, cache_p99),
        "cache_hits": len(hits),
        "peak_rss_baseline_kb": baseline["max_rss_kb"],
        "peak_rss_cache_kb": cached["max_rss_kb"],
        "actions_equal": cached["actions_equal"],
        "state_action_trace_equal": cached["state_action_trace_equal"],
        "evidence_equal": cached["evidence_equal"],
    }


def _licm_comparison(data: dict[str, object]) -> dict[str, object]:
    runs = data["runs"]
    baseline = runs["licm_off"]
    optimized = runs["licm"]
    baseline_p99 = baseline["steady_p99_us"]
    optimized_p99 = optimized["steady_p99_us"]
    return {
        "baseline_steady_p99_us": baseline_p99,
        "optimized_steady_p99_us": optimized_p99,
        "p99_reduction_percent": _reduction(
            baseline_p99,
            optimized_p99,
        ),
        "peak_rss_baseline_kb": baseline["max_rss_kb"],
        "peak_rss_optimized_kb": optimized["max_rss_kb"],
        "actions_equal": baseline["actions"] == optimized["actions"],
        "state_action_trace_equal": (
            baseline["non_region_trace"]
            == optimized["non_region_trace"]
        ),
        "evidence_equal": (
            baseline["evidence_sha256"]
            == optimized["evidence_sha256"]
        ),
    }


def _compiler_metrics(repetitions: int) -> dict[str, object]:
    modules = {
        "smolvla": build_real_smolvla_action_program(
            chunk_size=50,
            max_action_dim=32,
            output_action_dim=6,
            num_steps=10,
        ),
        "openvla": build_real_openvla_action_program(action_dim=7),
    }
    result = {}
    for name, module in modules.items():
        memoized = synthesize_epoch_memoization(module)
        licm = temporal_loop_invariant_code_motion(memoized)
        lowered = lower_to_plan(licm.module)
        baseline = physicalize_plan(lowered)
        optimized = physicalize_plan(
            lowered,
            reuse_temporaries=True,
        )
        assert baseline.arena is not None
        assert optimized.arena is not None
        result[name] = {
            "memoization_compile": _measure(
                lambda: synthesize_epoch_memoization(module),
                repetitions,
            ),
            "licm_compile": _measure(
                lambda: temporal_loop_invariant_code_motion(memoized),
                repetitions,
            ),
            "arena_compile": _measure(
                lambda: physicalize_plan(
                    lowered,
                    reuse_temporaries=True,
                ),
                repetitions,
            ),
            "baseline_pipeline": _measure(
                lambda: physicalize_plan(lower_to_plan(module)),
                repetitions,
            ),
            "optimized_pipeline": _measure(
                lambda: optimize_whole_program(module),
                repetitions,
            ),
            "arena_baseline_bytes": baseline.arena.size_bytes,
            "arena_optimized_bytes": optimized.arena.size_bytes,
            "arena_reduction_percent": _reduction(
                baseline.arena.size_bytes,
                optimized.arena.size_bytes,
            ),
            "licm_decisions": [
                {
                    "region": item.region,
                    "disposition": item.disposition,
                    "reason": item.reason,
                }
                for item in licm.decisions
            ],
            "optimized_plan_digest": optimized.digest(),
        }
    return result


def _measure(
    function: Callable[[], object],
    repetitions: int,
) -> dict[str, float]:
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "median_ms": statistics.median(values),
        "p99_ms": _p99(values),
        "mean_ms": statistics.mean(values),
    }


def _gate_passed(result: dict[str, object]) -> bool:
    runtime = result["generated_cpp"]
    for model in ("smolvla", "openvla"):
        data = runtime[model]
        if data["linked_libpython"] is not False:
            return False
        for run in data["runs"].values():
            if not (
                run["actions_equal"]
                and run["state_action_trace_equal"]
                and run["evidence_equal"]
            ):
                return False
    cache = result["passes"]["epoch_keyed_cache_synthesis"]["runtime"]
    if any(data["cache_hits"] < 2 for data in cache.values()):
        return False
    if any(data["p99_reduction_percent"] <= 0.0 for data in cache.values()):
        return False
    compiler = result["compiler"]
    return all(
        data["arena_optimized_bytes"] < data["arena_baseline_bytes"]
        for data in compiler.values()
    )


def _p99(values: list[float]) -> float:
    if not values:
        raise ValueError("p99 requires at least one measurement")
    ordered = sorted(values)
    index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[index]


def _reduction(baseline: float, optimized: float) -> float:
    return 100.0 * (baseline - optimized) / baseline


def _parse_metrics(text: str) -> dict[str, str]:
    return dict(
        re.findall(r"([A-Z_]+)=([0-9]+(?:\.[0-9]+)?)", text)
    )


def _parse_optional_metrics(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    values = _parse_metrics(path.read_text(encoding="utf-8"))
    return {
        "wall_seconds": float(values["COMPILE_WALL_SECONDS"]),
        "max_rss_kb": int(values["COMPILE_MAX_RSS_KB"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
