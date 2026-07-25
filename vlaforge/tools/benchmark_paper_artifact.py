#!/usr/bin/env python3
"""Invocation IR v0.2 host reference benchmark.

This benchmark measures the Python Semantic and Plan executors and validates
their committed-output parity. It intentionally does not report real model,
generated C++, CUDA, or Orin performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Iterable, Mapping

from vlaforge.adapters import (
    build_driving_diffusion_fixture,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.interpreter import InputBinding, InputStamp, Interpreter
from vlaforge.interpreter.trace import normalize_value
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan


_DTYPE_BYTES = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(
    values: list[float],
    bootstrap_resamples: int,
    *,
    seed: int,
) -> dict[str, object]:
    if not values:
        raise ValueError("latency samples must be non-empty")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    generator = random.Random(seed)
    result: dict[str, object] = {"samples": list(values)}
    for name, percentile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        bootstrap = []
        for _ in range(bootstrap_resamples):
            sample = [
                values[generator.randrange(len(values))]
                for _ in range(len(values))
            ]
            bootstrap.append(_percentile(sample, percentile))
        result[name] = {
            "estimate": _percentile(values, percentile),
            "ci95": [
                _percentile(bootstrap, 0.025),
                _percentile(bootstrap, 0.975),
            ],
        }
    return result


def _declared_tensor_bytes(value: object) -> int:
    if isinstance(value, Mapping):
        if "shape" in value and "dtype" in value:
            elements = math.prod(int(item) for item in value["shape"])
            dtype = str(value["dtype"]).removeprefix("torch.")
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"unknown tensor dtype: {dtype}")
            return elements * _DTYPE_BYTES[dtype]
        return sum(_declared_tensor_bytes(item) for item in value.values())
    if isinstance(value, tuple | list):
        return sum(_declared_tensor_bytes(item) for item in value)
    return 0


def _write_csv(path: Path, cells: Iterable[Mapping[str, object]]) -> None:
    fields = (
        "model",
        "workload",
        "mode",
        "measurement_reused_from",
        "post_warm_samples",
        "p50_us",
        "p95_us",
        "p99_us",
        "cache_hits",
        "cache_misses",
        "process_rss_peak_bytes",
        "process_vram_peak_bytes",
        "compiler_arena_bytes",
        "backend_declared_tensor_bytes",
        "exact_vs_off",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for cell in cells:
            latency = cell["latency_us"]
            memory = cell["memory"]
            writer.writerow(
                {
                    "model": cell["model"],
                    "workload": cell["workload"],
                    "mode": cell["mode"],
                    "measurement_reused_from": cell.get(
                        "measurement_reused_from", ""
                    ),
                    "post_warm_samples": cell["post_warm_samples"],
                    "p50_us": latency["p50"]["estimate"],
                    "p95_us": latency["p95"]["estimate"],
                    "p99_us": latency["p99"]["estimate"],
                    "cache_hits": cell["cache_hits"],
                    "cache_misses": cell["cache_misses"],
                    "process_rss_peak_bytes": memory[
                        "process_rss_peak_bytes"
                    ],
                    "process_vram_peak_bytes": memory[
                        "process_vram_peak_bytes"
                    ],
                    "compiler_arena_bytes": memory["compiler_arena_bytes"],
                    "backend_declared_tensor_bytes": memory[
                        "backend_declared_tensor_bytes"
                    ],
                    "exact_vs_off": cell["exact_vs_off"],
                }
            )


def _markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# VLAForge Invocation v0.2 Host Reference Benchmark",
        "",
        f"- Revision: `{result['revision']}`",
        f"- Gate passed: `{str(result['gate_passed']).lower()}`",
        f"- Exact Semantic/Plan outputs: "
        f"`{str(result['evidence_exact']).lower()}`",
        "- Scope: Python fixture reference executors only; not real model, "
        "generated C++, CUDA, or Orin performance",
        "",
        "| Model | Workload | Mode | Measurement | n | p50 us | p95 us | "
        "p99 us |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in result["measurements"]:
        latency = cell["latency_us"]
        reused = cell.get("measurement_reused_from")
        measurement = f"reused from {reused}" if reused else "measured"
        lines.append(
            f"| {cell['model']} | {cell['workload']} | {cell['mode']} | "
            f"{measurement} | {cell['post_warm_samples']} | "
            f"{latency['p50']['estimate']:.3f} | "
            f"{latency['p95']['estimate']:.3f} | "
            f"{latency['p99']['estimate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _bindings(
    fixture,
    iteration: int,
    workload: str,
) -> dict[str, InputBinding]:
    base = fixture.runs[0].inputs
    if workload == "repeat":
        return dict(base)
    if workload != "new-revision":
        raise ValueError(f"unknown workload: {workload}")
    return {
        name: InputBinding(
            binding.value,
            InputStamp(
                revision=1000 + iteration * 100 + index,
                timestamp_ns=binding.stamp.timestamp_ns,
            ),
        )
        for index, (name, binding) in enumerate(sorted(base.items()))
    }


def _run_cell(
    name: str,
    fixture,
    mode: str,
    workload: str,
    warmup: int,
    samples: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, object], list[object]]:
    plan = physicalize_plan(lower_to_plan(fixture.module))
    if mode == "semantic":
        executor = Interpreter(
            fixture.module,
            regions=fixture.regions,
            validators=fixture.validators,
            initial_state=fixture.initial_state,
        )
    elif mode == "plan":
        executor = PlanExecutor(
            plan,
            fixture.module,
            regions=fixture.regions,
            validators=fixture.validators,
            initial_state=fixture.initial_state,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    latency = []
    outputs = []
    start_hits = executor.cache.hits
    start_misses = executor.cache.misses
    for iteration in range(warmup + samples):
        started = time.perf_counter_ns()
        result = executor.run(
            inputs=_bindings(fixture, iteration, workload)
        )
        elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        if iteration >= warmup:
            latency.append(elapsed_us)
            outputs.append(normalize_value(result.committed_outputs))
    specification = {
        "inputs": [
            port.payload.to_dict()
            for port in fixture.module.inputs
            if hasattr(port.payload, "shape")
        ],
        "outputs": [
            port.payload.to_dict()
            for port in fixture.module.outputs
            if hasattr(port.payload, "shape")
        ],
    }
    cell = {
        "model": name,
        "workload": workload,
        "mode": mode,
        "post_warm_samples": samples,
        "latency_us": _summarize(
            latency,
            bootstrap_resamples,
            seed=int.from_bytes(
                hashlib.sha256(
                    f"{name}/{workload}/{mode}".encode()
                ).digest()[:4],
                "little",
            ),
        ),
        "cache_hits": executor.cache.hits - start_hits,
        "cache_misses": executor.cache.misses - start_misses,
        "memory": {
            "process_rss_peak_bytes": 0,
            "process_vram_peak_bytes": 0,
            "compiler_arena_bytes": (
                0 if plan.arena is None else int(plan.arena.size_bytes)
            ),
            "backend_declared_tensor_bytes": _declared_tensor_bytes(
                specification
            ),
        },
        "exact_vs_off": True,
    }
    return cell, outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    args = parser.parse_args()
    if args.samples < 30:
        parser.error("--samples must be at least 30")
    if args.warmup < 0 or args.bootstrap_resamples < 100:
        parser.error("warmup must be non-negative and bootstrap >= 100")

    fixtures = {
        "openvla_fixture": build_openvla_fixture(),
        "smolvla_fixture": build_smolvla_fixture(),
        "driving_diffusion_fixture": build_driving_diffusion_fixture(),
    }
    cells = []
    exact = True
    for name, fixture in fixtures.items():
        for workload in ("repeat", "new-revision"):
            semantic, semantic_outputs = _run_cell(
                name,
                fixture,
                "semantic",
                workload,
                args.warmup,
                args.samples,
                args.bootstrap_resamples,
            )
            plan, plan_outputs = _run_cell(
                name,
                fixture,
                "plan",
                workload,
                args.warmup,
                args.samples,
                args.bootstrap_resamples,
            )
            exact = exact and semantic_outputs == plan_outputs
            semantic["exact_vs_off"] = semantic_outputs == plan_outputs
            plan["exact_vs_off"] = semantic_outputs == plan_outputs
            cells.extend((semantic, plan))

    result = {
        "schema": "vlaforge.paper_benchmark/2",
        "revision": "working-tree",
        "evidence_scope": "deterministic v0.2 Python fixtures",
        "evidence_exact": exact,
        "gate_passed": exact,
        "measurements": cells,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vlaforge_paper_benchmark_v02.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / "vlaforge_paper_benchmark_v02.csv",
        cells,
    )
    (args.output_dir / "vlaforge_paper_benchmark_v02.md").write_text(
        _markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": exact, "cells": len(cells)}))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
