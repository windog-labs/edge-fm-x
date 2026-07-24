#!/usr/bin/env python3
"""Paper-grade VLAForge profile benchmark with resumable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

from vlaforge.adapters import (
    build_real_openvla_action_program,
    build_real_smolvla_action_program,
)
from vlaforge.compiler import compile_module


WORKLOADS = ("nominal", "repeat", "all-miss", "stale")
MODEL_MODES = {
    "smolvla": ("off", "cache", "licm", "combined"),
    "openvla": ("off", "cache", "combined"),
}
DTYPE_BYTES = {
    "bool": 1,
    "i32": 4,
    "i64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--compiler-repetitions", type=int, default=30)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--smol-runner", type=Path, required=True)
    parser.add_argument("--smol-prefix", type=Path, required=True)
    parser.add_argument("--smol-solver", type=Path, required=True)
    parser.add_argument("--smol-trim", type=Path, required=True)
    parser.add_argument("--smol-codegen-manifest", type=Path, required=True)
    parser.add_argument("--open-runner", type=Path, required=True)
    parser.add_argument("--open-archive", type=Path, required=True)
    parser.add_argument("--open-input-dir", type=Path, required=True)
    parser.add_argument("--open-codegen-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 30:
        parser.error("--samples must be at least 30 post-warm samples")
    if args.warmup < 1 or args.bootstrap_resamples < 100:
        parser.error("warmup must be positive and bootstrap needs >=100 resamples")

    output = args.output_dir.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    iterations = args.warmup + args.samples
    commands: list[dict[str, object]] = []
    cells = []
    for model in ("smolvla", "openvla"):
        for workload in WORKLOADS:
            baseline = None
            for mode in MODEL_MODES[model]:
                result = _run_cell(
                    args,
                    raw,
                    model,
                    workload,
                    mode,
                    iterations=iterations,
                    reuse=args.reuse_raw,
                )
                commands.append(result.pop("command"))
                samples = result.pop("all_tick_us")[args.warmup :]
                if len(samples) < args.samples:
                    raise RuntimeError(
                        f"{model}/{workload}/{mode}: only "
                        f"{len(samples)} post-warm samples"
                    )
                samples = samples[: args.samples]
                result["latency_us"] = _summarize(
                    samples,
                    args.bootstrap_resamples,
                    seed=_stable_seed(model, workload, mode),
                )
                result["post_warm_samples"] = len(samples)
                if baseline is None:
                    baseline = result
                else:
                    _require_exact_evidence(baseline, result)
                cells.append(result)

    compiler = _compiler_metrics(args.compiler_repetitions)
    manifests = {
        "smolvla": _load_json(args.smol_codegen_manifest),
        "openvla": _load_json(args.open_codegen_manifest),
    }
    activation_bytes = {
        model: _declared_tensor_bytes(manifest.get("spec", {}))
        for model, manifest in manifests.items()
    }
    for cell in cells:
        cell["memory"]["compiler_arena_bytes"] = compiler[cell["model"]][
            "verified"
        ]["arena_bytes"]
        cell["memory"]["backend_declared_tensor_bytes"] = activation_bytes[
            cell["model"]
        ]

    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"]))
    result = {
        "schema": "vlaforge.paper_benchmark/1",
        "revision": revision,
        "source_dirty": dirty,
        "sample_contract": {
            "warmup": args.warmup,
            "post_warm_samples": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence": 0.95,
        },
        "workloads": {
            "nominal": "every input epoch is reused by two adjacent ticks",
            "repeat": "one fresh input epoch is reused for all measured ticks",
            "all-miss": "every tick carries a new input epoch",
            "stale": "the repeated epoch exceeds its max_age_ns bound",
        },
        "modes": {
            "off": "memoization off, temporal LICM off",
            "cache": "memoization on, temporal LICM off",
            "licm": "memoization off, temporal LICM on",
            "combined": "verified memoization, LICM, and arena reuse",
        },
        "compiler": compiler,
        "measurements": cells,
        "openvla_licm": {
            "status": "already_prehoisted",
            "reason": (
                "prefill produces the autoregressive loop seed and is "
                "structurally required in the loop preheader"
            ),
        },
        "artifact_hashes": _artifact_hashes(args),
        "commands": commands,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES", ""
            ),
        },
    }
    result["evidence_exact"] = all(
        cell["exact_vs_off"]
        for cell in cells
        if cell["mode"] != "off"
    )
    result["gate_passed"] = (
        result["evidence_exact"]
        and all(
            cell["post_warm_samples"] >= 30
            for cell in cells
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "paper_benchmark.csv", cells)
    (output / "paper_benchmark.md").write_text(
        _markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": result["gate_passed"], "output": str(output)}))
    return 0 if result["gate_passed"] else 1


def _run_cell(
    args: argparse.Namespace,
    raw: Path,
    model: str,
    workload: str,
    mode: str,
    *,
    iterations: int,
    reuse: bool,
) -> dict[str, object]:
    stem = f"{model}-{workload}-{mode}"
    stdout_path = raw / f"{stem}.stdout"
    stderr_path = raw / f"{stem}.stderr"
    evidence_path = raw / f"{stem}.evidence.bin"
    metrics_path = raw / f"{stem}.run.json"
    if model == "smolvla":
        command = [
            str(args.smol_runner.resolve()),
            str(args.smol_prefix.resolve()),
            str(args.smol_solver.resolve()),
            str(args.smol_trim.resolve()),
            str(evidence_path),
        ]
    else:
        command = [
            str(args.open_runner.resolve()),
            str(args.open_archive.resolve()),
            str(args.open_input_dir.resolve()),
            str(evidence_path),
        ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
            "VLAFORGE_OPT_BENCHMARK": mode,
            "VLAFORGE_BENCH_WORKLOAD": workload,
            "VLAFORGE_BENCH_ITERATIONS": str(iterations),
        }
    )
    if not reuse or not (
        stdout_path.exists()
        and stderr_path.exists()
        and evidence_path.exists()
        and metrics_path.exists()
    ):
        metrics = _execute(
            command,
            environment,
            stdout_path,
            stderr_path,
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        metrics = _load_json(metrics_path)
    stdout = stdout_path.read_text(encoding="utf-8")
    tick_us = []
    cache_hits = 0
    for line in stdout.splitlines():
        if not line.startswith("BENCH_TICK_US,"):
            continue
        fields = line.split(",")
        if len(fields) != 5:
            raise ValueError(f"malformed benchmark line: {line}")
        tick_us.append(float(fields[3]))
        cache_hits += int(fields[4])
    actions = [
        line for line in stdout.splitlines() if line.startswith("ACTION,")
    ]
    non_region = [
        line
        for line in stdout.splitlines()
        if line.startswith("TRACE,") and not line.startswith("TRACE,11,")
    ]
    return {
        "model": model,
        "workload": workload,
        "mode": mode,
        "command": {
            "argv": command,
            "environment": {
                name: environment[name]
                for name in (
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "VLAFORGE_OPT_BENCHMARK",
                    "VLAFORGE_BENCH_WORKLOAD",
                    "VLAFORGE_BENCH_ITERATIONS",
                )
            },
        },
        "all_tick_us": tick_us,
        "cache_hits": cache_hits,
        "cache_misses": len(tick_us) - cache_hits,
        "action_digest": _text_digest(actions),
        "non_region_trace_digest": _text_digest(non_region),
        "evidence_digest": _sha256(evidence_path),
        "exact_vs_off": True,
        "memory": {
            "process_rss_peak_bytes": int(metrics["rss_peak_bytes"]),
            "process_vram_peak_bytes": int(metrics["vram_peak_bytes"]),
            "compiler_arena_bytes": None,
            "backend_declared_tensor_bytes": None,
            "scope_note": (
                "compiler arena, declared backend tensors, process RSS, and "
                "whole-process VRAM are distinct measurements"
            ),
        },
        "wall_seconds": float(metrics["wall_seconds"]),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _execute(
    command: list[str],
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, float | int]:
    rss_peak = 0
    vram_peak = 0
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            text=True,
        )
        next_vram_poll = 0.0
        while process.poll() is None:
            rss_peak = max(rss_peak, _process_rss(process.pid))
            now = time.monotonic()
            if now >= next_vram_poll:
                vram_peak = max(vram_peak, _process_vram(process.pid))
                next_vram_poll = now + 0.2
            time.sleep(0.02)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return {
        "rss_peak_bytes": rss_peak,
        "vram_peak_bytes": vram_peak,
        "wall_seconds": elapsed,
    }


def _process_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def _process_vram(pid: int) -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1]) * 1024 * 1024
            except ValueError:
                return 0
    return 0


def _require_exact_evidence(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> None:
    names = (
        "action_digest",
        "non_region_trace_digest",
        "evidence_digest",
    )
    mismatches = [
        name for name in names if baseline[name] != candidate[name]
    ]
    candidate["exact_vs_off"] = not mismatches
    candidate["evidence_mismatches"] = mismatches
    if mismatches:
        raise RuntimeError(
            f"{candidate['model']}/{candidate['workload']}/"
            f"{candidate['mode']} differs from off: {mismatches}"
        )


def _summarize(
    values: list[float],
    resamples: int,
    *,
    seed: int,
) -> dict[str, object]:
    metrics = {}
    for name, percentile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        estimator = lambda data, p=percentile: _percentile(data, p)
        estimate = estimator(values)
        rng = random.Random(seed + int(percentile * 100))
        bootstrap = []
        for _ in range(resamples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            bootstrap.append(estimator(sample))
        metrics[name] = {
            "estimate": estimate,
            "ci95": [
                _percentile(bootstrap, 0.025),
                _percentile(bootstrap, 0.975),
            ],
        }
    metrics["mean"] = statistics.fmean(values)
    metrics["minimum"] = min(values)
    metrics["maximum"] = max(values)
    metrics["samples"] = list(values)
    return metrics


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    for model, module in modules.items():
        profiles = {}
        for profile in ("off", "verified"):
            durations = []
            compilation = None
            for _ in range(repetitions):
                started = time.perf_counter_ns()
                compilation = compile_module(module, profile=profile)
                durations.append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
            assert compilation is not None
            profiles[profile] = {
                "compile_time_ms": _summarize(
                    durations,
                    1000,
                    seed=_stable_seed(model, profile),
                ),
                "arena_bytes": compilation.plan.arena.size_bytes,
                "plan_digest": compilation.plan.digest(),
                "certificate_digest": compilation.certificate.digest(),
            }
        result[model] = profiles
    return result


def _declared_tensor_bytes(value: object) -> int:
    if isinstance(value, dict):
        if set(value) >= {"shape", "dtype"}:
            elements = 1
            for dimension in value["shape"]:
                elements *= int(dimension)
            return elements * DTYPE_BYTES[str(value["dtype"])]
        return sum(_declared_tensor_bytes(item) for item in value.values())
    if isinstance(value, list):
        return sum(_declared_tensor_bytes(item) for item in value)
    return 0


def _artifact_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "smol_runner": args.smol_runner,
        "smol_prefix": args.smol_prefix,
        "smol_solver": args.smol_solver,
        "smol_trim": args.smol_trim,
        "smol_codegen_manifest": args.smol_codegen_manifest,
        "open_runner": args.open_runner,
        "open_archive": args.open_archive,
        "open_input_dir": args.open_input_dir,
        "open_codegen_manifest": args.open_codegen_manifest,
    }
    return {
        name: (
            _sha256_tree(path.resolve())
            if path.is_dir()
            else _sha256(path.resolve())
        )
        for name, path in paths.items()
    }


def _write_csv(path: Path, cells: list[dict[str, object]]) -> None:
    fields = [
        "model",
        "workload",
        "mode",
        "post_warm_samples",
        "p50_us",
        "p95_us",
        "p99_us",
        "p50_ci_low_us",
        "p50_ci_high_us",
        "p95_ci_low_us",
        "p95_ci_high_us",
        "p99_ci_low_us",
        "p99_ci_high_us",
        "cache_hits",
        "cache_misses",
        "process_rss_peak_bytes",
        "process_vram_peak_bytes",
        "compiler_arena_bytes",
        "backend_declared_tensor_bytes",
        "exact_vs_off",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            latency = cell["latency_us"]
            memory = cell["memory"]
            row = {
                "model": cell["model"],
                "workload": cell["workload"],
                "mode": cell["mode"],
                "post_warm_samples": cell["post_warm_samples"],
                "cache_hits": cell["cache_hits"],
                "cache_misses": cell["cache_misses"],
                "process_rss_peak_bytes": memory["process_rss_peak_bytes"],
                "process_vram_peak_bytes": memory["process_vram_peak_bytes"],
                "compiler_arena_bytes": memory["compiler_arena_bytes"],
                "backend_declared_tensor_bytes": memory[
                    "backend_declared_tensor_bytes"
                ],
                "exact_vs_off": cell["exact_vs_off"],
            }
            for metric in ("p50", "p95", "p99"):
                row[f"{metric}_us"] = latency[metric]["estimate"]
                row[f"{metric}_ci_low_us"] = latency[metric]["ci95"][0]
                row[f"{metric}_ci_high_us"] = latency[metric]["ci95"][1]
            writer.writerow(row)


def _markdown(result: dict[str, object]) -> str:
    lines = [
        "# VLAForge Paper Benchmark",
        "",
        f"- Revision: `{result['revision']}`",
        f"- Gate passed: `{str(result['gate_passed']).lower()}`",
        f"- Exact state/action/evidence: `{str(result['evidence_exact']).lower()}`",
        "",
        "| Model | Workload | Mode | n | p50 us | p95 us | p99 us | RSS MiB | VRAM MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in result["measurements"]:
        memory = cell["memory"]
        latency = cell["latency_us"]
        lines.append(
            "| {model} | {workload} | {mode} | {n} | {p50:.3f} | "
            "{p95:.3f} | {p99:.3f} | {rss:.1f} | {vram:.1f} |".format(
                model=cell["model"],
                workload=cell["workload"],
                mode=cell["mode"],
                n=cell["post_warm_samples"],
                p50=latency["p50"]["estimate"],
                p95=latency["p95"]["estimate"],
                p99=latency["p99"]["estimate"],
                rss=memory["process_rss_peak_bytes"] / 2**20,
                vram=memory["process_vram_peak_bytes"] / 2**20,
            )
        )
    lines.extend(
        [
            "",
            "Bootstrap 95% confidence intervals, exact commands, environment, "
            "hashes, compiler arena, declared backend tensors, process RSS, "
            "and whole-process VRAM are retained in `paper_benchmark.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _text_digest(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _stable_seed(*values: str) -> int:
    digest = hashlib.sha256("\0".join(values).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
