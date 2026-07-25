#!/usr/bin/env python3
"""Aggregate real-model host-CUDA benchmarks into paper-facing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


_SCHEMA = "vlaforge.real_cuda_evidence/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"expected JSON row array: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _archive(
    source: Path,
    archive_root: Path,
    relative: Path,
) -> dict[str, object]:
    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise RuntimeError(f"archived evidence digest mismatch: {source}")
    return _artifact(destination)


def _speedup(baseline_ns: float, optimized_ns: float) -> float:
    if baseline_ns <= 0 or optimized_ns <= 0:
        raise ValueError("latencies must be positive")
    return baseline_ns / optimized_ns


def _latency(report: dict[str, Any]) -> dict[str, float | int]:
    latency = report["latency"]
    return {
        "samples": int(report["samples"]),
        "mean_ns": float(latency["mean_ns"]),
        "p50_ns": int(latency["p50_ns"]),
        "p90_ns": int(latency["p90_ns"]),
        "p99_ns": int(latency["p99_ns"]),
        "throughput_runs_per_second": float(
            latency["throughput_runs_per_second"]
        ),
    }


def _validate_generated(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "passed"
        or report.get("path") != "generated_no_python_cpp_session"
        or report.get("runner", {}).get("links_libpython") is not False
        or report.get("bundle", {}).get("source_dirty") is not False
        or int(report.get("runtime", {}).get("transaction_aborts", -1)) != 0
        or int(report.get("memory", {}).get("cuda_used_drift_bytes", -1)) != 0
    ):
        raise ValueError("generated Session benchmark is not publishable")


def _validate_model_path(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "passed"
        or report.get("path") not in {"eager", "direct_artifact"}
        or report.get("reproduction", {}).get("source_dirty") is not False
        or int(report.get("memory", {}).get("cuda_used_drift_bytes", -1)) != 0
    ):
        raise ValueError("real model-path benchmark is not publishable")


def _load_generated_matrix(
    *,
    diffusion_root: Path,
    smolvla_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    declarations = {
        "diffusiondrive": {
            "full": diffusion_root / "diffusiondrive_full_100.json",
            "same": diffusion_root / "diffusiondrive_same_500.json",
            "new": diffusion_root / "diffusiondrive_new_500.json",
            "missing": diffusion_root / "diffusiondrive_missing_500.json",
            "soak": diffusion_root / "diffusiondrive_same_10000.json",
        },
        "smolvla": {
            "full": smolvla_root / "smolvla_full_100.json",
            "same": smolvla_root / "smolvla_same_500.json",
            "new": smolvla_root / "smolvla_new_500.json",
            "missing": smolvla_root / "smolvla_missing_500.json",
            "soak": smolvla_root / "smolvla_same_10000.json",
        },
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model, modes in declarations.items():
        result[model] = {}
        for mode, path in modes.items():
            report = _json(path)
            _validate_generated(report)
            if report.get("model") != model:
                raise ValueError(f"model mismatch in {path}")
            result[model][mode] = report
    return result


def _load_model_paths(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model in ("diffusiondrive", "smolvla"):
        result[model] = {}
        for path_name in ("eager", "direct_artifact"):
            path = root / f"{model}_{path_name}.json"
            report = _json(path)
            _validate_model_path(report)
            if (
                report.get("model") != model
                or report.get("path") != path_name
            ):
                raise ValueError(f"model/path mismatch in {path}")
            result[model][path_name] = report
    return result


def _nsight_summary(root: Path, model: str) -> dict[str, object]:
    api = _json_rows(root / f"{model}_stats_cuda_api_sum.json")
    kernels = _json_rows(root / f"{model}_stats_cuda_gpu_kern_sum.json")
    memory = _json_rows(
        root / f"{model}_stats_cuda_gpu_mem_time_sum.json"
    )
    nsys = root / f"{model}_full.nsys-rep"
    ncu = root / f"{model}_basic.ncu-rep"
    stderr = root / f"{model}_ncu.stderr"
    if stderr.read_text(encoding="utf-8"):
        raise ValueError(f"{model} NCU stderr is non-empty")
    return {
        "nsys": _artifact(nsys),
        "ncu": _artifact(ncu),
        "ncu_stderr_bytes": stderr.stat().st_size,
        "top_cuda_apis": api[:8],
        "top_cuda_kernels": kernels[:10],
        "gpu_memory_operations": memory,
        "scope": (
            "20 measured full-compute Runs after 2 warmups for NSYS; "
            "20 representative kernels with NCU basic metrics"
        ),
    }


def _checksum_per_run(report: dict[str, Any]) -> float:
    return float(report["runtime"]["checksum"]) / int(report["samples"])


def _raw_sources(
    *,
    generated: dict[str, dict[str, dict[str, Any]]],
    model_paths: dict[str, dict[str, dict[str, Any]]],
    diffusion_root: Path,
    smolvla_root: Path,
    model_path_root: Path,
    nsight_root: Path,
    archive_root: Path,
) -> dict[str, object]:
    records: dict[str, object] = {}
    roots = {
        "diffusiondrive": diffusion_root,
        "smolvla": smolvla_root,
    }
    for model, modes in generated.items():
        records[f"{model}_generated"] = {
            mode: {
                "summary": _archive(
                    roots[model]
                    / f"{model}_{report['mode']}_{report['samples']}.json",
                    archive_root,
                    Path("generated")
                    / f"{model}_{report['mode']}_{report['samples']}.json",
                ),
                "samples_csv": _archive(
                    roots[model]
                    / f"{model}_{report['mode']}_{report['samples']}.csv",
                    archive_root,
                    Path("generated")
                    / f"{model}_{report['mode']}_{report['samples']}.csv",
                ),
            }
            for mode, report in modes.items()
        }
    for model, paths in model_paths.items():
        records[f"{model}_model_paths"] = {
            path_name: _archive(
                model_path_root / f"{model}_{path_name}.json",
                archive_root,
                Path("model_paths") / f"{model}_{path_name}.json",
            )
            for path_name in paths
        }
    records["nsight_summaries"] = {
        model: {
            report_name: _archive(
                nsight_root / f"{model}_stats_{report_name}.json",
                archive_root,
                Path("nsight") / f"{model}_{report_name}.json",
            )
            for report_name in (
                "cuda_api_sum",
                "cuda_gpu_kern_sum",
                "cuda_gpu_mem_time_sum",
                "osrt_sum",
            )
        }
        for model in ("diffusiondrive", "smolvla")
    }
    records["nsight_runner_output"] = {
        model: {
            "nsys_stdout": _archive(
                nsight_root / f"{model}_full.stdout",
                archive_root,
                Path("nsight") / f"{model}_nsys.stdout",
            ),
            "ncu_stdout": _archive(
                nsight_root / f"{model}_ncu.stdout",
                archive_root,
                Path("nsight") / f"{model}_ncu.stdout",
            ),
        }
        for model in ("diffusiondrive", "smolvla")
    }
    return records


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VLAForge real-model host-CUDA evidence",
        "",
        "This report covers an RTX 3060 (sm_86), CUDA 12.8, "
        "PyTorch 2.10.0+cu128 host. It makes no Orin claim.",
        "",
        "## Full-compute path comparison",
        "",
        "| Model | Eager mean | Direct AOTI mean | Generated C++ mean | "
        "C++ vs eager | C++ overhead vs direct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("diffusiondrive", "smolvla"):
        item = report["comparisons"][model]
        lines.append(
            f"| {model} | {item['eager_mean_ms']:.3f} ms | "
            f"{item['direct_mean_ms']:.3f} ms | "
            f"{item['generated_mean_ms']:.3f} ms | "
            f"{item['generated_vs_eager_speedup']:.3f}x | "
            f"{item['generated_vs_direct_overhead_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "Generated Session and direct AOTI use the same compiled model "
            "artifacts. Their near-equal full-compute latency bounds framework "
            "overhead; eager speedups are not attributed to new CUDA kernels.",
            "",
            "## Stateful invocation and exact-reuse ablations",
            "",
            "| Model | Full mean | Same revision mean | New revision mean | "
            "Missing revision mean | Same/full speedup |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("diffusiondrive", "smolvla"):
        item = report["ablations"][model]
        lines.append(
            f"| {model} | {item['full_mean_ms']:.3f} ms | "
            f"{item['same_mean_ms']:.3f} ms | "
            f"{item['new_mean_ms']:.3f} ms | "
            f"{item['missing_mean_ms']:.3f} ms | "
            f"{item['same_vs_full_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "DiffusionDrive same-revision Runs reuse only the exact condition "
            "cache. SmolVLA also exercises Adapter-owned action queue/cursor "
            "state; this queue is not a core-IR assumption.",
            "",
            "## 10,000-Run soak",
            "",
            "| Model | Commits | Cache hit/miss | State commits | "
            "CUDA drift | RSS drift | Aborts |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("diffusiondrive", "smolvla"):
        soak = report["soak"][model]
        lines.append(
            f"| {model} | {soak['transaction_commits']} | "
            f"{soak['cache_hits']}/{soak['cache_misses']} | "
            f"{soak['state_commits']} | {soak['cuda_drift_bytes']} B | "
            f"{soak['rss_drift_kib']} KiB | "
            f"{soak['transaction_aborts']} |"
        )
    lines.extend(
        [
            "",
            "## Profiling interpretation",
            "",
            "- NSYS and NCU ran the generated no-Python C++ binaries, not "
            "Python wrappers.",
            "- Kernel time remains in upstream AOTI/cuDNN/CUTLASS/Triton "
            "kernels. VLAForge does not claim those kernels as contributions.",
            "- Scalar Region storage is now 16-byte aligned. The pre-fix "
            "SmolVLA bundle emitted two warning sites per Run; the current "
            "bundle emits none.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-generated-root", type=Path, required=True)
    parser.add_argument("--smolvla-generated-root", type=Path, required=True)
    parser.add_argument("--model-path-root", type=Path, required=True)
    parser.add_argument("--nsight-root", type=Path, required=True)
    parser.add_argument("--pre-alignment-ncu-stderr", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    generated = _load_generated_matrix(
        diffusion_root=args.diffusion_generated_root,
        smolvla_root=args.smolvla_generated_root,
    )
    model_paths = _load_model_paths(args.model_path_root)
    comparisons: dict[str, object] = {}
    ablations: dict[str, object] = {}
    soak: dict[str, object] = {}
    for model in ("diffusiondrive", "smolvla"):
        full = generated[model]["full"]
        same = generated[model]["same"]
        new = generated[model]["new"]
        missing = generated[model]["missing"]
        eager = model_paths[model]["eager"]
        direct = model_paths[model]["direct_artifact"]
        generated_mean = float(full["latency"]["mean_ns"])
        eager_mean = float(eager["latency"]["mean_ns"])
        direct_mean = float(direct["latency"]["mean_ns"])
        if _checksum_per_run(full) != _checksum_per_run(direct):
            raise ValueError(
                f"{model} direct and generated output probes differ"
            )
        comparisons[model] = {
            "eager_mean_ms": eager_mean / 1e6,
            "direct_mean_ms": direct_mean / 1e6,
            "generated_mean_ms": generated_mean / 1e6,
            "generated_vs_eager_speedup": _speedup(
                eager_mean, generated_mean
            ),
            "generated_vs_direct_overhead_percent": (
                generated_mean / direct_mean - 1.0
            ) * 100.0,
            "direct_generated_probe_exact": True,
        }
        ablations[model] = {
            "full_mean_ms": generated_mean / 1e6,
            "same_mean_ms": float(same["latency"]["mean_ns"]) / 1e6,
            "new_mean_ms": float(new["latency"]["mean_ns"]) / 1e6,
            "missing_mean_ms": float(missing["latency"]["mean_ns"]) / 1e6,
            "same_vs_full_speedup": _speedup(
                generated_mean,
                float(same["latency"]["mean_ns"]),
            ),
            "same_cache_hits": int(same["runtime"]["cache_hits"]),
            "same_cache_misses": int(same["runtime"]["cache_misses"]),
            "new_cache_hits": int(new["runtime"]["cache_hits"]),
            "new_cache_misses": int(new["runtime"]["cache_misses"]),
            "missing_cache_hits": int(missing["runtime"]["cache_hits"]),
            "missing_cache_misses": int(missing["runtime"]["cache_misses"]),
        }
        soak_report = generated[model]["soak"]
        if (
            int(soak_report["samples"]) != 10_000
            or int(soak_report["runtime"]["transaction_commits"]) != 10_000
        ):
            raise ValueError(f"{model} soak is incomplete")
        soak[model] = {
            "latency": _latency(soak_report),
            "checksum": float(soak_report["runtime"]["checksum"]),
            "cache_hits": int(soak_report["runtime"]["cache_hits"]),
            "cache_misses": int(soak_report["runtime"]["cache_misses"]),
            "state_commits": int(soak_report["runtime"]["state_commits"]),
            "transaction_commits": int(
                soak_report["runtime"]["transaction_commits"]
            ),
            "transaction_aborts": int(
                soak_report["runtime"]["transaction_aborts"]
            ),
            "output_commits": int(
                soak_report["runtime"]["output_commits"]
            ),
            "state_versions": [
                int(soak_report["runtime"]["state_0_version"]),
                int(soak_report["runtime"]["state_1_version"]),
            ],
            "cuda_drift_bytes": int(
                soak_report["memory"]["cuda_used_drift_bytes"]
            ),
            "rss_drift_kib": int(soak_report["memory"]["rss_drift_kib"]),
        }

    pre_alignment = args.pre_alignment_ncu_stderr.read_text(
        encoding="utf-8"
    )
    warning_count = pre_alignment.count("not aligned at run time")
    if warning_count < 1:
        raise ValueError("pre-alignment profile has no alignment warning")
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "scope": {
            "platform": "host CUDA RTX 3060 sm_86",
            "orin_in_scope": False,
            "models": ["SmolVLA-Base", "DiffusionDrive NAVSIM 88.1 PDMS"],
            "timing_boundary": (
                "one full Session::Run/model invocation through backend "
                "synchronization; setup, input upload, output probe, and "
                "reporting excluded"
            ),
        },
        "comparisons": comparisons,
        "ablations": ablations,
        "soak": soak,
        "alignment_fix": {
            "pre_fix_warning_count": warning_count,
            "post_fix_warning_count": 0,
            "minimum_scalar_region_storage_alignment": 16,
            "smolvla_verified_arena_bytes": int(
                generated["smolvla"]["full"]["memory"]["static_arena"][
                    "compiled_bytes"
                ]
            ),
            "interpretation": (
                "Removed AOTI implicit aligned copies for loop-carried cursor "
                "scalars; no model kernel changed."
            ),
        },
        "nsight": {
            model: _nsight_summary(args.nsight_root, model)
            for model in ("diffusiondrive", "smolvla")
        },
        "raw_evidence": _raw_sources(
            generated=generated,
            model_paths=model_paths,
            diffusion_root=args.diffusion_generated_root,
            smolvla_root=args.smolvla_generated_root,
            model_path_root=args.model_path_root,
            nsight_root=args.nsight_root,
            archive_root=args.archive_root,
        ),
        "claim_boundary": (
            "AOTI/cuDNN/CUTLASS/Triton kernels are upstream-generated. "
            "VLAForge claims whole-program state/cache/transaction semantics, "
            "bounded static memory, generated Session integration, and "
            "measured orchestration overhead—not ownership of model kernels."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
