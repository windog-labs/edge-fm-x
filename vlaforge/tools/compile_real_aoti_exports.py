#!/usr/bin/env python3
"""Compile saved real-model torch.export regions into AOTI packages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

from vlaforge.frontend import load_exported_region


_INDUCTOR_PROFILES: dict[str, dict[str, object]] = {
    "default": {
        "aot_inductor.force_mmap_weights": True,
    },
    # Keep GEMMs on ATen and disable epilogue fusion when a deployment audit
    # needs the closest practical match to exported eager BF16 numerics.
    # `emulate_precision_casts` is deliberately excluded: PyTorch 2.6 fails
    # while lowering the real SmolVLA prefix graph with that option.
    "conservative": {
        "aot_inductor.force_mmap_weights": True,
        "force_same_precision": True,
        "max_autotune_gemm_backends": "ATEN",
        "mixed_mm_choice": "aten",
        "epilogue_fusion": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--inductor-profile",
        choices=tuple(_INDUCTOR_PROFILES),
        default="default",
    )
    parser.add_argument(
        "--mmap-export-cache",
        type=Path,
        help=(
            "content-addressed extraction cache for memory-bounded loading "
            "of legacy torch.export archives"
        ),
    )
    parser.add_argument(
        "--normalized-export-dir",
        type=Path,
        help=(
            "re-export each loaded program with the active PyTorch version "
            "before AOTI compilation"
        ),
    )
    parser.add_argument("regions", nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.normalized_export_dir is not None:
        args.normalized_export_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name in args.regions:
        export_path = args.export_dir / f"{name}.pt2e"
        package_path = args.output_dir / f"{name}.pt2"
        load_started = time.perf_counter()
        program = load_exported_region(
            export_path,
            mmap_cache=args.mmap_export_cache,
        )
        load_seconds = time.perf_counter() - load_started
        load_peak_rss_kib = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        source_program = program
        normalized_path: Path | None = None
        normalize_seconds: float | None = None
        normalize_maximum_absolute_error: float | None = None
        normalize_exact: bool | None = None
        if args.normalized_export_dir is not None:
            example_args, example_kwargs = program.example_inputs
            with torch.inference_mode():
                source_outputs = _as_tuple(
                    program.module()(
                        *example_args,
                        **example_kwargs,
                    )
                )
            normalize_started = time.perf_counter()
            program = torch.export.export(
                program.module(),
                example_args,
                kwargs=example_kwargs,
                strict=False,
            )
            normalized_path = (
                args.normalized_export_dir / f"{name}.pt2"
            )
            torch.export.save(program, normalized_path)
            normalize_seconds = time.perf_counter() - normalize_started
            with torch.inference_mode():
                normalized_outputs = _as_tuple(
                    program.module()(
                        *example_args,
                        **example_kwargs,
                    )
                )
            if len(source_outputs) != len(normalized_outputs):
                raise RuntimeError(
                    f"{name}: normalized export output arity changed"
                )
            normalize_maximum_absolute_error = max(
                (
                    float(
                        (expected - actual)
                        .abs()
                        .float()
                        .max()
                        .item()
                    )
                    if expected.numel()
                    else 0.0
                )
                for expected, actual in zip(
                    source_outputs,
                    normalized_outputs,
                    strict=True,
                )
            )
            normalize_exact = all(
                torch.equal(expected, actual)
                for expected, actual in zip(
                    source_outputs,
                    normalized_outputs,
                    strict=True,
                )
            )
            if not normalize_exact:
                raise RuntimeError(
                    f"{name}: active-version re-export changed outputs "
                    f"(max_abs={normalize_maximum_absolute_error})"
                )
        started = time.perf_counter()
        actual = Path(
            torch._inductor.aoti_compile_and_package(
                program,
                package_path=str(package_path),
                inductor_configs=_INDUCTOR_PROFILES[
                    args.inductor_profile
                ],
            )
        )
        elapsed = time.perf_counter() - started
        compile_peak_rss_kib = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        if actual != package_path or not package_path.is_file():
            raise RuntimeError(
                f"AOTI package path mismatch: {actual} != {package_path}"
            )
        records.append(
            {
                "region": name,
                "export_path": str(export_path.resolve()),
                "export_sha256": _sha256(export_path),
                "normalized_export_path": (
                    str(normalized_path.resolve())
                    if normalized_path is not None
                    else None
                ),
                "normalized_export_sha256": (
                    _sha256(normalized_path)
                    if normalized_path is not None
                    else None
                ),
                "normalize_seconds": normalize_seconds,
                "normalize_exact": normalize_exact,
                "normalize_maximum_absolute_error": (
                    normalize_maximum_absolute_error
                ),
                "package_path": str(package_path.resolve()),
                "package_sha256": _sha256(package_path),
                "package_size_bytes": package_path.stat().st_size,
                "compile_seconds": elapsed,
                "compile_peak_rss_kib": compile_peak_rss_kib,
                "load_seconds": load_seconds,
                "load_peak_rss_kib": load_peak_rss_kib,
                "load_mode": (
                    "legacy_mmap"
                    if args.mmap_export_cache is not None
                    else "torch_export"
                ),
                "graph_nodes": len(tuple(program.graph_module.graph.nodes)),
            }
        )
        print(
            f"compiled region={name} seconds={elapsed:.3f} "
            f"bytes={package_path.stat().st_size}",
            flush=True,
        )
        del program, source_program
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema": "vlaforge.real_aoti_compile/1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inductor_profile": args.inductor_profile,
        "inductor_configs": _INDUCTOR_PROFILES[args.inductor_profile],
        "regions": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


if __name__ == "__main__":
    raise SystemExit(main())
