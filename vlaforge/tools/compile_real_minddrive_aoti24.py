#!/usr/bin/env python3
"""Compile one MindDrive export into a PyTorch 2.4 AOTI shared object."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

from vlaforge.frontend import load_exported_region


_PROFILES: dict[str, dict[str, object]] = {
    "default": {},
    "conservative": {
        "force_same_precision": True,
        "max_autotune_gemm_backends": "ATEN",
        "epilogue_fusion": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple | list):
        return tuple(value)
    return (value,)


def _tensor_contract(value: Any) -> dict[str, object]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "strides": [int(item) for item in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
        "layout": "contiguous" if value.is_contiguous() else "strided",
    }


def _require_contiguous(
    region: str,
    role: str,
    values: tuple[Any, ...],
) -> None:
    violations = [
        {"index": index, **_tensor_contract(value)}
        for index, value in enumerate(values)
        if not value.is_contiguous()
    ]
    if violations:
        raise ValueError(
            f"{region}: {role} violates the required contiguous physical "
            f"Region ABI: {violations}"
        )


def _comparison(
    reference: Any,
    candidate: Any,
    *,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
) -> dict[str, object]:
    import torch

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "passed": False,
            "shape_match": reference.shape == candidate.shape,
            "dtype_match": reference.dtype == candidate.dtype,
        }
    if reference.dtype == torch.bool or not reference.is_floating_point():
        exact = bool(torch.equal(reference, candidate))
        return {
            "passed": exact,
            "exact": exact,
            "shape_match": True,
            "dtype_match": True,
        }
    reference_fp64 = reference.detach().to(
        device="cpu", dtype=torch.float64
    )
    candidate_fp64 = candidate.detach().to(
        device="cpu", dtype=torch.float64
    )
    difference = candidate_fp64 - reference_fp64
    max_abs = (
        float(difference.abs().max().item())
        if difference.numel()
        else 0.0
    )
    rmse = (
        float(torch.sqrt(difference.square().mean()).item())
        if difference.numel()
        else 0.0
    )
    reference_abs_max = (
        float(reference_fp64.abs().max().item())
        if reference_fp64.numel()
        else 0.0
    )
    nrmse = rmse / max(reference_abs_max, 1.0e-12)
    return {
        "passed": (
            max_abs <= maximum_absolute_error
            and nrmse <= normalized_root_mean_square_error
        ),
        "exact": bool(torch.equal(reference, candidate)),
        "shape_match": True,
        "dtype_match": True,
        "maximum_absolute_error": max_abs,
        "root_mean_square_error": rmse,
        "normalized_root_mean_square_error": nrmse,
        "reference_absolute_maximum": reference_abs_max,
        "thresholds": {
            "maximum_absolute_error": maximum_absolute_error,
            "normalized_root_mean_square_error": (
                normalized_root_mean_square_error
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inductor-profile",
        choices=tuple(_PROFILES),
        default="default",
    )
    parser.add_argument("--maximum-absolute-error", type=float, required=True)
    parser.add_argument(
        "--normalized-root-mean-square-error",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--require-contiguous-abi",
        action="store_true",
        help=(
            "Reject captured inputs/outputs whose concrete stride is not the "
            "stable contiguous physical Region ABI."
        ),
    )
    parser.add_argument(
        "--numerical-output-count",
        type=int,
        help=(
            "Enforce numerical thresholds on only this leading output "
            "prefix. Remaining outputs must retain shape/dtype and repeated "
            "determinism, but may use a separate end-to-end semantic "
            "comparator such as proposal-state bundle alignment."
        ),
    )
    args = parser.parse_args()

    import torch

    if torch.__version__ != "2.4.1+cu118":
        raise RuntimeError(
            "MindDrive source-compatible AOTI compile requires "
            f"torch 2.4.1+cu118, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("MindDrive AOTI compile requires CUDA")
    export_path = args.export.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    program = load_exported_region(export_path)
    load_seconds = time.perf_counter() - load_started
    example_args, example_kwargs = program.example_inputs
    with torch.no_grad():
        reference = _as_tuple(
            program.module()(*example_args, **example_kwargs)
        )
    if args.require_contiguous_abi:
        if example_kwargs:
            raise ValueError(
                f"{args.region}: contiguous physical ABI accepts positional "
                "tensor arguments only"
            )
        _require_contiguous(args.region, "captured inputs", example_args)
        _require_contiguous(args.region, "captured outputs", reference)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    compile_started = time.perf_counter()
    compile_options = {
        "aot_inductor.output_path": str(output_path),
        "aot_inductor.force_mmap_weights": False,
        **_PROFILES[args.inductor_profile],
    }
    actual_path = Path(
        torch._export.aot_compile(
            program.module(),
            example_args,
            example_kwargs,
            options=compile_options,
        )
    ).resolve()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    compile_peak_cuda_allocated = torch.cuda.max_memory_allocated()
    compile_peak_rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    if actual_path != output_path or not output_path.is_file():
        raise RuntimeError(
            f"AOTI output path mismatch: {actual_path} != {output_path}"
        )

    load_artifact_started = time.perf_counter()
    runner = torch._export.aot_load(str(output_path), args.device)
    artifact_load_seconds = time.perf_counter() - load_artifact_started
    torch.cuda.reset_peak_memory_stats()
    first_started = time.perf_counter()
    with torch.no_grad():
        first = _as_tuple(runner(*example_args, **example_kwargs))
    torch.cuda.synchronize()
    first_run_seconds = time.perf_counter() - first_started
    first_peak_cuda_allocated = torch.cuda.max_memory_allocated()
    second_started = time.perf_counter()
    with torch.no_grad():
        second = _as_tuple(runner(*example_args, **example_kwargs))
    torch.cuda.synchronize()
    second_run_seconds = time.perf_counter() - second_started
    if len(reference) != len(first) or len(first) != len(second):
        raise RuntimeError(f"{args.region}: AOTI output arity changed")
    numerical_output_count = (
        len(reference)
        if args.numerical_output_count is None
        else args.numerical_output_count
    )
    if not 1 <= numerical_output_count <= len(reference):
        raise ValueError(
            f"{args.region}: invalid numerical output count "
            f"{numerical_output_count} for {len(reference)} outputs"
        )
    outputs = [
        _comparison(
            expected,
            actual,
            maximum_absolute_error=args.maximum_absolute_error,
            normalized_root_mean_square_error=(
                args.normalized_root_mean_square_error
            ),
        )
        for expected, actual in zip(reference, first, strict=True)
    ]
    for index, output in enumerate(outputs):
        output["threshold_enforced"] = index < numerical_output_count
    repeated_exact = [
        bool(torch.equal(left, right))
        for left, right in zip(first, second, strict=True)
    ]
    checks = {
        "all_enforced_outputs_within_predeclared_thresholds": all(
            item["passed"] for item in outputs[:numerical_output_count]
        ),
        "all_output_shapes_and_dtypes_match": all(
            item["shape_match"] and item["dtype_match"]
            for item in outputs
        ),
        "repeated_execution_exact": all(repeated_exact),
        "physical_tensor_abi": (
            not args.require_contiguous_abi
            or (
                all(value.is_contiguous() for value in example_args)
                and all(value.is_contiguous() for value in reference)
            )
        ),
    }
    report = {
        "schema": "vlaforge.minddrive_aoti24_compile/1",
        "passed": all(checks.values()),
        "evidence_role": "real-L3-compiled-region",
        "region": args.region,
        "checks": checks,
        "export": {
            "path": str(export_path),
            "size_bytes": export_path.stat().st_size,
            "sha256": _sha256(export_path),
        },
        "artifact": {
            "path": str(output_path),
            "kind": "aotinductor-shared-object",
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "force_mmap_weights": False,
        },
        "outputs": outputs,
        "numerical_output_count": numerical_output_count,
        "deferred_semantic_output_indices": list(
            range(numerical_output_count, len(reference))
        ),
        "repeated_execution_exact": repeated_exact,
        "timing_seconds": {
            "export_load": load_seconds,
            "compile": compile_seconds,
            "artifact_load": artifact_load_seconds,
            "first_run": first_run_seconds,
            "second_run": second_run_seconds,
        },
        "memory": {
            "compile_peak_host_rss_kib": compile_peak_rss_kib,
            "compile_peak_cuda_allocated_bytes": (
                compile_peak_cuda_allocated
            ),
            "first_run_peak_cuda_allocated_bytes": (
                first_peak_cuda_allocated
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "inductor_profile": args.inductor_profile,
        "inductor_options": compile_options,
        "physical_tensor_abi": {
            "required": args.require_contiguous_abi,
            "layout": (
                "contiguous" if args.require_contiguous_abi else "captured"
            ),
            "inputs": [
                _tensor_contract(value) for value in example_args
            ],
            "reference_outputs": [
                _tensor_contract(value) for value in reference
            ],
            "compiled_outputs": [
                _tensor_contract(value) for value in first
            ],
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise ValueError(
            f"{args.region}: AOTI compile audit failed: {report}"
        )
    print(
        f"compiled region={args.region} seconds={compile_seconds:.3f} "
        f"bytes={output_path.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
