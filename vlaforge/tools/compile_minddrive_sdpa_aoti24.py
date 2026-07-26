#!/usr/bin/env python3
"""Compile the two fixed MindDrive EVA SDPA profiles to raw CUDA AOTI."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any


_PROFILES = {
    "window": (54, 256, 16, 64),
    "global": (6, 1600, 16, 64),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _comparison(reference: Any, candidate: Any) -> dict[str, object]:
    import torch

    reference_fp64 = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_fp64 = candidate.detach().to(device="cpu", dtype=torch.float64)
    difference = candidate_fp64 - reference_fp64
    return {
        "shape_match": reference.shape == candidate.shape,
        "dtype_match": reference.dtype == candidate.dtype,
        "exact": bool(torch.equal(reference, candidate)),
        "maximum_absolute_error": float(difference.abs().max().item()),
        "root_mean_square_error": float(
            torch.sqrt(difference.square().mean()).item()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    if torch.__version__ != "2.4.1+cu118":
        raise RuntimeError(
            "MindDrive SDPA AOTI compile requires torch 2.4.1+cu118, got "
            f"{torch.__version__}"
        )
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MindDrive SDPA AOTI compile requires CUDA")

    class MindDriveSDPA(torch.nn.Module):
        def forward(self, query: Any, key_value: Any) -> Any:
            query_fp16 = query.to(torch.float16).permute(0, 2, 1, 3)
            key_fp16 = key_value[:, :, 0].to(torch.float16).permute(
                0, 2, 1, 3
            )
            value_fp16 = key_value[:, :, 1].to(torch.float16).permute(
                0, 2, 1, 3
            )
            output = functional.scaled_dot_product_attention(
                query_fp16,
                key_fp16,
                value_fp16,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )
            return (
                output.permute(0, 2, 1, 3)
                .to(torch.float32)
                .contiguous()
            )

    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    module = MindDriveSDPA().eval().to(args.device)
    profile_reports = []
    for profile_name, query_shape in _PROFILES.items():
        generator = torch.Generator(device=args.device)
        generator.manual_seed(20260727)
        query = torch.randn(
            query_shape,
            generator=generator,
            dtype=torch.float16,
            device=args.device,
        )
        key_value = torch.randn(
            (
                query_shape[0],
                query_shape[1],
                2,
                query_shape[2],
                query_shape[3],
            ),
            generator=generator,
            dtype=torch.float16,
            device=args.device,
        )
        example_args = (query.contiguous(), key_value.contiguous())
        with torch.inference_mode():
            reference = module(*example_args)
        if not all(item.is_contiguous() for item in example_args) or (
            not reference.is_contiguous()
        ):
            raise ValueError(
                f"MindDrive {profile_name} SDPA physical ABI is not contiguous"
            )

        output = output_root / f"sdpa_{profile_name}.so"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actual_path = Path(
            torch._export.aot_compile(
                module,
                example_args,
                options={
                    "aot_inductor.output_path": str(output),
                    "aot_inductor.force_mmap_weights": False,
                },
            )
        ).resolve()
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - started
        if actual_path != output or not output.is_file():
            raise RuntimeError(
                f"MindDrive SDPA output path mismatch: {actual_path} != "
                f"{output}"
            )
        runner = torch._export.aot_load(str(output), args.device)
        with torch.inference_mode():
            first = runner(*example_args)
            second = runner(*example_args)
        torch.cuda.synchronize()
        comparison = _comparison(reference, first)
        repeated_exact = bool(torch.equal(first, second))
        passed = (
            comparison["shape_match"]
            and comparison["dtype_match"]
            and comparison["exact"]
            and repeated_exact
            and first.is_contiguous()
        )
        profile_reports.append(
            {
                "profile": profile_name,
                "passed": passed,
                "artifact": {
                    "path": str(output),
                    "size_bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                    "kind": "aotinductor-shared-object",
                },
                "physical_tensor_abi": {
                    "inputs": [
                        _tensor_contract(value) for value in example_args
                    ],
                    "output": _tensor_contract(first),
                },
                "comparison": comparison,
                "repeated_execution_exact": repeated_exact,
                "compile_seconds": compile_seconds,
                "compile_peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                ),
            }
        )
        del runner
        del first
        del second
        del reference
        del query
        del key_value
        torch.cuda.empty_cache()

    report = {
        "schema": "vlaforge.minddrive_sdpa_aoti24/1",
        "passed": all(item["passed"] for item in profile_reports),
        "evidence_role": "real-L3-compiled-physical-region",
        "profiles": profile_reports,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "peak_host_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise ValueError(f"MindDrive SDPA compile audit failed: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
