#!/usr/bin/env python3
"""Standalone Qwen3.5 decode GateUp+SwiGLU feasibility benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "gateup_swiglu_kernels.cu"
DEFAULT_SHAPES = {
    "qwen3.5-0.8b": (1024, 3584),
    "qwen3.5-2b": (2048, 6144),
}


def load_extension():
    return load(
        name="qwen3_5_gateup_swiglu_iter121",
        sources=[str(SOURCE)],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    fused = torch.matmul(weight, hidden)
    up, gate = fused.chunk(2, dim=0)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def event_ms(fn, warmup: int, runs: int) -> dict[str, Any]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p50_ms": statistics.median(times),
        "stdev_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "last_checksum": float(out.float().sum().item()),
    }


def run_shape(ext, name: str, hidden_size: int, intermediate_size: int, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + hidden_size)
    hidden = torch.randn(hidden_size, device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    weight = torch.randn(2 * intermediate_size, hidden_size, device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    torch.cuda.synchronize()

    ref = reference(hidden, weight)
    variants = []
    for vec2 in (False, True):
        for warps in (8, 16, 24, 32):
            label = f"{'bf162' if vec2 else 'scalar'}_w{warps}"
            out = ext.gateup_swiglu_bf16(hidden, weight, warps, vec2)
            torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs()

            def fn(warps=warps, vec2=vec2):
                return ext.gateup_swiglu_bf16(hidden, weight, warps, vec2)

            metrics = event_ms(fn, args.warmup, args.runs)
            metrics.update(
                {
                    "variant": label,
                    "warps_per_block": warps,
                    "vec2": vec2,
                    "max_abs_diff": float(diff.max().item()),
                    "mean_abs_diff": float(diff.mean().item()),
                    "allclose_2e2_6e2": bool(torch.allclose(out, ref, rtol=2e-2, atol=6.25e-2)),
                }
            )
            variants.append(metrics)

    def torch_fn():
        return reference(hidden, weight)

    torch_metrics = event_ms(torch_fn, args.warmup, args.runs)
    best = min(variants, key=lambda item: item["mean_ms"])
    result = {
        "shape": name,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "torch_reference_mean_ms": torch_metrics["mean_ms"],
        "best_variant": best["variant"],
        "best_mean_ms": best["mean_ms"],
        "best_allclose": best["allclose_2e2_6e2"],
        "variants": variants,
    }
    del hidden, weight, ref
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "gateup_swiglu_iter121.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--shape", action="append", choices=sorted(DEFAULT_SHAPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ext = load_extension()
    results = []
    for name in args.shape or list(DEFAULT_SHAPES):
        hidden, intermediate = DEFAULT_SHAPES[name]
        print(f"[gateup-swiglu] shape={name} hidden={hidden} intermediate={intermediate}")
        result = run_shape(ext, name, hidden, intermediate, args)
        results.append(result)
        print(
            f"  torch={result['torch_reference_mean_ms']:.4f} ms "
            f"best={result['best_variant']} {result['best_mean_ms']:.4f} ms "
            f"allclose={result['best_allclose']}"
        )
    payload = {
        "run": "qwen3_5_gateup_swiglu_iter121",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[gateup-swiglu] wrote {args.out}")


if __name__ == "__main__":
    main()
