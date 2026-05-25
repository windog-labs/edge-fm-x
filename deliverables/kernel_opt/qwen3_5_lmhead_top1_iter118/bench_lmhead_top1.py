#!/usr/bin/env python3
"""Standalone LMHead top1 correctness and timing sweep for Qwen3.5 shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "lmhead_top1_kernels.cu"
DEFAULT_SHAPES = {
    "qwen3.5-0.8b": (248_320, 1024),
    "qwen3.5-2b": (248_320, 2048),
}


def load_extension():
    return load(
        name="qwen3_5_lmhead_top1_iter118",
        sources=[str(SOURCE)],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def cuda_event_ms(fn, warmup: int, runs: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    last_token = None
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_token = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    assert last_token is not None
    token_value = int(last_token.cpu().item())
    return {
        "token": token_value,
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p50_ms": statistics.median(times),
        "stdev_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "runs": runs,
    }


def torch_reference_token(hidden: torch.Tensor, weight: torch.Tensor) -> int:
    torch.cuda.empty_cache()
    logits = torch.mv(weight.float(), hidden.float())
    token = int(torch.argmax(logits).item())
    del logits
    torch.cuda.empty_cache()
    return token


def run_shape(ext, name: str, vocab: int, hidden_size: int, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + hidden_size)
    hidden = torch.randn(hidden_size, device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    weight = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    torch.cuda.synchronize()

    reference_token = None if args.skip_torch_ref else torch_reference_token(hidden, weight)
    scalar24_token = int(ext.top1_bf16(hidden, weight, 24, False).cpu().item())
    correctness_anchor = scalar24_token if reference_token is None else reference_token

    variants = []
    for vec2 in (False, True):
        for warps in (16, 24, 32):
            label = f"{'bf162' if vec2 else 'scalar'}_w{warps}"

            def fn(warps=warps, vec2=vec2):
                return ext.top1_bf16(hidden, weight, warps, vec2)

            metrics = cuda_event_ms(fn, args.warmup, args.runs)
            metrics.update(
                {
                    "variant": label,
                    "warps_per_block": warps,
                    "vec2": vec2,
                    "matches_scalar24": metrics["token"] == scalar24_token,
                    "matches_torch_ref": None if reference_token is None else metrics["token"] == reference_token,
                }
            )
            variants.append(metrics)

    best = min(variants, key=lambda item: item["mean_ms"])
    result = {
        "shape": name,
        "vocab_size": vocab,
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
        "torch_reference_token": reference_token,
        "scalar24_token": scalar24_token,
        "anchor_token": correctness_anchor,
        "all_variants_match_anchor": all(v["token"] == correctness_anchor for v in variants),
        "best_variant": best["variant"],
        "best_mean_ms": best["mean_ms"],
        "variants": variants,
    }
    del hidden, weight
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "lmhead_top1_iter118_baseline.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=118)
    parser.add_argument("--skip-torch-ref", action="store_true")
    parser.add_argument(
        "--shape",
        action="append",
        choices=sorted(DEFAULT_SHAPES),
        help="Shape name to run. Defaults to both Qwen3.5 shapes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ext = load_extension()
    shape_names = args.shape or list(DEFAULT_SHAPES)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results = []
    for name in shape_names:
        vocab, hidden = DEFAULT_SHAPES[name]
        print(f"[lmhead-top1] shape={name} vocab={vocab} hidden={hidden}")
        result = run_shape(ext, name, vocab, hidden, args)
        results.append(result)
        best = result["best_variant"]
        best_ms = result["best_mean_ms"]
        ok = result["all_variants_match_anchor"]
        print(f"  best={best} mean={best_ms:.4f} ms all_match={ok}")

    payload = {
        "run": "qwen3_5_lmhead_top1_iter118",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[lmhead-top1] wrote {args.out}")


if __name__ == "__main__":
    main()
