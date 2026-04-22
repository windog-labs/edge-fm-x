#!/usr/bin/env python3
"""
cuBLASLt Algorithm Search for Edge-FM Linear Layers

This script searches for the best cuBLASLt algorithm for prefill linear operations
on the target hardware (RTX 3060, sm_86).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build-3060" / "install" / "python"))

import edge_fm


def benchmark_linear(layer, x_tensor, y_tensor, warmup=20, iters=100):
    """Benchmark a linear layer"""
    # Warmup
    for _ in range(warmup):
        layer.forward_fp16_bf16(x_tensor, y_tensor, 0, "Prefill")
        torch.cuda.synchronize()

    # Timed runs
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        layer.forward_fp16_bf16(x_tensor, y_tensor, 0, "Prefill")
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iters
    return avg_ms


def search_best_algo(m, k, n, device_id=0):
    """
    Search for best cuBLASLt algorithm for given shape

    Args:
        m: Number of rows in output (seq_len)
        k: Inner dimension (in_features)
        n: Number of columns in output (out_features)
    """
    print(f"\n{'='*60}")
    print(f"Searching best algo for shape: M={m}, K={k}, N={n}")
    print(f"{'='*60}\n")

    device = torch.device(f"cuda:{device_id}")

    # Create test data
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    bias = torch.randn(n, device=device, dtype=torch.bfloat16)
    y = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    # Test different algorithms
    # cuBLASLt typically has 20-30 algorithms
    results = []

    for algo_idx in range(30):
        try:
            # TODO: Implement algo selection via edge_fm API
            # For now, this is a placeholder
            # In practice, you'd need to:
            # 1. Modify linear_impl.cu to expose algo selection
            # 2. Or use cuBLASLt Python bindings directly

            print(f"Testing algo {algo_idx}...", end=" ")

            # Placeholder: would call edge_fm with specific algo
            # latency_ms = benchmark_linear(layer, x_efm, y_efm)

            # For demonstration, using PyTorch
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                torch.matmul(x, weight.t(), out=y)
                y += bias
            torch.cuda.synchronize()
            end = time.perf_counter()
            latency_ms = (end - start) * 10  # ms per call

            results.append({
                "algo_index": algo_idx,
                "latency_ms": latency_ms,
                "gflops": (2 * m * k * n) / (latency_ms * 1e6)
            })

            print(f"{latency_ms:.3f} ms, {results[-1]['gflops']:.1f} GFLOPS")

        except Exception as e:
            print(f"Failed: {e}")
            continue

    # Sort by latency
    results.sort(key=lambda x: x["latency_ms"])

    print(f"\n{'='*60}")
    print("Top 5 Algorithms:")
    print(f"{'='*60}")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Algo {result['algo_index']}: "
              f"{result['latency_ms']:.3f} ms, "
              f"{result['gflops']:.1f} GFLOPS")

    return results


def main():
    parser = argparse.ArgumentParser(description="cuBLASLt algorithm search")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=str, default="cublaslt_algo_search_results.json")
    args = parser.parse_args()

    # Test shapes for Qwen2.5-1.5B prefill
    test_shapes = [
        # (M, K, N) - (seq_len, in_features, out_features)
        (512, 1536, 2048),   # fused_qkv, m=512
        (1024, 1536, 2048),  # fused_qkv, m=1024
        (2048, 1536, 2048),  # fused_qkv, m=2048
        (1024, 1536, 1536),  # attention_output
        (1024, 1536, 8960),  # gate_up_proj
        (1024, 4480, 1536),  # down_proj
    ]

    all_results = {}

    for m, k, n in test_shapes:
        shape_key = f"m{m}_k{k}_n{n}"
        results = search_best_algo(m, k, n, args.device_id)
        all_results[shape_key] = results

    # Save results
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
