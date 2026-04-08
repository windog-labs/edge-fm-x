#!/usr/bin/env python3
"""
Prototype benchmark for decode m=1 gate_up + silu_and_mul fusion.

Purpose:
- measure a fair Triton unfused-vs-fused A/B on the exact Qwen2.5-1.5B decode shape
- optionally compare the winning Triton prototype against the current EdgeFM runtime path

This script is an experiment driver. It does not change runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"
OP_TABLE_PATH = REPO_ROOT / "examples" / "config" / "operator_impl_table.json"
BUILD_PYTHON_CANDIDATES = [
    REPO_ROOT / "build" / "install" / "python",
    REPO_ROOT / "build" / "python",
]

HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class GemvConfig:
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class ActConfig:
    block_n: int
    num_warps: int
    num_stages: int


GEMV_CONFIGS = [
    GemvConfig(block_n=64, block_k=32, num_warps=4, num_stages=2),
    GemvConfig(block_n=64, block_k=64, num_warps=4, num_stages=2),
    GemvConfig(block_n=128, block_k=32, num_warps=4, num_stages=2),
    GemvConfig(block_n=128, block_k=64, num_warps=4, num_stages=2),
    GemvConfig(block_n=128, block_k=128, num_warps=8, num_stages=2),
    GemvConfig(block_n=256, block_k=32, num_warps=8, num_stages=2),
    GemvConfig(block_n=256, block_k=64, num_warps=8, num_stages=2),
]

ACT_CONFIGS = [
    ActConfig(block_n=128, num_warps=4, num_stages=2),
    ActConfig(block_n=256, num_warps=4, num_stages=2),
    ActConfig(block_n=512, num_warps=8, num_stages=2),
]


@triton.jit
def gate_up_unfused_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    packed_ptr,
    k,
    n,
    stride_wg_k,
    stride_wg_n,
    stride_wu_k,
    stride_wu_n,
    stride_pm,
    stride_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
        wg = tl.load(
            wg_ptr + offs_k[:, None] * stride_wg_k + offs_n[None, :] * stride_wg_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float32)
        wu = tl.load(
            wu_ptr + offs_k[:, None] * stride_wu_k + offs_n[None, :] * stride_wu_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float32)
        acc_g += tl.sum(wg * x[:, None], axis=0)
        acc_u += tl.sum(wu * x[:, None], axis=0)

    tl.store(packed_ptr + offs_n * stride_pn, acc_g.to(tl.bfloat16), mask=mask_n)
    tl.store(packed_ptr + stride_pm + offs_n * stride_pn, acc_u.to(tl.bfloat16), mask=mask_n)


@triton.jit
def silu_mul_kernel(
    packed_ptr,
    out_ptr,
    n,
    stride_pm,
    stride_pn,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n

    gate = tl.load(packed_ptr + offs_n * stride_pn, mask=mask_n, other=0).to(tl.float32)
    up = tl.load(packed_ptr + stride_pm + offs_n * stride_pn, mask=mask_n, other=0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    out = silu * up
    tl.store(out_ptr + offs_n * stride_on, out.to(tl.bfloat16), mask=mask_n)


@triton.jit
def gate_up_fused_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    out_ptr,
    k,
    n,
    stride_wg_k,
    stride_wg_n,
    stride_wu_k,
    stride_wu_n,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
        wg = tl.load(
            wg_ptr + offs_k[:, None] * stride_wg_k + offs_n[None, :] * stride_wg_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float32)
        wu = tl.load(
            wu_ptr + offs_k[:, None] * stride_wu_k + offs_n[None, :] * stride_wu_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float32)
        acc_g += tl.sum(wg * x[:, None], axis=0)
        acc_u += tl.sum(wu * x[:, None], axis=0)

    silu = acc_g / (1.0 + tl.exp(-acc_g))
    out = silu * acc_u
    tl.store(out_ptr + offs_n * stride_on, out.to(tl.bfloat16), mask=mask_n)


def median_cuda_ms(fn, *, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    measurements = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return statistics.median(measurements)


def load_weights(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model_file = MODEL_PATH / "model.safetensors"
    with safe_open(str(model_file), framework="pt", device=device) as handle:
        gate = handle.get_tensor("model.layers.0.mlp.gate_proj.weight").contiguous()
        up = handle.get_tensor("model.layers.0.mlp.up_proj.weight").contiguous()
    return gate.t().contiguous(), up.t().contiguous()


def make_reference(x: torch.Tensor, wg_t: torch.Tensor, wu_t: torch.Tensor) -> torch.Tensor:
    gate = torch.matmul(x.float(), wg_t.float())
    up = torch.matmul(x.float(), wu_t.float())
    return (torch.nn.functional.silu(gate) * up).to(DTYPE)


def benchmark_triton(device_id: int, warmup: int, iters: int) -> dict:
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    wg_t, wu_t = load_weights(device)

    torch.manual_seed(0)
    x = torch.randn((HIDDEN_SIZE,), device=device, dtype=DTYPE)
    packed = torch.empty((2, INTERMEDIATE_SIZE), device=device, dtype=DTYPE)
    out_unfused = torch.empty((INTERMEDIATE_SIZE,), device=device, dtype=DTYPE)
    out_fused = torch.empty((INTERMEDIATE_SIZE,), device=device, dtype=DTYPE)
    ref = make_reference(x, wg_t, wu_t)

    results = []
    for gemv_cfg in GEMV_CONFIGS:
        gemv_grid = lambda META, block_n=gemv_cfg.block_n: (triton.cdiv(INTERMEDIATE_SIZE, block_n),)

        def run_gemv_only():
            gate_up_unfused_kernel[gemv_grid](
                x,
                wg_t,
                wu_t,
                packed,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                wg_t.stride(0),
                wg_t.stride(1),
                wu_t.stride(0),
                wu_t.stride(1),
                packed.stride(0),
                packed.stride(1),
                BLOCK_N=gemv_cfg.block_n,
                BLOCK_K=gemv_cfg.block_k,
                num_warps=gemv_cfg.num_warps,
                num_stages=gemv_cfg.num_stages,
            )

        def run_fused():
            gate_up_fused_kernel[gemv_grid](
                x,
                wg_t,
                wu_t,
                out_fused,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                wg_t.stride(0),
                wg_t.stride(1),
                wu_t.stride(0),
                wu_t.stride(1),
                out_fused.stride(0),
                BLOCK_N=gemv_cfg.block_n,
                BLOCK_K=gemv_cfg.block_k,
                num_warps=gemv_cfg.num_warps,
                num_stages=gemv_cfg.num_stages,
            )

        run_gemv_only()
        run_fused()
        torch.testing.assert_close(out_fused, ref, rtol=1e-2, atol=1e-2)
        fused_ms = median_cuda_ms(run_fused, warmup=warmup, iters=iters)

        best_unfused = None
        best_act_cfg = None
        for act_cfg in ACT_CONFIGS:
            act_grid = lambda META, block_n=act_cfg.block_n: (triton.cdiv(INTERMEDIATE_SIZE, block_n),)

            def run_unfused():
                run_gemv_only()
                silu_mul_kernel[act_grid](
                    packed,
                    out_unfused,
                    INTERMEDIATE_SIZE,
                    packed.stride(0),
                    packed.stride(1),
                    out_unfused.stride(0),
                    BLOCK_N=act_cfg.block_n,
                    num_warps=act_cfg.num_warps,
                    num_stages=act_cfg.num_stages,
                )

            run_unfused()
            torch.testing.assert_close(out_unfused, ref, rtol=1e-2, atol=1e-2)
            unfused_ms = median_cuda_ms(run_unfused, warmup=warmup, iters=iters)
            if best_unfused is None or unfused_ms < best_unfused:
                best_unfused = unfused_ms
                best_act_cfg = act_cfg

        results.append(
            {
                "gemv_cfg": gemv_cfg,
                "act_cfg": best_act_cfg,
                "unfused_ms": best_unfused,
                "fused_ms": fused_ms,
                "gain_ms": best_unfused - fused_ms,
                "gain_pct": (best_unfused - fused_ms) / best_unfused * 100.0,
            }
        )

    best = min(results, key=lambda item: item["fused_ms"])
    best_unfused = min(results, key=lambda item: item["unfused_ms"])
    return {
        "all_results": results,
        "best_fused": best,
        "best_unfused": best_unfused,
    }


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def make_engine_config(device_id: int) -> Path:
    config = {
        "model_name": "Qwen2.5",
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": "cuda_sm80",
        },
        "prefill_model_path": str(MODEL_PATH),
        "operator_impl_table_path": str(OP_TABLE_PATH),
    }
    return write_json_file("efm_gate_up_proto_", "engine_config.json", config)


def benchmark_edge_fm(device_id: int, warmup: int, iters: int) -> float | None:
    config_path = make_engine_config(device_id)
    python_paths = [str(p) for p in BUILD_PYTHON_CANDIDATES if p.exists()]
    if not python_paths:
        return None

    code = f"""
import json
import statistics
import sys
from pathlib import Path

for p in reversed({python_paths!r}):
    if p not in sys.path:
        sys.path.insert(0, p)

import edge_fm
import torch

torch.cuda.set_device({device_id})
loader = edge_fm.WeightLoader.instance()
loader.clear_stage(edge_fm.ModelStage.Prefill)
loader.clear_stage(edge_fm.ModelStage.Decode)

fused_linear = edge_fm.FusedGateUpLinearLayer(
    'model.layers.0.mlp',
    {str(config_path)!r},
    {HIDDEN_SIZE},
    {INTERMEDIATE_SIZE},
    {INTERMEDIATE_SIZE},
)
activation = edge_fm.ActivationLayer({str(config_path)!r})
x = torch.randn((1, {HIDDEN_SIZE}), device='cuda:{device_id}', dtype=torch.bfloat16)
packed = torch.empty((1, {2 * INTERMEDIATE_SIZE}), device='cuda:{device_id}', dtype=torch.bfloat16)
out = torch.empty((1, {INTERMEDIATE_SIZE}), device='cuda:{device_id}', dtype=torch.bfloat16)
x_efm = edge_fm.Tensor.from_dlpack(x.contiguous().__dlpack__())
packed_efm = edge_fm.Tensor.from_dlpack(packed.contiguous().__dlpack__())
out_efm = edge_fm.Tensor.from_dlpack(out.contiguous().__dlpack__())

def run():
    fused_linear.forward_fp16_bf16(x_efm, packed_efm, 0, 'Decode')
    activation.forward_silu_and_mul(packed_efm, out_efm, 0, 'Decode')

for _ in range({warmup}):
    run()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
vals = []
for _ in range({iters}):
    start.record()
    run()
    end.record()
    end.synchronize()
    vals.append(start.elapsed_time(end))
print(statistics.median(vals))
"""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-edgefm", action="store_true")
    args = parser.parse_args()

    triton_result = benchmark_triton(args.device, args.warmup, args.iters)
    best_fused = triton_result["best_fused"]
    best_unfused = triton_result["best_unfused"]
    best_gain = max(triton_result["all_results"], key=lambda x: x["gain_ms"])

    print("== Triton gate_up decode prototype ==")
    print(
        f"best_fused: gemv={best_fused['gemv_cfg']} "
        f"fused={best_fused['fused_ms']:.6f} ms"
    )
    print(
        f"best_unfused: gemv={best_unfused['gemv_cfg']} act={best_unfused['act_cfg']} "
        f"unfused={best_unfused['unfused_ms']:.6f} ms"
    )
    print(
        f"best_paired_gain: gemv={best_gain['gemv_cfg']} act={best_gain['act_cfg']} "
        f"gain={best_gain['gain_ms']:.6f} ms ({best_gain['gain_pct']:.2f}%)"
    )

    if not args.skip_edgefm:
        edge_fm_ms = benchmark_edge_fm(args.device, args.warmup, args.iters)
        if edge_fm_ms is not None:
            print(f"edgefm_current_decode_path: {edge_fm_ms:.6f} ms")
            delta = edge_fm_ms - best_fused["fused_ms"]
            print(f"edgefm_minus_best_triton_fused: {delta:.6f} ms")
        else:
            print("edgefm_current_decode_path: unavailable")

    print("== Full sweep ==")
    for item in sorted(triton_result["all_results"], key=lambda x: x["fused_ms"]):
        print(
            f"gemv={item['gemv_cfg']} act={item['act_cfg']} "
            f"unfused={item['unfused_ms']:.6f} ms fused={item['fused_ms']:.6f} ms "
            f"gain={item['gain_ms']:.6f} ms ({item['gain_pct']:.2f}%)"
        )


if __name__ == "__main__":
    main()
