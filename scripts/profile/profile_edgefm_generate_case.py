#!/usr/bin/env python3
"""
Profile a single EdgeFM generate case with configurable prefill/decode lengths.

This script is intended to be used directly or under `nsys` / `ncu` so we can
focus on one model + one workload without dragging in the full benchmark suite.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for build_python in [
    PROJECT_ROOT / "build" / "python",
    PROJECT_ROOT / "build" / "install" / "python",
]:
    build_python_str = str(build_python)
    if build_python.is_dir() and build_python_str not in sys.path:
        sys.path.insert(0, build_python_str)

import edge_fm
from operator_table.utils import (
    resolve_engine_model_name,
    resolve_operator_table_path,
    resolve_target_hw_profile,
)
from temp_paths import make_temp_dir


CUDA_HW_PROFILE = resolve_target_hw_profile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a single EdgeFM generate case")
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument("--model-name", default="", help="Engine model_name field; empty means auto-detect")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefill-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="Hello, how are you today?")
    parser.add_argument(
        "--operator-impl-table",
        default="",
    )
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)
    parser.add_argument(
        "--profile-range",
        action="store_true",
        help="Wrap only the timed runs with cudaProfilerStart/Stop for nsys capture-range=cudaProfilerApi",
    )
    parser.add_argument("--json", action="store_true", help="Print only final JSON metrics")
    return parser.parse_args()


def build_prefill_token_ids(tokenizer, prompt: str, prefill_len: int) -> list[int]:
    if prefill_len <= 0:
        raise ValueError(f"prefill_len must be > 0, got {prefill_len}")
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not token_ids:
        raise ValueError("tokenizer returned no token ids")
    if len(token_ids) >= prefill_len:
        return token_ids[:prefill_len]
    repeat = (prefill_len + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeat)[:prefill_len]


def load_engine_text_config(model_path: Path) -> dict:
    config = json.loads((model_path / "config.json").read_text())
    return config.get("text_config", config)


def make_engine_config(
    *,
    model_path: Path,
    model_name: str,
    device_id: int,
    prefill_len: int,
    decode_len: int,
    operator_impl_table_path: str,
    use_cuda_graph: bool,
) -> Path:
    config = load_engine_text_config(model_path)
    torch_dtype = str(config.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    num_heads = int(config.get("num_attention_heads", 8))
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    max_tokens = prefill_len + decode_len - 1

    payload = {
        "model_name": resolve_engine_model_name(
            model_path,
            explicit_model_name=model_name or None,
            config=config,
        ),
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": CUDA_HW_PROFILE,
            "use_cuda_graph": use_cuda_graph,
        },
        "operator_impl_table_path": str(
            resolve_operator_table_path(
                Path(operator_impl_table_path).resolve() if operator_impl_table_path else None,
                model_path=model_path,
                model_name=model_name or None,
                config=config,
            )
        ),
        "prefill_model_path": str(model_path.resolve()),
        "kvcache": {
            "dtype": kvcache_dtype,
            "attention_type": attention_type,
            "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
        },
        "sampling": {
            "temperature": 0.0,
            "seed": 42,
        },
    }

    temp_dir = make_temp_dir("edgefm_profile_case_")
    path = temp_dir / "engine_config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    torch.cuda.set_device(args.device_id)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    token_ids = build_prefill_token_ids(tokenizer, args.prompt, args.prefill_len)
    engine_config_path = make_engine_config(
        model_path=model_path,
        model_name=args.model_name,
        device_id=args.device_id,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        operator_impl_table_path=args.operator_impl_table,
        use_cuda_graph=args.use_cuda_graph,
    )
    engine = edge_fm.EdgeFM(str(engine_config_path))

    def make_request() -> edge_fm.Request:
        req = edge_fm.Request(0, token_ids)
        req.set_ignore_stop_tokens(True)
        return req

    warmup_generated = []
    for _ in range(args.warmup):
        response = engine.generate(make_request())
        warmup_generated.append(len(response.token_ids()))
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart() if args.profile_range else None
    if cudart is not None:
        err = cudart.cudaProfilerStart()
        if err != 0:
            raise RuntimeError(f"cudaProfilerStart failed with error code {err}")

    times_ms = []
    stage_metrics = []
    generated_counts = []
    try:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            response = engine.generate(make_request())
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            generated_counts.append(len(response.token_ids()))
            stage_metrics.append(engine.last_generate_metrics())
    finally:
        if cudart is not None:
            err = cudart.cudaProfilerStop()
            if err != 0:
                raise RuntimeError(f"cudaProfilerStop failed with error code {err}")

    result = {
        "model_path": str(model_path),
        "device_id": args.device_id,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "use_cuda_graph": args.use_cuda_graph,
        "warmup": args.warmup,
        "runs": args.runs,
        "warmup_generated": warmup_generated,
        "generated_counts": generated_counts,
        "times_ms": times_ms,
        "avg_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "stage_metrics": stage_metrics,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(
        f"[profile] model={model_path.name} device=cuda:{args.device_id} "
        f"prefill={args.prefill_len} decode={args.decode_len} "
        f"cuda_graph={args.use_cuda_graph}"
    )
    print(f"[profile] warmup generated counts: {warmup_generated}")
    print(f"[profile] timed generated counts:  {generated_counts}")
    print(f"[profile] times_ms: {times_ms}")
    print(f"[profile] avg_ms: {result['avg_ms']:.3f}")
    print("[profile] stage metrics:")
    for idx, metrics in enumerate(stage_metrics):
        print(f"  run{idx}: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
