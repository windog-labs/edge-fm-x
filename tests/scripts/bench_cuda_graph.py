#!/usr/bin/env python3
"""
Benchmark: EdgeFM decode 性能对比 (use_cuda_graph=False vs True)

Usage:
  EDGE_FM_DEVICE_ID=0 python tests/scripts/bench_cuda_graph.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
for _p in [project_root / "build" / "install" / "python", project_root / "build" / "python"]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import edge_fm

DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "0"))
WARMUP = 5
RUNS = 50
PROMPT = "Hello, how are you today?"
SEQ_LEN = 6
NUM_STEPS = 100


def find_model():
    candidates = [
        os.environ.get("EDGE_FM_QWEN_MODEL_PATH"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
    ]
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "config.json").exists():
            return str(Path(p).resolve())
    return None


def create_config(model_path: str, use_cuda_graph: bool, prefix_token_ids: list | None = None) -> str:
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    nh = text_cfg.get("num_attention_heads", 8)
    nkv = text_cfg.get("num_key_value_heads", nh)
    att = "gqa" if nkv < nh else "mha"
    max_tok = SEQ_LEN + NUM_STEPS + 2
    d = tempfile.mkdtemp()
    p = Path(d) / "engine_config.json"
    runtime = {"device": "cuda", "device_id": DEVICE_ID}
    if use_cuda_graph:
        runtime["use_cuda_graph"] = True
    prefix = prefix_token_ids if prefix_token_ids is not None else []
    with open(p, "w") as f:
        json.dump({
            "model_name": "Qwen2.5",
            "runtime": runtime,
            "prefill_model_path": model_path,
            "kvcache": {
                "dtype": "fp16",
                "attention_type": att,
                "requests": [{"request_id": 0, "prefix_token_ids": prefix, "max_tokens": max_tok}],
            },
            "sampling": {"temperature": 0.0, "seed": 42},
        }, f, indent=2)
    return str(p)


def bench(engine, request_fn, warmup, runs):
    for _ in range(warmup):
        engine.generate(request_fn())
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(request_fn())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    return avg * 1000, NUM_STEPS / avg


def main():
    model_path = find_model()
    if not model_path:
        print("Model not found. Set EDGE_FM_QWEN_MODEL_PATH or place model in examples/")
        sys.exit(1)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)[:SEQ_LEN]
    if len(token_ids) < SEQ_LEN:
        token_ids += [0] * (SEQ_LEN - len(token_ids))

    def make_req():
        return edge_fm.Request(0, token_ids)

    print(f"Model: {model_path}")
    print(f"Device: cuda:{DEVICE_ID}, prefill={len(token_ids)}, decode_steps={NUM_STEPS}")
    print(f"Warmup={WARMUP}, runs={RUNS}\n")

    # Baseline (no CUDA graph)
    cfg_baseline = create_config(model_path, use_cuda_graph=False)
    engine_baseline = edge_fm.EdgeFM(cfg_baseline)
    lat_baseline, tps_baseline = bench(engine_baseline, make_req, WARMUP, RUNS)

    # CUDA graph: use prefix so warmup runs decode dry-run (required for capture)
    prefix = token_ids[: min(4, len(token_ids))]
    cfg_graph = create_config(model_path, use_cuda_graph=True, prefix_token_ids=prefix)
    engine_graph = edge_fm.EdgeFM(cfg_graph)
    lat_graph, tps_graph = bench(engine_graph, make_req, WARMUP, RUNS)

    print("=" * 60)
    print("  Decode Performance: Baseline vs CUDA Graph")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Baseline':>12} {'CUDA Graph':>12} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Total latency (ms)':<30} {lat_baseline:>12.2f} {lat_graph:>12.2f} {lat_baseline/lat_graph:>9.2f}x")
    print(f"  {'Decode throughput (tok/s)':<30} {tps_baseline:>12.1f} {tps_graph:>12.1f} {tps_graph/tps_baseline:>9.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
