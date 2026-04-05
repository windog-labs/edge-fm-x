"""
EdgeFM nsys profiling script.

Usage:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,cublas \
      -o nsys_reports/edgefm python tests/scripts/profile_edgefm.py

Env vars:
  EDGE_FM_DEVICE_ID      – GPU device id (default 1)
  PROFILE_PREFILL_LEN    – prefill length to profile (default 512)
  PROFILE_NUM_STEPS      – decode steps to profile (default 20)
  EDGE_FM_USE_CUDA_GRAPH – whether to profile cuda-graph mode (default 1)
"""

import json, os, sys, tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
for _p in [project_root / "build" / "python", project_root / "build" / "install" / "python"]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import edge_fm

DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
PREFILL_LEN = int(os.environ.get("PROFILE_PREFILL_LEN", "512"))
NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", "20"))
WARMUP_RUNS = 3
USE_CUDA_GRAPH = os.environ.get("EDGE_FM_USE_CUDA_GRAPH", "1").strip() not in {"0", "false", "False"}

PROMPT = "Hello, how are you today?"
BASE_SEQ_LEN = 6

def find_model_path():
    candidates = [
        os.environ.get("EDGE_FM_QWEN_MODEL_PATH"),
        str(project_root / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
        str(project_root / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
    ]
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "config.json").exists():
            return str(Path(p).resolve())
    return None


def create_engine_config(model_path: str) -> str:
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    torch_dtype = str(text_cfg.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    nh = text_cfg.get("num_attention_heads", 8)
    nkv = text_cfg.get("num_key_value_heads", nh)
    att = "gqa" if nkv < nh else "mha"
    max_tok = PREFILL_LEN + NUM_STEPS - 1
    d = tempfile.mkdtemp()
    p = Path(d) / "engine_config.json"
    runtime = {"device": "cuda", "device_id": DEVICE_ID, "hw_profile": "cuda_sm80"}
    if USE_CUDA_GRAPH:
        runtime["use_cuda_graph"] = True
    with open(p, "w") as f:
        json.dump({
            "model_name": "Qwen2.5",
            "runtime": runtime,
            "operator_impl_table_path": str((project_root / "examples" / "config" / "operator_impl_table.json").resolve()),
            "prefill_model_path": model_path,
            "kvcache": {
                "dtype": kvcache_dtype,
                "attention_type": att,
                "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tok}],
            },
            "sampling": {"temperature": 0.0, "seed": 42},
        }, f, indent=2)
    return str(p)


def build_prefill_token_ids(model_path: str) -> list[int]:
    dump_dir = project_root / "tests" / "data" / "decode_dump"
    manifest_path = dump_dir / "manifest.json"
    token_ids_path = dump_dir / "token_ids.npy"
    if manifest_path.exists() and token_ids_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_model_path = manifest.get("model_path", "")
        if manifest_model_path and Path(manifest_model_path).exists():
            model_path = manifest_model_path
        token_ids = np.load(token_ids_path).flatten().tolist()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)[:BASE_SEQ_LEN]
        if len(token_ids) < BASE_SEQ_LEN:
            token_ids += [0] * (BASE_SEQ_LEN - len(token_ids))

    if len(token_ids) < PREFILL_LEN:
        repeat = (PREFILL_LEN + len(token_ids) - 1) // len(token_ids)
        token_ids = (token_ids * repeat)[:PREFILL_LEN]
    else:
        token_ids = token_ids[:PREFILL_LEN]
    return token_ids


def main():
    model_path = find_model_path()
    assert model_path, "Model not found"
    print(f"Model: {model_path}")
    print(f"Device: cuda:{DEVICE_ID}, Prefill: {PREFILL_LEN}, Steps: {NUM_STEPS}, cuda_graph={USE_CUDA_GRAPH}")

    token_ids = build_prefill_token_ids(model_path)
    print(f"Prefill tokens: {len(token_ids)}")

    cfg_path = create_engine_config(model_path)
    engine = edge_fm.EdgeFM(cfg_path)

    def make_request():
        req = edge_fm.Request(0, token_ids)
        req.set_ignore_stop_tokens(True)
        return req

    print(f"Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        engine.generate(make_request())
    torch.cuda.synchronize()
    print("Warmup done.")

    print("Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()

    resp = engine.generate(make_request())
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()

    out_tokens = resp.token_ids()
    print(f"Generated {len(out_tokens)} tokens: {out_tokens[:10]}...")
    print("Profiling complete.")


if __name__ == "__main__":
    main()
