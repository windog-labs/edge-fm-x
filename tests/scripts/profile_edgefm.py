"""
EdgeFM nsys profiling script.

Usage:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,cublas \
      -o nsys_reports/edgefm python tests/scripts/profile_edgefm.py

Env vars:
  EDGE_FM_DEVICE_ID  – GPU device id (default 1)
  PROFILE_NUM_STEPS  – decode steps to profile (default 20)
"""

import json, os, sys, tempfile
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
for _p in [project_root / "build" / "install" / "python", project_root / "build" / "python"]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import edge_fm

DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", "20"))
WARMUP_RUNS = 3

PROMPT = "Hello, how are you today?"
SEQ_LEN = 6

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
    nh = text_cfg.get("num_attention_heads", 8)
    nkv = text_cfg.get("num_key_value_heads", nh)
    att = "gqa" if nkv < nh else "mha"
    max_tok = SEQ_LEN + NUM_STEPS + 5
    d = tempfile.mkdtemp()
    p = Path(d) / "engine_config.json"
    with open(p, "w") as f:
        json.dump({
            "model_name": "Qwen2.5",
            "runtime": {"device": "cuda", "device_id": DEVICE_ID},
            "prefill_model_path": model_path,
            "kvcache": {
                "dtype": "fp16",
                "attention_type": att,
                "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tok}],
            },
            "sampling": {"temperature": 0.0, "seed": 42},
        }, f, indent=2)
    return str(p)


def main():
    model_path = find_model_path()
    assert model_path, "Model not found"
    print(f"Model: {model_path}")
    print(f"Device: cuda:{DEVICE_ID}, Steps: {NUM_STEPS}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)[:SEQ_LEN]
    if len(token_ids) < SEQ_LEN:
        token_ids += [0] * (SEQ_LEN - len(token_ids))
    print(f"Prefill tokens: {len(token_ids)}")

    cfg_path = create_engine_config(model_path)
    engine = edge_fm.EdgeFM(cfg_path)

    def make_request():
        return edge_fm.Request(0, token_ids)

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
