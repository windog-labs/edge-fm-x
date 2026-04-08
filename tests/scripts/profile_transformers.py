"""
Transformers nsys profiling script (KV-cached greedy decode).

Usage:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,cublas \
      -o nsys_reports/transformers python tests/scripts/profile_transformers.py

Env vars:
  EDGE_FM_DEVICE_ID  – GPU device id (default 1)
  PROFILE_PREFILL_LEN – prefill length to profile (default 512)
  PROFILE_NUM_STEPS  – decode steps to profile (default 20)
"""

import json, os, sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
PREFILL_LEN = int(os.environ.get("PROFILE_PREFILL_LEN", "512"))
NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", "20"))
WARMUP_RUNS = 3
CUDA_DEVICE = f"cuda:{DEVICE_ID}"

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


def run_once(model, input_ids, device):
    with torch.no_grad():
        out = model(input_ids, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    tok = out.logits[0, -1].argmax().item()
    decode_input = torch.tensor([[tok]], dtype=torch.long, device=device)
    for _ in range(NUM_STEPS - 1):
        with torch.no_grad():
            out = model(decode_input, past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        tok = out.logits[0, -1].argmax().item()
        decode_input = torch.tensor([[tok]], dtype=torch.long, device=device)
    return tok


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
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    model_path = find_model_path()
    assert model_path, "Model not found"
    print(f"Model: {model_path}")
    print(f"Device: {CUDA_DEVICE}, Prefill: {PREFILL_LEN}, Steps: {NUM_STEPS}")

    config = AutoConfig.from_pretrained(model_path)
    dtype_str = str(getattr(config, "torch_dtype", "float16")).lower()
    model_dtype = torch.bfloat16 if "bfloat" in dtype_str or "bf16" in dtype_str else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=model_dtype, device_map=CUDA_DEVICE)
    model.eval()

    token_ids = build_prefill_token_ids(model_path)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=CUDA_DEVICE)
    print(f"Prefill tokens: {len(token_ids)}")

    print(f"Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        run_once(model, input_ids, CUDA_DEVICE)
    torch.cuda.synchronize()
    print("Warmup done.")

    print("Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()

    run_once(model, input_ids, CUDA_DEVICE)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()

    print("Profiling complete.")


if __name__ == "__main__":
    main()
