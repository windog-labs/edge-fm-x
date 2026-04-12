"""
TRT-Edge-LLM nsys profiling script.

Usage:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,cublas \
      -o nsys_reports/trt_edgellm python tests/scripts/profile_trt_edgellm.py

Env vars:
  EDGE_FM_DEVICE_ID   – GPU device id (default 1)
  PROFILE_PREFILL_LEN – prefill length to profile (default 512)
  PROFILE_NUM_STEPS   – decode steps to profile (default 32)
  TRT_EDGELLM_ENGINE_DIR  – optional TRT engine dir override
  TRT_EDGELLM_PLUGIN_PATH – optional plugin path override
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(project_root)

import edge_fm_trt

DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
PREFILL_LEN = int(os.environ.get("PROFILE_PREFILL_LEN", "512"))
NUM_STEPS = int(os.environ.get("PROFILE_NUM_STEPS", "32"))
WARMUP_RUNS = 3

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


def get_engine_dir() -> Path:
    raw = os.environ.get("TRT_EDGELLM_ENGINE_DIR", "").strip()
    if raw:
        return Path(raw).resolve()

    workspace_dir = (project_root / "tests" / "data" / "trt_edgellm_workspace" / "qwen2.5-1.5b").resolve()
    candidates = []
    for name in ["engines_mxil2048", "engines"]:
        engine_dir = workspace_dir / name
        if engine_dir.exists() and (engine_dir / "llm.engine").exists():
            config = json.loads((engine_dir / "config.json").read_text())
            max_input_len = int(config.get("builder_config", {}).get("max_input_len", 0) or 0)
            if max_input_len >= PREFILL_LEN:
                candidates.append((max_input_len, engine_dir.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"No TRT-Edge-LLM engine under {workspace_dir} supports PROFILE_PREFILL_LEN={PREFILL_LEN}"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def get_plugin_path() -> Path:
    raw = os.environ.get("TRT_EDGELLM_PLUGIN_PATH", "").strip()
    if raw:
        return Path(raw).resolve()
    return (project_root / "third_party" / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so").resolve()


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

    engine_dir = get_engine_dir()
    plugin_path = get_plugin_path()
    assert engine_dir.exists() and (engine_dir / "llm.engine").exists(), f"Invalid engine dir: {engine_dir}"
    assert plugin_path.exists(), f"Invalid plugin path: {plugin_path}"
    os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)

    token_ids = build_prefill_token_ids(model_path)
    print(f"Model: {model_path}")
    print(f"TRT engine: {engine_dir}")
    print(f"Device: cuda:{DEVICE_ID}, Prefill: {len(token_ids)}, Steps: {NUM_STEPS}")

    runtime = edge_fm_trt.TrtEdgeLlmRuntime(str(engine_dir), "", DEVICE_ID)

    print(f"Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        runtime.generate_from_token_ids(
            token_ids,
            NUM_STEPS,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            ignore_stop_tokens=True,
        )
    torch.cuda.synchronize()
    print("Warmup done.")

    print("Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()

    output_ids, _ = runtime.generate_from_token_ids(
        token_ids,
        NUM_STEPS,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        ignore_stop_tokens=True,
    )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()

    generated = output_ids[0] if output_ids else []
    print(f"Generated {len(generated)} tokens: {generated[:10]}...")
    print("Profiling complete.")


if __name__ == "__main__":
    main()
