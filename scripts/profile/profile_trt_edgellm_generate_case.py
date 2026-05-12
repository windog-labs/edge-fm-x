#!/usr/bin/env python3
"""Profile one TRT-Edge-LLM generate case under nsys/ncu.

This mirrors the token-id based TRT path used by tests/engine/test_qwen2_generate.py
so the profiled prefill input is directly comparable to EdgeFM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for build_python in [
    PROJECT_ROOT / "build-3060" / "install" / "python",
    PROJECT_ROOT / "build" / "install" / "python",
    PROJECT_ROOT / "build" / "python",
]:
    build_python_str = str(build_python)
    if build_python.is_dir() and build_python_str not in sys.path:
        sys.path.insert(0, build_python_str)


DEFAULT_PROMPT = "Hello, how are you today?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a single TRT-Edge-LLM generate case")
    parser.add_argument("--model-path", required=True, help="HF model directory used for tokenization")
    parser.add_argument("--engine-dir", required=True, help="TRT-Edge-LLM engine directory containing llm.engine")
    parser.add_argument("--plugin-path", required=True, help="TensorRT-Edge-LLM plugin library")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefill-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--profile-range",
        action="store_true",
        help="Wrap only timed runs with cudaProfilerStart/Stop for nsys capture-range=cudaProfilerApi",
    )
    parser.add_argument("--json", action="store_true", help="Print final JSON metrics")
    parser.add_argument("--output-json", default="", help="Optional path to write final JSON metrics")
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


def load_engine_config(engine_dir: Path) -> dict:
    config_path = engine_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"TRT-Edge-LLM config not found at {config_path}")
    return json.loads(config_path.read_text())


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = Path(args.model_path).expanduser().resolve()
    engine_dir = Path(args.engine_dir).expanduser().resolve()
    plugin_path = Path(args.plugin_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    if not (engine_dir / "llm.engine").exists():
        raise FileNotFoundError(f"TRT-Edge-LLM engine not found at {engine_dir / 'llm.engine'}")
    if not plugin_path.exists():
        raise FileNotFoundError(f"TRT-Edge-LLM plugin not found at {plugin_path}")
    max_input_len = int(load_engine_config(engine_dir).get("builder_config", {}).get("max_input_len", 0) or 0)
    if max_input_len and args.prefill_len > max_input_len:
        raise RuntimeError(
            f"TRT-Edge-LLM engine at {engine_dir} supports max_input_len={max_input_len}, "
            f"but requested prefill_len={args.prefill_len}"
        )
    return model_path, engine_dir, plugin_path


def generated_count(output_ids) -> int:
    if not output_ids:
        return 0
    return len(output_ids[0])


def main() -> None:
    args = parse_args()
    model_path, engine_dir, plugin_path = validate_inputs(args)
    os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)

    import edge_fm_trt

    torch.cuda.set_device(args.device_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    token_ids = build_prefill_token_ids(tokenizer, args.prompt, args.prefill_len)
    runtime = edge_fm_trt.TrtEdgeLlmRuntime(str(engine_dir), "", args.device_id)

    warmup_counts: list[int] = []
    for _ in range(args.warmup):
        output_ids, _ = runtime.generate_from_token_ids(
            token_ids,
            args.decode_len,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            ignore_stop_tokens=True,
        )
        warmup_counts.append(generated_count(output_ids))
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart() if args.profile_range else None
    if cudart is not None:
        err = cudart.cudaProfilerStart()
        if err != 0:
            raise RuntimeError(f"cudaProfilerStart failed with error code {err}")

    times_ms: list[float] = []
    generated_counts: list[int] = []
    stage_metrics: list[dict[str, float]] = []
    try:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output_ids, _ = runtime.generate_from_token_ids(
                token_ids,
                args.decode_len,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                ignore_stop_tokens=True,
            )
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            generated_counts.append(generated_count(output_ids))
            stage_metrics.append({k: float(v) for k, v in runtime.last_generate_metrics().items()})
    finally:
        if cudart is not None:
            err = cudart.cudaProfilerStop()
            if err != 0:
                raise RuntimeError(f"cudaProfilerStop failed with error code {err}")

    result = {
        "framework": "trt-edge-llm",
        "model_path": str(model_path),
        "engine_dir": str(engine_dir),
        "plugin_path": str(plugin_path),
        "device_id": args.device_id,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "warmup": args.warmup,
        "runs": args.runs,
        "warmup_generated": warmup_counts,
        "generated_counts": generated_counts,
        "times_ms": times_ms,
        "avg_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "stage_metrics": stage_metrics,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(
        f"[profile-trt] model={model_path.name} device=cuda:{args.device_id} "
        f"prefill={args.prefill_len} decode={args.decode_len}"
    )
    print(f"[profile-trt] warmup generated counts: {warmup_counts}")
    print(f"[profile-trt] timed generated counts:  {generated_counts}")
    print(f"[profile-trt] times_ms: {times_ms}")
    print(f"[profile-trt] avg_ms: {result['avg_ms']:.3f}")
    print("[profile-trt] stage metrics:")
    for idx, metrics in enumerate(stage_metrics):
        print(f"  run{idx}: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
