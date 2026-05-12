#!/usr/bin/env python3
"""
Profile a single prepared-multimodal Qwen2.5-VL case for EdgeFM or TRT-Edge-LLM.

This keeps ViT/image encoder work outside the timed/profiled region so the
measured workload matches the current VLM benchmark contract:
  - prepared multimodal
  - image embeddings already prepared
  - do not count ViT

Typical usage under nsys:

  nsys profile -o .tmp_codex/nsys/vlm7b_51232_edgefm_prepared \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    /xs-train-nas/zzm/conda/e2e_zk/bin/python scripts/profile/profile_vlm_prepared_case.py \
      --framework edgefm \
      --model-path examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct \
      --model-name Qwen2.5-VL \
      --model-size 7b \
      --prefill-len 512 \
      --decode-len 32 \
      --use-cuda-graph \
      --profile-range

  nsys profile -o .tmp_codex/nsys/vlm7b_51232_trt_prepared \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    /xs-train-nas/zzm/conda/e2e_zk/bin/python scripts/profile/profile_vlm_prepared_case.py \
      --framework trt \
      --model-path examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct \
      --model-size 7b \
      --prefill-len 512 \
      --decode-len 32 \
      --profile-range
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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
from tests.engine import test_qwen2_generate as qbench

try:
    import edge_fm_trt
except ImportError:
    edge_fm_trt = None


DEFAULT_VLM_IMAGE = PROJECT_ROOT / "tests" / "data" / "candy.JPG"
CUDA_HW_PROFILE = resolve_target_hw_profile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a prepared-multimodal VLM case")
    parser.add_argument("--framework", choices=["edgefm", "trt"], required=True)
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument("--model-size", default="", help="Used for TRT engine auto-discovery, e.g. 0.5b/3b/7b")
    parser.add_argument("--model-name", default="Qwen2.5-VL", help="Engine model_name field for EdgeFM")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefill-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="What animal is on the candy?")
    parser.add_argument("--image-path", default=str(DEFAULT_VLM_IMAGE))
    parser.add_argument("--operator-impl-table", default="", help="Optional explicit EdgeFM operator table")
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)
    parser.add_argument("--engine-dir", default="", help="Optional TRT llm engine dir override")
    parser.add_argument("--multimodal-engine-dir", default="", help="Optional TRT visual engine dir override")
    parser.add_argument("--plugin-path", default="", help="Optional TRT plugin override")
    parser.add_argument(
        "--profile-range",
        action="store_true",
        help="Wrap only timed runs with cudaProfilerStart/Stop for nsys capture-range=cudaProfilerApi",
    )
    parser.add_argument("--json", action="store_true", help="Print only final JSON metrics")
    return parser.parse_args()


def make_edgefm_engine_config(
    *,
    model_path: Path,
    model_name: str,
    device_id: int,
    prefill_len: int,
    decode_len: int,
    operator_impl_table_path: str,
    use_cuda_graph: bool,
) -> Path:
    config = json.loads((model_path / "config.json").read_text())
    text_config = config.get("text_config", config)
    torch_dtype = str(text_config.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    num_heads = int(text_config.get("num_attention_heads", 8))
    num_kv_heads = int(text_config.get("num_key_value_heads", num_heads))
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    max_tokens = prefill_len + decode_len - 1

    resolved_model_name = resolve_engine_model_name(
        model_path,
        explicit_model_name=model_name or None,
        config=text_config,
    )
    operator_table = resolve_operator_table_path(
        Path(operator_impl_table_path).resolve() if operator_impl_table_path else None,
        model_path=model_path,
        model_name=resolved_model_name,
        config=text_config,
    )

    payload = {
        "model_name": resolved_model_name,
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": CUDA_HW_PROFILE,
            "use_cuda_graph": use_cuda_graph,
        },
        "operator_impl_table_path": str(operator_table),
        "prefill_model_path": str(model_path.resolve()),
        "kvcache": {
            "dtype": kvcache_dtype,
            "attention_type": attention_type,
            "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
        },
        "sampling": {
            "max_new_tokens": decode_len,
            "temperature": 0.0,
            "seed": 42,
        },
    }
    out_dir = make_temp_dir("edgefm_vlm_profile_case_")
    path = out_dir / "engine_config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def resolve_trt_paths(
    *,
    model_size: str,
    requested_engine_dir: str,
    requested_multimodal_engine_dir: str,
    requested_plugin_path: str,
    required_prefill_len: int,
) -> tuple[Path, Path, Path]:
    if edge_fm_trt is None:
        raise RuntimeError("edge_fm_trt not available. Build with BUILD_TRT_EDGELLM_PYBIND=ON.")
    if not model_size:
        raise ValueError("--model-size is required for TRT auto-discovery")

    engine_dir, multimodal_engine_dir = qbench._resolve_trt_vlm_engine_dirs(
        required_prefill_len,
        requested_engine_dir=Path(requested_engine_dir).resolve() if requested_engine_dir else None,
        requested_multimodal_engine_dir=(
            Path(requested_multimodal_engine_dir).resolve() if requested_multimodal_engine_dir else None
        ),
        model_size=model_size,
    )

    if requested_plugin_path:
        plugin_path = Path(requested_plugin_path).resolve()
    else:
        plugin_path = (
            PROJECT_ROOT / "third_party" / "TensorRT-Edge-LLM" / "build" / "libNvInfer_edgellm_plugin.so"
        ).resolve()
    if not plugin_path.exists():
        raise FileNotFoundError(f"TRT plugin not found at {plugin_path}")
    return engine_dir, multimodal_engine_dir, plugin_path


def prepare_vlm_inputs(model_path: Path, prefill_len: int, image_path: str, prompt: str) -> dict:
    model, processor = qbench._load_transformers_vlm_model(str(model_path))
    try:
        prepared = qbench._prepare_vlm_bench_case(
            model,
            processor,
            prefill_len=prefill_len,
            image_path=image_path,
            prompt=prompt,
        )
    finally:
        del model
        del processor
        torch.cuda.empty_cache()
    return prepared


def maybe_start_profiler(enabled: bool):
    if not enabled:
        return None
    cudart = torch.cuda.cudart()
    err = cudart.cudaProfilerStart()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStart failed with error code {err}")
    return cudart


def maybe_stop_profiler(cudart) -> None:
    if cudart is None:
        return
    err = cudart.cudaProfilerStop()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStop failed with error code {err}")


def profile_edgefm(args: argparse.Namespace, prepared_inputs: dict) -> dict:
    engine_config_path = make_edgefm_engine_config(
        model_path=Path(args.model_path).resolve(),
        model_name=args.model_name,
        device_id=args.device_id,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        operator_impl_table_path=args.operator_impl_table,
        use_cuda_graph=args.use_cuda_graph,
    )
    engine = edge_fm.EdgeFM(str(engine_config_path))

    make_request = qbench._make_vl_request_factory(
        prepared_inputs["edgefm_token_ids"],
        prepared_inputs["image_embeddings"],
        int(prepared_inputs["embed_token_id"]),
        prepared_inputs["position_ids"],
        ignore_stop_tokens=True,
    )
    prepared_request = make_request()

    warmup_generated = []
    for _ in range(args.warmup):
        response = engine.generate(prepared_request)
        warmup_generated.append(len(response.token_ids()))
    torch.cuda.synchronize()

    cudart = maybe_start_profiler(args.profile_range)
    times_ms = []
    generated_counts = []
    stage_metrics = []
    try:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            response = engine.generate(prepared_request)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            generated_counts.append(len(response.token_ids()))
            stage_metrics.append(engine.last_generate_metrics())
    finally:
        maybe_stop_profiler(cudart)

    return {
        "framework": "edgefm",
        "model_path": str(Path(args.model_path).resolve()),
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
        "prepared_inputs_meta": {
            "prompt": prepared_inputs["prompt"],
            "image_path": prepared_inputs["image_path"],
            "prepared_prefill_tokens": prepared_inputs["prefill_tokens"],
        },
        "request_mode": "reuse_prepared_request",
    }


def profile_trt(args: argparse.Namespace, prepared_inputs: dict) -> dict:
    engine_dir, multimodal_engine_dir, plugin_path = resolve_trt_paths(
        model_size=args.model_size,
        requested_engine_dir=args.engine_dir,
        requested_multimodal_engine_dir=args.multimodal_engine_dir,
        requested_plugin_path=args.plugin_path,
        required_prefill_len=args.prefill_len,
    )
    os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)
    runtime = edge_fm_trt.TrtEdgeLlmRuntime(str(engine_dir), str(multimodal_engine_dir), args.device_id)

    trt_token_ids = qbench._build_trt_vlm_token_ids(prepared_inputs, args.model_path)
    image_grid_thw = prepared_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("prepared_inputs.image_grid_thw is required for TRT prepared multimodal profiling")

    runtime.prepare_multimodal_from_token_ids(
        trt_token_ids,
        prepared_inputs["image_embeddings"],
        image_grid_thw.tolist(),
    )

    warmup_generated = []
    for _ in range(args.warmup):
        output_ids, _ = runtime.generate_from_prepared_multimodal(
            args.decode_len, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=True
        )
        warmup_generated.append(len(output_ids[0]) if output_ids else 0)

    cudart = maybe_start_profiler(args.profile_range)
    times_ms = []
    generated_counts = []
    stage_metrics = []
    try:
        for _ in range(args.runs):
            t0 = time.perf_counter()
            output_ids, _ = runtime.generate_from_prepared_multimodal(
                args.decode_len, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=True
            )
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            generated_counts.append(len(output_ids[0]) if output_ids else 0)
            stage_metrics.append(runtime.last_generate_metrics())
    finally:
        maybe_stop_profiler(cudart)

    return {
        "framework": "trt",
        "model_path": str(Path(args.model_path).resolve()),
        "device_id": args.device_id,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "warmup": args.warmup,
        "runs": args.runs,
        "warmup_generated": warmup_generated,
        "generated_counts": generated_counts,
        "times_ms": times_ms,
        "avg_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "stage_metrics": stage_metrics,
        "engine_dir": str(engine_dir),
        "multimodal_engine_dir": str(multimodal_engine_dir),
        "plugin_path": str(plugin_path),
        "prepared_inputs_meta": {
            "prompt": prepared_inputs["prompt"],
            "image_path": prepared_inputs["image_path"],
            "prepared_prefill_tokens": prepared_inputs["prefill_tokens"],
        },
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device_id)

    prepared_inputs = prepare_vlm_inputs(
        Path(args.model_path).resolve(),
        args.prefill_len,
        args.image_path,
        args.prompt,
    )
    if prepared_inputs["prefill_tokens"] != args.prefill_len:
        raise RuntimeError(
            f"Prepared prefill token count mismatch: requested={args.prefill_len}, "
            f"prepared={prepared_inputs['prefill_tokens']}"
        )

    if args.framework == "edgefm":
        result = profile_edgefm(args, prepared_inputs)
    else:
        result = profile_trt(args, prepared_inputs)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(
        f"[profile] framework={result['framework']} model={Path(args.model_path).name} "
        f"device=cuda:{args.device_id} prefill={args.prefill_len} decode={args.decode_len}"
    )
    print(f"[profile] warmup generated counts: {result['warmup_generated']}")
    print(f"[profile] timed generated counts:  {result['generated_counts']}")
    print(f"[profile] times_ms: {result['times_ms']}")
    print(f"[profile] avg_ms: {result['avg_ms']:.3f}")
    print("[profile] stage metrics:")
    for idx, metrics in enumerate(result["stage_metrics"]):
        print(f"  run{idx}: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
