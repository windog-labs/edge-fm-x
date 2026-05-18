#!/usr/bin/env python3
"""
Profile a single EdgeFM generate case with configurable prefill/decode lengths.

This script is intended to be used directly or under `nsys` / `ncu` so we can
focus on one model + one workload without dragging in the full benchmark suite.
"""

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
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for build_python in [
    PROJECT_ROOT / "build" / "python",
    PROJECT_ROOT / "build" / "install" / "python",
]:
    build_python_str = str(build_python)
    if build_python.is_dir() and build_python_str not in sys.path:
        sys.path.insert(0, build_python_str)

from edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(PROJECT_ROOT)

import edge_fm
from operator_table.utils import (
    resolve_engine_model_name,
    resolve_operator_model_name,
    resolve_operator_table_path,
    resolve_target_hw_profile,
)
from temp_paths import make_temp_dir


CUDA_HW_PROFILE = resolve_target_hw_profile()
EDGEFM_REQUIRED_STAGE_METRIC_KEYS = {
    "prefill_ms",
    "decode_ms",
    "decode_step_avg_ms",
    "tokens_per_second",
    "decode_tokens_per_second",
    "executed_generated_tokens_total",
    "returned_generated_tokens_total",
    "cuda_graph_enabled",
    "lm_head_top1_enabled",
    "lm_head_top1_decode_steps",
}
EDGEFM_PLUGIN_ATTENTION_ENV_KEY = "EDGE_FM_PREFILL_TRT_FMHA_PLUGIN"
EDGEFM_PLUGIN_ATTENTION_ALLOW_BF16_CAST_ENV_KEY = (
    "EDGE_FM_PREFILL_TRT_FMHA_PLUGIN_ALLOW_BF16_FP16_CAST"
)


def _mean_metric(stage_metrics: list[dict], key: str) -> float:
    if not stage_metrics:
        return 0.0
    return sum(float(metrics.get(key, 0.0)) for metrics in stage_metrics) / len(stage_metrics)


def _pct(part: float, total: float) -> float:
    return (part * 100.0 / total) if total > 0.0 else 0.0


def build_owner_a_decode_breakdown(stage_metrics: list[dict]) -> dict:
    """Summarize decode timing and whether the default-off top1 path was active."""
    decode_ms = _mean_metric(stage_metrics, "decode_ms")
    decode_model_ms = _mean_metric(stage_metrics, "decode_model_ms")
    decode_sampler_ms = _mean_metric(stage_metrics, "decode_sampler_ms")
    decode_finalize_ms = _mean_metric(stage_metrics, "decode_finalize_ms")
    decode_graph_replay_ms = _mean_metric(stage_metrics, "decode_graph_replay_ms")
    lm_head_top1_enabled = _mean_metric(stage_metrics, "lm_head_top1_enabled") > 0.5
    lm_head_top1_decode_steps = _mean_metric(stage_metrics, "lm_head_top1_decode_steps")

    return {
        "decode_ms": decode_ms,
        "decode_model_including_lm_head_ms": decode_model_ms,
        "decode_sampler_ms": decode_sampler_ms,
        "decode_finalize_ms": decode_finalize_ms,
        "decode_graph_replay_ms": decode_graph_replay_ms,
        "lm_head_top1_enabled": lm_head_top1_enabled,
        "lm_head_top1_decode_steps": lm_head_top1_decode_steps,
        "model_pct": _pct(decode_model_ms, decode_ms),
        "sampler_pct": _pct(decode_sampler_ms, decode_ms),
        "finalize_pct": _pct(decode_finalize_ms, decode_ms),
        "graph_replay_pct": _pct(decode_graph_replay_ms, decode_ms),
        "full_logits_default": not lm_head_top1_enabled,
        "lm_head_top1_status": "enabled_experimental" if lm_head_top1_enabled else "available_default_off",
        "decision_gate": "Keep full logits as default unless lm_head_top1 shows >=1% end-to-end CUDA graph gain plus token alignment.",
    }


def format_owner_a_decode_breakdown(breakdown: dict) -> str:
    return (
        "Owner A decode breakdown: "
        f"decode model+lm_head={float(breakdown.get('decode_model_including_lm_head_ms', 0.0)):.3f} ms "
        f"({float(breakdown.get('model_pct', 0.0)):.1f}%), "
        f"sampler={float(breakdown.get('decode_sampler_ms', 0.0)):.3f} ms "
        f"({float(breakdown.get('sampler_pct', 0.0)):.1f}%), "
        f"finalize={float(breakdown.get('decode_finalize_ms', 0.0)):.3f} ms "
        f"({float(breakdown.get('finalize_pct', 0.0)):.1f}%), "
        f"graph_replay={float(breakdown.get('decode_graph_replay_ms', 0.0)):.3f} ms "
        f"({float(breakdown.get('graph_replay_pct', 0.0)):.1f}%). "
        f"lm_head_top1={breakdown.get('lm_head_top1_status', 'unknown')} "
        f"steps={float(breakdown.get('lm_head_top1_decode_steps', 0.0)):.1f}; "
        f"full_logits_default={bool(breakdown.get('full_logits_default', True))}."
    )


def cuda_memory_snapshot(device_id: int) -> dict:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        return {"available": False, "error": str(exc)}
    used_bytes = total_bytes - free_bytes
    bytes_per_mb = 1024.0 * 1024.0
    return {
        "available": True,
        "free_mb": free_bytes / bytes_per_mb,
        "used_mb": used_bytes / bytes_per_mb,
        "total_mb": total_bytes / bytes_per_mb,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_loaded_edgefm_runtime() -> dict:
    """Catch LD_LIBRARY_PATH mistakes that make this script load the wrong libedge_fm.so."""
    build_dir_raw = os.environ.get("EDGE_FM_BUILD_DIR", "").strip()
    if not build_dir_raw:
        return {"checked": False, "reason": "EDGE_FM_BUILD_DIR not set"}

    expected_build_dir = Path(build_dir_raw).expanduser().resolve()
    loaded_libs: list[str] = []
    maps_path = Path("/proc/self/maps")
    if maps_path.exists():
        for line in maps_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            candidate = Path(parts[-1])
            if candidate.name == "libedge_fm.so":
                resolved = str(candidate.resolve())
                if resolved not in loaded_libs:
                    loaded_libs.append(resolved)

    mismatched = [
        path for path in loaded_libs if not _is_relative_to(Path(path), expected_build_dir)
    ]
    result = {
        "checked": True,
        "expected_build_dir": str(expected_build_dir),
        "loaded_libedge_fm": loaded_libs,
        "mismatched_libedge_fm": mismatched,
    }
    if mismatched:
        message = (
            "Loaded libedge_fm.so does not match EDGE_FM_BUILD_DIR. "
            f"expected under {expected_build_dir}, got {mismatched}. "
            "Fix LD_LIBRARY_PATH so the selected build's lib/ or install/lib comes first."
        )
        print(f"[profile][warning] {message}", file=sys.stderr)
    return result


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
        "--lm-head-top1",
        action="store_true",
        default=False,
        help="Enable the default-off experimental greedy decode lm_head_top1 path.",
    )
    parser.add_argument(
        "--edgefm-mode",
        choices=("as-is", "native", "plugin-op"),
        default="as-is",
        help=(
            "Select EdgeFM runtime mode. as-is preserves the caller environment; "
            "native disables default-off plugin env gates; plugin-op enables the "
            "default-off direct TRT FMHA plugin attention op."
        ),
    )
    parser.add_argument(
        "--plugin-op-allow-bf16-fp16-cast",
        action="store_true",
        default=False,
        help=(
            "Diagnostic-only: allow plugin-op attention to cast BF16 Q/K/V through the TRT "
            "FP16 ContextFMHA runner. This path has not passed token alignment and is never "
            "enabled by --edgefm-mode plugin-op alone."
        ),
    )
    parser.add_argument(
        "--profile-range",
        action="store_true",
        help="Wrap only the timed runs with cudaProfilerStart/Stop for nsys capture-range=cudaProfilerApi",
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


def load_engine_text_config(model_path: Path) -> dict:
    config = json.loads((model_path / "config.json").read_text())
    return config.get("text_config", config)


def attention_shape_sig_from_config(config: dict) -> str:
    num_heads = int(config.get("num_attention_heads", 8))
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    hidden_size = int(config.get("hidden_size", num_heads * 128))
    if num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError(
            f"Cannot derive attention shape signature from num_attention_heads={num_heads}, "
            f"hidden_size={hidden_size}"
        )
    head_dim = hidden_size // num_heads
    return f"num_qo_heads={num_heads}|num_kv_heads={num_kv_heads}|head_dim={head_dim}"


def write_attention_plugin_operator_table(
    *,
    base_table_path: Path,
    model_path: Path,
    model_name: str,
    config: dict,
) -> Path:
    payload = json.loads(base_table_path.read_text())
    records = list(payload.get("records", []))
    records.append(
        {
            "model_name": resolve_operator_model_name(
                model_path=model_path,
                model_name=model_name or None,
                config=config,
            ),
            "hw_profile": CUDA_HW_PROFILE,
            "op_kind": "attention",
            "layer_role": "",
            "op_name": "attention",
            "stage": "prefill",
            "shape_sig": attention_shape_sig_from_config(config),
            "impl_id": "trt_context_fmha_plugin_attention",
            "impl_params": {},
        }
    )
    payload["records"] = records
    payload.setdefault("metadata", {})
    payload["metadata"]["edgefm_mode_overlay"] = "plugin-op"
    payload["metadata"]["source_operator_table_path"] = str(base_table_path)

    temp_dir = make_temp_dir("edgefm_plugin_op_table_")
    output_path = temp_dir / "operator_impl_table_plugin_op.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def configure_edgefm_mode(
    *,
    mode: str,
    model_path: Path,
    model_name: str,
    config: dict,
    operator_impl_table_path: str,
    plugin_op_allow_bf16_fp16_cast: bool,
) -> dict:
    base_operator_table = resolve_operator_table_path(
        Path(operator_impl_table_path).resolve() if operator_impl_table_path else None,
        model_path=model_path,
        model_name=model_name or None,
        config=config,
    )
    result = {
        "mode": mode,
        "base_operator_impl_table": str(base_operator_table),
        "effective_operator_impl_table": str(base_operator_table),
        "cleared_env": {},
        "set_env": {},
    }

    if mode == "as-is":
        return result

    for name in (
        EDGEFM_PLUGIN_ATTENTION_ENV_KEY,
        EDGEFM_PLUGIN_ATTENTION_ALLOW_BF16_CAST_ENV_KEY,
    ):
        if name in os.environ:
            result["cleared_env"][name] = os.environ.pop(name)

    if mode == "native":
        return result

    os.environ[EDGEFM_PLUGIN_ATTENTION_ENV_KEY] = "1"
    result["set_env"][EDGEFM_PLUGIN_ATTENTION_ENV_KEY] = "1"
    if plugin_op_allow_bf16_fp16_cast:
        os.environ[EDGEFM_PLUGIN_ATTENTION_ALLOW_BF16_CAST_ENV_KEY] = "1"
        result["set_env"][EDGEFM_PLUGIN_ATTENTION_ALLOW_BF16_CAST_ENV_KEY] = "1"
    plugin_operator_table = write_attention_plugin_operator_table(
        base_table_path=base_operator_table,
        model_path=model_path,
        model_name=model_name,
        config=config,
    )
    result["effective_operator_impl_table"] = str(plugin_operator_table)
    return result


def add_edgefm_json_contract(result: dict, expected_generated_tokens: int) -> dict:
    generated_counts = result.get("generated_counts", [])
    bad_counts = [count for count in generated_counts if count != expected_generated_tokens]
    if bad_counts:
        raise RuntimeError(
            f"Generated token count mismatch: expected each run to return {expected_generated_tokens}, "
            f"got {generated_counts}"
        )

    stage_metrics = result.get("stage_metrics", [])
    if not stage_metrics:
        raise RuntimeError("EdgeFM profile result has no stage_metrics")
    for idx, metrics in enumerate(stage_metrics):
        missing = sorted(EDGEFM_REQUIRED_STAGE_METRIC_KEYS - set(metrics.keys()))
        if missing:
            raise RuntimeError(f"EdgeFM profile run{idx} missing required stage metric keys: {missing}")

    result["json_contract"] = "edgefm.generate_profile.v1"
    for key in sorted(EDGEFM_REQUIRED_STAGE_METRIC_KEYS):
        result[key] = sum(float(metrics[key]) for metrics in stage_metrics) / len(stage_metrics)
    result["owner_a_decode_breakdown"] = build_owner_a_decode_breakdown(stage_metrics)
    return result


def make_engine_config(
    *,
    model_path: Path,
    model_name: str,
    device_id: int,
    prefill_len: int,
    decode_len: int,
    operator_impl_table_path: str,
    use_cuda_graph: bool,
    lm_head_top1: bool,
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
            "lm_head_top1": {"enabled": lm_head_top1},
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
            "max_new_tokens": decode_len,
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
    model_config = load_engine_text_config(model_path)
    edgefm_mode = configure_edgefm_mode(
        mode=args.edgefm_mode,
        model_path=model_path,
        model_name=args.model_name,
        config=model_config,
        operator_impl_table_path=args.operator_impl_table,
        plugin_op_allow_bf16_fp16_cast=args.plugin_op_allow_bf16_fp16_cast,
    )
    runtime_library_check = validate_loaded_edgefm_runtime()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    token_ids = build_prefill_token_ids(tokenizer, args.prompt, args.prefill_len)
    engine_config_path = make_engine_config(
        model_path=model_path,
        model_name=args.model_name,
        device_id=args.device_id,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        operator_impl_table_path=edgefm_mode["effective_operator_impl_table"],
        use_cuda_graph=args.use_cuda_graph,
        lm_head_top1=args.lm_head_top1,
    )
    engine = edge_fm.EdgeFM(str(engine_config_path))
    torch.cuda.synchronize()
    memory_after_engine_init = cuda_memory_snapshot(args.device_id)

    def make_request() -> edge_fm.Request:
        req = edge_fm.Request(0, token_ids)
        req.set_ignore_stop_tokens(True)
        return req

    warmup_generated = []
    for _ in range(args.warmup):
        response = engine.generate(make_request())
        warmup_generated.append(len(response.token_ids()))
    torch.cuda.synchronize()
    memory_after_warmup = cuda_memory_snapshot(args.device_id)

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
    memory_after_runs = cuda_memory_snapshot(args.device_id)

    result = {
        "model_path": str(model_path),
        "device_id": args.device_id,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "use_cuda_graph": args.use_cuda_graph,
        "lm_head_top1": args.lm_head_top1,
        "edgefm_mode": edgefm_mode,
        "warmup": args.warmup,
        "runs": args.runs,
        "warmup_generated": warmup_generated,
        "generated_counts": generated_counts,
        "times_ms": times_ms,
        "avg_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "stage_metrics": stage_metrics,
        "memory_snapshots": {
            "after_engine_init": memory_after_engine_init,
            "after_warmup": memory_after_warmup,
            "after_runs": memory_after_runs,
        },
        "runtime_library_check": runtime_library_check,
    }
    add_edgefm_json_contract(result, args.decode_len)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(
        f"[profile] model={model_path.name} device=cuda:{args.device_id} "
        f"prefill={args.prefill_len} decode={args.decode_len} "
        f"cuda_graph={args.use_cuda_graph} lm_head_top1={args.lm_head_top1} "
        f"edgefm_mode={args.edgefm_mode}"
    )
    print(f"[profile] effective operator table: {edgefm_mode['effective_operator_impl_table']}")
    print(f"[profile] warmup generated counts: {warmup_generated}")
    print(f"[profile] timed generated counts:  {generated_counts}")
    print(f"[profile] times_ms: {times_ms}")
    print(f"[profile] avg_ms: {result['avg_ms']:.3f}")
    print("[profile] stage metrics:")
    for idx, metrics in enumerate(stage_metrics):
        print(f"  run{idx}: {json.dumps(metrics, sort_keys=True)}")
    print("[profile] owner-a breakdown:")
    print(f"  {format_owner_a_decode_breakdown(result['owner_a_decode_breakdown'])}")


if __name__ == "__main__":
    main()
