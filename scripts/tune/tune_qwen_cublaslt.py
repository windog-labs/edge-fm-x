#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(REPO_ROOT)

import edge_fm
from operator_table.utils import (
    resolve_engine_model_name,
    resolve_operator_model_name,
    resolve_operator_table_path,
    resolve_target_hw_profile,
)
from temp_paths import make_temp_dir


LAYER_ROLE_BY_KIND = {
    "fused_qkv": "fused_qkv",
    "attention_output": "attention_output",
    "mlp_down": "mlp_down",
    "fused_gate_up": "fused_gate_up",
    "lm_head": "lm_head",
}
DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
EDGE_FM_DTYPE_ID_BY_TORCH_DTYPE = {
    torch.float16: 1,
    torch.bfloat16: 2,
}


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = make_temp_dir(prefix)
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def load_operator_impl_table(path: Path) -> dict:
    return json.loads(path.read_text())


def write_operator_impl_table(records: list[dict]) -> Path:
    return write_json_file(
        "efm_qwen_cublaslt_tune_",
        "operator_impl_table.json",
        {
            "schema": "edgefm_operator_impl_table_v1",
            "records": records,
        },
    )


def make_engine_config(
    model_path: Path,
    device_id: int,
    operator_impl_table_path: Path,
    *,
    hw_profile: str,
) -> Path:
    return write_json_file(
        "efm_qwen_cublaslt_cfg_",
        "engine_config.json",
        {
            "model_name": resolve_engine_model_name(model_path),
            "runtime": {
                "device": "cuda",
                "device_id": device_id,
                "hw_profile": hw_profile,
            },
            "prefill_model_path": str(model_path),
            "operator_impl_table_path": str(operator_impl_table_path),
        },
    )


def _edge_fm_dtype(torch_dtype: torch.dtype) -> edge_fm.DType:
    if torch_dtype == torch.bfloat16:
        return edge_fm.DType.BFloat16
    if torch_dtype == torch.float16:
        return edge_fm.DType.Float16
    if torch_dtype == torch.float32:
        return edge_fm.DType.Float32
    if torch_dtype == torch.int32:
        return edge_fm.DType.Int32
    if torch_dtype == torch.int64:
        return edge_fm.DType.Int64
    if torch_dtype == torch.int8:
        return edge_fm.DType.Int8
    if torch_dtype == torch.uint8:
        return edge_fm.DType.UInt8
    raise TypeError(f"Unsupported torch dtype for edge_fm.Tensor view: {torch_dtype}")


def _edge_fm_device(torch_tensor: torch.Tensor) -> tuple[edge_fm.Device, int]:
    if torch_tensor.device.type == "cuda":
        return edge_fm.Device.GPU, torch_tensor.device.index or 0
    if torch_tensor.device.type == "cpu":
        return edge_fm.Device.CPU, 0
    raise TypeError(f"Unsupported torch device for edge_fm.Tensor view: {torch_tensor.device}")


def tensor_to_edge_fm_tensor(torch_tensor: torch.Tensor) -> edge_fm.Tensor:
    if not torch_tensor.is_contiguous():
        raise ValueError("tensor_to_edge_fm_tensor expects a contiguous torch.Tensor")
    device, device_id = _edge_fm_device(torch_tensor)
    return edge_fm.Tensor(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        _edge_fm_dtype(torch_tensor.dtype),
        device,
        device_id,
        False,
    )


def reset_weight_loader() -> None:
    loader = edge_fm.WeightLoader.instance()
    loader.clear_stage(edge_fm.ModelStage.Prefill)
    loader.clear_stage(edge_fm.ModelStage.Decode)


def bench_cuda_ms(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def median_cuda_ms(fn, *, warmup: int, iters: int) -> float:
    return statistics.median(bench_cuda_ms(fn, warmup=warmup, iters=iters))


def load_model_dims(model_path: Path) -> dict:
    cfg = json.loads((model_path / "config.json").read_text())
    text_cfg = cfg.get("text_config", {})

    def _resolve_int(key: str) -> int:
        value = cfg.get(key, text_cfg.get(key))
        if value is None:
            raise KeyError(key)
        return int(value)

    hidden = _resolve_int("hidden_size")
    intermediate = _resolve_int("intermediate_size")
    head_dim = hidden // _resolve_int("num_attention_heads")
    kv = _resolve_int("num_key_value_heads") * head_dim
    return {
        "hidden": hidden,
        "intermediate": intermediate,
        "kv": kv,
        "vocab": _resolve_int("vocab_size"),
    }


def shape_sig_for(kind: str, *, m: int, dims: dict, dtype_id: int = 2) -> str:
    if kind == "fused_qkv":
        in_features = dims["hidden"]
        out_features = dims["hidden"] + 2 * dims["kv"]
    elif kind == "attention_output":
        in_features = dims["hidden"]
        out_features = dims["hidden"]
    elif kind == "mlp_down":
        in_features = dims["intermediate"]
        out_features = dims["hidden"]
    elif kind == "fused_gate_up":
        in_features = dims["hidden"]
        out_features = 2 * dims["intermediate"]
    elif kind == "lm_head":
        in_features = dims["hidden"]
        out_features = dims["vocab"]
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    return (
        f"m={m}|input={dtype_id}|weight={dtype_id}|output={dtype_id}|"
        f"in_features={in_features}|out_features={out_features}"
    )


def build_tuned_records(
    base_records: list[dict],
    *,
    operator_model_name: str,
    hw_profile: str,
    kind: str,
    stage: str,
    m: int,
    dims: dict,
    dtype_id: int = 2,
    impl_params: dict | None,
) -> list[dict]:
    layer_role = LAYER_ROLE_BY_KIND[kind]
    shape_sig = shape_sig_for(kind, m=m, dims=dims, dtype_id=dtype_id)

    kept = []
    for record in base_records:
        if (
            record.get("model_name") == operator_model_name
            and record.get("hw_profile") == hw_profile
            and record.get("op_kind") == "linear"
            and record.get("layer_role") == layer_role
            and record.get("stage") == stage
            and record.get("shape_sig") == shape_sig
        ):
            continue
        kept.append(record)

    if impl_params is not None:
        kept.append(
            {
                "model_name": operator_model_name,
                "hw_profile": hw_profile,
                "op_kind": "linear",
                "layer_role": layer_role,
                "op_name": "",
                "stage": stage,
                "shape_sig": shape_sig,
                "impl_id": "cublasLt",
                "impl_params": impl_params,
            }
        )

    return kept


def make_layer(kind: str, engine_config_path: Path, dims: dict):
    if kind == "fused_qkv":
        return edge_fm.FusedQKVLinearLayer(
            "model.layers.0.self_attn",
            str(engine_config_path),
            dims["hidden"],
            dims["hidden"],
            dims["kv"],
            dims["kv"],
        )
    if kind == "attention_output":
        return edge_fm.LinearLayer(
            "model.layers.0.self_attn.o_proj",
            str(engine_config_path),
            dims["hidden"],
            dims["hidden"],
        )
    if kind == "mlp_down":
        return edge_fm.LinearLayer(
            "model.layers.0.mlp.down_proj",
            str(engine_config_path),
            dims["intermediate"],
            dims["hidden"],
        )
    if kind == "fused_gate_up":
        return edge_fm.FusedGateUpLinearLayer(
            "model.layers.0.mlp",
            str(engine_config_path),
            dims["hidden"],
            dims["intermediate"],
            dims["intermediate"],
        )
    if kind == "lm_head":
        return edge_fm.LMHeadLinearLayer(
            str(engine_config_path),
            dims["hidden"],
            dims["vocab"],
            "lm_head",
        )
    raise ValueError(f"Unsupported layer kind: {kind}")


def input_output_shapes(kind: str, *, m: int, dims: dict) -> tuple[tuple[int, int], int]:
    if kind == "mlp_down":
        return (m, dims["intermediate"]), dims["hidden"]
    if kind == "fused_qkv":
        return (m, dims["hidden"]), dims["hidden"] + 2 * dims["kv"]
    if kind == "attention_output":
        return (m, dims["hidden"]), dims["hidden"]
    if kind == "fused_gate_up":
        return (m, dims["hidden"]), 2 * dims["intermediate"]
    if kind == "lm_head":
        return (m, dims["hidden"]), dims["vocab"]
    raise ValueError(f"Unsupported layer kind: {kind}")


def debug_info(layer, *, stage: str, m: int) -> dict:
    stage_name = "Decode" if stage == "decode" else "Prefill"
    return json.loads(layer.debug_cached_impl_info(stage_name, m))


def enumerate_explicit_candidates(
    *,
    model_path: Path,
    dims: dict,
    base_records: list[dict],
    operator_model_name: str,
    hw_profile: str,
    kind: str,
    stage: str,
    m: int,
    torch_dtype: torch.dtype,
    dtype_id: int,
    device_id: int,
    max_algo_ids: int,
    top_k: int,
) -> list[dict]:
    table_path = write_operator_impl_table(
        build_tuned_records(
            base_records,
            operator_model_name=operator_model_name,
            hw_profile=hw_profile,
            kind=kind,
            stage=stage,
            m=m,
            dims=dims,
            dtype_id=dtype_id,
            impl_params=None,
        )
    )
    reset_weight_loader()
    engine_config_path = make_engine_config(model_path, device_id, table_path, hw_profile=hw_profile)
    layer = make_layer(kind, engine_config_path, dims)

    in_shape, out_dim = input_output_shapes(kind, m=m, dims=dims)
    x = torch.randn(*in_shape, device=f"cuda:{device_id}", dtype=torch_dtype)
    y = torch.empty(in_shape[0], out_dim, device=f"cuda:{device_id}", dtype=torch_dtype)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)
    stage_name = "Decode" if stage == "decode" else "Prefill"

    return json.loads(
        layer.debug_enumerate_cublaslt_candidates(
            x_efm,
            y_efm,
            stage_name,
            max_algo_ids,
            top_k,
        )
    )["candidates"]


def candidate_label(impl_params: dict | None) -> str:
    if impl_params is None:
        return "baseline"
    if set(impl_params.keys()) == {"algo_index"}:
        return f"algo_{impl_params['algo_index']}"
    parts = []
    for key in [
        "algo_id",
        "tile_id",
        "stages_id",
        "splitk_num",
        "reduction_scheme",
        "cta_swizzling",
        "custom_option",
    ]:
        if key in impl_params:
            parts.append(f"{key}={impl_params[key]}")
    return "explicit:" + ",".join(parts[:4]) if parts else "explicit"


def benchmark_candidate(
    *,
    model_path: Path,
    dims: dict,
    base_records: list[dict],
    operator_model_name: str,
    hw_profile: str,
    kind: str,
    stage: str,
    m: int,
    impl_params: dict | None,
    torch_dtype: torch.dtype,
    dtype_id: int,
    device_id: int,
    warmup: int,
    iters: int,
) -> dict:
    table_path = write_operator_impl_table(
        build_tuned_records(
            base_records,
            operator_model_name=operator_model_name,
            hw_profile=hw_profile,
            kind=kind,
            stage=stage,
            m=m,
            dims=dims,
            dtype_id=dtype_id,
            impl_params=impl_params,
        )
    )
    reset_weight_loader()
    engine_config_path = make_engine_config(model_path, device_id, table_path, hw_profile=hw_profile)
    layer = make_layer(kind, engine_config_path, dims)

    in_shape, out_dim = input_output_shapes(kind, m=m, dims=dims)
    x = torch.randn(*in_shape, device=f"cuda:{device_id}", dtype=torch_dtype)
    y = torch.empty(in_shape[0], out_dim, device=f"cuda:{device_id}", dtype=torch_dtype)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)
    stage_name = "Decode" if stage == "decode" else "Prefill"

    layer.forward_fp16_bf16(x_efm, y_efm, 0, stage_name)
    torch.cuda.synchronize()
    info = debug_info(layer, stage=stage, m=m)
    median_ms = median_cuda_ms(
        lambda: layer.forward_fp16_bf16(x_efm, y_efm, 0, stage_name),
        warmup=warmup,
        iters=iters,
    )
    return {
        "candidate": candidate_label(impl_params),
        "algo_index": None if impl_params is None else impl_params.get("algo_index"),
        "impl_params": impl_params,
        "median_ms": median_ms,
        "checksum_abs_mean": float(y.float().abs().mean().item()),
        "debug": {
            "selected_impl_id": info.get("selected_impl_id"),
            "selected_impl_params": info.get("selected_impl_params"),
            "selected_algo_config": info.get("selected_algo_config"),
            "best_algo_index": info.get("best_algo_index"),
            "heuristic_candidate_count": info.get("heuristic_candidate_count"),
            "workspace_bytes": info.get("workspace_bytes"),
            "waves_count": info.get("waves_count"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Qwen cublasLt tactics for a single linear shape")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--layer-kind",
        required=True,
        choices=sorted(LAYER_ROLE_BY_KIND.keys()),
    )
    parser.add_argument("--stage", required=True, choices=["decode", "prefill"])
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument(
        "--operator-table",
        default="",
    )
    parser.add_argument(
        "--candidate-indices",
        default="auto",
        help="Comma-separated algo indices to test, or 'auto' to use baseline heuristic_candidate_count",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["heuristic", "explicit", "all"],
        default="heuristic",
        help="Search heuristic algo_index candidates, explicit low-level configs, or both",
    )
    parser.add_argument(
        "--explicit-max-algo-ids",
        type=int,
        default=64,
        help="For candidate-mode explicit/all: max algo_id values to enumerate",
    )
    parser.add_argument(
        "--explicit-top-k",
        type=int,
        default=32,
        help="For candidate-mode explicit/all: max explicit configs to benchmark after enumeration",
    )
    parser.add_argument("--hw-profile", default="", help="Target runtime hw_profile, defaults to current platform")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_BY_NAME.keys()),
        default="bf16",
        help="Activation/weight/output dtype to tune. Defaults to bf16 for existing Qwen HF checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    model_path = Path(args.model_path).resolve()
    hw_profile = resolve_target_hw_profile(args.hw_profile)
    operator_table_path = resolve_operator_table_path(
        Path(args.operator_table).resolve() if args.operator_table else None,
        model_path=model_path,
    )
    dims = load_model_dims(model_path)
    base_records = load_operator_impl_table(operator_table_path)["records"]
    operator_model_name = resolve_operator_model_name(model_path=model_path)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    dtype_id = EDGE_FM_DTYPE_ID_BY_TORCH_DTYPE[torch_dtype]

    reset_weight_loader()
    baseline = benchmark_candidate(
        model_path=model_path,
        dims=dims,
        base_records=base_records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        kind=args.layer_kind,
        stage=args.stage,
        m=args.m,
        impl_params=None,
        torch_dtype=torch_dtype,
        dtype_id=dtype_id,
        device_id=args.device_id,
        warmup=args.warmup,
        iters=args.iters,
    )

    candidates = [baseline]
    candidate_impl_params = []

    if args.candidate_mode in {"heuristic", "all"}:
        if args.candidate_indices == "auto":
            count = int(baseline["debug"]["heuristic_candidate_count"])
            candidate_indices = list(range(count))
        else:
            candidate_indices = [
                int(item.strip()) for item in args.candidate_indices.split(",") if item.strip()
            ]
        candidate_impl_params.extend({"algo_index": algo_index} for algo_index in candidate_indices)

    if args.candidate_mode in {"explicit", "all"}:
        explicit_candidates = enumerate_explicit_candidates(
            model_path=model_path,
            dims=dims,
            base_records=base_records,
            operator_model_name=operator_model_name,
            hw_profile=hw_profile,
            kind=args.layer_kind,
            stage=args.stage,
            m=args.m,
            torch_dtype=torch_dtype,
            dtype_id=dtype_id,
            device_id=args.device_id,
            max_algo_ids=args.explicit_max_algo_ids,
            top_k=args.explicit_top_k,
        )
        candidate_impl_params.extend(candidate["config"] for candidate in explicit_candidates)

    seen_impl_params = set()
    for impl_params in candidate_impl_params:
        key = json.dumps(impl_params, sort_keys=True)
        if key in seen_impl_params:
            continue
        seen_impl_params.add(key)
        candidates.append(
            benchmark_candidate(
                model_path=model_path,
                dims=dims,
                base_records=base_records,
                operator_model_name=operator_model_name,
                hw_profile=hw_profile,
                kind=args.layer_kind,
                stage=args.stage,
                m=args.m,
                impl_params=impl_params,
                torch_dtype=torch_dtype,
                dtype_id=dtype_id,
                device_id=args.device_id,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    best = min(candidates, key=lambda item: item["median_ms"])
    report = {
        "model_path": str(model_path),
        "layer_kind": args.layer_kind,
        "stage": args.stage,
        "m": args.m,
        "dtype": args.dtype,
        "shape_sig": shape_sig_for(args.layer_kind, m=args.m, dims=dims, dtype_id=dtype_id),
        "operator_model_name": operator_model_name,
        "hw_profile": hw_profile,
        "candidates": candidates,
        "best": best,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
