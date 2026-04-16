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


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = make_temp_dir(prefix)
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def load_operator_impl_table(path: Path) -> dict:
    return json.loads(path.read_text())


def write_operator_impl_table(records: list[dict]) -> Path:
    return write_json_file(
        "efm_qwen_attn_prefill_tune_",
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
        "efm_qwen_attn_prefill_cfg_",
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


def load_model_attention_dims(model_path: Path) -> dict:
    config = json.loads((model_path / "config.json").read_text())
    if "text_config" in config and isinstance(config["text_config"], dict):
        config = config["text_config"]
    num_qo_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    hidden_size = int(config["hidden_size"])
    head_dim = hidden_size // num_qo_heads
    return {
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def attention_shape_sig(dims: dict) -> str:
    return (
        f"num_qo_heads={dims['num_qo_heads']}|"
        f"num_kv_heads={dims['num_kv_heads']}|"
        f"head_dim={dims['head_dim']}"
    )


def build_tuned_records(
    base_records: list[dict],
    *,
    operator_model_name: str,
    hw_profile: str,
    dims: dict,
    impl_params: dict,
) -> list[dict]:
    shape_sig = attention_shape_sig(dims)
    kept = []
    for record in base_records:
        if (
            record.get("model_name") == operator_model_name
            and record.get("hw_profile") == hw_profile
            and record.get("op_kind") == "attention"
            and record.get("stage") == "prefill"
            and record.get("shape_sig") == shape_sig
        ):
            continue
        kept.append(record)

    if impl_params:
        kept.append(
            {
                "model_name": operator_model_name,
                "hw_profile": hw_profile,
                "op_kind": "attention",
                "layer_role": "",
                "op_name": "",
                "stage": "prefill",
                "shape_sig": shape_sig,
                "impl_id": "flashinfer_attention",
                "impl_params": impl_params,
            }
        )

    return kept


def benchmark_candidate(
    *,
    model_path: Path,
    operator_model_name: str,
    dims: dict,
    base_records: list[dict],
    impl_params: dict,
    seq_lens: list[int],
    device_id: int,
    warmup: int,
    iters: int,
    hw_profile: str,
) -> dict:
    table_path = write_operator_impl_table(
        build_tuned_records(
            base_records,
            operator_model_name=operator_model_name,
            hw_profile=hw_profile,
            dims=dims,
            impl_params=impl_params,
        )
    )
    engine_config_path = make_engine_config(model_path, device_id, table_path, hw_profile=hw_profile)
    layer = edge_fm.AttentionLayer(str(engine_config_path))

    results = []
    for seq_len in seq_lens:
        q = torch.randn(
            (seq_len, dims["num_qo_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            (seq_len, dims["num_kv_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            (seq_len, dims["num_kv_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        o = torch.empty_like(q)

        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(o)

        def run() -> None:
            layer.forward_prefill(q_efm, k_efm, v_efm, o_efm, True, 0)

        run()
        torch.cuda.synchronize()
        median_ms = median_cuda_ms(run, warmup=warmup, iters=iters)
        results.append(
            {
                "seq_len": seq_len,
                "median_ms": median_ms,
                "checksum_abs_mean": float(o.float().abs().mean().item()),
            }
        )

    total_ms = sum(item["median_ms"] for item in results)
    avg_ms = total_ms / len(results)
    return {
        "candidate_label": candidate_label(impl_params),
        "impl_params": impl_params,
        "shape_sig": attention_shape_sig(dims),
        "seq_results": results,
        "total_median_ms": total_ms,
        "avg_median_ms": avg_ms,
    }


def parse_u32_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def candidate_label(impl_params: dict) -> str:
    if not impl_params:
        return "baseline"
    if "prefill_cta_tile_q" in impl_params:
        return f"global_cta_tile_q={impl_params['prefill_cta_tile_q']}"
    if (
        "prefill_short_qo_len_threshold" in impl_params
        and "prefill_short_cta_tile_q" in impl_params
        and "prefill_long_cta_tile_q" in impl_params
    ):
        return (
            "split_cta_tile_q"
            f"(threshold={impl_params['prefill_short_qo_len_threshold']},"
            f"short={impl_params['prefill_short_cta_tile_q']},"
            f"long={impl_params['prefill_long_cta_tile_q']})"
        )
    return json.dumps(impl_params, sort_keys=True)


def unique_impl_param_candidates(candidates: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for impl_params in candidates:
        key = json.dumps(impl_params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(impl_params)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Qwen prefill attention impl_params for a single attention shape"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seq-lens", type=parse_u32_list, default=[512, 1024, 2048])
    parser.add_argument(
        "--operator-table",
        default="",
    )
    parser.add_argument(
        "--cta-tile-q-list",
        type=parse_u32_list,
        default=[16, 64, 128],
        help="Comma separated candidate prefill_cta_tile_q values",
    )
    parser.add_argument(
        "--skip-global-cta-sweep",
        action="store_true",
        help="Skip the global prefill_cta_tile_q sweep and only run baseline/split candidates",
    )
    parser.add_argument(
        "--include-short-long-sweep",
        action="store_true",
        help="Also sweep prefill_short_qo_len_threshold + short/long cta_tile_q combinations",
    )
    parser.add_argument(
        "--short-threshold-list",
        type=parse_u32_list,
        default=[512, 1024],
        help="Comma separated candidate prefill_short_qo_len_threshold values",
    )
    parser.add_argument(
        "--short-cta-tile-q-list",
        type=parse_u32_list,
        default=[64, 128],
        help="Comma separated candidate prefill_short_cta_tile_q values",
    )
    parser.add_argument(
        "--long-cta-tile-q-list",
        type=parse_u32_list,
        default=[64, 128],
        help="Comma separated candidate prefill_long_cta_tile_q values",
    )
    parser.add_argument("--hw-profile", default="", help="Target runtime hw_profile, defaults to current platform")
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
    dims = load_model_attention_dims(model_path)
    base_records = load_operator_impl_table(operator_table_path)["records"]
    operator_model_name = resolve_operator_model_name(model_path=model_path)

    impl_param_candidates = [{}]
    if not args.skip_global_cta_sweep:
        for cta_tile_q in args.cta_tile_q_list:
            impl_param_candidates.append({"prefill_cta_tile_q": cta_tile_q})

    if args.include_short_long_sweep:
        for short_threshold in args.short_threshold_list:
            for short_cta_tile_q in args.short_cta_tile_q_list:
                for long_cta_tile_q in args.long_cta_tile_q_list:
                    impl_param_candidates.append(
                        {
                            "prefill_short_qo_len_threshold": short_threshold,
                            "prefill_short_cta_tile_q": short_cta_tile_q,
                            "prefill_long_cta_tile_q": long_cta_tile_q,
                        }
                    )

    candidates = []
    for impl_params in unique_impl_param_candidates(impl_param_candidates):
        candidates.append(
            benchmark_candidate(
                model_path=model_path,
                operator_model_name=operator_model_name,
                dims=dims,
                base_records=base_records,
                impl_params=impl_params,
                seq_lens=args.seq_lens,
                device_id=args.device_id,
                warmup=args.warmup,
                iters=args.iters,
                hw_profile=hw_profile,
            )
        )

    best = min(candidates, key=lambda item: item["total_median_ms"])
    report = {
        "model_path": str(model_path),
        "shape_sig": attention_shape_sig(dims),
        "hw_profile": hw_profile,
        "seq_lens": args.seq_lens,
        "candidates": candidates,
        "best": best,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
