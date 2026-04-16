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
        "efm_qwen_attn_tune_",
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
        "efm_qwen_attn_cfg_",
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
            and record.get("stage") == "decode"
            and record.get("shape_sig") == shape_sig
        ):
            continue
        kept.append(record)

    kept.append(
        {
            "model_name": operator_model_name,
            "hw_profile": hw_profile,
            "op_kind": "attention",
            "layer_role": "",
            "op_name": "",
            "stage": "decode",
            "shape_sig": shape_sig,
            "impl_id": "flashinfer_attention_decode_sm80_tuned",
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
    kv_lens: list[int],
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
    for kv_len in kv_lens:
        q = torch.randn(
            (1, dims["num_qo_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            (kv_len, dims["num_kv_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            (kv_len, dims["num_kv_heads"], dims["head_dim"]),
            device=f"cuda:{device_id}",
            dtype=torch.bfloat16,
        )
        o = torch.empty_like(q)
        d_kv_len = torch.tensor([kv_len], device=f"cuda:{device_id}", dtype=torch.int32)

        q_efm = tensor_to_edge_fm_tensor(q)
        k_efm = tensor_to_edge_fm_tensor(k)
        v_efm = tensor_to_edge_fm_tensor(v)
        o_efm = tensor_to_edge_fm_tensor(o)
        d_kv_len_efm = tensor_to_edge_fm_tensor(d_kv_len)

        def run() -> None:
            layer.forward_decode(q_efm, k_efm, v_efm, o_efm, 0, kv_len, d_kv_len_efm)

        run()
        torch.cuda.synchronize()
        median_ms = median_cuda_ms(run, warmup=warmup, iters=iters)
        results.append(
            {
                "kv_len": kv_len,
                "median_ms": median_ms,
                "checksum_abs_mean": float(o.float().abs().mean().item()),
            }
        )

    total_ms = sum(item["median_ms"] for item in results)
    avg_ms = total_ms / len(results)
    return {
        "impl_params": impl_params,
        "shape_sig": attention_shape_sig(dims),
        "kv_results": results,
        "total_median_ms": total_ms,
        "avg_median_ms": avg_ms,
    }


def parse_chunk_candidates(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("chunk-candidates must contain exactly 4 comma-separated integers")
    return values


def parse_kv_lens(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("kv-lens must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Qwen decode attention impl_params for a single attention shape")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--kv-lens", type=parse_kv_lens, default=[512, 1024, 2048])
    parser.add_argument(
        "--operator-table",
        default="",
    )
    parser.add_argument("--short-seq-bdz", type=int, default=3)
    parser.add_argument("--long-seq-bdz", type=int, default=4)
    parser.add_argument("--long-seq-threshold", type=int, default=1536)
    parser.add_argument("--no-split-kv-threshold", type=int, default=384)
    parser.add_argument("--min-chunk-size", type=int, default=128)
    parser.add_argument("--chunk-alignment", type=int, default=128)
    parser.add_argument("--chunk-candidates", type=parse_chunk_candidates, default=[128, 256, 512, 1024])
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
    operator_model_name = resolve_operator_model_name(model_path=model_path)
    base_records = load_operator_impl_table(operator_table_path)["records"]

    report = benchmark_candidate(
        model_path=model_path,
        operator_model_name=operator_model_name,
        dims=dims,
        base_records=base_records,
        impl_params={
            "short_seq_bdz": args.short_seq_bdz,
            "long_seq_bdz": args.long_seq_bdz,
            "long_seq_threshold": args.long_seq_threshold,
            "no_split_kv_threshold": args.no_split_kv_threshold,
            "min_chunk_size": args.min_chunk_size,
            "chunk_alignment": args.chunk_alignment,
            "chunk_candidates": args.chunk_candidates,
        },
        kv_lens=args.kv_lens,
        device_id=args.device_id,
        warmup=args.warmup,
        iters=args.iters,
        hw_profile=hw_profile,
    )
    report["hw_profile"] = hw_profile
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
