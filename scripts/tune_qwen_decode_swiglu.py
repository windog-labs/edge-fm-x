#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for build_python in [REPO_ROOT / "build" / "python", REPO_ROOT / "build" / "install" / "python"]:
    if build_python.exists() and str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

import edge_fm
from _repo_temp import make_temp_dir
from operator_table_utils import resolve_engine_model_name, resolve_operator_table_path


ALL_CONFIG_NAMES = [
    "16x128x64_s2",
    "16x256x64_s2",
    "32x128x64_s2",
    "64x128x64_s2",
    "128x128x64_s2",
    "16x128x64_s3",
    "16x256x64_s3",
]


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = make_temp_dir(prefix)
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def make_engine_config(model_path: Path, device_id: int, operator_impl_table_path: Path | None) -> Path:
    payload = {
        "model_name": resolve_engine_model_name(model_path),
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": "cuda_sm80",
        },
        "prefill_model_path": str(model_path),
    }
    if operator_impl_table_path is not None:
        payload["operator_impl_table_path"] = str(operator_impl_table_path)
    return write_json_file("efm_qwen_decode_swiglu_cfg_", "engine_config.json", payload)


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


def reset_weight_loader() -> None:
    loader = edge_fm.WeightLoader.instance()
    loader.clear_stage(edge_fm.ModelStage.Prefill)
    loader.clear_stage(edge_fm.ModelStage.Decode)


def load_model_dims(model_path: Path) -> dict:
    config = json.loads((model_path / "config.json").read_text())
    if "text_config" in config and isinstance(config["text_config"], dict):
        config = config["text_config"]
    return {
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
    }


def parse_candidates(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("candidates must not be empty")
    for value in values:
        if value not in {"auto", "default"} and value not in ALL_CONFIG_NAMES:
            raise argparse.ArgumentTypeError(f"unsupported candidate '{value}'")
    return values


def apply_env(candidate: str) -> dict[str, str | None]:
    previous = {
        "EDGE_FM_DECODE_SWIGLU_AUTOTUNE": os.environ.get("EDGE_FM_DECODE_SWIGLU_AUTOTUNE"),
        "EDGE_FM_DECODE_SWIGLU_CONFIG": os.environ.get("EDGE_FM_DECODE_SWIGLU_CONFIG"),
    }
    if candidate == "auto":
        os.environ.pop("EDGE_FM_DECODE_SWIGLU_CONFIG", None)
        os.environ["EDGE_FM_DECODE_SWIGLU_AUTOTUNE"] = "1"
    elif candidate == "default":
        os.environ.pop("EDGE_FM_DECODE_SWIGLU_CONFIG", None)
        os.environ["EDGE_FM_DECODE_SWIGLU_AUTOTUNE"] = "0"
    else:
        os.environ["EDGE_FM_DECODE_SWIGLU_CONFIG"] = candidate
        os.environ["EDGE_FM_DECODE_SWIGLU_AUTOTUNE"] = "1"
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def benchmark_candidate(
    *,
    model_path: Path,
    dims: dict,
    candidate: str,
    device_id: int,
    warmup: int,
    iters: int,
    operator_impl_table_path: Path | None,
) -> dict:
    previous_env = apply_env(candidate)
    try:
        reset_weight_loader()
        engine_config_path = make_engine_config(model_path, device_id, operator_impl_table_path)
        layer = edge_fm.FusedGateUpLinearLayer(
            "model.layers.0.mlp",
            str(engine_config_path),
            dims["hidden_size"],
            dims["intermediate_size"],
            dims["intermediate_size"],
        )

        x = torch.randn(1, dims["hidden_size"], device=f"cuda:{device_id}", dtype=torch.bfloat16)
        y = torch.empty(1, dims["intermediate_size"], device=f"cuda:{device_id}", dtype=torch.bfloat16)
        x_efm = tensor_to_edge_fm_tensor(x)
        y_efm = tensor_to_edge_fm_tensor(y)

        def run() -> None:
            ok = layer.try_forward_decode_swiglu_fused(x_efm, y_efm, 0)
            if not ok:
                raise RuntimeError("decode_swiglu_fused fast path unavailable")

        run()
        torch.cuda.synchronize()
        median_ms = median_cuda_ms(run, warmup=warmup, iters=iters)
        return {
            "candidate": candidate,
            "median_ms": median_ms,
            "checksum_abs_mean": float(y.float().abs().mean().item()),
        }
    finally:
        restore_env(previous_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Qwen decode fused gate_up + SwiGLU kernel config via existing env override"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--operator-table",
        default="",
    )
    parser.add_argument(
        "--candidates",
        type=parse_candidates,
        default=["auto", "default"] + ALL_CONFIG_NAMES,
        help="Comma-separated list from auto,default and concrete kernel config names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    model_path = Path(args.model_path).resolve()
    dims = load_model_dims(model_path)
    operator_impl_table_path = resolve_operator_table_path(
        Path(args.operator_table).resolve() if args.operator_table else None,
        model_path=model_path,
    )

    candidates = []
    for candidate in args.candidates:
        candidates.append(
            benchmark_candidate(
                model_path=model_path,
                dims=dims,
                candidate=candidate,
                device_id=args.device_id,
                warmup=args.warmup,
                iters=args.iters,
                operator_impl_table_path=operator_impl_table_path,
            )
        )

    best = min(candidates, key=lambda item: item["median_ms"])
    report = {
        "model_path": str(model_path),
        "device_id": args.device_id,
        "hidden_size": dims["hidden_size"],
        "intermediate_size": dims["intermediate_size"],
        "candidates": candidates,
        "best": best,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
