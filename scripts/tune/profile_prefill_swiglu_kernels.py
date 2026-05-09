#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for build_python in [REPO_ROOT / "build" / "python", REPO_ROOT / "build" / "install" / "python"]:
    if build_python.exists() and str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

from edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(REPO_ROOT)

import edge_fm
from operator_table.utils import (
    resolve_engine_model_name,
    resolve_operator_table_path,
    resolve_target_hw_profile,
)
from temp_paths import make_temp_dir


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


def make_engine_config(
    model_path: Path,
    *,
    device_id: int,
    operator_impl_table_path: Path,
    hw_profile: str,
) -> Path:
    return write_json_file(
        "efm_prefill_swiglu_cfg_",
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
    raise TypeError(f"Unsupported dtype for edge_fm.Tensor view: {torch_dtype}")


def tensor_to_edge_fm_tensor(torch_tensor: torch.Tensor) -> edge_fm.Tensor:
    if not torch_tensor.is_contiguous():
        raise ValueError("tensor_to_edge_fm_tensor expects a contiguous torch.Tensor")
    if torch_tensor.device.type != "cuda":
        raise TypeError("tensor_to_edge_fm_tensor expects a CUDA tensor")
    return edge_fm.Tensor(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        _edge_fm_dtype(torch_tensor.dtype),
        edge_fm.Device.GPU,
        torch_tensor.device.index or 0,
        False,
    )


def edge_fm_tensor_to_torch(tensor: edge_fm.Tensor) -> torch.Tensor:
    return torch.from_dlpack(tensor.to_dlpack())


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
    config = json.loads((model_path / "config.json").read_text())
    if "text_config" in config and isinstance(config["text_config"], dict):
        config = config["text_config"]
    return {
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
    }


def parse_seq_lens(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("seq lens must not be empty")
    return values


def make_outputs(seq_len: int, hidden_size: int, intermediate_size: int, device: str, dtype: torch.dtype):
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    fused_linear_out = torch.empty(seq_len, 2 * intermediate_size, device=device, dtype=dtype)
    two_stage_out = torch.empty(seq_len, intermediate_size, device=device, dtype=dtype)
    fused_out = torch.empty(seq_len, intermediate_size, device=device, dtype=dtype)
    return x, fused_linear_out, two_stage_out, fused_out


def tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 6.25e-2
    return 1e-3, 1e-3


def benchmark_case(
    *,
    model_path: Path,
    dims: dict,
    operator_impl_table_path: Path,
    hw_profile: str,
    device_id: int,
    seq_len: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict:
    reset_weight_loader()
    engine_config_path = make_engine_config(
        model_path,
        device_id=device_id,
        operator_impl_table_path=operator_impl_table_path,
        hw_profile=hw_profile,
    )

    fused_linear = edge_fm.FusedGateUpLinearLayer(
        "model.layers.0.mlp",
        str(engine_config_path),
        dims["hidden_size"],
        dims["intermediate_size"],
        dims["intermediate_size"],
    )
    activation = edge_fm.ActivationLayer(str(engine_config_path))

    device = f"cuda:{device_id}"
    torch.manual_seed(seq_len)
    x, fused_linear_out, two_stage_out, fused_out = make_outputs(
        seq_len,
        dims["hidden_size"],
        dims["intermediate_size"],
        device,
        dtype,
    )

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_linear_out_efm = tensor_to_edge_fm_tensor(fused_linear_out)
    two_stage_out_efm = tensor_to_edge_fm_tensor(two_stage_out)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)

    fused_available = fused_linear.try_forward_prefill_swiglu_fused(x_efm, fused_out_efm, 0)
    if fused_available:
        fused_ref_out = torch.empty_like(fused_out)
        fused_ref_out_efm = tensor_to_edge_fm_tensor(fused_ref_out)
        fused_linear.forward_fp16_bf16(x_efm, fused_linear_out_efm, 0, "Prefill")
        activation.forward_silu_and_mul_up_gate(fused_linear_out_efm, fused_ref_out_efm, 0, "Prefill")
        torch.cuda.synchronize()
        rtol, atol = tolerances_for_dtype(dtype)
        torch.testing.assert_close(
            edge_fm_tensor_to_torch(fused_out_efm),
            edge_fm_tensor_to_torch(fused_ref_out_efm),
            rtol=rtol,
            atol=atol,
        )
    else:
        fused_ref_out = None
        fused_ref_out_efm = None

    def run_two_stage() -> None:
        fused_linear.forward_fp16_bf16(x_efm, fused_linear_out_efm, 0, "Prefill")
        activation.forward_silu_and_mul_up_gate(fused_linear_out_efm, two_stage_out_efm, 0, "Prefill")

    def run_fused() -> None:
        if not fused_linear.try_forward_prefill_swiglu_fused(x_efm, fused_out_efm, 0):
            raise RuntimeError("prefill fused SwiGLU fast path unavailable")

    two_stage_ms = median_cuda_ms(run_two_stage, warmup=warmup, iters=iters)
    fused_ms = None
    if fused_available:
        fused_ms = median_cuda_ms(run_fused, warmup=warmup, iters=iters)
        torch.cuda.synchronize()
        rtol, atol = tolerances_for_dtype(dtype)
        torch.testing.assert_close(
            edge_fm_tensor_to_torch(fused_out_efm),
            edge_fm_tensor_to_torch(two_stage_out_efm),
            rtol=rtol,
            atol=atol,
        )

    result = {
        "seq_len": seq_len,
        "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
        "shape_sig": (
            f"m={seq_len}|input={EDGE_FM_DTYPE_ID_BY_TORCH_DTYPE[dtype]}|"
            f"weight={EDGE_FM_DTYPE_ID_BY_TORCH_DTYPE[dtype]}|"
            f"output={EDGE_FM_DTYPE_ID_BY_TORCH_DTYPE[dtype]}|"
            f"in_features={dims['hidden_size']}|out_features={2 * dims['intermediate_size']}"
        ),
        "fused_available": fused_available,
        "two_stage": {
            "median_ms": two_stage_ms,
        },
        "fused": {
            "median_ms": fused_ms,
            "delta_ms": None if fused_ms is None else fused_ms - two_stage_ms,
            "delta_pct": None if fused_ms is None else ((fused_ms - two_stage_ms) / two_stage_ms * 100.0),
        },
    }
    if fused_available:
        rtol, atol = tolerances_for_dtype(dtype)
        result["correctness"] = {
            "rtol": rtol,
            "atol": atol,
            "passed": True,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile prefill SwiGLU fused path against the two-stage fallback"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a Qwen2.5 model directory",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BY_NAME.keys()), default="bf16")
    parser.add_argument("--seq-lens", default="512,1024,2048", type=parse_seq_lens)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--hw-profile",
        default=resolve_target_hw_profile(),
        help="Runtime hw_profile used in the generated engine config",
    )
    parser.add_argument(
        "--operator-table",
        default="",
        help="Optional operator table path; defaults to the resolved platform table",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / ".tmp_codex" / "bench" / "profile_prefill_swiglu_kernels.json"),
        help="Structured JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("EDGE_FM_PREFILL_SWIGLU_FUSION", "1")
    torch.cuda.set_device(args.device_id)

    model_path = Path(args.model_path).resolve()
    operator_table_path = resolve_operator_table_path(
        Path(args.operator_table).resolve() if args.operator_table else None,
        model_path=model_path,
    )
    dims = load_model_dims(model_path)
    dtype = DTYPE_BY_NAME[args.dtype]

    results = []
    for seq_len in args.seq_lens:
        result = benchmark_case(
            model_path=model_path,
            dims=dims,
            operator_impl_table_path=operator_table_path,
            hw_profile=args.hw_profile,
            device_id=args.device_id,
            seq_len=seq_len,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(result)

        fused = result["fused"]["median_ms"]
        if fused is None:
            print(f"seq_len={seq_len}: fused unavailable, two-stage={result['two_stage']['median_ms']:.3f} ms")
        else:
            print(
                f"seq_len={seq_len}: fused={fused:.3f} ms, "
                f"two-stage={result['two_stage']['median_ms']:.3f} ms, "
                f"delta={result['fused']['delta_ms']:+.3f} ms"
            )

    payload = {
        "model_path": str(model_path),
        "model_name": resolve_engine_model_name(model_path),
        "hw_profile": args.hw_profile,
        "device_id": args.device_id,
        "dtype": args.dtype,
        "seq_lens": args.seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "operator_table_path": str(operator_table_path),
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
