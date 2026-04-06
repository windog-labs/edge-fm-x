#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for build_python in [REPO_ROOT / "build" / "python", REPO_ROOT / "build" / "install" / "python"]:
    if build_python.exists() and str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

import edge_fm


MODEL_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
}


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def load_operator_impl_table(path: Path) -> dict:
    return json.loads(path.read_text())


def write_operator_impl_table(records: list[dict]) -> Path:
    return write_json_file(
        "efm_fused_gate_up_tune_",
        "operator_impl_table.json",
        {
            "schema": "edgefm_operator_impl_table_v1",
            "records": records,
        },
    )


def make_engine_config(model_path: Path, device_id: int, operator_impl_table_path: Path) -> Path:
    config = {
        "model_name": "Qwen2.5",
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": "cuda_sm80",
        },
        "prefill_model_path": str(model_path),
        "operator_impl_table_path": str(operator_impl_table_path),
    }
    return write_json_file("efm_fused_gate_up_cfg_", "engine_config.json", config)


def tensor_to_edge_fm_tensor(torch_tensor: torch.Tensor) -> edge_fm.Tensor:
    return edge_fm.Tensor.from_dlpack(torch_tensor.contiguous().__dlpack__())


def edge_fm_tensor_to_torch(tensor: edge_fm.Tensor) -> torch.Tensor:
    return torch.from_dlpack(tensor.to_dlpack())


def reset_weight_loader() -> None:
    loader = edge_fm.WeightLoader.instance()
    loader.clear_stage(edge_fm.ModelStage.Prefill)
    loader.clear_stage(edge_fm.ModelStage.Decode)


def bench_cuda_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end))
    return vals


def median_cuda_ms(fn, warmup: int, iters: int) -> float:
    return statistics.median(bench_cuda_ms(fn, warmup=warmup, iters=iters))


def fused_gate_up_shape_sig(seq_len: int) -> str:
    return (
        f"m={seq_len}|input=2|weight=2|output=2|"
        f"in_features={MODEL_CONFIG['hidden_size']}|out_features={2 * MODEL_CONFIG['intermediate_size']}"
    )


def build_tuned_records(
    base_records: list[dict],
    *,
    seq_len: int,
    stage: str,
    algo_index: int | None,
) -> list[dict]:
    shape_sig = fused_gate_up_shape_sig(seq_len)
    kept = []
    for record in base_records:
        if (
            record.get("model_name") == "qwen2_5"
            and record.get("hw_profile") == "cuda_sm80"
            and record.get("op_kind") == "linear"
            and record.get("layer_role") == "fused_gate_up"
            and record.get("stage") == stage.lower()
            and record.get("shape_sig") == shape_sig
        ):
            continue
        kept.append(record)

    if algo_index is not None:
        kept.append(
            {
                "model_name": "qwen2_5",
                "hw_profile": "cuda_sm80",
                "op_kind": "linear",
                "layer_role": "fused_gate_up",
                "op_name": "",
                "stage": stage.lower(),
                "shape_sig": shape_sig,
                "impl_id": "cublasLt",
                "impl_params": {
                    "algo_index": algo_index,
                },
            }
        )

    return kept


def benchmark_candidate(
    *,
    model_path: Path,
    device_id: int,
    base_records: list[dict],
    seq_len: int,
    stage: str,
    algo_index: int | None,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    operator_table_path = write_operator_impl_table(
        build_tuned_records(
            base_records,
            seq_len=seq_len,
            stage=stage,
            algo_index=algo_index,
        )
    )
    engine_config_path = make_engine_config(model_path, device_id, operator_table_path)

    reset_weight_loader()
    layer = edge_fm.FusedGateUpLinearLayer(
        "model.layers.0.mlp",
        str(engine_config_path),
        MODEL_CONFIG["hidden_size"],
        MODEL_CONFIG["intermediate_size"],
        MODEL_CONFIG["intermediate_size"],
    )

    device = f"cuda:{device_id}"
    torch.manual_seed(seq_len)
    x = torch.randn(seq_len, MODEL_CONFIG["hidden_size"], device=device, dtype=torch.bfloat16)
    y = torch.empty(
        seq_len,
        2 * MODEL_CONFIG["intermediate_size"],
        device=device,
        dtype=torch.bfloat16,
    )
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    median_ms = median_cuda_ms(
        lambda: layer.forward_fp16_bf16(x_efm, y_efm, 0, stage),
        warmup=warmup,
        iters=iters,
    )
    checksum = float(edge_fm_tensor_to_torch(y_efm).float().abs().mean().item())
    return median_ms, checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune cublasLt algo_index for FusedGateUpLinearLayer")
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
    )
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--seq-lens", default="1,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--operator-table", default=str(REPO_ROOT / "examples" / "config" / "operator_impl_table.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    model_path = Path(args.model_path).resolve()
    base_table = load_operator_impl_table(Path(args.operator_table))
    base_records = base_table["records"]
    seq_lens = [int(item.strip()) for item in args.seq_lens.split(",") if item.strip()]

    report = []
    for seq_len in seq_lens:
        stage = "Decode" if seq_len == 1 else "Prefill"
        candidates: list[dict] = []
        for algo_index in [None, 0, 1, 2, 3, 4]:
            label = "baseline" if algo_index is None else f"algo_{algo_index}"
            median_ms, checksum = benchmark_candidate(
                model_path=model_path,
                device_id=args.device_id,
                base_records=base_records,
                seq_len=seq_len,
                stage=stage,
                algo_index=algo_index,
                warmup=args.warmup,
                iters=args.iters,
            )
            candidates.append(
                {
                    "candidate": label,
                    "median_ms": median_ms,
                    "checksum_abs_mean": checksum,
                }
            )

        best = min(candidates, key=lambda item: item["median_ms"])
        report.append(
            {
                "seq_len": seq_len,
                "stage": stage,
                "shape_sig": fused_gate_up_shape_sig(seq_len),
                "candidates": candidates,
                "best": best,
            }
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
