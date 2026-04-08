import math

import pytest
import torch
import torch.testing

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    QWEN_3B_MODEL_PATH,
    dtype_tolerances,
    edge_fm,
    ensure_cuda,
    load_operator_impl_table,
    make_engine_config,
    median_cuda_ms,
    reset_weight_loader,
    tensor_to_edge_fm_tensor,
    torch_device,
    write_operator_impl_table,
)


def _is_tuned_record(record: dict, *, layer_role: str, shape_sig: str, algo_index: int) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == layer_role
        and record.get("stage") == "prefill"
        and record.get("shape_sig") == shape_sig
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == algo_index
    )


PREFILL_TUNED_CASES = [
    {
        "name": "1p5b_fused_qkv_m512",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 512,
        "in_features": 1536,
        "out_features": 2048,
        "q_out_features": 1536,
        "k_out_features": 256,
        "v_out_features": 256,
        "shape_sig": "m=512|input=2|weight=2|output=2|in_features=1536|out_features=2048",
        "algo_index": 0,
    },
    {
        "name": "1p5b_fused_qkv_m1024",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 1024,
        "in_features": 1536,
        "out_features": 2048,
        "q_out_features": 1536,
        "k_out_features": 256,
        "v_out_features": 256,
        "shape_sig": "m=1024|input=2|weight=2|output=2|in_features=1536|out_features=2048",
        "algo_index": 4,
    },
    {
        "name": "1p5b_fused_qkv_m2048",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 2048,
        "in_features": 1536,
        "out_features": 2048,
        "q_out_features": 1536,
        "k_out_features": 256,
        "v_out_features": 256,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=1536|out_features=2048",
        "algo_index": 3,
    },
    {
        "name": "1p5b_attention_output_m512",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 512,
        "in_features": 1536,
        "out_features": 1536,
        "shape_sig": "m=512|input=2|weight=2|output=2|in_features=1536|out_features=1536",
        "algo_index": 1,
    },
    {
        "name": "3b_attention_output_m1024",
        "model_path": QWEN_3B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 1024,
        "in_features": 2048,
        "out_features": 2048,
        "shape_sig": "m=1024|input=2|weight=2|output=2|in_features=2048|out_features=2048",
        "algo_index": 3,
    },
    {
        "name": "3b_attention_output_m2048",
        "model_path": QWEN_3B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 2048,
        "in_features": 2048,
        "out_features": 2048,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=2048|out_features=2048",
        "algo_index": 4,
    },
]


def _make_layer(case: dict, engine_config_path: str):
    if case["layer_kind"] == "fused_qkv":
        return edge_fm.FusedQKVLinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            case["q_out_features"],
            case["k_out_features"],
            case["v_out_features"],
        )
    if case["layer_kind"] == "attention_output":
        return edge_fm.LinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            case["out_features"],
        )
    raise ValueError(f"Unsupported layer_kind: {case['layer_kind']}")


@pytest.mark.parametrize("case", PREFILL_TUNED_CASES, ids=[case["name"] for case in PREFILL_TUNED_CASES])
def test_prefill_tuned_record_matches_baseline_output_and_latency(case):
    ensure_cuda()
    device = torch_device()

    base_table = load_operator_impl_table()
    current_records = base_table["records"]
    matcher = lambda record: _is_tuned_record(
        record,
        layer_role=case["layer_role"],
        shape_sig=case["shape_sig"],
        algo_index=case["algo_index"],
    )
    assert any(matcher(record) for record in current_records)
    baseline_records = [record for record in current_records if not matcher(record)]
    baseline_table_path = write_operator_impl_table(baseline_records)

    torch.manual_seed(case["seq_len"])
    x = torch.randn(case["seq_len"], case["in_features"], device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    baseline_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=baseline_table_path,
            )
        ),
    )
    y_baseline = torch.empty(
        case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16
    )
    x_baseline = tensor_to_edge_fm_tensor(x)
    y_baseline_efm = tensor_to_edge_fm_tensor(y_baseline)
    baseline_ms = median_cuda_ms(
        lambda: baseline_layer.forward_fp16_bf16(x_baseline, y_baseline_efm, 0, "Prefill"),
        warmup=20,
        iters=120,
    )

    reset_weight_loader()
    tuned_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
            )
        ),
    )
    y_tuned = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    x_tuned = tensor_to_edge_fm_tensor(x)
    y_tuned_efm = tensor_to_edge_fm_tensor(y_tuned)
    tuned_ms = median_cuda_ms(
        lambda: tuned_layer.forward_fp16_bf16(x_tuned, y_tuned_efm, 0, "Prefill"),
        warmup=20,
        iters=120,
    )

    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(y_tuned, y_baseline, rtol=rtol, atol=atol)
    assert math.isfinite(baseline_ms)
    assert math.isfinite(tuned_ms)
    # cublasLt prefill microbench has small run-to-run jitter on this host.
    # Keep the gate strict enough to reject real regressions while tolerating
    # low-single-digit noise that can appear when multiple tuned cases run in one suite.
    assert tuned_ms <= baseline_ms * 1.03, (
        f"{case['name']} tuned prefill record regressed beyond noise tolerance: "
        f"tuned={tuned_ms:.6f} ms baseline={baseline_ms:.6f} ms"
    )
