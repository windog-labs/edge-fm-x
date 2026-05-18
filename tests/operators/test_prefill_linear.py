import math
from pathlib import Path

import pytest
import torch
import torch.testing

from ._test_utils import (
    CUDA_HW_PROFILE,
    DEFAULT_DEVICE_ID,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_0P5B_MODEL_PATH,
    QWEN_1P5B_MODEL_PATH,
    QWEN_3B_MODEL_PATH,
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


def _is_tuned_record(record: dict, case: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == case["hw_profile"]
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == case["layer_role"]
        and record.get("stage") == "prefill"
        and record.get("shape_sig") == case["shape_sig"]
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}) == case["impl_params"]
    )


PREFILL_TUNED_CASES_SM80 = [
    {
        "name": "1p5b_fused_qkv_m512",
        "hw_profile": "cuda_sm80",
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
        "impl_params": {"algo_index": 0},
    },
    {
        "name": "1p5b_fused_qkv_m1024",
        "hw_profile": "cuda_sm80",
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
        "impl_params": {"algo_index": 4},
    },
    {
        "name": "1p5b_fused_qkv_m2048",
        "hw_profile": "cuda_sm80",
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
        "impl_params": {"algo_index": 3},
    },
    {
        "name": "1p5b_attention_output_m512",
        "hw_profile": "cuda_sm80",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 512,
        "in_features": 1536,
        "out_features": 1536,
        "shape_sig": "m=512|input=2|weight=2|output=2|in_features=1536|out_features=1536",
        "impl_params": {"algo_index": 1},
    },
    {
        "name": "3b_attention_output_m1024",
        "hw_profile": "cuda_sm80",
        "model_path": QWEN_3B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 1024,
        "in_features": 2048,
        "out_features": 2048,
        "shape_sig": "m=1024|input=2|weight=2|output=2|in_features=2048|out_features=2048",
        "impl_params": {"algo_index": 3},
    },
    {
        "name": "3b_attention_output_m2048",
        "hw_profile": "cuda_sm80",
        "model_path": QWEN_3B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 2048,
        "in_features": 2048,
        "out_features": 2048,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=2048|out_features=2048",
        "impl_params": {"algo_index": 4},
    },
]


PREFILL_TUNED_CASES_SM86 = [
    {
        "name": "0p5b_fused_qkv_m2048",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 2048,
        "in_features": 896,
        "out_features": 1152,
        "q_out_features": 896,
        "k_out_features": 128,
        "v_out_features": 128,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=896|out_features=1152",
        "impl_params": {
            "algo_id": 30,
            "cluster_shape_id": 0,
            "cta_swizzling": 0,
            "custom_option": 0,
            "inner_shape_id": 0,
            "reduction_scheme": 0,
            "splitk_num": 1,
            "stages_id": 0,
            "tile_id": 18,
        },
    },
    {
        "name": "0p5b_mlp_down_m2048",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "mlp_down",
        "layer_role": "mlp_down",
        "layer_prefix": "model.layers.0.mlp.down_proj",
        "seq_len": 2048,
        "in_features": 4864,
        "out_features": 896,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=4864|out_features=896",
        "impl_params": {
            "algo_id": 5,
            "cluster_shape_id": 0,
            "cta_swizzling": 0,
            "custom_option": 0,
            "inner_shape_id": 0,
            "reduction_scheme": 0,
            "splitk_num": 1,
            "stages_id": 7,
            "tile_id": 23,
        },
    },
    {
        "name": "0p5b_fused_gate_up_m2048",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "fused_gate_up",
        "layer_role": "fused_gate_up",
        "layer_prefix": "model.layers.0.mlp",
        "seq_len": 2048,
        "in_features": 896,
        "out_features": 9728,
        "gate_out_features": 4864,
        "up_out_features": 4864,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=896|out_features=9728",
        "impl_params": {"algo_index": 3},
    },
    {
        "name": "1p5b_fused_gate_up_m1024",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "fused_gate_up",
        "layer_role": "fused_gate_up",
        "layer_prefix": "model.layers.0.mlp",
        "seq_len": 1024,
        "in_features": 1536,
        "out_features": 17920,
        "gate_out_features": 8960,
        "up_out_features": 8960,
        "shape_sig": "m=1024|input=2|weight=2|output=2|in_features=1536|out_features=17920",
        "impl_params": {"algo_index": 3},
    },
    {
        "name": "1p5b_fused_qkv_m2048",
        "hw_profile": "cuda_sm86",
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
        "impl_params": {"algo_index": 4},
    },
    {
        "name": "1p5b_mlp_down_m2048",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "mlp_down",
        "layer_role": "mlp_down",
        "layer_prefix": "model.layers.0.mlp.down_proj",
        "seq_len": 2048,
        "in_features": 8960,
        "out_features": 1536,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=8960|out_features=1536",
        "impl_params": {"algo_index": 5},
    },
    {
        "name": "1p5b_fused_gate_up_m2048",
        "hw_profile": "cuda_sm86",
        "model_path": QWEN_1P5B_MODEL_PATH,
        "layer_kind": "fused_gate_up",
        "layer_role": "fused_gate_up",
        "layer_prefix": "model.layers.0.mlp",
        "seq_len": 2048,
        "in_features": 1536,
        "out_features": 17920,
        "gate_out_features": 8960,
        "up_out_features": 8960,
        "shape_sig": "m=2048|input=2|weight=2|output=2|in_features=1536|out_features=17920",
        "impl_params": {"algo_index": 4},
    },
]


PREFILL_TUNED_CASES = (
    PREFILL_TUNED_CASES_SM86 if CUDA_HW_PROFILE == "cuda_sm86" else PREFILL_TUNED_CASES_SM80
)


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
    if case["layer_kind"] == "mlp_down":
        return edge_fm.LinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            case["out_features"],
        )
    if case["layer_kind"] == "fused_gate_up":
        return edge_fm.FusedGateUpLinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            case["gate_out_features"],
            case["up_out_features"],
        )
    raise ValueError(f"Unsupported layer_kind: {case['layer_kind']}")


def _source_op_table(
    case: dict,
    input_mode: str,
    weight_mode: str = "fp16_cast",
    overlap_casts: bool = False,
) -> Path:
    record = {
        "model_name": "qwen2_5",
        "hw_profile": CUDA_HW_PROFILE,
        "op_kind": "linear",
        "layer_role": case["layer_role"],
        "op_name": "",
        "stage": "prefill",
        "shape_sig": case["shape_sig"],
        "impl_id": "cutlass_prefill_linear_source_op",
        "impl_params": {
            "enabled": True,
            "tile": "auto",
            "input_mode": input_mode,
            "weight_mode": weight_mode,
            "overlap_casts": overlap_casts,
            "persistent_weights": False,
            "min_m": 1,
        },
    }
    return write_operator_impl_table([record])


@pytest.mark.parametrize("case", PREFILL_TUNED_CASES, ids=[case["name"] for case in PREFILL_TUNED_CASES])
def test_prefill_tuned_record_matches_baseline_output_and_latency(case):
    ensure_cuda()
    device = torch_device()

    base_table = load_operator_impl_table()
    current_records = base_table["records"]
    matcher = lambda record: _is_tuned_record(record, case)
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

    # Different cublasLt tactics can round BF16 GEMMs differently while remaining
    # valid for model-level generation. Keep this local to tuned-vs-heuristic
    # equivalence instead of relaxing the global dtype tolerance.
    torch.testing.assert_close(y_tuned, y_baseline, rtol=8e-2, atol=8e-2)
    assert math.isfinite(baseline_ms)
    assert math.isfinite(tuned_ms)
    # cublasLt prefill microbench has small run-to-run jitter on this host.
    # Keep the gate strict enough to reject real regressions while tolerating
    # the small run-to-run jitter that can appear on 3060 BF16 prefill microbenches.
    assert tuned_ms <= baseline_ms * 1.06, (
        f"{case['name']} tuned prefill record regressed beyond noise tolerance: "
        f"tuned={tuned_ms:.6f} ms baseline={baseline_ms:.6f} ms"
    )


def test_prefill_linear_source_op_mixed_bf16_matches_fp16_cast_source_op():
    ensure_cuda()
    device = torch_device()
    case = {
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "attention_output",
        "layer_role": "attention_output",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "seq_len": 128,
        "in_features": 896,
        "out_features": 896,
        "shape_sig": "m=128|input=2|weight=2|output=2|in_features=896|out_features=896",
    }

    torch.manual_seed(case["seq_len"])
    x = torch.randn(case["seq_len"], case["in_features"], device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    cast_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "fp16_cast"),
            )
        ),
    )
    y_cast = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    cast_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_cast), 0, "Prefill"
    )

    reset_weight_loader()
    mixed_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "mixed_bf16"),
            )
        ),
    )
    y_mixed = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    mixed_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_mixed), 0, "Prefill"
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(y_mixed, y_cast, rtol=8e-2, atol=8e-2)


def test_prefill_fused_qkv_source_op_mixed_bf16_bias_matches_fp16_cast_source_op():
    ensure_cuda()
    device = torch_device()
    case = {
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 128,
        "in_features": 896,
        "out_features": 1152,
        "q_out_features": 896,
        "k_out_features": 128,
        "v_out_features": 128,
        "shape_sig": "m=128|input=2|weight=2|output=2|in_features=896|out_features=1152",
    }

    torch.manual_seed(case["seq_len"] + 1)
    x = torch.randn(case["seq_len"], case["in_features"], device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    cast_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "fp16_cast"),
            )
        ),
    )
    y_cast = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    cast_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_cast), 0, "Prefill"
    )

    reset_weight_loader()
    mixed_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "mixed_bf16"),
            )
        ),
    )
    y_mixed = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    mixed_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_mixed), 0, "Prefill"
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(y_mixed, y_cast, rtol=8e-2, atol=8e-2)


def test_prefill_fused_qkv_source_op_overlap_casts_matches_same_stream_casts():
    ensure_cuda()
    device = torch_device()
    case = {
        "model_path": QWEN_0P5B_MODEL_PATH,
        "layer_kind": "fused_qkv",
        "layer_role": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "seq_len": 128,
        "in_features": 896,
        "out_features": 1152,
        "q_out_features": 896,
        "k_out_features": 128,
        "v_out_features": 128,
        "shape_sig": "m=128|input=2|weight=2|output=2|in_features=896|out_features=1152",
    }

    torch.manual_seed(case["seq_len"] + 7)
    x = torch.randn(case["seq_len"], case["in_features"], device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    cast_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "fp16_cast"),
            )
        ),
    )
    y_cast = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    cast_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_cast), 0, "Prefill"
    )

    reset_weight_loader()
    overlap_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(
                    case,
                    "fp16_cast",
                    overlap_casts=True,
                ),
            )
        ),
    )
    y_overlap = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    overlap_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_overlap), 0, "Prefill"
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(y_overlap, y_cast, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "case",
    [
        {
            "model_path": QWEN_0P5B_MODEL_PATH,
            "layer_kind": "attention_output",
            "layer_role": "attention_output",
            "layer_prefix": "model.layers.0.self_attn.o_proj",
            "seq_len": 128,
            "in_features": 896,
            "out_features": 896,
            "shape_sig": "m=128|input=2|weight=2|output=2|in_features=896|out_features=896",
        },
        {
            "model_path": QWEN_0P5B_MODEL_PATH,
            "layer_kind": "fused_qkv",
            "layer_role": "fused_qkv",
            "layer_prefix": "model.layers.0.self_attn",
            "seq_len": 128,
            "in_features": 896,
            "out_features": 1152,
            "q_out_features": 896,
            "k_out_features": 128,
            "v_out_features": 128,
            "shape_sig": "m=128|input=2|weight=2|output=2|in_features=896|out_features=1152",
        },
    ],
    ids=["attention_output", "fused_qkv_bias"],
)
def test_prefill_linear_source_op_bf16_direct_weight_matches_mixed_bf16(case):
    ensure_cuda()
    device = torch_device()

    torch.manual_seed(case["seq_len"] + 17)
    x = torch.randn(case["seq_len"], case["in_features"], device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    mixed_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "mixed_bf16"),
            )
        ),
    )
    y_mixed = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    mixed_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_mixed), 0, "Prefill"
    )

    reset_weight_loader()
    direct_layer = _make_layer(
        case,
        str(
            make_engine_config(
                case["model_path"],
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=_source_op_table(case, "mixed_bf16", "bf16_direct"),
            )
        ),
    )
    y_direct = torch.empty(case["seq_len"], case["out_features"], device=device, dtype=torch.bfloat16)
    direct_layer.forward_fp16_bf16(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(y_direct), 0, "Prefill"
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(y_direct, y_mixed, rtol=1.5e-1, atol=1.5e-1)
