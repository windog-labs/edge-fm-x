import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    dtype_tolerances,
    edge_fm_tensor_to_torch,
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
from scripts.operator_table.utils import resolve_operator_table_path

MODEL_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "q_out_features": 1536,
    "k_out_features": 256,
    "v_out_features": 256,
    "vocab_size": 151936,
}

QWEN_VL_3B_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "qwen2.5-vl-3b-instruct"
    / "qwen2.5-vl-3b-instruct"
)
VLM_OPERATOR_IMPL_TABLE_PATH = resolve_operator_table_path(
    model_path=QWEN_VL_3B_MODEL_PATH,
    model_name="Qwen2.5-VL",
)

ATTENTION_OUTPUT_DECODE_SHAPE_SIG = "m=1|input=2|weight=2|output=2|in_features=1536|out_features=1536"
MLP_DOWN_DECODE_SHAPE_SIG = "m=1|input=2|weight=2|output=2|in_features=8960|out_features=1536"
FUSED_QKV_DECODE_SHAPE_SIG = "m=1|input=2|weight=2|output=2|in_features=1536|out_features=2048"
LM_HEAD_DECODE_SHAPE_SIG = "m=1|input=2|weight=2|output=2|in_features=1536|out_features=151936"

DECODE_CASES = [
    {
        "name": "attention_output",
        "kind": "linear",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "weight_names": ["model.layers.0.self_attn.o_proj.weight"],
        "bias_names": ["model.layers.0.self_attn.o_proj.bias"],
        "max_median_ms": 0.5,
    },
    {
        "name": "mlp_down",
        "kind": "linear",
        "layer_prefix": "model.layers.0.mlp.down_proj",
        "in_features": MODEL_CONFIG["intermediate_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "weight_names": ["model.layers.0.mlp.down_proj.weight"],
        "bias_names": ["model.layers.0.mlp.down_proj.bias"],
        "max_median_ms": 0.5,
    },
    {
        "name": "fused_qkv",
        "kind": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": (
            MODEL_CONFIG["q_out_features"]
            + MODEL_CONFIG["k_out_features"]
            + MODEL_CONFIG["v_out_features"]
        ),
        "weight_names": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ],
        "bias_names": [
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
        ],
        "max_median_ms": 0.5,
    },
    {
        "name": "lm_head",
        "kind": "lm_head",
        "layer_prefix": "lm_head",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": MODEL_CONFIG["vocab_size"],
        "weight_names": [
            "lm_head.weight",
            "model.lm_head.weight",
            "model.embed_tokens.weight",
        ],
        "bias_names": [],
        "max_median_ms": 1.5,
    },
]


def _load_first_available_tensor(name_candidates: list[str], *, device: str):
    with safe_open(
        str(QWEN_1P5B_MODEL_PATH / "model.safetensors"), framework="pt", device=device
    ) as handle:
        for name in name_candidates:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"None of {name_candidates} found in model.safetensors")


def _load_optional_tensors(name_candidates: list[str], *, device: str) -> list[torch.Tensor]:
    tensors = []
    with safe_open(
        str(QWEN_1P5B_MODEL_PATH / "model.safetensors"), framework="pt", device=device
    ) as handle:
        for name in name_candidates:
            if name in handle.keys():
                tensors.append(handle.get_tensor(name))
    return tensors


def _make_decode_layer(case: dict, engine_config_path: str):
    if case["kind"] == "linear":
        return edge_fm.LinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            case["out_features"],
        )
    if case["kind"] == "fused_qkv":
        return edge_fm.FusedQKVLinearLayer(
            case["layer_prefix"],
            engine_config_path,
            case["in_features"],
            MODEL_CONFIG["q_out_features"],
            MODEL_CONFIG["k_out_features"],
            MODEL_CONFIG["v_out_features"],
        )
    if case["kind"] == "lm_head":
        return edge_fm.LMHeadLinearLayer(
            engine_config_path,
            case["in_features"],
            case["out_features"],
            case["layer_prefix"],
        )
    raise ValueError(f"Unsupported case kind: {case['kind']}")


def _reference_output(case: dict, x: torch.Tensor) -> torch.Tensor:
    device = str(x.device)
    if case["kind"] == "linear":
        weight = _load_first_available_tensor(case["weight_names"], device=device)
        bias_tensors = _load_optional_tensors(case["bias_names"], device=device)
        bias = bias_tensors[0] if bias_tensors else None
    elif case["kind"] == "fused_qkv":
        weights = [
            _load_first_available_tensor([name], device=device) for name in case["weight_names"]
        ]
        weight = torch.cat(weights, dim=0)
        bias_tensors = _load_optional_tensors(case["bias_names"], device=device)
        bias = torch.cat(bias_tensors, dim=0) if bias_tensors else None
    elif case["kind"] == "lm_head":
        weight = _load_first_available_tensor(case["weight_names"], device=device)
        bias = None
    else:
        raise ValueError(f"Unsupported case kind: {case['kind']}")

    ref = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
    return ref.to(x.dtype)


@pytest.mark.parametrize("case", DECODE_CASES, ids=[case["name"] for case in DECODE_CASES])
def test_decode_linear_correctness(case):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )

    layer = _make_decode_layer(case, str(engine_config_path))
    torch.manual_seed(0)
    x = torch.randn(1, case["in_features"], device=device, dtype=torch.bfloat16)
    y = torch.empty(1, case["out_features"], device=device, dtype=torch.bfloat16)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    layer.forward_fp16_bf16(
        x_efm,
        y_efm,
        0,
        "Decode",
    )
    torch.cuda.synchronize()

    y_torch = edge_fm_tensor_to_torch(y_efm)
    ref = _reference_output(case, x)
    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(y_torch, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", DECODE_CASES, ids=[case["name"] for case in DECODE_CASES])
def test_decode_linear_performance_smoke(case):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )

    layer = _make_decode_layer(case, str(engine_config_path))
    x = torch.randn(1, case["in_features"], device=device, dtype=torch.bfloat16)
    y = torch.empty(1, case["out_features"], device=device, dtype=torch.bfloat16)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    median_ms = median_cuda_ms(
        lambda: layer.forward_fp16_bf16(x_efm, y_efm, 0, "Decode"),
        warmup=30,
        iters=200,
    )
    assert math.isfinite(median_ms)
    assert median_ms < case["max_median_ms"], (
        f"{case['name']} decode median latency regressed: "
        f"{median_ms:.6f} ms >= {case['max_median_ms']:.6f} ms"
    )


def _is_tuned_mlp_down_record(record: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == "mlp_down"
        and record.get("stage") == "decode"
        and record.get("shape_sig") == MLP_DOWN_DECODE_SHAPE_SIG
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == 2
    )


def _is_tuned_attention_output_record(record: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == "attention_output"
        and record.get("stage") == "decode"
        and record.get("shape_sig") == ATTENTION_OUTPUT_DECODE_SHAPE_SIG
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == 1
    )


def _is_tuned_fused_qkv_record(record: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == "fused_qkv"
        and record.get("stage") == "decode"
        and record.get("shape_sig") == FUSED_QKV_DECODE_SHAPE_SIG
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == 0
    )


def _is_tuned_lm_head_record(record: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == "lm_head"
        and record.get("stage") == "decode"
        and record.get("shape_sig") == LM_HEAD_DECODE_SHAPE_SIG
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == 2
    )


TUNED_DECODE_LINEAR_CASES = [
    {
        "name": "fused_qkv",
        "kind": "fused_qkv",
        "layer_prefix": "model.layers.0.self_attn",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": (
            MODEL_CONFIG["q_out_features"]
            + MODEL_CONFIG["k_out_features"]
            + MODEL_CONFIG["v_out_features"]
        ),
        "matcher": _is_tuned_fused_qkv_record,
    },
    {
        "name": "attention_output",
        "kind": "linear",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "matcher": _is_tuned_attention_output_record,
    },
    {
        "name": "mlp_down",
        "kind": "linear",
        "layer_prefix": "model.layers.0.mlp.down_proj",
        "in_features": MODEL_CONFIG["intermediate_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "matcher": _is_tuned_mlp_down_record,
    },
    {
        "name": "lm_head",
        "kind": "lm_head",
        "layer_prefix": "lm_head",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": MODEL_CONFIG["vocab_size"],
        "matcher": _is_tuned_lm_head_record,
    },
]


@pytest.mark.parametrize("case", TUNED_DECODE_LINEAR_CASES, ids=[case["name"] for case in TUNED_DECODE_LINEAR_CASES])
def test_decode_tuned_record_matches_baseline_output(case):
    ensure_cuda()
    device = torch_device()

    base_table = load_operator_impl_table()
    current_records = base_table["records"]
    assert any(case["matcher"](record) for record in current_records)
    baseline_records = [record for record in current_records if not case["matcher"](record)]
    baseline_table_path = write_operator_impl_table(baseline_records)

    x = torch.randn(
        1, case["in_features"], device=device, dtype=torch.bfloat16
    )

    reset_weight_loader()
    baseline_layer = _make_decode_layer(
        case,
        str(
            make_engine_config(
                QWEN_1P5B_MODEL_PATH,
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=baseline_table_path,
            )
        ),
    )
    y_baseline = torch.empty(1, case["out_features"], device=device, dtype=torch.bfloat16)
    x_baseline = tensor_to_edge_fm_tensor(x)
    y_baseline_efm = tensor_to_edge_fm_tensor(y_baseline)
    baseline_ms = median_cuda_ms(
        lambda: baseline_layer.forward_fp16_bf16(x_baseline, y_baseline_efm, 0, "Decode"),
        warmup=40,
        iters=250,
    )

    reset_weight_loader()
    tuned_layer = _make_decode_layer(
        case,
        str(
            make_engine_config(
                QWEN_1P5B_MODEL_PATH,
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
            )
        ),
    )
    y_tuned = torch.empty(1, case["out_features"], device=device, dtype=torch.bfloat16)
    x_tuned = tensor_to_edge_fm_tensor(x)
    y_tuned_efm = tensor_to_edge_fm_tensor(y_tuned)
    tuned_ms = median_cuda_ms(
        lambda: tuned_layer.forward_fp16_bf16(x_tuned, y_tuned_efm, 0, "Decode"),
        warmup=40,
        iters=250,
    )

    y_baseline_torch = edge_fm_tensor_to_torch(y_baseline_efm)
    y_tuned_torch = edge_fm_tensor_to_torch(y_tuned_efm)
    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(y_tuned_torch, y_baseline_torch, rtol=rtol, atol=atol)
    assert math.isfinite(baseline_ms)
    assert math.isfinite(tuned_ms)


def test_vlm_decode_linear_uses_shape_tuned_record():
    ensure_cuda()
    if not QWEN_VL_3B_MODEL_PATH.exists():
        pytest.skip("Qwen2.5-VL-3B model path not found")

    reset_weight_loader()
    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_VL_3B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=VLM_OPERATOR_IMPL_TABLE_PATH,
        model_name="Qwen2.5-VL",
    )

    layer = edge_fm.LinearLayer(
        "model.layers.0.self_attn.o_proj",
        str(engine_config_path),
        2048,
        2048,
    )
    x = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
    y = torch.empty(1, 2048, device=device, dtype=torch.bfloat16)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    layer.forward_fp16_bf16(x_efm, y_efm, 0, "Decode")
    torch.cuda.synchronize()

    info = json.loads(layer.debug_cached_impl_info("Decode", 1))
    assert info["selected_impl_id"] == "cublasLt"
    assert info["selected_impl_params"].get("algo_index") == 1
