import math

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    DEFAULT_PREFILL_LENGTHS,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    dtype_tolerances,
    edge_fm_tensor_to_torch,
    edge_fm,
    ensure_cuda,
    make_engine_config,
    median_cuda_ms,
    reset_weight_loader,
    tensor_to_edge_fm_tensor,
    torch_device,
)

MODEL_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "q_out_features": 1536,
    "k_out_features": 256,
    "v_out_features": 256,
}

PREFILL_CASES = [
    {
        "name": "attention_output",
        "kind": "linear",
        "layer_prefix": "model.layers.0.self_attn.o_proj",
        "in_features": MODEL_CONFIG["hidden_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "weight_names": ["model.layers.0.self_attn.o_proj.weight"],
        "bias_names": ["model.layers.0.self_attn.o_proj.bias"],
        "rtol_atol_bf16": (1e-2, 1e-2),
        "max_median_ms_by_seq_len": {512: 0.5, 1024: 0.8, 2048: 1.2},
    },
    {
        "name": "mlp_down",
        "kind": "linear",
        "layer_prefix": "model.layers.0.mlp.down_proj",
        "in_features": MODEL_CONFIG["intermediate_size"],
        "out_features": MODEL_CONFIG["hidden_size"],
        "weight_names": ["model.layers.0.mlp.down_proj.weight"],
        "bias_names": ["model.layers.0.mlp.down_proj.bias"],
        "rtol_atol_bf16": (2e-2, 2e-2),
        "max_median_ms_by_seq_len": {512: 0.8, 1024: 1.2, 2048: 2.0},
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
        "rtol_atol_bf16": (1e-2, 1e-2),
        "max_median_ms_by_seq_len": {512: 0.8, 1024: 1.2, 2048: 1.8},
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


def _make_prefill_layer(case: dict, engine_config_path: str):
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
    else:
        raise ValueError(f"Unsupported case kind: {case['kind']}")

    ref = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
    return ref.to(x.dtype)


@pytest.mark.parametrize("seq_len", DEFAULT_PREFILL_LENGTHS)
@pytest.mark.parametrize("case", PREFILL_CASES, ids=[case["name"] for case in PREFILL_CASES])
def test_prefill_linear_correctness(case, seq_len):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )

    layer = _make_prefill_layer(case, str(engine_config_path))
    torch.manual_seed(seq_len)
    x = torch.randn(seq_len, case["in_features"], device=device, dtype=torch.bfloat16)
    y = torch.empty(seq_len, case["out_features"], device=device, dtype=torch.bfloat16)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    layer.forward_fp16_bf16(x_efm, y_efm, 0, "Prefill")
    torch.cuda.synchronize()

    y_torch = edge_fm_tensor_to_torch(y_efm)
    ref = _reference_output(case, x)
    rtol, atol = case.get("rtol_atol_bf16", dtype_tolerances(torch.bfloat16))
    torch.testing.assert_close(y_torch, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("seq_len", DEFAULT_PREFILL_LENGTHS)
@pytest.mark.parametrize("case", PREFILL_CASES, ids=[case["name"] for case in PREFILL_CASES])
def test_prefill_linear_performance_smoke(case, seq_len):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )

    layer = _make_prefill_layer(case, str(engine_config_path))
    x = torch.randn(seq_len, case["in_features"], device=device, dtype=torch.bfloat16)
    y = torch.empty(seq_len, case["out_features"], device=device, dtype=torch.bfloat16)
    x_efm = tensor_to_edge_fm_tensor(x)
    y_efm = tensor_to_edge_fm_tensor(y)

    median_ms = median_cuda_ms(
        lambda: layer.forward_fp16_bf16(x_efm, y_efm, 0, "Prefill"),
        warmup=20,
        iters=100,
    )
    assert math.isfinite(median_ms)
    assert median_ms < case["max_median_ms_by_seq_len"][seq_len], (
        f"{case['name']} prefill seq_len={seq_len} median latency regressed: "
        f"{median_ms:.6f} ms >= {case['max_median_ms_by_seq_len'][seq_len]:.6f} ms"
    )
