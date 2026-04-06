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
    load_operator_impl_table,
    make_engine_config,
    median_cuda_ms,
    reset_weight_loader,
    tensor_to_edge_fm_tensor,
    torch_device,
    write_operator_impl_table,
)

HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960
FUSED_GATE_UP_DECODE_SHAPE_SIG = (
    "m=1|input=2|weight=2|output=2|in_features=1536|out_features=17920"
)
FUSED_GATE_UP_PREFILL_512_SHAPE_SIG = (
    "m=512|input=2|weight=2|output=2|in_features=1536|out_features=17920"
)
FUSED_GATE_UP_PREFILL_1024_SHAPE_SIG = (
    "m=1024|input=2|weight=2|output=2|in_features=1536|out_features=17920"
)
FUSED_GATE_UP_PREFILL_2048_SHAPE_SIG = (
    "m=2048|input=2|weight=2|output=2|in_features=1536|out_features=17920"
)


def _load_tensor(name: str, *, device: str) -> torch.Tensor:
    with safe_open(
        str(QWEN_1P5B_MODEL_PATH / "model.safetensors"), framework="pt", device=device
    ) as handle:
        return handle.get_tensor(name)


def _make_layers():
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )

    fused_linear = edge_fm.FusedGateUpLinearLayer(
        "model.layers.0.mlp",
        str(engine_config_path),
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        INTERMEDIATE_SIZE,
    )
    activation = edge_fm.ActivationLayer(str(engine_config_path))
    return fused_linear, activation


def _is_tuned_fused_gate_up_record(record: dict, *, stage: str, shape_sig: str) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "linear"
        and record.get("layer_role") == "fused_gate_up"
        and record.get("stage") == stage
        and record.get("shape_sig") == shape_sig
        and record.get("impl_id") == "cublasLt"
        and record.get("impl_params", {}).get("algo_index") == 2
    )


def _run_correctness_case(seq_len: int, stage: str):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    torch.manual_seed(0)
    x = torch.randn(seq_len, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(seq_len, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    activation_out = torch.empty(seq_len, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    activation_out_efm = tensor_to_edge_fm_tensor(activation_out)

    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, stage)
    activation.forward_silu_and_mul(fused_out_efm, activation_out_efm, 0, stage)
    torch.cuda.synchronize()

    fused_out_torch = edge_fm_tensor_to_torch(fused_out_efm)
    activation_out_torch = edge_fm_tensor_to_torch(activation_out_efm)

    gate_weight = _load_tensor("model.layers.0.mlp.gate_proj.weight", device=device)
    up_weight = _load_tensor("model.layers.0.mlp.up_proj.weight", device=device)
    weight = torch.cat([gate_weight, up_weight], dim=0)

    ref_fused = F.linear(x.float(), weight.float()).to(torch.bfloat16)
    gate_ref, up_ref = ref_fused.split(INTERMEDIATE_SIZE, dim=-1)
    ref_activation = (F.silu(gate_ref.float()) * up_ref.float()).to(torch.bfloat16)

    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(fused_out_torch, ref_fused, rtol=rtol, atol=atol)
    torch.testing.assert_close(activation_out_torch, ref_activation, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("seq_len", "stage"),
    [(1, "Decode")] + [(seq_len, "Prefill") for seq_len in DEFAULT_PREFILL_LENGTHS],
    ids=lambda value: str(value),
)
def test_fused_gate_up_and_activation_correctness(seq_len, stage):
    _run_correctness_case(seq_len, stage)


@pytest.mark.parametrize(
    ("seq_len", "stage", "fused_linear_cap_ms", "activation_cap_ms"),
    [
        (1, "Decode", 1.0, 0.5),
        (512, "Prefill", 0.8, 0.5),
        (1024, "Prefill", 1.2, 0.8),
        (2048, "Prefill", 2.0, 1.2),
    ],
    ids=lambda value: str(value),
)
def test_fused_gate_up_and_activation_performance_smoke(
    seq_len, stage, fused_linear_cap_ms, activation_cap_ms
):
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    x = torch.randn(seq_len, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(seq_len, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    activation_out = torch.empty(seq_len, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    activation_out_efm = tensor_to_edge_fm_tensor(activation_out)

    fused_linear_ms = median_cuda_ms(
        lambda: fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, stage),
        warmup=20,
        iters=100,
    )
    activation_ms = median_cuda_ms(
        lambda: activation.forward_silu_and_mul(fused_out_efm, activation_out_efm, 0, stage),
        warmup=20,
        iters=100,
    )

    assert math.isfinite(fused_linear_ms)
    assert math.isfinite(activation_ms)
    assert fused_linear_ms < fused_linear_cap_ms, (
        f"fused_gate_up {stage} seq_len={seq_len} latency regressed to {fused_linear_ms:.6f} ms"
    )
    assert activation_ms < activation_cap_ms, (
        f"silu_and_mul {stage} seq_len={seq_len} latency regressed to {activation_ms:.6f} ms"
    )


@pytest.mark.parametrize(
    ("seq_len", "stage", "shape_sig"),
    [
        (1, "Decode", FUSED_GATE_UP_DECODE_SHAPE_SIG),
        (512, "Prefill", FUSED_GATE_UP_PREFILL_512_SHAPE_SIG),
        (1024, "Prefill", FUSED_GATE_UP_PREFILL_1024_SHAPE_SIG),
        (2048, "Prefill", FUSED_GATE_UP_PREFILL_2048_SHAPE_SIG),
    ],
    ids=["decode_1", "prefill_512", "prefill_1024", "prefill_2048"],
)
def test_fused_gate_up_tuned_record_matches_baseline_output(seq_len, stage, shape_sig):
    ensure_cuda()
    device = torch_device()

    base_table = load_operator_impl_table()
    current_records = base_table["records"]
    assert any(
        _is_tuned_fused_gate_up_record(
            record, stage=stage.lower(), shape_sig=shape_sig
        )
        for record in current_records
    )
    baseline_records = [
        record
        for record in current_records
        if not _is_tuned_fused_gate_up_record(
            record, stage=stage.lower(), shape_sig=shape_sig
        )
    ]
    baseline_table_path = write_operator_impl_table(baseline_records)

    x = torch.randn(seq_len, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    reset_weight_loader()
    baseline_layer = edge_fm.FusedGateUpLinearLayer(
        "model.layers.0.mlp",
        str(
            make_engine_config(
                QWEN_1P5B_MODEL_PATH,
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=baseline_table_path,
            )
        ),
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        INTERMEDIATE_SIZE,
    )
    y_baseline = torch.empty(
        seq_len, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16
    )
    x_baseline = tensor_to_edge_fm_tensor(x)
    y_baseline_efm = tensor_to_edge_fm_tensor(y_baseline)
    baseline_ms = median_cuda_ms(
        lambda: baseline_layer.forward_fp16_bf16(x_baseline, y_baseline_efm, 0, stage),
        warmup=40,
        iters=250,
    )

    reset_weight_loader()
    tuned_layer = edge_fm.FusedGateUpLinearLayer(
        "model.layers.0.mlp",
        str(
            make_engine_config(
                QWEN_1P5B_MODEL_PATH,
                device_id=DEFAULT_DEVICE_ID,
                operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
            )
        ),
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        INTERMEDIATE_SIZE,
    )
    y_tuned = torch.empty(seq_len, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    x_tuned = tensor_to_edge_fm_tensor(x)
    y_tuned_efm = tensor_to_edge_fm_tensor(y_tuned)
    tuned_ms = median_cuda_ms(
        lambda: tuned_layer.forward_fp16_bf16(x_tuned, y_tuned_efm, 0, stage),
        warmup=40,
        iters=250,
    )

    y_baseline_torch = edge_fm_tensor_to_torch(y_baseline_efm)
    y_tuned_torch = edge_fm_tensor_to_torch(y_tuned_efm)
    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(y_tuned_torch, y_baseline_torch, rtol=rtol, atol=atol)
    assert math.isfinite(baseline_ms)
    assert math.isfinite(tuned_ms)
