import json
import math

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._test_utils import (
    CUDA_HW_PROFILE,
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


def _make_layers(*, hw_profile: str = CUDA_HW_PROFILE):
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
        hw_profile=hw_profile,
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


def _current_sm() -> int:
    major, minor = torch.cuda.get_device_capability(DEFAULT_DEVICE_ID)
    return major * 10 + minor


def _prefill_swiglu_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    rtol, atol = dtype_tolerances(dtype)
    if dtype == torch.bfloat16:
        return max(rtol, 2e-2), max(atol, 6.25e-2)
    return rtol, atol


def _is_tuned_fused_gate_up_record(record: dict, *, stage: str, shape_sig: str) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == CUDA_HW_PROFILE
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
    activation.forward_silu_and_mul_up_gate(fused_out_efm, activation_out_efm, 0, stage)
    torch.cuda.synchronize()

    fused_out_torch = edge_fm_tensor_to_torch(fused_out_efm)
    activation_out_torch = edge_fm_tensor_to_torch(activation_out_efm)

    gate_weight = _load_tensor("model.layers.0.mlp.gate_proj.weight", device=device)
    up_weight = _load_tensor("model.layers.0.mlp.up_proj.weight", device=device)
    weight = torch.cat([up_weight, gate_weight], dim=0)

    ref_fused = F.linear(x.float(), weight.float()).to(torch.bfloat16)
    up_ref, gate_ref = ref_fused.split(INTERMEDIATE_SIZE, dim=-1)
    ref_activation = (F.silu(gate_ref.float()) * up_ref.float()).to(torch.bfloat16)

    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
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
    "seq_len",
    [512, 1024, 2048],
    ids=lambda value: f"prefill_{value}",
)
def test_prefill_fused_gate_up_swiglu_matches_two_stage_output(seq_len, monkeypatch):
    ensure_cuda()
    monkeypatch.setenv("EDGE_FM_PREFILL_SWIGLU_FUSION", "1")
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    torch.manual_seed(seq_len)
    x = torch.randn(seq_len, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(seq_len, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    two_stage_out = torch.empty(seq_len, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fused_prefill_out = torch.empty(seq_len, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    two_stage_out_efm = tensor_to_edge_fm_tensor(two_stage_out)
    fused_prefill_out_efm = tensor_to_edge_fm_tensor(fused_prefill_out)

    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, "Prefill")
    activation.forward_silu_and_mul_up_gate(fused_out_efm, two_stage_out_efm, 0, "Prefill")

    assert fused_linear.try_forward_prefill_swiglu_fused(x_efm, fused_prefill_out_efm, 0)
    torch.cuda.synchronize()

    two_stage_out_torch = edge_fm_tensor_to_torch(two_stage_out_efm)
    fused_prefill_out_torch = edge_fm_tensor_to_torch(fused_prefill_out_efm)
    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
    torch.testing.assert_close(fused_prefill_out_torch, two_stage_out_torch, rtol=rtol, atol=atol)


def test_prefill_fused_gate_up_swiglu_disabled_by_env_falls_back(monkeypatch):
    ensure_cuda()
    monkeypatch.setenv("EDGE_FM_PREFILL_SWIGLU_FUSION", "0")
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    torch.manual_seed(0)
    x = torch.randn(512, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(512, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    reference_out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fallback_out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    reference_out_efm = tensor_to_edge_fm_tensor(reference_out)
    fallback_out_efm = tensor_to_edge_fm_tensor(fallback_out)

    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, "Prefill")
    activation.forward_silu_and_mul_up_gate(fused_out_efm, reference_out_efm, 0, "Prefill")

    assert not fused_linear.try_forward_prefill_swiglu_fused(x_efm, fallback_out_efm, 0)

    fallback_gate_up = torch.empty(512, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fallback_gate_up_efm = tensor_to_edge_fm_tensor(fallback_gate_up)
    fused_linear.forward_fp16_bf16(x_efm, fallback_gate_up_efm, 0, "Prefill")
    activation.forward_silu_and_mul_up_gate(fallback_gate_up_efm, fallback_out_efm, 0, "Prefill")
    torch.cuda.synchronize()

    reference_out_torch = edge_fm_tensor_to_torch(reference_out_efm)
    fallback_out_torch = edge_fm_tensor_to_torch(fallback_out_efm)
    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
    torch.testing.assert_close(fallback_out_torch, reference_out_torch, rtol=rtol, atol=atol)


def test_prefill_fused_gate_up_swiglu_default_off(monkeypatch):
    ensure_cuda()
    monkeypatch.delenv("EDGE_FM_PREFILL_SWIGLU_FUSION", raising=False)
    reset_weight_loader()

    device = torch_device()
    fused_linear, _ = _make_layers()

    x = torch.randn(512, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    assert not fused_linear.try_forward_prefill_swiglu_fused(
        tensor_to_edge_fm_tensor(x), tensor_to_edge_fm_tensor(out), 0
    )


def test_fused_gate_up_weight_info_exposes_edgefm_resident_layout():
    ensure_cuda()
    reset_weight_loader()

    fused_linear, _ = _make_layers()

    info = json.loads(fused_linear.debug_weight_tensor_info("Prefill"))

    assert info["has_weight"]
    assert info["shape"] == [2 * INTERMEDIATE_SIZE, HIDDEN_SIZE]
    assert info["dtype_name"] == "bfloat16"
    assert info["layout"] == "edgefm_fused_gate_up_up_gate_out_in"


def test_prefill_fused_gate_up_swiglu_unsupported_shape_returns_false_without_cache_poisoning(monkeypatch):
    ensure_cuda()
    monkeypatch.setenv("EDGE_FM_PREFILL_SWIGLU_FUSION", "1")
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    torch.manual_seed(0)
    x = torch.randn(512, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(512, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    reference_out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fused_prefill_out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    reference_out_efm = tensor_to_edge_fm_tensor(reference_out)
    fused_prefill_out_efm = tensor_to_edge_fm_tensor(fused_prefill_out)

    assert fused_linear.try_forward_prefill_swiglu_fused(x_efm, fused_prefill_out_efm, 0)
    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, "Prefill")
    activation.forward_silu_and_mul_up_gate(fused_out_efm, reference_out_efm, 0, "Prefill")
    torch.cuda.synchronize()

    reference_out_torch = edge_fm_tensor_to_torch(reference_out_efm)
    fused_prefill_out_torch = edge_fm_tensor_to_torch(fused_prefill_out_efm)
    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
    torch.testing.assert_close(fused_prefill_out_torch, reference_out_torch, rtol=rtol, atol=atol)

    small_x = torch.randn(32, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    small_out = torch.empty(32, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    small_x_efm = tensor_to_edge_fm_tensor(small_x)
    small_out_efm = tensor_to_edge_fm_tensor(small_out)
    assert not fused_linear.try_forward_prefill_swiglu_fused(small_x_efm, small_out_efm, 0)

    second_prefill_out = torch.empty(512, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    second_prefill_out_efm = tensor_to_edge_fm_tensor(second_prefill_out)
    assert fused_linear.try_forward_prefill_swiglu_fused(x_efm, second_prefill_out_efm, 0)
    torch.cuda.synchronize()

    second_prefill_out_torch = edge_fm_tensor_to_torch(second_prefill_out_efm)
    torch.testing.assert_close(second_prefill_out_torch, reference_out_torch, rtol=rtol, atol=atol)


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
    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
    torch.testing.assert_close(y_tuned_torch, y_baseline_torch, rtol=rtol, atol=atol)
    assert math.isfinite(baseline_ms)
    assert math.isfinite(tuned_ms)


def test_decode_fused_gate_up_swiglu_matches_two_stage_output():
    ensure_cuda()
    reset_weight_loader()

    device = torch_device()
    fused_linear, activation = _make_layers()

    torch.manual_seed(0)
    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(1, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    two_stage_out = torch.empty(1, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fused_decode_out = torch.empty(1, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    two_stage_out_efm = tensor_to_edge_fm_tensor(two_stage_out)
    fused_decode_out_efm = tensor_to_edge_fm_tensor(fused_decode_out)

    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, "Decode")
    activation.forward_silu_and_mul_up_gate(fused_out_efm, two_stage_out_efm, 0, "Decode")

    if not fused_linear.try_forward_decode_swiglu_fused(x_efm, fused_decode_out_efm, 0):
        pytest.skip("decode fused SwiGLU fast path is unavailable on this device/config")

    torch.cuda.synchronize()

    two_stage_out_torch = edge_fm_tensor_to_torch(two_stage_out_efm)
    fused_decode_out_torch = edge_fm_tensor_to_torch(fused_decode_out_efm)
    rtol, atol = _prefill_swiglu_tolerances(torch.bfloat16)
    torch.testing.assert_close(fused_decode_out_torch, two_stage_out_torch, rtol=rtol, atol=atol)


def test_decode_fused_gate_up_swiglu_disabled_by_default_on_non_sm80_ampere():
    ensure_cuda()
    sm = _current_sm()
    if sm == 80 or sm < 80 or sm >= 90:
        pytest.skip("SM80-only decode fused SwiGLU guard is only relevant to non-SM80 Ampere GPUs")

    reset_weight_loader()

    device = torch_device()
    fused_linear, _ = _make_layers(hw_profile=f"cuda_sm{sm}")

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    fused_out = torch.empty(1, 2 * INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    fused_decode_out = torch.empty(1, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)

    x_efm = tensor_to_edge_fm_tensor(x)
    fused_out_efm = tensor_to_edge_fm_tensor(fused_out)
    fused_decode_out_efm = tensor_to_edge_fm_tensor(fused_decode_out)

    fused_linear.forward_fp16_bf16(x_efm, fused_out_efm, 0, "Decode")

    assert not fused_linear.try_forward_decode_swiglu_fused(x_efm, fused_decode_out_efm, 0)
