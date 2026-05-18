import math
import sys
from pathlib import Path

import pytest
import torch
import torch.testing

from ._test_utils import (
    CUDA_HW_PROFILE,
    DEFAULT_DEVICE_ID,
    DEFAULT_PREFILL_LENGTHS,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    dtype_tolerances,
    edge_fm,
    ensure_cuda,
    make_engine_config,
    median_cuda_ms,
    tensor_to_edge_fm_tensor,
    torch_device,
    write_operator_impl_table,
)

_SHIM_DIR = Path(__file__).resolve().parent / "_vendor_shims"
if str(_SHIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SHIM_DIR))

NUM_QO_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128


def _make_attention_layer():
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH,
    )
    return edge_fm.AttentionLayer(str(engine_config_path))


def _make_attention_layer_with_table(records: list[dict]):
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=write_operator_impl_table(records),
    )
    return edge_fm.AttentionLayer(str(engine_config_path))


@pytest.mark.parametrize("seq_len", DEFAULT_PREFILL_LENGTHS, ids=lambda value: f"seq{value}")
def test_attention_prefill_correctness(seq_len):
    ensure_cuda()
    device = torch_device()
    layer = _make_attention_layer()

    torch.manual_seed(0)
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(out),
        True,
    )
    rerun_out = torch.empty_like(q)
    layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(rerun_out),
        True,
    )
    torch.cuda.synchronize()

    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(out, rerun_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("seq_len", "max_median_ms"),
    [(512, 0.80), (1024, 1.50), (2048, 3.00)],
    ids=lambda value: f"seq{value}" if isinstance(value, int) else str(value),
)
def test_attention_prefill_performance_smoke(seq_len, max_median_ms):
    ensure_cuda()
    device = torch_device()
    layer = _make_attention_layer()

    torch.manual_seed(0)
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    q_efm = tensor_to_edge_fm_tensor(q)
    k_efm = tensor_to_edge_fm_tensor(k)
    v_efm = tensor_to_edge_fm_tensor(v)
    out_efm = tensor_to_edge_fm_tensor(out)

    median_ms = median_cuda_ms(
        lambda: layer.forward_prefill(q_efm, k_efm, v_efm, out_efm, True),
        warmup=20,
        iters=120,
    )

    assert math.isfinite(median_ms)
    assert median_ms < max_median_ms, (
        f"prefill attention seq_len={seq_len} latency regressed to {median_ms:.6f} ms"
    )


def test_attention_prefill_prerotate_impl_matches_flashinfer_rope():
    ensure_cuda()
    device = torch_device()
    seq_len = 64

    torch.manual_seed(17)
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

    baseline_layer = _make_attention_layer()
    baseline_out = torch.empty_like(q)
    baseline_layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(baseline_out),
        True,
    )

    candidate_layer = _make_attention_layer_with_table([
        {
            "model_name": "qwen2_5",
            "hw_profile": CUDA_HW_PROFILE,
            "op_kind": "attention",
            "layer_role": "",
            "op_name": "",
            "stage": "prefill",
            "shape_sig": "num_qo_heads=12|num_kv_heads=2|head_dim=128",
            "impl_id": "flashinfer_attention_prefill_prerotate",
            "impl_params": {
                "prefill_cta_tile_q": 64,
                "prefill_max_mma_kv_cap": 2,
            },
        }
    ])
    candidate_out = torch.empty_like(q)
    candidate_layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(candidate_out),
        True,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(candidate_out, baseline_out, rtol=8e-2, atol=8e-2)


def test_attention_prefill_accepts_strided_qkv_views():
    ensure_cuda()
    device = torch_device()
    seq_len = 64
    q_dim = NUM_QO_HEADS * HEAD_DIM
    k_dim = NUM_KV_HEADS * HEAD_DIM
    v_dim = NUM_KV_HEADS * HEAD_DIM
    qkv_total = q_dim + k_dim + v_dim
    dtype_size = 2

    torch.manual_seed(23)
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

    layer = _make_attention_layer()
    baseline_out = torch.empty_like(q)
    layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(baseline_out),
        True,
    )

    packed_qkv = torch.empty(seq_len, qkv_total, device=device, dtype=torch.bfloat16)
    packed_qkv[:, :q_dim].copy_(q.reshape(seq_len, q_dim))
    packed_qkv[:, q_dim:q_dim + k_dim].copy_(k.reshape(seq_len, k_dim))
    packed_qkv[:, q_dim + k_dim:].copy_(v.reshape(seq_len, v_dim))

    base_ptr = packed_qkv.data_ptr()
    q_view = edge_fm.Tensor(
        base_ptr,
        [seq_len, NUM_QO_HEADS, HEAD_DIM],
        edge_fm.DType.BFloat16,
        edge_fm.Device.GPU,
        DEFAULT_DEVICE_ID,
        False,
    )
    k_view = edge_fm.Tensor(
        base_ptr + q_dim * dtype_size,
        [seq_len, NUM_KV_HEADS, HEAD_DIM],
        edge_fm.DType.BFloat16,
        edge_fm.Device.GPU,
        DEFAULT_DEVICE_ID,
        False,
    )
    v_view = edge_fm.Tensor(
        base_ptr + (q_dim + k_dim) * dtype_size,
        [seq_len, NUM_KV_HEADS, HEAD_DIM],
        edge_fm.DType.BFloat16,
        edge_fm.Device.GPU,
        DEFAULT_DEVICE_ID,
        False,
    )
    strided_out = torch.empty_like(q)
    layer.forward_prefill(
        q_view,
        k_view,
        v_view,
        tensor_to_edge_fm_tensor(strided_out),
        True,
        q_stride_n=qkv_total,
        q_stride_h=HEAD_DIM,
        kv_stride_n=qkv_total,
        kv_stride_h=HEAD_DIM,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(strided_out, baseline_out, rtol=8e-2, atol=8e-2)


def test_attention_prefill_accepts_k_only_strided_view():
    ensure_cuda()
    device = torch_device()
    seq_len = 64
    q_dim = NUM_QO_HEADS * HEAD_DIM
    k_dim = NUM_KV_HEADS * HEAD_DIM
    v_dim = NUM_KV_HEADS * HEAD_DIM
    qkv_total = q_dim + k_dim + v_dim
    dtype_size = 2

    torch.manual_seed(29)
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

    layer = _make_attention_layer()
    baseline_out = torch.empty_like(q)
    layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(baseline_out),
        True,
    )

    packed_qkv = torch.empty(seq_len, qkv_total, device=device, dtype=torch.bfloat16)
    packed_qkv[:, :q_dim].copy_(q.reshape(seq_len, q_dim))
    packed_qkv[:, q_dim:q_dim + k_dim].copy_(k.reshape(seq_len, k_dim))
    packed_qkv[:, q_dim + k_dim:].copy_(v.reshape(seq_len, v_dim))

    base_ptr = packed_qkv.data_ptr()
    q_view = edge_fm.Tensor(
        base_ptr,
        [seq_len, NUM_QO_HEADS, HEAD_DIM],
        edge_fm.DType.BFloat16,
        edge_fm.Device.GPU,
        DEFAULT_DEVICE_ID,
        False,
    )
    k_view = edge_fm.Tensor(
        base_ptr + q_dim * dtype_size,
        [seq_len, NUM_KV_HEADS, HEAD_DIM],
        edge_fm.DType.BFloat16,
        edge_fm.Device.GPU,
        DEFAULT_DEVICE_ID,
        False,
    )
    k_only_out = torch.empty_like(q)
    layer.forward_prefill(
        q_view,
        k_view,
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(k_only_out),
        True,
        q_stride_n=qkv_total,
        q_stride_h=HEAD_DIM,
        k_stride_n=qkv_total,
        k_stride_h=HEAD_DIM,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(k_only_out, baseline_out, rtol=8e-2, atol=8e-2)
