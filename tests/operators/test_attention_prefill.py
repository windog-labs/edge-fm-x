import math
import sys
from pathlib import Path

import pytest
import torch
import torch.testing

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    DEFAULT_PREFILL_LENGTHS,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    dtype_tolerances,
    edge_fm,
    ensure_cuda,
    load_operator_impl_table,
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


def _is_tuned_decode_attention_record(record: dict) -> bool:
    return (
        record.get("model_name") == "qwen2_5"
        and record.get("hw_profile") == "cuda_sm80"
        and record.get("op_kind") == "attention"
        and record.get("stage") == "decode"
        and record.get("impl_id") == "flashinfer_attention_decode_sm80_tuned"
    )


def _make_baseline_prefill_layer():
    base_table = load_operator_impl_table()
    baseline_records = [
        record for record in base_table["records"] if not _is_tuned_decode_attention_record(record)
    ]
    baseline_table_path = write_operator_impl_table(baseline_records)
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=baseline_table_path,
    )
    return edge_fm.AttentionLayer(str(engine_config_path))


@pytest.mark.parametrize("seq_len", DEFAULT_PREFILL_LENGTHS, ids=lambda value: f"seq{value}")
def test_attention_prefill_correctness(seq_len):
    ensure_cuda()
    device = torch_device()
    layer = _make_attention_layer()
    baseline_layer = _make_baseline_prefill_layer()

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
    baseline_out = torch.empty_like(q)
    baseline_layer.forward_prefill(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(baseline_out),
        True,
    )
    torch.cuda.synchronize()

    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(out, baseline_out, rtol=rtol, atol=atol)


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
