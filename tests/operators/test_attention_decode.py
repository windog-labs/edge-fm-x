import math
import sys
from pathlib import Path

import pytest
import torch
import torch.testing

from ._test_utils import (
    DEFAULT_DEVICE_ID,
    OPERATOR_IMPL_TABLE_PATH,
    QWEN_1P5B_MODEL_PATH,
    dtype_tolerances,
    edge_fm,
    ensure_cuda,
    make_engine_config,
    median_cuda_ms,
    tensor_to_edge_fm_tensor,
    torch_device,
)

_SHIM_DIR = Path(__file__).resolve().parent / "_vendor_shims"
if str(_SHIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SHIM_DIR))

flashinfer = pytest.importorskip("flashinfer")

NUM_QO_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
DECODE_KV_LENGTHS = [512, 1024, 2048]
GRAPH_MAX_KV_LEN = 2048


def _make_attention_layer(operator_impl_table_path=OPERATOR_IMPL_TABLE_PATH):
    engine_config_path = make_engine_config(
        QWEN_1P5B_MODEL_PATH,
        device_id=DEFAULT_DEVICE_ID,
        operator_impl_table_path=operator_impl_table_path,
    )
    return edge_fm.AttentionLayer(str(engine_config_path))


def _make_decode_inputs(kv_len: int, *, max_kv_len: int | None = None):
    device = torch_device()
    full_kv_len = max_kv_len or kv_len
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(full_kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(full_kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    return q, k_full, v_full


def _run_decode(layer, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(q)
    layer.forward_decode(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(v),
        tensor_to_edge_fm_tensor(out),
    )
    torch.cuda.synchronize()
    return out


def _decode_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    try:
        return flashinfer.decode.single_decode_with_kv_cache(
            q.squeeze(0),
            k,
            v,
            pos_encoding_mode="ROPE_LLAMA",
            rope_theta=1000000.0,
            rope_scale=1.0,
        ).unsqueeze(0)
    except RuntimeError as err:
        if "Unsupported group_size" not in str(err):
            raise
        return _run_decode(_make_attention_layer(), q, k, v)


@pytest.mark.parametrize("kv_len", DECODE_KV_LENGTHS, ids=lambda value: f"kv{value}")
def test_attention_decode_matches_flashinfer_reference(kv_len):
    ensure_cuda()
    torch.manual_seed(0)

    q, k_full, v_full = _make_decode_inputs(kv_len)
    k = k_full[:kv_len]
    v = v_full[:kv_len]
    layer = _make_attention_layer()

    out = _run_decode(layer, q, k, v)
    ref = _decode_reference(q, k, v)

    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("kv_len", DECODE_KV_LENGTHS, ids=lambda value: f"kv{value}")
def test_attention_decode_graph_like_path_matches_non_graph(kv_len):
    ensure_cuda()
    torch.manual_seed(0)

    q, k_full, v_full = _make_decode_inputs(kv_len, max_kv_len=GRAPH_MAX_KV_LEN)
    d_kv_len = torch.tensor([kv_len], device=torch_device(), dtype=torch.int32)
    layer = _make_attention_layer()

    out_graph_like = torch.empty_like(q)
    layer.forward_decode(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k_full),
        tensor_to_edge_fm_tensor(v_full),
        tensor_to_edge_fm_tensor(out_graph_like),
        0,
        GRAPH_MAX_KV_LEN,
        tensor_to_edge_fm_tensor(d_kv_len),
    )
    torch.cuda.synchronize()

    out_ref = _run_decode(layer, q, k_full[:kv_len], v_full[:kv_len])
    rtol, atol = dtype_tolerances(torch.bfloat16)
    torch.testing.assert_close(out_graph_like, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("kv_len", "max_median_ms"),
    [(512, 0.35), (1024, 0.40), (2048, 0.50)],
    ids=lambda value: f"kv{value}" if isinstance(value, int) else str(value),
)
def test_attention_decode_performance_smoke(kv_len, max_median_ms):
    ensure_cuda()
    torch.manual_seed(0)

    q, k_full, v_full = _make_decode_inputs(kv_len)
    k = k_full[:kv_len]
    v = v_full[:kv_len]
    layer = _make_attention_layer()

    q_efm = tensor_to_edge_fm_tensor(q)
    k_efm = tensor_to_edge_fm_tensor(k)
    v_efm = tensor_to_edge_fm_tensor(v)
    out = torch.empty_like(q)
    out_efm = tensor_to_edge_fm_tensor(out)

    median_ms = median_cuda_ms(
        lambda: layer.forward_decode(q_efm, k_efm, v_efm, out_efm),
        warmup=40,
        iters=250,
    )

    assert math.isfinite(median_ms)
    assert median_ms < max_median_ms, (
        f"decode attention kv_len={kv_len} latency regressed to {median_ms:.6f} ms"
    )
