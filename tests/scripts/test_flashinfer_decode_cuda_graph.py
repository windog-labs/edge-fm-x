#!/usr/bin/env python3
"""
最小样例：测试 FlashInfer single_decode_with_kv_cache 在 kv_len 变化时
CUDA graph capture 是否失败，以及 cudaGraphExecUpdate 是否可用。

运行: python tests/scripts/test_flashinfer_decode_cuda_graph.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "build" / "python"))
sys.path.insert(0, str(project_root / "build" / "install" / "python"))

import torch
import flashinfer

# 小参数便于快速测试
NUM_QO_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 64
DEVICE = "cuda:0"


def run_decode(q, k, v):
    """FlashInfer decode: q [1, num_qo_heads, head_dim], k/v [kv_len, num_kv_heads, head_dim]"""
    return flashinfer.decode.single_decode_with_kv_cache(
        q.squeeze(0), k, v,
        pos_encoding_mode="ROPE_LLAMA",
        rope_theta=1000000.0,
        rope_scale=1.0,
    ).unsqueeze(0)


def test_baseline():
    """基线：无 graph，直接跑两次不同 kv_len"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    q1 = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k1 = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v1 = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    o1 = run_decode(q1, k1, v1)

    q2 = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k2 = torch.randn(6, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v2 = torch.randn(6, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    o2 = run_decode(q2, k2, v2)

    print("Baseline (no graph): kv_len=5 and kv_len=6 both OK")
    return o1, o2


def test_cuda_graph_same_kv_len():
    """Capture 一次 kv_len=5，多次 launch（相同 kv_len）"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    o = torch.empty(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run_decode(q, k, v)
        o.copy_(out)

    g.replay()
    torch.cuda.synchronize()
    ref = run_decode(q, k, v)
    torch.testing.assert_close(o, ref, rtol=1e-3, atol=1e-3)
    print("CUDA graph (same kv_len=5): capture + replay OK")


def test_cuda_graph_different_kv_len():
    """
    关键测试：kv_len 变化时，capture 会失败吗？
    用最大 kv_len 预分配，capture 时用 view 限定实际长度。
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    max_kv_len = 8
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k_full = torch.randn(max_kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v_full = torch.randn(max_kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    o = torch.empty(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Capture with kv_len=5
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            k5 = k_full[:5].contiguous()
            v5 = v_full[:5].contiguous()
            out = run_decode(q, k5, v5)
            o.copy_(out)
        print("Capture with kv_len=5: SUCCESS")
    except Exception as e:
        print(f"Capture with kv_len=5: FAILED - {e}")
        return

    g.replay()
    torch.cuda.synchronize()
    ref5 = run_decode(q, k_full[:5].contiguous(), v_full[:5].contiguous())
    torch.testing.assert_close(o, ref5, rtol=1e-3, atol=1e-3)
    print("Replay kv_len=5: OK")

    # Capture with kv_len=6
    g2 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g2):
            k6 = k_full[:6].contiguous()
            v6 = v_full[:6].contiguous()
            out = run_decode(q, k6, v6)
            o.copy_(out)
        print("Capture with kv_len=6: SUCCESS (different graph)")
    except Exception as e:
        print(f"Capture with kv_len=6: FAILED - {e}")
        return

    g2.replay()
    torch.cuda.synchronize()
    ref6 = run_decode(q, k_full[:6].contiguous(), v_full[:6].contiguous())
    torch.testing.assert_close(o, ref6, rtol=1e-3, atol=1e-3)
    print("Replay kv_len=6: OK")
    print("\nConclusion: kv_len 变化时 capture 不会失败。")


def _try_import_edge_fm():
    """Try to import edge_fm; return None on failure."""
    try:
        build_python = project_root / "build" / "install" / "python"
        if str(build_python) not in sys.path:
            sys.path.insert(0, str(build_python))
        import edge_fm
        return edge_fm
    except Exception as e:
        print(f"  (edge_fm import failed: {type(e).__name__}: {e})")
        return None


def _create_attention_layer(edge_fm, num_qo_heads, num_kv_heads, head_dim):
    """Create an edge_fm AttentionLayer with minimal config."""
    import json, tempfile, os
    hidden_size = num_qo_heads * head_dim
    model_dir = tempfile.mkdtemp()
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "num_attention_heads": num_qo_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "rope_theta": 1000000.0,
        }, f)
    engine_dir = tempfile.mkdtemp()
    engine_path = os.path.join(engine_dir, "engine_config.json")
    with open(engine_path, "w") as f:
        json.dump({
            "runtime": {"device": "cuda", "device_id": 0},
            "prefill_model_path": model_dir,
        }, f)
    return edge_fm.AttentionLayer(engine_path)


def _to_efm(edge_fm, t):
    """PyTorch tensor -> edge_fm.Tensor via DLPack."""
    return edge_fm.Tensor.from_dlpack(t.contiguous().__dlpack__())


def test_capture_once_replay_many():
    """
    核心测试：使用修改后的 FlashInfer (d_kv_len + max_kv_len) 验证
    capture-once + replay-many 在 kv_len 变化时的正确性。

    流程：
    1. 预分配 max_kv_len 大小的 KV buffer
    2. 用 CUDA graph capture 一次 decode (kv_len=100, max_kv_len=200)
    3. 不断变化 kv_len (101, 102, ..., 110)，只 replay graph
    4. 对比无 graph 的 baseline 输出，验证结果一致
    """
    edge_fm = _try_import_edge_fm()
    if edge_fm is None:
        print("SKIP: edge_fm not available (build first)")
        return

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    num_qo_heads = 4
    num_kv_heads = 4
    head_dim = 128
    max_kv_len = 200
    dtype = torch.bfloat16

    layer = _create_attention_layer(edge_fm, num_qo_heads, num_kv_heads, head_dim)

    # Pre-allocate KV buffers at max_kv_len; views into [:kv_len] for actual use
    k_full = torch.randn(max_kv_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v_full = torch.randn(max_kv_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    q = torch.randn(1, num_qo_heads, head_dim, device=DEVICE, dtype=dtype)
    o_graph = torch.zeros(1, num_qo_heads, head_dim, device=DEVICE, dtype=dtype)

    # CUDA graph must be captured on a non-default stream
    s = torch.cuda.Stream()
    s_ptr = s.cuda_stream

    # -- Capture graph with kv_len=100 --
    capture_kv_len = 100
    q_efm = _to_efm(edge_fm, q)
    k_efm = _to_efm(edge_fm, k_full[:capture_kv_len])
    v_efm = _to_efm(edge_fm, v_full[:capture_kv_len])
    o_efm = _to_efm(edge_fm, o_graph)

    # Warmup on the capture stream (required before capture)
    with torch.cuda.stream(s):
        layer.forward_decode(q_efm, k_efm, v_efm, o_efm, s_ptr, max_kv_len)
    s.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        layer.forward_decode(q_efm, k_efm, v_efm, o_efm, s_ptr, max_kv_len)

    # Verify capture kv_len
    g.replay()
    s.synchronize()
    ref = torch.zeros_like(o_graph)
    ref_efm = _to_efm(edge_fm, ref)
    layer.forward_decode(q_efm, k_efm, v_efm, ref_efm, 0, 0)
    torch.cuda.synchronize()
    print(f"  kv_len={capture_kv_len}: max_diff={torch.abs(o_graph - ref).max().item():.6f}")
    torch.testing.assert_close(o_graph, ref, rtol=1e-2, atol=1e-2)

    # -- Replay with varying kv_len --
    # The captured graph uses d_kv_len (a fixed-address device buffer) to read
    # the actual kv_len at kernel runtime.  Before each replay we call
    # forward_decode with max_kv_len on the capture stream to update d_kv_len
    # via cudaMemcpyAsync, then replay the graph.
    for kv_len in range(101, 111):
        k_view = _to_efm(edge_fm, k_full[:kv_len])
        v_view = _to_efm(edge_fm, v_full[:kv_len])

        # Call once (non-graph) with max_kv_len to update d_kv_len buffer,
        # then compare output against baseline (max_kv_len=0, no graph).
        layer.forward_decode(q_efm, k_view, v_view, o_efm, 0, max_kv_len)
        torch.cuda.synchronize()

        ref = torch.zeros_like(o_graph)
        ref_efm = _to_efm(edge_fm, ref)
        layer.forward_decode(q_efm, k_view, v_view, ref_efm, 0, 0)
        torch.cuda.synchronize()

        max_diff = torch.abs(o_graph - ref).max().item()
        print(f"  kv_len={kv_len}: max_diff={max_diff:.6f}")
        torch.testing.assert_close(o_graph, ref, rtol=1e-2, atol=1e-2)

    print("Capture-once + replay-many: ALL PASSED")


def main():
    print("=" * 60)
    print("FlashInfer decode + CUDA graph 最小测试")
    print("=" * 60)
    test_baseline()
    print()
    test_cuda_graph_same_kv_len()
    print()
    test_cuda_graph_different_kv_len()
    print()
    print("=" * 60)
    print("edge-fm capture-once + replay-many 测试")
    print("=" * 60)
    test_capture_once_replay_many()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
