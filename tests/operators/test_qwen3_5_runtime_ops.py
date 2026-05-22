import math

import pytest
import torch
import torch.nn.functional as F

from ._test_utils import (
    edge_fm,
    edge_fm_tensor_to_torch,
    ensure_cuda,
    tensor_to_edge_fm_tensor,
    torch_device,
)


def _qwen3_5_rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def _gated_rmsnorm_ref(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out * weight.float()
    out = out * F.silu(gate.float())
    return out.to(x.dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _l2norm_transformers_dtype(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Transformers Qwen3.5 applies this before casting q/k to float32, so
    # bf16 inputs keep bf16 square/sum/rsqrt/mul semantics.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _gated_delta_step_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _l2norm(query.float()) * (1.0 / math.sqrt(query.shape[-1]))
    key = _l2norm(key.float())
    value = value.float()
    g = g.float().exp()
    beta = beta.float()
    next_state = state.float().clone()
    out = torch.empty(query.shape[0], value.shape[-1], device=query.device, dtype=torch.float32)

    for h in range(query.shape[0]):
        next_state[h] = next_state[h] * g[h]
        kv_mem = (next_state[h] * key[h, :, None]).sum(dim=0)
        delta = (value[h] - kv_mem) * beta[h]
        next_state[h] = next_state[h] + key[h, :, None] * delta[None, :]
        out[h] = (next_state[h] * query[h, :, None]).sum(dim=0)

    return out.to(value.dtype), next_state.to(state.dtype)


def _depthwise_conv_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, channels = x.shape
    kernel = weight.shape[-1]
    out = torch.empty_like(x)
    for t in range(seq_len):
        for c in range(channels):
            acc = 0.0
            for j in range(kernel):
                source = t + j - (kernel - 1)
                if source >= 0:
                    val = x[source, c].float()
                else:
                    state_idx = t + 1 + j
                    val = state[c, state_idx].float() if 0 <= state_idx < kernel else torch.tensor(0.0, device=x.device)
                acc = acc + weight[c, 0, j].float() * val
            out[t, c] = F.silu(torch.as_tensor(acc, device=x.device)).to(x.dtype)

    cat = torch.cat([state.T, x], dim=0)
    next_state = cat[-kernel:].T.contiguous().to(state.dtype)
    return out, next_state


def _gated_delta_sequence_ref(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, k_dim, v_dim = state.shape
    key_total = heads * k_dim
    out = torch.empty(mixed_qkv.shape[0], heads * v_dim, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    next_state = state.clone()
    for t in range(mixed_qkv.shape[0]):
        q_raw = mixed_qkv[t, :key_total].reshape(heads, k_dim)
        k_raw = mixed_qkv[t, key_total : 2 * key_total].reshape(heads, k_dim)
        q = _l2norm_transformers_dtype(q_raw).float() * (1.0 / math.sqrt(k_dim))
        k = _l2norm_transformers_dtype(k_raw).float()
        v = mixed_qkv[t, 2 * key_total :].float().reshape(heads, v_dim)
        beta_t = beta[t].float()
        g_t = g[t].float().exp()
        step_out = torch.empty(heads, v_dim, device=mixed_qkv.device, dtype=torch.float32)
        for h in range(heads):
            next_state[h] = next_state[h] * g_t[h]
            kv_mem = (next_state[h] * k[h, :, None]).sum(dim=0)
            delta = (v[h] - kv_mem) * beta_t[h]
            next_state[h] = next_state[h] + k[h, :, None] * delta[None, :]
            step_out[h] = (next_state[h] * q[h, :, None]).sum(dim=0)
        out[t] = step_out.reshape(-1).to(mixed_qkv.dtype)
    return out, next_state


def _partial_interleaved_mrope_ref(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: list[int],
    rope_theta: float,
    rotary_dim: int,
) -> torch.Tensor:
    out = x.clone()
    freq_dim = rotary_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / float(rotary_dim))
    )
    freqs = position_ids.to(device=x.device, dtype=torch.float32)[:, :, None] * inv_freq[None, None, :]
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = torch.arange(offset, min(length, freq_dim), 3, device=x.device)
        if idx.numel() > 0:
            freqs_t[:, idx] = freqs[dim, :, idx]

    emb = torch.cat([freqs_t, freqs_t], dim=-1)
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    x_rot = out[..., :rotary_dim].float()
    x_pass = out[..., rotary_dim:]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    out_rot = (x_rot * cos) + (rotated * sin)
    out = torch.cat([out_rot.to(x.dtype), x_pass], dim=-1)
    return out


def test_qwen3_5_rmsnorm_uses_one_plus_weight():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(10)
    x = torch.randn(5, 33, device=device, dtype=torch.bfloat16)
    weight = torch.randn(33, device=device, dtype=torch.bfloat16) * 0.05
    y = torch.empty_like(x)

    edge_fm.qwen3_5_rmsnorm(
        tensor_to_edge_fm_tensor(x),
        tensor_to_edge_fm_tensor(weight),
        tensor_to_edge_fm_tensor(y),
        1e-6,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(y)), _qwen3_5_rmsnorm_ref(x, weight, 1e-6), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden", [1024, 2048])
def test_qwen3_5_rmsnorm_large_hidden_matches_reference(hidden):
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(101 + hidden)
    x = torch.randn(3, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=device, dtype=torch.bfloat16) * 0.05
    y = torch.empty_like(x)

    edge_fm.qwen3_5_rmsnorm(
        tensor_to_edge_fm_tensor(x),
        tensor_to_edge_fm_tensor(weight),
        tensor_to_edge_fm_tensor(y),
        1e-6,
    )
    torch.cuda.synchronize()

    ref = _qwen3_5_rmsnorm_ref(x, weight, 1e-6)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(y)), ref, rtol=2e-2, atol=2e-2)


def test_qwen3_5_gated_rmsnorm_matches_transformers_reference():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(11)
    x = torch.randn(7, 16, device=device, dtype=torch.bfloat16)
    gate = torch.randn(7, 16, device=device, dtype=torch.bfloat16)
    weight = torch.randn(16, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    edge_fm.qwen3_5_gated_rmsnorm(
        tensor_to_edge_fm_tensor(x),
        tensor_to_edge_fm_tensor(gate),
        tensor_to_edge_fm_tensor(weight),
        tensor_to_edge_fm_tensor(y),
        1e-6,
    )
    torch.cuda.synchronize()

    ref = _gated_rmsnorm_ref(x, gate, weight, 1e-6)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(y)), ref, rtol=2e-2, atol=2e-2)


def test_qwen3_5_add_supports_inplace_lhs_output_alias():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(112)
    lhs = torch.randn(3, 511, device=device, dtype=torch.bfloat16)
    rhs = torch.randn(3, 511, device=device, dtype=torch.bfloat16)
    lhs_inplace = lhs.clone()

    edge_fm.qwen3_5_add(
        tensor_to_edge_fm_tensor(lhs_inplace),
        tensor_to_edge_fm_tensor(rhs),
        tensor_to_edge_fm_tensor(lhs_inplace),
    )
    torch.cuda.synchronize()

    ref = (lhs.float() + rhs.float()).to(lhs.dtype)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(lhs_inplace)), ref, rtol=0, atol=0)


def test_qwen3_5_gated_delta_recurrent_single_step_updates_state():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(12)
    heads, k_dim, v_dim = 3, 8, 5
    query = torch.randn(heads, k_dim, device=device, dtype=torch.float32)
    key = torch.randn(heads, k_dim, device=device, dtype=torch.float32)
    value = torch.randn(heads, v_dim, device=device, dtype=torch.float32)
    g = torch.randn(heads, device=device, dtype=torch.float32) * -0.2
    beta = torch.sigmoid(torch.randn(heads, device=device, dtype=torch.float32))
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.1
    state_ref_input = state.clone()
    out = torch.empty(heads, v_dim, device=device, dtype=torch.float32)

    edge_fm.qwen3_5_gated_delta_recurrent_step(
        tensor_to_edge_fm_tensor(query),
        tensor_to_edge_fm_tensor(key),
        tensor_to_edge_fm_tensor(value),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    torch.cuda.synchronize()

    out_ref, state_ref = _gated_delta_step_ref(query, key, value, g, beta, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), out_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), state_ref, rtol=2e-5, atol=2e-5)


def test_qwen3_5_partial_interleaved_mrope_only_rotates_prefix_dims():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(13)
    seq_len, q_heads, kv_heads, head_dim, rotary_dim = 6, 2, 1, 32, 8
    mrope_section = [1, 1, 2]
    position_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
        device=device,
        dtype=torch.int32,
    )
    q = torch.randn(seq_len, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_ref_input = q.clone()
    k_ref_input = k.clone()

    edge_fm.qwen3_5_apply_partial_interleaved_mrope(
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(k),
        tensor_to_edge_fm_tensor(position_ids),
        mrope_section,
        10000000.0,
        rotary_dim,
    )
    torch.cuda.synchronize()

    q_ref = _partial_interleaved_mrope_ref(q_ref_input, position_ids, mrope_section, 10000000.0, rotary_dim)
    k_ref = _partial_interleaved_mrope_ref(k_ref_input, position_ids, mrope_section, 10000000.0, rotary_dim)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(q)), q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(k)), k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(q[..., rotary_dim:], q_ref_input[..., rotary_dim:])
    torch.testing.assert_close(k[..., rotary_dim:], k_ref_input[..., rotary_dim:])


def test_qwen3_5_split_q_gate_and_sigmoid_gate_match_reference():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(14)
    seq_len, heads, head_dim = 3, 4, 8
    q_proj = torch.randn(seq_len, heads * head_dim * 2, device=device, dtype=torch.bfloat16)
    q = torch.empty(seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    gate = torch.empty(seq_len, heads * head_dim, device=device, dtype=torch.bfloat16)
    attn = torch.randn(seq_len, heads * head_dim, device=device, dtype=torch.bfloat16)
    gated = torch.empty_like(attn)

    edge_fm.qwen3_5_split_q_gate(
        tensor_to_edge_fm_tensor(q_proj),
        tensor_to_edge_fm_tensor(q),
        tensor_to_edge_fm_tensor(gate),
        heads,
        head_dim,
    )
    edge_fm.qwen3_5_mul_sigmoid(
        tensor_to_edge_fm_tensor(attn),
        tensor_to_edge_fm_tensor(gate),
        tensor_to_edge_fm_tensor(gated),
    )
    torch.cuda.synchronize()

    ref = q_proj.view(seq_len, heads, head_dim * 2)
    q_ref, gate_ref = torch.chunk(ref, 2, dim=-1)
    gate_ref = gate_ref.reshape(seq_len, heads * head_dim)
    gated_ref = (attn.float() * torch.sigmoid(gate_ref.float())).to(attn.dtype)

    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(q)), q_ref.contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(gate)), gate_ref, rtol=0, atol=0)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(gated)), gated_ref, rtol=2e-2, atol=2e-2)


def test_qwen3_5_depthwise_conv_updates_state():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(15)
    seq_len, channels, kernel = 5, 7, 4
    x = torch.randn(seq_len, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, kernel, device=device, dtype=torch.bfloat16) * 0.2
    state = torch.randn(channels, kernel, device=device, dtype=torch.bfloat16) * 0.1
    state_ref_input = state.clone()
    out = torch.empty_like(x)

    edge_fm.qwen3_5_depthwise_causal_conv1d(
        tensor_to_edge_fm_tensor(x),
        tensor_to_edge_fm_tensor(weight),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
        True,
    )
    torch.cuda.synchronize()

    ref_out, ref_state = _depthwise_conv_ref(x, weight, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), ref_state, rtol=0, atol=0)


def test_qwen3_5_depthwise_conv_single_token_updates_state():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(151)
    seq_len, channels, kernel = 1, 11, 4
    x = torch.randn(seq_len, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, kernel, device=device, dtype=torch.bfloat16) * 0.2
    state = torch.randn(channels, kernel, device=device, dtype=torch.bfloat16) * 0.1
    state_ref_input = state.clone()
    out = torch.empty_like(x)

    edge_fm.qwen3_5_depthwise_causal_conv1d(
        tensor_to_edge_fm_tensor(x),
        tensor_to_edge_fm_tensor(weight),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
        True,
    )
    torch.cuda.synchronize()

    ref_out, ref_state = _depthwise_conv_ref(x, weight, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), ref_state, rtol=0, atol=0)


def test_qwen3_5_compute_g_beta_and_sequence_delta_match_reference():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(16)
    seq_len, heads, k_dim, v_dim = 4, 3, 5, 6
    mixed = torch.randn(seq_len, heads * (2 * k_dim + v_dim), device=device, dtype=torch.bfloat16)
    a = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16)
    b = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    beta = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.01
    state_ref_input = state.clone()
    out = torch.empty(seq_len, heads * v_dim, device=device, dtype=torch.bfloat16)

    edge_fm.qwen3_5_compute_g_beta(
        tensor_to_edge_fm_tensor(a),
        tensor_to_edge_fm_tensor(b),
        tensor_to_edge_fm_tensor(a_log),
        tensor_to_edge_fm_tensor(dt_bias),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
    )
    edge_fm.qwen3_5_gated_delta_sequence(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    torch.cuda.synchronize()

    g_ref = -torch.exp(a_log.float())[None, :] * F.softplus(a.float() + dt_bias.float()[None, :])
    beta_ref = torch.sigmoid(b.float())
    out_ref, state_ref = _gated_delta_sequence_ref(mixed, g_ref, beta_ref, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(g)), g_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(beta)), beta_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), state_ref, rtol=2e-5, atol=2e-5)


def test_qwen3_5_sequence_from_ab_matches_separate_decode_path():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(161)
    seq_len, heads, k_dim, v_dim = 1, 16, 128, 128
    mixed = torch.randn(seq_len, heads * (2 * k_dim + v_dim), device=device, dtype=torch.bfloat16) * 0.1
    a = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16) * 0.1
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    beta = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.001
    state_fused = state.clone()
    out = torch.empty(seq_len, heads * v_dim, device=device, dtype=torch.bfloat16)
    out_fused = torch.empty_like(out)

    edge_fm.qwen3_5_compute_g_beta(
        tensor_to_edge_fm_tensor(a),
        tensor_to_edge_fm_tensor(b),
        tensor_to_edge_fm_tensor(a_log),
        tensor_to_edge_fm_tensor(dt_bias),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
    )
    edge_fm.qwen3_5_gated_delta_sequence(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    edge_fm.qwen3_5_gated_delta_sequence_from_ab(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(a),
        tensor_to_edge_fm_tensor(b),
        tensor_to_edge_fm_tensor(a_log),
        tensor_to_edge_fm_tensor(dt_bias),
        tensor_to_edge_fm_tensor(state_fused),
        tensor_to_edge_fm_tensor(out_fused),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_fused, out, rtol=0, atol=0)
    torch.testing.assert_close(state_fused, state, rtol=0, atol=0)


def test_qwen3_5_sequence_delta_matches_reference_at_model_geometry():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(17)
    seq_len, heads, k_dim, v_dim = 3, 16, 128, 128
    mixed = torch.randn(seq_len, heads * (2 * k_dim + v_dim), device=device, dtype=torch.bfloat16) * 0.1
    g = -torch.rand(seq_len, heads, device=device, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn(seq_len, heads, device=device, dtype=torch.float32))
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.001
    state_ref_input = state.clone()
    out = torch.empty(seq_len, heads * v_dim, device=device, dtype=torch.bfloat16)

    edge_fm.qwen3_5_gated_delta_sequence(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    torch.cuda.synchronize()

    out_ref, state_ref = _gated_delta_sequence_ref(mixed, g, beta, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), state_ref, rtol=2e-5, atol=2e-5)


def test_qwen3_5_precomputed_sequence_delta_matches_reference_at_model_geometry():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(18)
    seq_len, heads, k_dim, v_dim = 3, 16, 128, 128
    mixed = torch.randn(seq_len, heads * (2 * k_dim + v_dim), device=device, dtype=torch.bfloat16) * 0.1
    g = -torch.rand(seq_len, heads, device=device, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn(seq_len, heads, device=device, dtype=torch.float32))
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.001
    state_ref_input = state.clone()
    out = torch.empty(seq_len, heads * v_dim, device=device, dtype=torch.bfloat16)
    q_norm = torch.empty(seq_len, heads, k_dim, device=device, dtype=torch.float32)
    k_norm = torch.empty(seq_len, heads, k_dim, device=device, dtype=torch.float32)

    edge_fm.qwen3_5_precompute_gated_delta_qk(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(q_norm),
        tensor_to_edge_fm_tensor(k_norm),
    )
    edge_fm.qwen3_5_gated_delta_sequence_precomputed(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(q_norm),
        tensor_to_edge_fm_tensor(k_norm),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    torch.cuda.synchronize()

    key_total = heads * k_dim
    q_raw = mixed[:, :key_total].reshape(seq_len, heads, k_dim)
    k_raw = mixed[:, key_total : 2 * key_total].reshape(seq_len, heads, k_dim)
    q_ref = _l2norm_transformers_dtype(q_raw).float() * (1.0 / math.sqrt(k_dim))
    k_ref = _l2norm_transformers_dtype(k_raw).float()
    out_ref, state_ref = _gated_delta_sequence_ref(mixed, g, beta, state_ref_input)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(q_norm)), q_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(k_norm)), k_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(out)), out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(edge_fm_tensor_to_torch(tensor_to_edge_fm_tensor(state)), state_ref, rtol=2e-5, atol=2e-5)


def test_qwen3_5_precomputed_sequence_accepts_precomputed_decay():
    ensure_cuda()
    device = torch_device()
    torch.manual_seed(19)
    seq_len, heads, k_dim, v_dim = 3, 16, 128, 128
    mixed = torch.randn(seq_len, heads * (2 * k_dim + v_dim), device=device, dtype=torch.bfloat16) * 0.1
    a = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(seq_len, heads, device=device, dtype=torch.bfloat16) * 0.1
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    beta = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    decay = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    beta_decay = torch.empty(seq_len, heads, device=device, dtype=torch.float32)
    state = torch.randn(heads, k_dim, v_dim, device=device, dtype=torch.float32) * 0.001
    state_decay = state.clone()
    out = torch.empty(seq_len, heads * v_dim, device=device, dtype=torch.bfloat16)
    out_decay = torch.empty_like(out)
    q_norm = torch.empty(seq_len, heads, k_dim, device=device, dtype=torch.float32)
    k_norm = torch.empty(seq_len, heads, k_dim, device=device, dtype=torch.float32)

    edge_fm.qwen3_5_compute_g_beta(
        tensor_to_edge_fm_tensor(a),
        tensor_to_edge_fm_tensor(b),
        tensor_to_edge_fm_tensor(a_log),
        tensor_to_edge_fm_tensor(dt_bias),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
    )
    edge_fm.qwen3_5_compute_decay_beta(
        tensor_to_edge_fm_tensor(a),
        tensor_to_edge_fm_tensor(b),
        tensor_to_edge_fm_tensor(a_log),
        tensor_to_edge_fm_tensor(dt_bias),
        tensor_to_edge_fm_tensor(decay),
        tensor_to_edge_fm_tensor(beta_decay),
    )
    edge_fm.qwen3_5_precompute_gated_delta_qk(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(q_norm),
        tensor_to_edge_fm_tensor(k_norm),
    )
    edge_fm.qwen3_5_gated_delta_sequence_precomputed(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(g),
        tensor_to_edge_fm_tensor(beta),
        tensor_to_edge_fm_tensor(q_norm),
        tensor_to_edge_fm_tensor(k_norm),
        tensor_to_edge_fm_tensor(state),
        tensor_to_edge_fm_tensor(out),
    )
    edge_fm.qwen3_5_gated_delta_sequence_precomputed_decay(
        tensor_to_edge_fm_tensor(mixed),
        tensor_to_edge_fm_tensor(decay),
        tensor_to_edge_fm_tensor(beta_decay),
        tensor_to_edge_fm_tensor(q_norm),
        tensor_to_edge_fm_tensor(k_norm),
        tensor_to_edge_fm_tensor(state_decay),
        tensor_to_edge_fm_tensor(out_decay),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(decay, torch.exp(g), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(beta_decay, beta, rtol=0, atol=0)
    torch.testing.assert_close(out_decay, out, rtol=0, atol=0)
    torch.testing.assert_close(state_decay, state, rtol=0, atol=0)
