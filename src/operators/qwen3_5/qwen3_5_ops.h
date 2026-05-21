#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace edge_fm {

void qwen3_5_rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps,
    cudaStream_t stream = nullptr);

void qwen3_5_gated_rmsnorm_forward(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    Tensor& output,
    float eps,
    cudaStream_t stream = nullptr);

void qwen3_5_add_forward(
    const Tensor& lhs,
    const Tensor& rhs,
    Tensor& output,
    cudaStream_t stream = nullptr);

void qwen3_5_mul_sigmoid_forward(
    const Tensor& input,
    const Tensor& gate,
    Tensor& output,
    cudaStream_t stream = nullptr);

void qwen3_5_split_q_gate_forward(
    const Tensor& q_proj,
    Tensor& query,
    Tensor& gate,
    int32_t num_heads,
    int32_t head_dim,
    cudaStream_t stream = nullptr);

void qwen3_5_depthwise_causal_conv1d_forward(
    const Tensor& input,
    const Tensor& weight,
    Tensor& conv_state,
    Tensor& output,
    bool update_state,
    cudaStream_t stream = nullptr);

void qwen3_5_compute_g_beta_forward(
    const Tensor& a,
    const Tensor& b,
    const Tensor& a_log,
    const Tensor& dt_bias,
    Tensor& g,
    Tensor& beta,
    cudaStream_t stream = nullptr);

void qwen3_5_gated_delta_recurrent_step(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream = nullptr);

void qwen3_5_gated_delta_sequence_forward(
    const Tensor& mixed_qkv,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream = nullptr);

void qwen3_5_apply_partial_interleaved_mrope(
    Tensor& query,
    Tensor& key,
    const Tensor& position_ids,
    const std::vector<int32_t>& mrope_section,
    float rope_theta,
    int32_t rotary_dim,
    cudaStream_t stream = nullptr);

} // namespace edge_fm
