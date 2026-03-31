#pragma once

#include <edge-fm/core.h>

#include <cstdint>

#include <cuda_runtime.h>

namespace edge_fm {

enum class AttentionPosEncoding {
    kNone,
    kRoPELlama,
};

struct AttentionOpContext {
    uint32_t num_qo_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    float rope_scale = 1.0f;
    float rope_theta = 1000000.0f;
    DType dtype = DType::Float16;
    AttentionPosEncoding pos_encoding = AttentionPosEncoding::kRoPELlama;
    int32_t device_id = 0;
};

void attention_forward_prefill(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream);

void attention_forward_decode(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len);

} // namespace edge_fm
