#pragma once

#include <edge-fm/core.h>

#include <cstdint>

#include <cuda_runtime.h>

namespace edge_fm {

struct RMSNormOpContext {
    uint32_t batch_size = 0;
    uint32_t hidden_size = 0;
    float eps = 1e-6f;
    DType dtype = DType::Float16;
};

void rms_norm_forward(
    const RMSNormOpContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    cudaStream_t stream);

void fused_add_rms_norm_forward(
    const RMSNormOpContext& ctx,
    Tensor& inout,
    Tensor& residual,
    const Tensor& weight,
    cudaStream_t stream);

} // namespace edge_fm
