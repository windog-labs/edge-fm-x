#pragma once

#include <edge-fm/core.h>

#include <cstdint>

#include <cuda_runtime.h>

namespace edge_fm {

enum class ActivationKind {
    kSilu,
};

struct ActivationOpContext {
    int64_t batch_size = 0;
    int64_t hidden_size = 0;
    DType dtype = DType::Float16;
    ActivationKind kind = ActivationKind::kSilu;
};

void activation_act_and_mul(
    const ActivationOpContext& ctx,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream);

} // namespace edge_fm
