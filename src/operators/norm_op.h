#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace edge_fm {

struct RMSNormOpContext {
    uint32_t batch_size = 0;
    uint32_t hidden_size = 0;
    float eps = 1e-6f;
    DType dtype = DType::Float16;
};

using RMSNormForwardFn = void (*)(
    const RMSNormOpContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    cudaStream_t stream);

using FusedAddRMSNormForwardFn = void (*)(
    const RMSNormOpContext& ctx,
    Tensor& inout,
    Tensor& residual,
    const Tensor& weight,
    cudaStream_t stream);

class NormOp {
public:
    virtual ~NormOp() = default;

    virtual std::string impl_id() const = 0;
    virtual bool supports(const RMSNormOpContext& ctx) const = 0;
    virtual RMSNormForwardFn rms_norm_forward_fn() const = 0;
    virtual FusedAddRMSNormForwardFn fused_add_rms_norm_forward_fn() const = 0;
    virtual void rms_norm_forward(
        const RMSNormOpContext& ctx,
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        cudaStream_t stream) = 0;
    virtual void fused_add_rms_norm_forward(
        const RMSNormOpContext& ctx,
        Tensor& inout,
        Tensor& residual,
        const Tensor& weight,
        cudaStream_t stream) = 0;
};

class NormOpRegistry {
public:
    static NormOpRegistry& instance();

    NormOp* find_impl_by_id(const std::string& impl_id) const;
    NormOp* default_impl(const RMSNormOpContext& ctx) const;

private:
    NormOpRegistry();

    std::vector<std::unique_ptr<NormOp>> impls_;
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
