#include "operators/norm_op.h"

#include "utils/check.h"
#include "utils/device/cuda_utils.h"

#include <flashinfer/norm.cuh>
#include <flashinfer/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <memory>
#include <string>

using namespace flashinfer;

namespace edge_fm {
namespace {

void flashinfer_rms_norm_forward_impl(
    const RMSNormOpContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    cudaStream_t stream)
{
    const uint32_t stride_input = ctx.hidden_size;
    const uint32_t stride_output = ctx.hidden_size;
    cudaError_t err = cudaSuccess;

    if (ctx.dtype == DType::Float16) {
        using NormDType = half;
        err = norm::RMSNorm<NormDType>(
            static_cast<NormDType*>(const_cast<void*>(input.data_ptr())),
            static_cast<NormDType*>(const_cast<void*>(weight.data_ptr())),
            static_cast<NormDType*>(output.data_ptr()),
            ctx.batch_size,
            ctx.hidden_size,
            stride_input,
            stride_output,
            ctx.eps,
            false,
            stream);
    } else if (ctx.dtype == DType::BFloat16) {
        using NormDType = __nv_bfloat16;
        err = norm::RMSNorm<NormDType>(
            static_cast<NormDType*>(const_cast<void*>(input.data_ptr())),
            static_cast<NormDType*>(const_cast<void*>(weight.data_ptr())),
            static_cast<NormDType*>(output.data_ptr()),
            ctx.batch_size,
            ctx.hidden_size,
            stride_input,
            stride_output,
            ctx.eps,
            false,
            stream);
    } else {
        throw ConfigurationError("rms_norm operator only supports Float16 / BFloat16");
    }

    CUDA_CHECK_THROW(err, "RMSNorm failed");
}

void flashinfer_fused_add_rms_norm_forward_impl(
    const RMSNormOpContext& ctx,
    Tensor& inout,
    Tensor& residual,
    const Tensor& weight,
    cudaStream_t stream)
{
    const uint32_t stride_input = ctx.hidden_size;
    const uint32_t stride_residual = ctx.hidden_size;
    cudaError_t err = cudaSuccess;

    if (ctx.dtype == DType::Float16) {
        using NormDType = half;
        err = norm::FusedAddRMSNorm<NormDType>(
            static_cast<NormDType*>(inout.data_ptr()),
            static_cast<NormDType*>(residual.data_ptr()),
            static_cast<NormDType*>(const_cast<void*>(weight.data_ptr())),
            ctx.batch_size,
            ctx.hidden_size,
            stride_input,
            stride_residual,
            ctx.eps,
            false,
            stream);
    } else if (ctx.dtype == DType::BFloat16) {
        using NormDType = __nv_bfloat16;
        err = norm::FusedAddRMSNorm<NormDType>(
            static_cast<NormDType*>(inout.data_ptr()),
            static_cast<NormDType*>(residual.data_ptr()),
            static_cast<NormDType*>(const_cast<void*>(weight.data_ptr())),
            ctx.batch_size,
            ctx.hidden_size,
            stride_input,
            stride_residual,
            ctx.eps,
            false,
            stream);
    } else {
        throw ConfigurationError("rms_norm operator only supports Float16 / BFloat16");
    }

    CUDA_CHECK_THROW(err, "FusedAddRMSNorm failed");
}

class FlashInferNormOp final : public NormOp {
public:
    std::string impl_id() const override { return "flashinfer_norm"; }

    bool supports(const RMSNormOpContext& ctx) const override {
        return (ctx.dtype == DType::Float16 || ctx.dtype == DType::BFloat16) && ctx.hidden_size > 0;
    }

    RMSNormForwardFn rms_norm_forward_fn() const override {
        return &flashinfer_rms_norm_forward_impl;
    }

    FusedAddRMSNormForwardFn fused_add_rms_norm_forward_fn() const override {
        return &flashinfer_fused_add_rms_norm_forward_impl;
    }

    void rms_norm_forward(
        const RMSNormOpContext& ctx,
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        cudaStream_t stream) override
    {
        flashinfer_rms_norm_forward_impl(ctx, input, weight, output, stream);
    }

    void fused_add_rms_norm_forward(
        const RMSNormOpContext& ctx,
        Tensor& inout,
        Tensor& residual,
        const Tensor& weight,
        cudaStream_t stream) override
    {
        flashinfer_fused_add_rms_norm_forward_impl(ctx, inout, residual, weight, stream);
    }
};

} // namespace

NormOpRegistry::NormOpRegistry() {
    impls_.emplace_back(std::make_unique<FlashInferNormOp>());
}

NormOpRegistry& NormOpRegistry::instance() {
    static NormOpRegistry registry;
    return registry;
}

NormOp* NormOpRegistry::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
}

NormOp* NormOpRegistry::default_impl(const RMSNormOpContext& ctx) const {
    for (const auto& impl : impls_) {
        if (impl->supports(ctx)) {
            return impl.get();
        }
    }
    return nullptr;
}

void rms_norm_forward(
    const RMSNormOpContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    cudaStream_t stream)
{
    NormOp* impl = NormOpRegistry::instance().default_impl(ctx);
    check<ConfigurationError>(impl != nullptr, "rms_norm operator only supports Float16 / BFloat16");
    impl->rms_norm_forward(ctx, input, weight, output, stream);
}

void fused_add_rms_norm_forward(
    const RMSNormOpContext& ctx,
    Tensor& inout,
    Tensor& residual,
    const Tensor& weight,
    cudaStream_t stream)
{
    NormOp* impl = NormOpRegistry::instance().default_impl(ctx);
    check<ConfigurationError>(impl != nullptr, "rms_norm operator only supports Float16 / BFloat16");
    impl->fused_add_rms_norm_forward(ctx, inout, residual, weight, stream);
}

} // namespace edge_fm
