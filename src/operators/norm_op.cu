#include "operators/norm_op.h"

#include "utils/device/cuda_utils.h"

#include <flashinfer/norm.cuh>
#include <flashinfer/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace flashinfer;

namespace edge_fm {
void rms_norm_forward(
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

void fused_add_rms_norm_forward(
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

} // namespace edge_fm
