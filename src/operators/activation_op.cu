#include "operators/activation_op.h"

#include "utils/device/cuda_utils.h"

#include <flashinfer/activation.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
using namespace flashinfer;

namespace edge_fm {
namespace {

__device__ __forceinline__ float silu(const float& val) {
    return val / (1.0f + __expf(-val));
}

template <typename T, float (*Activation)(const float&)>
void launch_activation(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream)
{
    const T* input_data = static_cast<const T*>(input);
    T* output_data = static_cast<T*>(output);

    constexpr uint32_t vec_size = 16 / sizeof(T);
    uint32_t block_size = std::min(static_cast<uint32_t>(hidden_size) / vec_size, 1024U);
    if (block_size == 0) {
        block_size = 1;
    }

    dim3 grid(static_cast<uint32_t>(batch_size));
    dim3 block(block_size);

    activation::act_and_mul_kernel<T, Activation><<<grid, block, 0, stream>>>(
        output_data,
        input_data,
        static_cast<int>(hidden_size));

    CUDA_CHECK_THROW(cudaGetLastError(), "act_and_mul_kernel launch failed");
}

} // namespace

void activation_act_and_mul(
    const ActivationOpContext& ctx,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    if (ctx.kind != ActivationKind::kSilu) {
        throw ConfigurationError("activation operator only supports silu");
    }

    if (ctx.dtype == DType::Float16) {
        launch_activation<half, silu>(
            output.data_ptr(), input.data_ptr(), ctx.batch_size, ctx.hidden_size, stream);
        return;
    }

    if (ctx.dtype == DType::BFloat16) {
        launch_activation<__nv_bfloat16, silu>(
            output.data_ptr(), input.data_ptr(), ctx.batch_size, ctx.hidden_size, stream);
        return;
    }

    throw InvalidRequestError(
        "Unsupported dtype for activation operator. Only Float16 and BFloat16 are supported.");
}

} // namespace edge_fm
