#include "operators/activation_op.h"

#include "utils/device/cuda_utils.h"

#include <flashinfer/activation.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <memory>
#include <string>
using namespace flashinfer;

namespace edge_fm {
namespace {

__device__ __forceinline__ float silu(const float& val) {
    return val / (1.0f + __expf(-val));
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float value);

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T, float (*Activation)(const float&), ActivationInputLayout kInputLayout>
__global__ void act_and_mul_kernel_layout(
    T* __restrict__ out,
    const T* __restrict__ input,
    const int d)
{
    constexpr uint32_t vec_size = 16 / sizeof(T);
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t offset = token_idx * 2 * d;
    constexpr int64_t kGateBase = (kInputLayout == ActivationInputLayout::kGateUp) ? 0 : 1;
    constexpr int64_t kUpBase = (kInputLayout == ActivationInputLayout::kGateUp) ? 1 : 0;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
    for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
        vec_t<float, vec_size> gate_vec, up_vec, out_vec;
        gate_vec.cast_load(input + offset + kGateBase * d + idx * vec_size);
        up_vec.cast_load(input + offset + kUpBase * d + idx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            out_vec[i] = Activation(gate_vec[i]) * up_vec[i];
        }
        out_vec.cast_store(out + token_idx * d + idx * vec_size);
    }

    const int64_t remaining_offset = d - d % vec_size;
#pragma unroll 1
    for (int64_t idx = thread_idx; idx < d % vec_size; idx += stride) {
        float gate = input[offset + kGateBase * d + remaining_offset + idx];
        float up = input[offset + kUpBase * d + remaining_offset + idx];
        out[token_idx * d + remaining_offset + idx] = float_to_scalar<T>(Activation(gate) * up);
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, float (*Activation)(const float&)>
void launch_activation(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t hidden_size,
    ActivationInputLayout input_layout,
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

    if (input_layout == ActivationInputLayout::kGateUp) {
        act_and_mul_kernel_layout<T, Activation, ActivationInputLayout::kGateUp>
            <<<grid, block, 0, stream>>>(output_data, input_data, static_cast<int>(hidden_size));
    } else {
        act_and_mul_kernel_layout<T, Activation, ActivationInputLayout::kUpGate>
            <<<grid, block, 0, stream>>>(output_data, input_data, static_cast<int>(hidden_size));
    }

    CUDA_CHECK_THROW(cudaGetLastError(), "act_and_mul_kernel launch failed");
}

class FlashInferSiluAndMulOp final : public ActivationOp {
public:
    std::string impl_id() const override { return "flashinfer_silu_and_mul"; }

    bool supports(const ActivationOpContext& ctx) const override {
        return ctx.kind == ActivationKind::kSilu &&
            (ctx.dtype == DType::Float16 || ctx.dtype == DType::BFloat16);
    }

    void act_and_mul(
        const ActivationOpContext& ctx,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream) override
    {
        if (ctx.kind != ActivationKind::kSilu) {
            throw ConfigurationError("activation operator only supports silu");
        }

        if (ctx.dtype == DType::Float16) {
            launch_activation<half, silu>(
                output.data_ptr(),
                input.data_ptr(),
                ctx.batch_size,
                ctx.hidden_size,
                ctx.input_layout,
                stream);
            return;
        }

        if (ctx.dtype == DType::BFloat16) {
            launch_activation<__nv_bfloat16, silu>(
                output.data_ptr(),
                input.data_ptr(),
                ctx.batch_size,
                ctx.hidden_size,
                ctx.input_layout,
                stream);
            return;
        }

        throw InvalidRequestError(
            "Unsupported dtype for activation operator. Only Float16 and BFloat16 are supported.");
    }
};

} // namespace

ActivationOpRegistry::ActivationOpRegistry() {
    impls_.emplace_back(std::make_unique<FlashInferSiluAndMulOp>());
}

ActivationOpRegistry& ActivationOpRegistry::instance() {
    static ActivationOpRegistry registry;
    return registry;
}

ActivationOp* ActivationOpRegistry::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
}

ActivationOp* ActivationOpRegistry::default_impl(const ActivationOpContext& ctx) const {
    for (const auto& impl : impls_) {
        if (impl->supports(ctx)) {
            return impl.get();
        }
    }
    return nullptr;
}

void activation_act_and_mul(
    const ActivationOpContext& ctx,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    ActivationOp* impl = ActivationOpRegistry::instance().default_impl(ctx);
    if (impl == nullptr) {
        throw InvalidRequestError(
            "Unsupported dtype for activation operator. Only Float16 and BFloat16 are supported.");
    }
    impl->act_and_mul(ctx, input, output, stream);
}

} // namespace edge_fm
