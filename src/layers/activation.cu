#include "layers/activation.h"
#include "utils/device/nvtx.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <flashinfer/activation.cuh>
#include "utils/device/cuda_utils.h"

using namespace flashinfer;

namespace edge_fm {

// SiLU activation function
// Note: __device__ functions can be used as template parameters (only type info is needed)
__device__ __forceinline__ float silu(const float& val) {
    return val / (1.0f + __expf(-val));
}

ActivationLayer::ActivationLayer(const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name))
{
    nlohmann::json model_config = engine_config_.prefill_model_config();
    // MLP 中 SwiGLU 的输入为 gate+up 拼接，维度 2*intermediate_size；输出为 intermediate_size
    uint32_t intermediate_size = model_config.value("intermediate_size", 0U);
    hidden_size_ = intermediate_size != 0U ? intermediate_size : model_config.value("hidden_size", 4096U);
    activation_type_ = model_config.value("hidden_act", std::string("silu"));
    
    // Validate that the activation type is supported
    if (activation_type_ != "silu") {
        throw ConfigurationError(
            "Unsupported activation type: \"" + activation_type_ + 
            "\". Currently only \"silu\" is supported.");
    }
}

void ActivationLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    [[maybe_unused]] ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("ActivationLayer is not initialized");
    }

    const auto& input = inputs.at("input");
    auto& output = outputs.at("output");
    
    forward_silu_and_mul(input, output, stream);
}

void ActivationLayer::forward_silu_and_mul(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    auto input_device = input.device();
    auto output_device = output.device();
    
    if (std::get<0>(input_device) != device_ || std::get<1>(input_device) != device_id_) {
        throw DeviceError("Input tensor must be on the same device as the layer.");
    }
    if (std::get<0>(output_device) != device_ || std::get<1>(output_device) != device_id_) {
        throw DeviceError("Output tensor must be on the same device as the layer.");
    }
    
    const auto& input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw InvalidRequestError("Input tensor must have at least 2 dimensions");
    }
    
    // Get the last dimension (hidden_size * 2)
    int64_t input_last_dim = input_shape.back();
    if (input_last_dim % 2 != 0) {
        throw InvalidRequestError(
            "Input tensor last dimension must be even (2 * hidden_size). Got: " + 
            std::to_string(input_last_dim));
    }
    
    // Verify that the input last dimension matches the configured hidden_size * 2
    int64_t expected_input_last_dim = static_cast<int64_t>(hidden_size_) * 2;
    if (input_last_dim != expected_input_last_dim) {
        throw InvalidRequestError(
            "Input tensor last dimension mismatch. Expected " + 
            std::to_string(expected_input_last_dim) + " (2 * hidden_size from config: " + 
            std::to_string(hidden_size_) + "), got " + std::to_string(input_last_dim));
    }
    
    // Calculate batch size (product of all dimensions except the last)
    int64_t batch_size = std::accumulate(
        input_shape.begin(), 
        input_shape.end() - 1, 
        int64_t(1), 
        std::multiplies<int64_t>()
    );
    
    const auto& output_shape = output.shape();
    if (output_shape.size() != input_shape.size()) {
        throw InvalidRequestError(
            "Output tensor must have the same number of dimensions as input. "
            "Input: " + std::to_string(input_shape.size()) + 
            ", Output: " + std::to_string(output_shape.size()));
    }
    
    // Check all dimensions except the last
    for (size_t i = 0; i < output_shape.size() - 1; ++i) {
        if (output_shape[i] != input_shape[i]) {
            throw InvalidRequestError(
                "Output tensor shape mismatch at dimension " + std::to_string(i) + 
                ". Expected " + std::to_string(input_shape[i]) + 
                ", got " + std::to_string(output_shape[i]));
        }
    }
    
    // Check last dimension - should match configured hidden_size
    if (static_cast<uint32_t>(output_shape.back()) != hidden_size_) {
        throw InvalidRequestError(
            "Output tensor last dimension mismatch. Expected " + std::to_string(hidden_size_) + 
            " (from config), got " + std::to_string(output_shape.back()));
    }
    
    // Check dtype
    DType input_dtype = input.dtype();
    if (input.dtype() != output.dtype()) {
        throw InvalidRequestError(
            "Input and output tensors must have the same dtype. "
            "Input: " + std::to_string(static_cast<int>(input.dtype())) +
            ", Output: " + std::to_string(static_cast<int>(output.dtype())));
    }
    
    // Launch kernel based on dtype, using configured hidden_size_
    if (input_dtype == DType::Float16) {
        launch_activation<half, silu>(
            output.data_ptr(), input.data_ptr(), batch_size, static_cast<int64_t>(hidden_size_), stream
        );
    } else if (input_dtype == DType::BFloat16) {
        launch_activation<__nv_bfloat16, silu>(
            output.data_ptr(), input.data_ptr(), batch_size, static_cast<int64_t>(hidden_size_), stream
        );
    } else {
        throw InvalidRequestError(
            "Unsupported dtype for ActivationLayer. Only Float16 and BFloat16 are supported. "
            "Got dtype: " + std::to_string(static_cast<int>(input_dtype)));
    }
}

template <typename T, float (*Activation)(const float&)>
void ActivationLayer::launch_activation(
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
    if (block_size == 0) block_size = 1;
    
    dim3 grid(static_cast<uint32_t>(batch_size));
    dim3 block(block_size);
    
    activation::act_and_mul_kernel<T, Activation><<<grid, block, 0, stream>>>(
        output_data,
        input_data,
        static_cast<int>(hidden_size)
    );
    
    CUDA_CHECK_THROW(cudaGetLastError(), "act_and_mul_kernel launch failed");
}

// Explicit template instantiations
template void ActivationLayer::launch_activation<half, silu>(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream
);

template void ActivationLayer::launch_activation<__nv_bfloat16, silu>(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream
);

} // namespace edge_fm

