#include "layers/layernorm.h"
#include "utils/device/nvtx.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <tuple>
#include <flashinfer/utils.cuh>
#include <flashinfer/norm.cuh>
#include "utils/device/cuda_utils.h"
#include "utils/device/weight_loader.h"

using namespace flashinfer;

namespace edge_fm {

RMSNormLayer::RMSNormLayer(uint32_t layer_id, NormWeightType weight_type, const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name)), layer_id_(layer_id), weight_type_(weight_type)
{
    nlohmann::json model_config = engine_config_.prefill_model_config();
    hidden_size_ = model_config.value("hidden_size", 4096U);
    eps_ = model_config.value("rms_norm_eps", 1e-6);  // 默认值通常为 1e-6
}

void RMSNormLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor>& decode_weights)
{
    // Norm 权重通常共享，只从 prefill_weights 加载
    // 按 weight_type_ 加载对应权重，避免 input/post_attention 互相加载错误
    std::vector<std::string> possible_names;
    if (layer_id_ == UINT32_MAX || weight_type_ == NormWeightType::Final) {
        possible_names = {"model.norm.weight"};
    } else if (weight_type_ == NormWeightType::Input) {
        possible_names = {"model.layers." + std::to_string(layer_id_) + ".input_layernorm.weight"};
    } else {
        possible_names = {"model.layers." + std::to_string(layer_id_) + ".post_attention_layernorm.weight"};
    }
    
    for (const auto& name : possible_names) {
        auto it = prefill_weights.find(name);
        if (it != prefill_weights.end()) {
            weight_ = &it->second;
            
            const auto& shape = weight_->shape();
            if (shape.size() != 1 || shape[0] != static_cast<int64_t>(hidden_size_)) {
                throw ConfigurationError(
                    "RMSNorm weight shape mismatch. Expected [" + 
                    std::to_string(hidden_size_) + "], got [" + 
                    std::to_string(shape[0]) + "] for weight: " + name);
            }
            weights_loaded_ = true;
            return;
        }
    }
    
    weights_loaded_ = false;
    std::string tried_names = "";
    for (size_t i = 0; i < possible_names.size(); i++) {
        if (i > 0) tried_names += ", ";
        tried_names += possible_names[i];
    }
    throw ConfigurationError(
        "RMSNorm weight not found for layer_id=" + std::to_string(layer_id_) + 
        ". Tried: " + tried_names);
}

void RMSNormLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    [[maybe_unused]] ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("RMSNormLayer is not initialized");
    }

    const auto& input = inputs.at("input");
    auto& output = outputs.at("output");
    
    auto input_device = input.device();
    auto output_device = output.device();
    auto weight_device = weight_->device();
    
    if (std::get<0>(input_device) != device_ || std::get<1>(input_device) != device_id_) {
        throw DeviceError("Input tensor must be on the same device as the layer.");
    }
    if (std::get<0>(output_device) != device_ || std::get<1>(output_device) != device_id_) {
        throw DeviceError("Output tensor must be on the same device as the layer.");
    }
    if (std::get<0>(weight_device) != device_ || std::get<1>(weight_device) != device_id_) {
        throw DeviceError("Weight tensor must be on the same device as the layer.");
    }
    
    const auto& input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw InvalidRequestError("Input tensor must be 2D [batch_size, hidden_size]");
    }
    
    uint32_t hidden_size = static_cast<uint32_t>(input_shape[1]);
    if (hidden_size != hidden_size_) {
        throw InvalidRequestError(
            "Input hidden_size mismatch. Expected " + std::to_string(hidden_size_) + 
            ", got " + std::to_string(hidden_size));
    }
    
    const auto& output_shape = output.shape();
    if (output_shape.size() != 2 || 
        output_shape[0] != input_shape[0] || 
        output_shape[1] != input_shape[1]) {
        throw InvalidRequestError(
            "Output tensor shape mismatch. Expected [" + 
            std::to_string(input_shape[0]) + ", " + std::to_string(input_shape[1]) + 
            "], got [" + std::to_string(output_shape[0]) + ", " + 
            std::to_string(output_shape[1]) + "]");
    }
    
    DType weight_dtype = weight_->dtype();
    if (input.dtype() != weight_dtype || output.dtype() != weight_dtype) {
        throw ConfigurationError(
            "Tensor dtype must match weight dtype. Weight dtype: " + 
            std::to_string(static_cast<int>(weight_dtype)) +
            ", Input dtype: " + std::to_string(static_cast<int>(input.dtype())) +
            ", Output dtype: " + std::to_string(static_cast<int>(output.dtype())));
    }
    
    auto residual_it = inputs.find("residual");
    if (residual_it != inputs.end()) {
        Tensor& residual = const_cast<Tensor&>(residual_it->second);
        
        auto residual_device = residual.device();
        if (std::get<0>(residual_device) != device_ || std::get<1>(residual_device) != device_id_) {
            throw DeviceError("Residual tensor must be on the same device as the layer.");
        }
        
        const auto& residual_shape = residual.shape();
        if (residual_shape.size() != 2 || 
            residual_shape[0] != input_shape[0] || 
            residual_shape[1] != input_shape[1]) {
            throw InvalidRequestError(
                "Residual tensor shape mismatch. Expected [" + 
                std::to_string(input_shape[0]) + ", " + std::to_string(input_shape[1]) + 
                "], got [" + std::to_string(residual_shape[0]) + ", " + 
                std::to_string(residual_shape[1]) + "]");
        }
        
        if (residual.dtype() != weight_dtype) {
            throw ConfigurationError(
                "Residual dtype must match weight dtype. Weight dtype: " + 
                std::to_string(static_cast<int>(weight_dtype)) +
                ", Residual dtype: " + std::to_string(static_cast<int>(residual.dtype())));
        }
        
        // 检查 input 和 output 是否是同一个 tensor
        const Tensor& output_const = output;
        if (!(input == output_const)) {
            throw InvalidRequestError(
                "For fused_add_rmsnorm, input and output must be the same tensor. "
                "Got different tensors (different data pointers).");
        }
        
        // input 和 output 是同一个 tensor，直接使用 output 作为 inout 参数
        forward_fused_add_rmsnorm(output, residual, stream);
    } else {
        // 没有 residual，调用单纯的 rmsnorm
        forward_rmsnorm(input, output, stream);
    }
}

void RMSNormLayer::forward_rmsnorm(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    const auto& input_shape = input.shape();
    uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
    uint32_t hidden_size = static_cast<uint32_t>(input_shape[1]);
    uint32_t stride_input = hidden_size;
    uint32_t stride_output = hidden_size;
    
    DType weight_dtype = weight_->dtype();
    cudaError_t err = cudaSuccess;
    
    if (weight_dtype == DType::Float16) {
        using DTypeNorm = half;
        DTypeNorm* input_data = static_cast<DTypeNorm*>(const_cast<void*>(input.data_ptr()));
        DTypeNorm* weight_data = static_cast<DTypeNorm*>(const_cast<void*>(weight_->data_ptr()));
        DTypeNorm* output_data = static_cast<DTypeNorm*>(output.data_ptr());
        
        err = norm::RMSNorm<DTypeNorm>(
            input_data,
            weight_data,
            output_data,
            batch_size,
            hidden_size,
            stride_input,
            stride_output,
            eps_,
            false,
            stream
        );
    } else if (weight_dtype == DType::BFloat16) {
        using DTypeNorm = __nv_bfloat16;
        DTypeNorm* input_data = static_cast<DTypeNorm*>(const_cast<void*>(input.data_ptr()));
        DTypeNorm* weight_data = static_cast<DTypeNorm*>(const_cast<void*>(weight_->data_ptr()));
        DTypeNorm* output_data = static_cast<DTypeNorm*>(output.data_ptr());
        
        err = norm::RMSNorm<DTypeNorm>(
            input_data,
            weight_data,
            output_data,
            batch_size,
            hidden_size,
            stride_input,
            stride_output,
            eps_,
            false,
            stream
        );
    } else {
        throw ConfigurationError(
            "Unsupported weight dtype for RMSNorm. Only Float16 and BFloat16 are supported. "
            "Got dtype: " + std::to_string(static_cast<int>(weight_dtype)));
    }
    
    CUDA_CHECK_THROW(err, "RMSNorm failed");
}

void RMSNormLayer::forward_fused_add_rmsnorm(
    Tensor& inout,
    Tensor& residual,
    cudaStream_t stream)
{
    const auto& input_shape = inout.shape();
    uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
    uint32_t hidden_size = static_cast<uint32_t>(input_shape[1]);
    uint32_t stride_input = hidden_size;
    uint32_t stride_residual = hidden_size;
    
    DType weight_dtype = weight_->dtype();
    cudaError_t err = cudaSuccess;
    
    if (weight_dtype == DType::Float16) {
        using DTypeNorm = half;
        DTypeNorm* inout_data = static_cast<DTypeNorm*>(inout.data_ptr());
        DTypeNorm* residual_data = static_cast<DTypeNorm*>(residual.data_ptr());
        DTypeNorm* weight_data = static_cast<DTypeNorm*>(const_cast<void*>(weight_->data_ptr()));
        
        err = norm::FusedAddRMSNorm<DTypeNorm>(
            inout_data,
            residual_data,
            weight_data,
            batch_size,
            hidden_size,
            stride_input,
            stride_residual,
            eps_,
            false,
            stream
        );
    } else if (weight_dtype == DType::BFloat16) {
        using DTypeNorm = __nv_bfloat16;
        DTypeNorm* inout_data = static_cast<DTypeNorm*>(inout.data_ptr());
        DTypeNorm* residual_data = static_cast<DTypeNorm*>(residual.data_ptr());
        DTypeNorm* weight_data = static_cast<DTypeNorm*>(const_cast<void*>(weight_->data_ptr()));
        
        err = norm::FusedAddRMSNorm<DTypeNorm>(
            inout_data,
            residual_data,
            weight_data,
            batch_size,
            hidden_size,
            stride_input,
            stride_residual,
            eps_,
            false,
            stream
        );
        
    } else {
        throw ConfigurationError(
            "Unsupported weight dtype for FusedAddRMSNorm. Only Float16 and BFloat16 are supported. "
            "Got dtype: " + std::to_string(static_cast<int>(weight_dtype)));
    }
    
    CUDA_CHECK_THROW(err, "FusedAddRMSNorm failed");
}

}  // namespace edge_fm