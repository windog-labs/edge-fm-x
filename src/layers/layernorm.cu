#include "layers/layernorm.h"

#include "operators/norm_op.h"
#include "utils/device/nvtx.h"

#include <string>
#include <vector>

namespace edge_fm {
namespace {

void validate_tensor_device(const Tensor& tensor, Device device, int32_t device_id, const std::string& tensor_name) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    if (tensor_device != device || tensor_device_id != device_id) {
        throw DeviceError(tensor_name + " tensor must be on the same device as the layer.");
    }
}

void validate_tensor_shape_2d(const Tensor& tensor, uint32_t hidden_size, const std::string& tensor_name) {
    const auto& shape = tensor.shape();
    if (shape.size() != 2) {
        throw InvalidRequestError(tensor_name + " tensor must be 2D [batch_size, hidden_size]");
    }
    if (static_cast<uint32_t>(shape[1]) != hidden_size) {
        throw InvalidRequestError(
            tensor_name + " hidden_size mismatch. Expected " + std::to_string(hidden_size) +
            ", got " + std::to_string(shape[1]));
    }
}

void validate_tensor_dtype(const Tensor& tensor, DType dtype, const std::string& tensor_name) {
    if (tensor.dtype() != dtype) {
        throw ConfigurationError(
            tensor_name + " dtype mismatch. Expected " + std::to_string(static_cast<int>(dtype)) +
            ", got " + std::to_string(static_cast<int>(tensor.dtype())));
    }
}

} // namespace

RMSNormLayer::RMSNormLayer(
    uint32_t layer_id,
    NormWeightType weight_type,
    const EngineConfig& engine_config,
    std::string layer_name)
    : Layer(engine_config, std::move(layer_name)),
      layer_id_(layer_id),
      weight_type_(weight_type),
      hidden_size_(0),
      eps_(1e-6f),
      weight_(nullptr)
{
    const nlohmann::json model_config = engine_config_.prefill_model_config();
    hidden_size_ = model_config.value("hidden_size", 4096U);
    eps_ = model_config.value("rms_norm_eps", 1e-6f);
}

RMSNormLayer::~RMSNormLayer() = default;

void RMSNormLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor>& decode_weights)
{
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
        if (it == prefill_weights.end()) {
            continue;
        }

        weight_ = &it->second;
        const auto& shape = weight_->shape();
        if (shape.size() != 1 || shape[0] != static_cast<int64_t>(hidden_size_)) {
            throw ConfigurationError(
                "RMSNorm weight shape mismatch. Expected [" + std::to_string(hidden_size_) +
                "], got [" + std::to_string(shape[0]) + "] for weight: " + name);
        }
        weights_loaded_ = true;
        return;
    }

    weights_loaded_ = false;
    std::string tried_names;
    for (size_t i = 0; i < possible_names.size(); ++i) {
        if (i > 0) {
            tried_names += ", ";
        }
        tried_names += possible_names[i];
    }
    throw ConfigurationError(
        "RMSNorm weight not found for layer_id=" + std::to_string(layer_id_) + ". Tried: " + tried_names);
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

    auto residual_it = inputs.find("residual");
    if (residual_it == inputs.end()) {
        forward_rmsnorm(input, output, stream);
        return;
    }

    const Tensor& output_const = output;
    if (!(input == output_const)) {
        throw InvalidRequestError(
            "For fused_add_rmsnorm, input and output must be the same tensor. "
            "Got different tensors (different data pointers).");
    }

    Tensor& residual = const_cast<Tensor&>(residual_it->second);
    forward_fused_add_rmsnorm(output, residual, stream);
}

void RMSNormLayer::forward_rmsnorm(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    if (weight_ == nullptr) {
        throw ConfigurationError("RMSNormLayer weight is not loaded");
    }

    validate_tensor_device(input, device_, device_id_, "Input");
    validate_tensor_device(output, device_, device_id_, "Output");
    validate_tensor_device(*weight_, device_, device_id_, "Weight");
    validate_tensor_shape_2d(input, hidden_size_, "Input");
    validate_tensor_shape_2d(output, hidden_size_, "Output");

    const DType weight_dtype = weight_->dtype();
    validate_tensor_dtype(input, weight_dtype, "Input");
    validate_tensor_dtype(output, weight_dtype, "Output");

    const auto& input_shape = input.shape();
    if (output.shape()[0] != input_shape[0]) {
        throw InvalidRequestError(
            "Output tensor shape mismatch. Expected [" + std::to_string(input_shape[0]) + ", " +
            std::to_string(input_shape[1]) + "], got [" + std::to_string(output.shape()[0]) + ", " +
            std::to_string(output.shape()[1]) + "]");
    }

    RMSNormOpContext ctx;
    ctx.batch_size = static_cast<uint32_t>(input_shape[0]);
    ctx.hidden_size = hidden_size_;
    ctx.eps = eps_;
    ctx.dtype = weight_dtype;

    rms_norm_forward(ctx, input, *weight_, output, stream);
}

void RMSNormLayer::forward_fused_add_rmsnorm(
    Tensor& inout,
    Tensor& residual,
    cudaStream_t stream)
{
    if (weight_ == nullptr) {
        throw ConfigurationError("RMSNormLayer weight is not loaded");
    }

    validate_tensor_device(inout, device_, device_id_, "Inout");
    validate_tensor_device(residual, device_, device_id_, "Residual");
    validate_tensor_device(*weight_, device_, device_id_, "Weight");
    validate_tensor_shape_2d(inout, hidden_size_, "Inout");
    validate_tensor_shape_2d(residual, hidden_size_, "Residual");

    const auto& inout_shape = inout.shape();
    const auto& residual_shape = residual.shape();
    if (residual_shape[0] != inout_shape[0] || residual_shape[1] != inout_shape[1]) {
        throw InvalidRequestError(
            "Residual tensor shape mismatch. Expected [" + std::to_string(inout_shape[0]) + ", " +
            std::to_string(inout_shape[1]) + "], got [" + std::to_string(residual_shape[0]) + ", " +
            std::to_string(residual_shape[1]) + "]");
    }

    const DType weight_dtype = weight_->dtype();
    validate_tensor_dtype(inout, weight_dtype, "Inout");
    validate_tensor_dtype(residual, weight_dtype, "Residual");

    RMSNormOpContext ctx;
    ctx.batch_size = static_cast<uint32_t>(inout_shape[0]);
    ctx.hidden_size = hidden_size_;
    ctx.eps = eps_;
    ctx.dtype = weight_dtype;

    fused_add_rms_norm_forward(ctx, inout, residual, *weight_, stream);
}

} // namespace edge_fm
