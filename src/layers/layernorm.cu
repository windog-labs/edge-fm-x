#include "layers/layernorm.h"

#include "operators/norm_op.h"
#include "operators/operator_impl_table.h"
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

std::string infer_layer_role(NormWeightType weight_type) {
    switch (weight_type) {
        case NormWeightType::Input:
            return "input_norm";
        case NormWeightType::PostAttention:
            return "post_attention_norm";
        case NormWeightType::Final:
            return "final_norm";
        default:
            return "norm";
    }
}

size_t stage_slot(ModelStage stage) {
    return stage == ModelStage::Decode ? 1U : 0U;
}

std::string stage_key(ModelStage stage) {
    return stage == ModelStage::Decode ? "decode" : "prefill";
}

std::string model_name_for_operator_resolution(const EngineConfig& engine_config) {
    try {
        return engine_config.resolved_model_name();
    } catch (const ConfigurationError&) {
        return std::string();
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
      weight_(nullptr),
      layer_role_(infer_layer_role(weight_type))
{
    const nlohmann::json model_config = engine_config_.prefill_model_config();
    hidden_size_ = model_config.value("hidden_size", 4096U);
    eps_ = model_config.value("rms_norm_eps", 1e-6f);
}

RMSNormLayer::~RMSNormLayer() = default;

NormOp* RMSNormLayer::resolve_impl(const RMSNormOpContext& ctx, ModelStage stage) {
    const size_t slot = stage_slot(stage);
    if (NormOp* impl = selected_impls_[slot]; impl != nullptr) {
        return impl;
    }

    auto& selected_impl_id = selected_impl_ids_[slot];
    if (!selected_impl_id.empty()) {
        if (NormOp* impl = NormOpRegistry::instance().find_impl_by_id(selected_impl_id); impl != nullptr) {
            selected_impls_[slot] = impl;
            selected_rms_norm_fns_[slot] = impl->rms_norm_forward_fn();
            selected_fused_add_rms_norm_fns_[slot] = impl->fused_add_rms_norm_forward_fn();
            return impl;
        }
        selected_impl_id.clear();
    }

    OperatorQuery query;
    query.op_kind = "norm";
    query.layer_role = layer_role_;
    query.op_name = "rms_norm";
    query.stage = stage_key(stage);

    auto resolved = OperatorImplTable::instance().resolve(
        model_name_for_operator_resolution(engine_config_),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (NormOp* impl = NormOpRegistry::instance().find_impl_by_id(resolved->impl_id); impl != nullptr) {
            if (!impl->supports(ctx)) {
                throw ConfigurationError(
                    "RMSNormLayer: operator_impl_table selected unsupported impl '" + resolved->impl_id + "'");
            }
            selected_impl_id = impl->impl_id();
            selected_impls_[slot] = impl;
            selected_rms_norm_fns_[slot] = impl->rms_norm_forward_fn();
            selected_fused_add_rms_norm_fns_[slot] = impl->fused_add_rms_norm_forward_fn();
            return impl;
        }
        throw ConfigurationError(
            "RMSNormLayer: operator_impl_table selected unknown impl '" + resolved->impl_id + "'");
    }

    if (NormOp* impl = NormOpRegistry::instance().default_impl(ctx); impl != nullptr) {
        selected_impl_id = impl->impl_id();
        selected_impls_[slot] = impl;
        selected_rms_norm_fns_[slot] = impl->rms_norm_forward_fn();
        selected_fused_add_rms_norm_fns_[slot] = impl->fused_add_rms_norm_forward_fn();
        return impl;
    }

    throw ConfigurationError("RMSNormLayer: no supported implementation found");
}

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
        selected_impl_ids_.fill(std::string());
        selected_impls_.fill(nullptr);
        selected_rms_norm_fns_.fill(nullptr);
        selected_fused_add_rms_norm_fns_.fill(nullptr);
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
    ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("RMSNormLayer is not initialized");
    }

    const auto& input = inputs.at("input");
    auto& output = outputs.at("output");

    auto residual_it = inputs.find("residual");
    if (residual_it == inputs.end()) {
        forward_rmsnorm_impl(input, output, stream, stage);
        return;
    }

    const Tensor& output_const = output;
    if (!(input == output_const)) {
        throw InvalidRequestError(
            "For fused_add_rmsnorm, input and output must be the same tensor. "
            "Got different tensors (different data pointers).");
    }

    Tensor& residual = const_cast<Tensor&>(residual_it->second);
    forward_fused_add_rmsnorm_impl(output, residual, stream, stage);
}

void RMSNormLayer::forward_rmsnorm(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    forward_rmsnorm_impl(input, output, stream, ModelStage::Prefill);
}

void RMSNormLayer::forward_rmsnorm_impl(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
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

    const size_t slot = stage_slot(stage);
    if (selected_rms_norm_fns_[slot] == nullptr) {
        resolve_impl(ctx, stage);
    }

    RMSNormForwardFn rms_norm_fn = selected_rms_norm_fns_[slot];
    if (rms_norm_fn == nullptr) {
        throw ConfigurationError("RMSNormLayer: no bound implementation found");
    }
    rms_norm_fn(ctx, input, *weight_, output, stream);
}

void RMSNormLayer::forward_fused_add_rmsnorm(
    Tensor& inout,
    Tensor& residual,
    cudaStream_t stream)
{
    forward_fused_add_rmsnorm_impl(inout, residual, stream, ModelStage::Prefill);
}

void RMSNormLayer::forward_fused_add_rmsnorm_impl(
    Tensor& inout,
    Tensor& residual,
    cudaStream_t stream,
    ModelStage stage)
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

    const size_t slot = stage_slot(stage);
    if (selected_fused_add_rms_norm_fns_[slot] == nullptr) {
        resolve_impl(ctx, stage);
    }

    FusedAddRMSNormForwardFn fused_add_rms_norm_fn = selected_fused_add_rms_norm_fns_[slot];
    if (fused_add_rms_norm_fn == nullptr) {
        throw ConfigurationError("RMSNormLayer: no bound implementation found");
    }
    fused_add_rms_norm_fn(ctx, inout, residual, *weight_, stream);
}

} // namespace edge_fm
