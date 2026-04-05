#include "layers/activation.h"

#include "operators/activation_op.h"
#include "operators/operator_impl_table.h"
#include "utils/device/nvtx.h"

#include <functional>
#include <numeric>
#include <string>

namespace edge_fm {
namespace {

void validate_tensor_device(const Tensor& tensor, Device device, int32_t device_id, const std::string& tensor_name) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    if (tensor_device != device || tensor_device_id != device_id) {
        throw DeviceError(tensor_name + " tensor must be on the same device as the layer.");
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

ActivationLayer::ActivationLayer(const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name)),
      layer_role_("mlp_activation")
{
    const nlohmann::json model_config = engine_config_.prefill_model_config();
    const uint32_t intermediate_size = model_config.value("intermediate_size", 0U);
    hidden_size_ = intermediate_size != 0U ? intermediate_size : model_config.value("hidden_size", 4096U);
    activation_type_ = model_config.value("hidden_act", std::string("silu"));

    if (activation_type_ != "silu") {
        throw ConfigurationError(
            "Unsupported activation type: '" + activation_type_ +
            "'. Currently only 'silu' is supported.");
    }
}

ActivationLayer::~ActivationLayer() = default;

ActivationOp* ActivationLayer::resolve_impl(const ActivationOpContext& ctx, ModelStage stage) {
    const size_t slot = stage_slot(stage);
    if (ActivationOp* impl = selected_impls_[slot]; impl != nullptr) {
        return impl;
    }

    auto& selected_impl_id = selected_impl_ids_[slot];
    if (!selected_impl_id.empty()) {
        if (ActivationOp* impl = ActivationOpRegistry::instance().find_impl_by_id(selected_impl_id); impl != nullptr) {
            selected_impls_[slot] = impl;
            return impl;
        }
        selected_impl_id.clear();
    }

    OperatorQuery query;
    query.op_kind = "activation";
    query.layer_role = layer_role_;
    query.op_name = "silu_and_mul";
    query.stage = stage_key(stage);

    auto resolved = OperatorImplTable::instance().resolve(
        model_name_for_operator_resolution(engine_config_),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (ActivationOp* impl = ActivationOpRegistry::instance().find_impl_by_id(resolved->impl_id); impl != nullptr) {
            if (!impl->supports(ctx)) {
                throw ConfigurationError(
                    "ActivationLayer: operator_impl_table selected unsupported impl '" + resolved->impl_id + "'");
            }
            selected_impl_id = impl->impl_id();
            selected_impls_[slot] = impl;
            return impl;
        }
        throw ConfigurationError(
            "ActivationLayer: operator_impl_table selected unknown impl '" + resolved->impl_id + "'");
    }

    if (ActivationOp* impl = ActivationOpRegistry::instance().default_impl(ctx); impl != nullptr) {
        selected_impl_id = impl->impl_id();
        selected_impls_[slot] = impl;
        return impl;
    }

    throw ConfigurationError("ActivationLayer: no supported implementation found");
}

void ActivationLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("ActivationLayer is not initialized");
    }

    const auto& input = inputs.at("input");
    auto& output = outputs.at("output");
    forward_silu_and_mul_impl(input, output, stream, stage);
}

void ActivationLayer::forward_silu_and_mul(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    forward_silu_and_mul_impl(input, output, stream, stage);
}

void ActivationLayer::forward_silu_and_mul_impl(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    validate_tensor_device(input, device_, device_id_, "Input");
    validate_tensor_device(output, device_, device_id_, "Output");

    const auto& input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw InvalidRequestError("Input tensor must have at least 2 dimensions");
    }

    const int64_t input_last_dim = input_shape.back();
    if (input_last_dim % 2 != 0) {
        throw InvalidRequestError(
            "Input tensor last dimension must be even (2 * hidden_size). Got: " +
            std::to_string(input_last_dim));
    }

    const int64_t expected_input_last_dim = static_cast<int64_t>(hidden_size_) * 2;
    if (input_last_dim != expected_input_last_dim) {
        throw InvalidRequestError(
            "Input tensor last dimension mismatch. Expected " + std::to_string(expected_input_last_dim) +
            " (2 * hidden_size from config: " + std::to_string(hidden_size_) + "), got " +
            std::to_string(input_last_dim));
    }

    const auto& output_shape = output.shape();
    if (output_shape.size() != input_shape.size()) {
        throw InvalidRequestError(
            "Output tensor must have the same number of dimensions as input. Input: " +
            std::to_string(input_shape.size()) + ", Output: " + std::to_string(output_shape.size()));
    }

    for (size_t i = 0; i + 1 < output_shape.size(); ++i) {
        if (output_shape[i] != input_shape[i]) {
            throw InvalidRequestError(
                "Output tensor shape mismatch at dimension " + std::to_string(i) + ". Expected " +
                std::to_string(input_shape[i]) + ", got " + std::to_string(output_shape[i]));
        }
    }
    if (static_cast<uint32_t>(output_shape.back()) != hidden_size_) {
        throw InvalidRequestError(
            "Output tensor last dimension mismatch. Expected " + std::to_string(hidden_size_) +
            " (from config), got " + std::to_string(output_shape.back()));
    }

    const DType input_dtype = input.dtype();
    if (input_dtype != output.dtype()) {
        throw InvalidRequestError(
            "Input and output tensors must have the same dtype. Input: " +
            std::to_string(static_cast<int>(input_dtype)) + ", Output: " +
            std::to_string(static_cast<int>(output.dtype())));
    }

    const int64_t batch_size = std::accumulate(
        input_shape.begin(), input_shape.end() - 1, int64_t(1), std::multiplies<int64_t>());

    ActivationOpContext ctx;
    ctx.batch_size = batch_size;
    ctx.hidden_size = static_cast<int64_t>(hidden_size_);
    ctx.dtype = input_dtype;
    ctx.kind = ActivationKind::kSilu;

    ActivationOp* impl = resolve_impl(ctx, stage);
    impl->act_and_mul(ctx, input, output, stream);
}

} // namespace edge_fm
