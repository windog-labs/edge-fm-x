#include "models/qwen3_5/qwen3_5.h"

#include "engine/tasks/token_generation/cuda/scheduler.h"
#include "operators/qwen3_5/qwen3_5_ops.h"
#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/device/weight_loader.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <string>

namespace edge_fm {

namespace {

std::vector<std::string> parse_layer_types(const nlohmann::json& model_config, int32_t num_layers) {
    std::vector<std::string> layer_types(static_cast<size_t>(num_layers), "full_attention");
    if (!model_config.contains("layer_types") || !model_config["layer_types"].is_array()) {
        return layer_types;
    }

    const auto& raw_layer_types = model_config["layer_types"];
    check<ConfigurationError>(
        static_cast<int32_t>(raw_layer_types.size()) == num_layers,
        "Qwen3_5: layer_types length must equal num_hidden_layers");
    for (int32_t i = 0; i < num_layers; ++i) {
        layer_types[static_cast<size_t>(i)] = raw_layer_types[static_cast<size_t>(i)].get<std::string>();
    }
    return layer_types;
}

std::vector<int32_t> parse_mrope_section(const nlohmann::json& model_config) {
    std::vector<int32_t> section = {11, 11, 10};
    if (!model_config.contains("rope_parameters") || !model_config["rope_parameters"].is_object()) {
        return section;
    }
    const auto& rope_parameters = model_config["rope_parameters"];
    if (!rope_parameters.contains("mrope_section") || !rope_parameters["mrope_section"].is_array()) {
        return section;
    }
    section.clear();
    for (const auto& value : rope_parameters["mrope_section"]) {
        section.push_back(value.get<int32_t>());
    }
    check<ConfigurationError>(section.size() == 3, "Qwen3_5: rope_parameters.mrope_section must have 3 entries");
    return section;
}

size_t shape_nbytes(const std::vector<int64_t>& shape, DType dtype) {
    size_t elements = 1;
    for (int64_t dim : shape) {
        elements *= static_cast<size_t>(dim);
    }
    return elements * get_dtype_size(dtype);
}

} // namespace

Qwen3_5::Qwen3_5(const EngineConfig& config)
    : Model(config)
{
    const nlohmann::json model_config = engine_config_.prefill_model_config();

    check<ConfigurationError>(
        dtype_ == DType::Float16 || dtype_ == DType::BFloat16,
        "Qwen3_5 text runtime currently supports Float16/BFloat16 weights");

    intermediate_size_ = model_config.value("intermediate_size", 0);
    num_attention_heads_ = model_config.value("num_attention_heads", 0);
    num_kv_heads_ = model_config.value("num_key_value_heads", num_attention_heads_);
    head_dim_ = runtime_spec_.head_dim;
    full_attention_dim_ = num_attention_heads_ * head_dim_;
    linear_num_key_heads_ = model_config.value("linear_num_key_heads", 0);
    linear_num_value_heads_ = model_config.value("linear_num_value_heads", linear_num_key_heads_);
    linear_key_head_dim_ = model_config.value("linear_key_head_dim", 0);
    linear_value_head_dim_ = model_config.value("linear_value_head_dim", linear_key_head_dim_);
    linear_key_dim_ = linear_num_key_heads_ * linear_key_head_dim_;
    linear_value_dim_ = linear_num_value_heads_ * linear_value_head_dim_;
    linear_conv_kernel_dim_ = model_config.value("linear_conv_kernel_dim", 4);
    linear_conv_dim_ = linear_key_dim_ * 2 + linear_value_dim_;
    rms_norm_eps_ = model_config.value("rms_norm_eps", 1e-6f);
    layer_types_ = parse_layer_types(model_config, num_layers_);
    mrope_section_ = parse_mrope_section(model_config);
    if (model_config.contains("rope_parameters") && model_config["rope_parameters"].is_object()) {
        const auto& rope_parameters = model_config["rope_parameters"];
        rope_theta_ = rope_parameters.value("rope_theta", rope_theta_);
        const float partial_rotary_factor = rope_parameters.value("partial_rotary_factor", 1.0f);
        rotary_dim_ = static_cast<int32_t>(static_cast<float>(head_dim_) * partial_rotary_factor);
    } else {
        rope_theta_ = model_config.value("rope_theta", rope_theta_);
        rotary_dim_ = head_dim_;
    }

    check<ConfigurationError>(intermediate_size_ > 0, "Qwen3_5: intermediate_size must be positive");
    check<ConfigurationError>(full_attention_dim_ > 0, "Qwen3_5: attention dimensions must be positive");
    check<ConfigurationError>(
        linear_key_dim_ > 0 && linear_value_dim_ > 0 && linear_conv_dim_ == linear_key_dim_ * 2 + linear_value_dim_,
        "Qwen3_5: linear attention dimensions are invalid");
    check<ConfigurationError>(
        linear_num_key_heads_ == linear_num_value_heads_,
        "Qwen3_5 v1 requires equal linear_num_key_heads and linear_num_value_heads");

    embed_head_ = std::make_unique<EmbedHeadLayer>(config, "Qwen3_5_Embedding");
    lm_head_ = std::make_unique<LMHeadLinearLayer>(
        "lm_head",
        config,
        static_cast<uint32_t>(hidden_size_),
        static_cast<uint32_t>(vocab_size_),
        "Qwen3_5_LMHead");
    activation_layer_ = std::make_unique<ActivationLayer>(config, "Qwen3_5_Activation");

    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        const std::string layer_prefix = "layers." + std::to_string(layer_id);
        const std::string hf_layer_prefix = "model.layers." + std::to_string(layer_id);

        if (is_full_attention_layer(layer_id)) {
            linear_[layer_prefix + ".self_attn.q_proj"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".self_attn.q_proj",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(full_attention_dim_ * 2),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_QProj");
            linear_[layer_prefix + ".self_attn.k_proj"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".self_attn.k_proj",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(num_kv_heads_ * head_dim_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_KProj");
            linear_[layer_prefix + ".self_attn.v_proj"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".self_attn.v_proj",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(num_kv_heads_ * head_dim_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_VProj");
            linear_[layer_prefix + ".self_attn.o_proj"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".self_attn.o_proj",
                config,
                static_cast<uint32_t>(full_attention_dim_),
                static_cast<uint32_t>(hidden_size_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_OProj");
            attentions_[layer_id] = std::make_unique<AttentionLayer>(
                config,
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_Attention");
        } else {
            linear_[layer_prefix + ".linear_attn.in_proj_qkv"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".linear_attn.in_proj_qkv",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(linear_conv_dim_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_LinearQKV");
            linear_[layer_prefix + ".linear_attn.in_proj_z"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".linear_attn.in_proj_z",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(linear_value_dim_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_LinearZ");
            linear_[layer_prefix + ".linear_attn.in_proj_b"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".linear_attn.in_proj_b",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(linear_num_value_heads_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_LinearB");
            linear_[layer_prefix + ".linear_attn.in_proj_a"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".linear_attn.in_proj_a",
                config,
                static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(linear_num_value_heads_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_LinearA");
            linear_[layer_prefix + ".linear_attn.out_proj"] = std::make_unique<LinearLayer>(
                hf_layer_prefix + ".linear_attn.out_proj",
                config,
                static_cast<uint32_t>(linear_value_dim_),
                static_cast<uint32_t>(hidden_size_),
                "Qwen3_5_Layer_" + std::to_string(layer_id) + "_LinearOut");
        }

        linear_[layer_prefix + ".mlp.gate_up_fused"] = std::make_unique<FusedGateUpLinearLayer>(
            hf_layer_prefix + ".mlp",
            config,
            static_cast<uint32_t>(hidden_size_),
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(intermediate_size_),
            "Qwen3_5_Layer_" + std::to_string(layer_id) + "_GateUp");
        linear_[layer_prefix + ".mlp.down_proj"] = std::make_unique<LinearLayer>(
            hf_layer_prefix + ".mlp.down_proj",
            config,
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(hidden_size_),
            "Qwen3_5_Layer_" + std::to_string(layer_id) + "_DownProj");
    }

    WeightLoader& loader = WeightLoader::instance();
    prefill_weights_ = loader.take_stage(ModelStage::Prefill);
    decode_weights_ = loader.take_stage_or_empty(ModelStage::Decode);

    embed_head_->load_weights(prefill_weights_, decode_weights_);
    lm_head_->load_weights(prefill_weights_, decode_weights_);
    activation_layer_->load_weights(prefill_weights_, decode_weights_);
    for (auto& [key, layer] : attentions_) {
        (void)key;
        layer->load_weights(prefill_weights_, decode_weights_);
    }
    for (auto& [key, layer] : linear_) {
        (void)key;
        layer->load_weights(prefill_weights_, decode_weights_);
    }
    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        const std::string hf_layer_prefix = "model.layers." + std::to_string(layer_id);
        (void)required_weight(hf_layer_prefix + ".input_layernorm.weight");
        (void)required_weight(hf_layer_prefix + ".post_attention_layernorm.weight");
        if (is_full_attention_layer(layer_id)) {
            (void)required_weight(hf_layer_prefix + ".self_attn.q_norm.weight");
            (void)required_weight(hf_layer_prefix + ".self_attn.k_norm.weight");
        } else {
            (void)required_weight(hf_layer_prefix + ".linear_attn.conv1d.weight");
            (void)required_weight(hf_layer_prefix + ".linear_attn.dt_bias");
            (void)required_weight(hf_layer_prefix + ".linear_attn.A_log");
            (void)required_weight(hf_layer_prefix + ".linear_attn.norm.weight");
        }
    }
    (void)required_weight("model.norm.weight");

    model_loaded_ = true;
}

bool Qwen3_5::is_full_attention_layer(int32_t layer_id) const {
    check<InvalidRequestError>(
        layer_id >= 0 && layer_id < static_cast<int32_t>(layer_types_.size()),
        "Qwen3_5: invalid layer id");
    return layer_types_[static_cast<size_t>(layer_id)] == "full_attention";
}

const Tensor& Qwen3_5::required_weight(const std::string& name) const {
    auto it = prefill_weights_.find(name);
    check<ConfigurationError>(it != prefill_weights_.end(), "Qwen3_5: missing weight '" + name + "'");
    return it->second;
}

LinearLayer& Qwen3_5::linear_layer(const std::string& key) const {
    auto it = linear_.find(key);
    check<InternalError>(it != linear_.end(), "Qwen3_5: missing linear layer '" + key + "'");
    return *it->second;
}

AttentionLayer& Qwen3_5::attention_layer(int32_t layer_id) const {
    auto it = attentions_.find(layer_id);
    check<InternalError>(it != attentions_.end(), "Qwen3_5: missing attention layer");
    return *it->second;
}

Tensor Qwen3_5::workspace_tensor(
    const std::string& name,
    const std::vector<int64_t>& shape,
    DType dtype,
    int32_t device_id) const
{
    void* ptr = StaticBufferManager::get_cache_buf(
        "qwen3_5_" + name,
        shape_nbytes(shape, dtype),
        device_id);
    return Tensor::view(ptr, shape, dtype, Device::GPU, device_id);
}

Tensor Qwen3_5::make_position_ids(
    const Context& context,
    int32_t seq_len,
    cudaStream_t stream,
    ModelStage stage) const
{
    int32_t start = 0;
    if (stage == ModelStage::Prefill) {
        start = static_cast<int32_t>(context.prefix_size());
    } else {
        const Request* request = context.request();
        check<InternalError>(request != nullptr, "Qwen3_5: context request must not be null");
        start = static_cast<int32_t>(request->token_ids().size()) + context.get_generated_tokens() - 1;
    }

    std::vector<int32_t> host(static_cast<size_t>(3 * seq_len));
    for (int32_t axis = 0; axis < 3; ++axis) {
        for (int32_t i = 0; i < seq_len; ++i) {
            host[static_cast<size_t>(axis * seq_len + i)] = start + i;
        }
    }

    const std::string name = (stage == ModelStage::Prefill)
        ? "prefill_position_ids"
        : "decode_position_ids";
    Tensor position_ids = workspace_tensor(name, {3, seq_len}, DType::Int32, engine_config_.runtime_device_id());
    CUDA_CHECK_THROW(cudaMemcpyAsync(
                         position_ids.data_ptr(),
                         host.data(),
                         static_cast<size_t>(host.size()) * sizeof(int32_t),
                         cudaMemcpyHostToDevice,
                         stream),
                     "Qwen3_5: copy position_ids");
    return position_ids;
}

void Qwen3_5::run_mlp(
    int32_t layer_id,
    int32_t seq_len,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    const int32_t device_id = engine_config_.runtime_device_id();
    Tensor activation_input = workspace_tensor(
        "mlp_activation_input_L" + std::to_string(layer_id),
        {seq_len, 2 * intermediate_size_},
        dtype_,
        device_id);
    Tensor intermediate = workspace_tensor(
        "mlp_intermediate_L" + std::to_string(layer_id),
        {seq_len, intermediate_size_},
        dtype_,
        device_id);

    const std::string layer_prefix = "layers." + std::to_string(layer_id);
    auto* gate_up = dynamic_cast<FusedGateUpLinearLayer*>(
        &linear_layer(layer_prefix + ".mlp.gate_up_fused"));
    check<InternalError>(gate_up != nullptr, "Qwen3_5: gate_up layer has unexpected type");

    const bool fused = (stage == ModelStage::Prefill)
        ? gate_up->try_forward_prefill_swiglu_fused(input, intermediate, stream)
        : gate_up->try_forward_decode_swiglu_fused(input, intermediate, stream);
    if (!fused) {
        gate_up->forward_fp16_bf16(input, activation_input, stream, stage);
        activation_layer_->forward_silu_and_mul_up_gate(
            activation_input,
            intermediate,
            stream,
            stage);
    }
    linear_layer(layer_prefix + ".mlp.down_proj").forward_fp16_bf16(
        intermediate,
        output,
        stream,
        stage);
}

void Qwen3_5::run_full_attention_layer(
    const Context& context,
    int32_t layer_id,
    int32_t seq_len,
    const Tensor& norm_output,
    Tensor& mixer_output,
    Tensor& position_ids,
    cudaStream_t stream,
    ModelStage stage)
{
    auto& tensors = const_cast<std::unordered_map<std::string, Tensor>&>(context.tensors());
    const int32_t device_id = engine_config_.runtime_device_id();
    const std::string layer_prefix = "layers." + std::to_string(layer_id);
    const std::string weight_prefix = "model.layers." + std::to_string(layer_id);

    Tensor q_proj = workspace_tensor(
        "full_q_proj_L" + std::to_string(layer_id),
        {seq_len, full_attention_dim_ * 2},
        dtype_,
        device_id);
    Tensor query = workspace_tensor(
        "full_query_L" + std::to_string(layer_id),
        {seq_len, num_attention_heads_, head_dim_},
        dtype_,
        device_id);
    Tensor gate = workspace_tensor(
        "full_gate_L" + std::to_string(layer_id),
        {seq_len, full_attention_dim_},
        dtype_,
        device_id);
    Tensor attention_out = workspace_tensor(
        "full_attention_out_L" + std::to_string(layer_id),
        {seq_len, num_attention_heads_, head_dim_},
        dtype_,
        device_id);

    linear_layer(layer_prefix + ".self_attn.q_proj").forward_fp16_bf16(
        norm_output, q_proj, stream, stage);
    qwen3_5_split_q_gate_forward(q_proj, query, gate, num_attention_heads_, head_dim_, stream);

    Tensor query_2d = Tensor::view(
        query.data_ptr(),
        {static_cast<int64_t>(seq_len) * num_attention_heads_, head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    qwen3_5_rmsnorm_forward(
        query_2d,
        required_weight(weight_prefix + ".self_attn.q_norm.weight"),
        query_2d,
        rms_norm_eps_,
        stream);

    Tensor& k_write = tensors.at(ModelTensors::k_write_layer(layer_id));
    Tensor& v_write = tensors.at(ModelTensors::v_write_layer(layer_id));
    Tensor k_write_2d = Tensor::view(
        k_write.data_ptr(),
        {seq_len, num_kv_heads_ * head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor v_write_2d = Tensor::view(
        v_write.data_ptr(),
        {seq_len, num_kv_heads_ * head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    linear_layer(layer_prefix + ".self_attn.k_proj").forward_fp16_bf16(
        norm_output, k_write_2d, stream, stage);
    linear_layer(layer_prefix + ".self_attn.v_proj").forward_fp16_bf16(
        norm_output, v_write_2d, stream, stage);

    Tensor key_2d = Tensor::view(
        k_write.data_ptr(),
        {static_cast<int64_t>(seq_len) * num_kv_heads_, head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    qwen3_5_rmsnorm_forward(
        key_2d,
        required_weight(weight_prefix + ".self_attn.k_norm.weight"),
        key_2d,
        rms_norm_eps_,
        stream);
    qwen3_5_apply_partial_interleaved_mrope(
        query,
        k_write,
        position_ids,
        mrope_section_,
        rope_theta_,
        rotary_dim_,
        stream);

    if (stage == ModelStage::Prefill) {
        Tensor& k_cache = tensors.at(ModelTensors::k_cache_layer(layer_id));
        Tensor& v_cache = tensors.at(ModelTensors::v_cache_layer(layer_id));
        attention_layer(layer_id).forward_prefill(
            query,
            k_cache,
            v_cache,
            attention_out,
            true,
            stream);
    } else {
        Tensor& k_cache = tensors.at(ModelTensors::k_cache_layer(layer_id));
        Tensor& v_cache = tensors.at(ModelTensors::v_cache_layer(layer_id));
        uint32_t* d_kv_len = nullptr;
        uint32_t max_kv_len = 0;
        auto dkv_it = tensors.find(ModelTensors::D_KV_LEN);
        if (dkv_it != tensors.end()) {
            d_kv_len = static_cast<uint32_t*>(dkv_it->second.data_ptr());
            max_kv_len = static_cast<uint32_t>(k_cache.shape()[0]);
        }
        attention_layer(layer_id).forward_decode(
            query,
            k_cache,
            v_cache,
            attention_out,
            stream,
            d_kv_len,
            max_kv_len);
    }

    Tensor attention_flat = Tensor::view(
        attention_out.data_ptr(),
        {seq_len, full_attention_dim_},
        dtype_,
        Device::GPU,
        device_id);
    qwen3_5_mul_sigmoid_forward(attention_flat, gate, attention_flat, stream);
    linear_layer(layer_prefix + ".self_attn.o_proj").forward_fp16_bf16(
        attention_flat,
        mixer_output,
        stream,
        stage);
}

void Qwen3_5::run_linear_attention_layer(
    const Context& context,
    int32_t layer_id,
    int32_t seq_len,
    const Tensor& norm_output,
    Tensor& mixer_output,
    cudaStream_t stream,
    ModelStage stage)
{
    const int32_t device_id = engine_config_.runtime_device_id();
    const std::string layer_prefix = "layers." + std::to_string(layer_id);
    const std::string weight_prefix = "model.layers." + std::to_string(layer_id) + ".linear_attn";

    Tensor mixed_qkv = workspace_tensor(
        "linear_mixed_qkv_L" + std::to_string(layer_id),
        {seq_len, linear_conv_dim_},
        dtype_,
        device_id);
    Tensor conv_out = workspace_tensor(
        "linear_conv_out_L" + std::to_string(layer_id),
        {seq_len, linear_conv_dim_},
        dtype_,
        device_id);
    Tensor z = workspace_tensor(
        "linear_z_L" + std::to_string(layer_id),
        {seq_len, linear_value_dim_},
        dtype_,
        device_id);
    Tensor a = workspace_tensor(
        "linear_a_L" + std::to_string(layer_id),
        {seq_len, linear_num_value_heads_},
        dtype_,
        device_id);
    Tensor b = workspace_tensor(
        "linear_b_L" + std::to_string(layer_id),
        {seq_len, linear_num_value_heads_},
        dtype_,
        device_id);
    Tensor g = workspace_tensor(
        "linear_g_L" + std::to_string(layer_id),
        {seq_len, linear_num_value_heads_},
        DType::Float32,
        device_id);
    Tensor beta = workspace_tensor(
        "linear_beta_L" + std::to_string(layer_id),
        {seq_len, linear_num_value_heads_},
        DType::Float32,
        device_id);
    Tensor delta_out = workspace_tensor(
        "linear_delta_out_L" + std::to_string(layer_id),
        {seq_len, linear_value_dim_},
        dtype_,
        device_id);
    Tensor gated_norm_out = workspace_tensor(
        "linear_gated_norm_out_L" + std::to_string(layer_id),
        {seq_len, linear_value_dim_},
        dtype_,
        device_id);

    linear_layer(layer_prefix + ".linear_attn.in_proj_qkv").forward_fp16_bf16(
        norm_output, mixed_qkv, stream, stage);
    linear_layer(layer_prefix + ".linear_attn.in_proj_z").forward_fp16_bf16(
        norm_output, z, stream, stage);
    linear_layer(layer_prefix + ".linear_attn.in_proj_b").forward_fp16_bf16(
        norm_output, b, stream, stage);
    linear_layer(layer_prefix + ".linear_attn.in_proj_a").forward_fp16_bf16(
        norm_output, a, stream, stage);

    auto& arena = const_cast<Context&>(context).runtime_state_arena();
    Tensor& conv_state = arena.get_or_create(
        "qwen3_5.layer." + std::to_string(layer_id) + ".conv_state",
        {linear_conv_dim_, linear_conv_kernel_dim_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor& recurrent_state = arena.get_or_create(
        "qwen3_5.layer." + std::to_string(layer_id) + ".recurrent_state",
        {linear_num_value_heads_, linear_key_head_dim_, linear_value_head_dim_},
        DType::Float32,
        Device::GPU,
        device_id);

    qwen3_5_depthwise_causal_conv1d_forward(
        mixed_qkv,
        required_weight(weight_prefix + ".conv1d.weight"),
        conv_state,
        conv_out,
        true,
        stream);
    qwen3_5_compute_g_beta_forward(
        a,
        b,
        required_weight(weight_prefix + ".A_log"),
        required_weight(weight_prefix + ".dt_bias"),
        g,
        beta,
        stream);
    qwen3_5_gated_delta_sequence_forward(
        conv_out,
        g,
        beta,
        recurrent_state,
        delta_out,
        stream);

    Tensor delta_2d = Tensor::view(
        delta_out.data_ptr(),
        {static_cast<int64_t>(seq_len) * linear_num_value_heads_, linear_value_head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor z_2d = Tensor::view(
        z.data_ptr(),
        {static_cast<int64_t>(seq_len) * linear_num_value_heads_, linear_value_head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor gated_norm_2d = Tensor::view(
        gated_norm_out.data_ptr(),
        {static_cast<int64_t>(seq_len) * linear_num_value_heads_, linear_value_head_dim_},
        dtype_,
        Device::GPU,
        device_id);
    qwen3_5_gated_rmsnorm_forward(
        delta_2d,
        z_2d,
        required_weight(weight_prefix + ".norm.weight"),
        gated_norm_2d,
        rms_norm_eps_,
        stream);
    linear_layer(layer_prefix + ".linear_attn.out_proj").forward_fp16_bf16(
        gated_norm_out,
        mixer_output,
        stream,
        stage);
}

void Qwen3_5::forward_impl(const Context& context, int32_t seq_len, ModelStage stage) {
    auto& tensors = const_cast<std::unordered_map<std::string, Tensor>&>(context.tensors());
    cudaStream_t stream = cuda_stream(context);
    const int32_t device_id = engine_config_.runtime_device_id();
    const size_t hidden_bytes =
        static_cast<size_t>(seq_len) * static_cast<size_t>(hidden_size_) * get_dtype_size(dtype_);

    auto embed_inputs = context.make_layer_inputs({{"token_ids", ModelTensors::TOKEN_IDS}});
    auto embed_outputs = context.make_layer_outputs({{"output", ModelTensors::HIDDEN_STATES}});
    embed_head_->forward(embed_inputs, embed_outputs, stream, stage);

    Tensor& hidden_3d = tensors.at(ModelTensors::HIDDEN_STATES);
    Tensor hidden = Tensor::view(
        hidden_3d.data_ptr(),
        {seq_len, hidden_size_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor residual = workspace_tensor("residual", {seq_len, hidden_size_}, dtype_, device_id);
    Tensor norm_output = workspace_tensor("norm_output", {seq_len, hidden_size_}, dtype_, device_id);
    Tensor post_norm_output = workspace_tensor("post_norm_output", {seq_len, hidden_size_}, dtype_, device_id);
    Tensor mixer_output = workspace_tensor("mixer_output", {seq_len, hidden_size_}, dtype_, device_id);
    Tensor position_ids = make_position_ids(context, seq_len, stream, stage);

    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        const std::string weight_prefix = "model.layers." + std::to_string(layer_id);
        CUDA_CHECK_THROW(cudaMemcpyAsync(
                             residual.data_ptr(),
                             hidden.data_ptr(),
                             hidden_bytes,
                             cudaMemcpyDeviceToDevice,
                             stream),
                         "Qwen3_5: copy attention residual");
        qwen3_5_rmsnorm_forward(
            hidden,
            required_weight(weight_prefix + ".input_layernorm.weight"),
            norm_output,
            rms_norm_eps_,
            stream);
        if (is_full_attention_layer(layer_id)) {
            run_full_attention_layer(
                context,
                layer_id,
                seq_len,
                norm_output,
                mixer_output,
                position_ids,
                stream,
                stage);
        } else {
            run_linear_attention_layer(
                context,
                layer_id,
                seq_len,
                norm_output,
                mixer_output,
                stream,
                stage);
        }
        qwen3_5_add_forward(residual, mixer_output, hidden, stream);

        CUDA_CHECK_THROW(cudaMemcpyAsync(
                             residual.data_ptr(),
                             hidden.data_ptr(),
                             hidden_bytes,
                             cudaMemcpyDeviceToDevice,
                             stream),
                         "Qwen3_5: copy mlp residual");
        qwen3_5_rmsnorm_forward(
            hidden,
            required_weight(weight_prefix + ".post_attention_layernorm.weight"),
            post_norm_output,
            rms_norm_eps_,
            stream);
        run_mlp(layer_id, seq_len, post_norm_output, mixer_output, stream, stage);
        qwen3_5_add_forward(residual, mixer_output, hidden, stream);
    }

    qwen3_5_rmsnorm_forward(
        hidden,
        required_weight("model.norm.weight"),
        norm_output,
        rms_norm_eps_,
        stream);

    Tensor& logits = tensors.at(ModelTensors::LOGITS);
    void* lm_head_input_ptr = norm_output.data_ptr();
    int32_t lm_head_rows = seq_len;
    if (stage == ModelStage::Prefill) {
        lm_head_rows = 1;
        lm_head_input_ptr = static_cast<char*>(norm_output.data_ptr()) +
            static_cast<size_t>(seq_len - 1) * static_cast<size_t>(hidden_size_) * get_dtype_size(dtype_);
    }
    Tensor lm_head_input = Tensor::view(
        lm_head_input_ptr,
        {lm_head_rows, hidden_size_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor logits_2d = Tensor::view(
        logits.data_ptr(),
        {lm_head_rows, vocab_size_},
        logits.dtype(),
        Device::GPU,
        device_id);
    lm_head_->forward_fp16_bf16(lm_head_input, logits_2d, stream, stage);
}

void Qwen3_5::prefill(const Context& context) {
    const auto& token_ids = context.tensors().at(ModelTensors::TOKEN_IDS);
    const int32_t seq_len = static_cast<int32_t>(token_ids.shape().back());
    forward_impl(context, seq_len, ModelStage::Prefill);
}

void Qwen3_5::decode_step(const Context& context) {
    forward_impl(context, 1, ModelStage::Decode);
}

void Qwen3_5::reset_operator_impl_caches() {
    if (activation_layer_ != nullptr) {
        activation_layer_->reset_operator_impl_cache();
    }
    if (lm_head_ != nullptr) {
        lm_head_->reset_operator_impl_cache();
    }
    for (auto& [key, layer] : attentions_) {
        (void)key;
        layer->reset_operator_impl_cache();
    }
    for (auto& [key, layer] : linear_) {
        (void)key;
        layer->reset_operator_impl_cache();
    }
}

} // namespace edge_fm
