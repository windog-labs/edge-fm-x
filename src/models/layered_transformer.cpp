#include "models/layered_transformer.h"

#include "engine/engine.h"
#include "engine/scheduler.h"
#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/device/weight_loader.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace edge_fm {

LayeredTransformerModel::LayeredTransformerModel(const EngineConfig& config)
    : Model(config)
{
    load_plan_from_config();

    const nlohmann::json model_config = engine_config_.prefill_model_config();
    intermediate_size_ = model_config.value("intermediate_size", 0);
    num_attention_heads_ = model_config.value("num_attention_heads", 32);
    num_kv_heads_ = model_config.value("num_key_value_heads", num_attention_heads_);
    head_dim_ = hidden_size_ / num_attention_heads_;

    check<ConfigurationError>(
        dtype_ == DType::Float16 || dtype_ == DType::BFloat16,
        "LayeredTransformerModel attention only supports Float16 or BFloat16");

    embed_head_ = std::make_unique<EmbedHeadLayer>(config, "Symbolic_Embedding");
    for (int32_t i = 0; i < num_layers_; ++i) {
        const std::string key = "layers." + std::to_string(i) + ".attn";
        attentions_[key] = std::make_unique<AttentionLayer>(
            config,
            "Symbolic_Layer_" + std::to_string(i) + "_Attention");
    }

    for (int32_t i = 0; i < num_layers_; ++i) {
        const std::string input_key = "layers." + std::to_string(i) + ".input_layernorm";
        layernorms_[input_key] = std::make_unique<RMSNormLayer>(
            static_cast<uint32_t>(i),
            NormWeightType::Input,
            config,
            "Symbolic_Layer_" + std::to_string(i) + "_InputNorm");

        const std::string post_key = "layers." + std::to_string(i) + ".post_attention_layernorm";
        layernorms_[post_key] = std::make_unique<RMSNormLayer>(
            static_cast<uint32_t>(i),
            NormWeightType::PostAttention,
            config,
            "Symbolic_Layer_" + std::to_string(i) + "_PostNorm");
    }

    layernorms_["final_norm"] = std::make_unique<RMSNormLayer>(
        UINT32_MAX,
        NormWeightType::Final,
        config,
        "Symbolic_FinalNorm");

    const int32_t q_dim = num_attention_heads_ * head_dim_;
    const int32_t k_dim = num_kv_heads_ * head_dim_;
    const int32_t v_dim = num_kv_heads_ * head_dim_;
    for (int32_t i = 0; i < num_layers_; ++i) {
        const std::string layer_prefix = "layers." + std::to_string(i);
        const std::string hf_attn_prefix = "model.layers." + std::to_string(i) + ".self_attn";
        linear_[layer_prefix + ".attn.qkv_fused"] = std::make_unique<FusedQKVLinearLayer>(
            hf_attn_prefix,
            config,
            static_cast<uint32_t>(hidden_size_),
            static_cast<uint32_t>(q_dim),
            static_cast<uint32_t>(k_dim),
            static_cast<uint32_t>(v_dim),
            "Symbolic_Layer_" + std::to_string(i) + "_QKV");

        linear_[layer_prefix + ".attn.o_proj"] = std::make_unique<LinearLayer>(
            hf_attn_prefix + ".o_proj",
            config,
            static_cast<uint32_t>(num_attention_heads_ * head_dim_),
            static_cast<uint32_t>(hidden_size_),
            "Symbolic_Layer_" + std::to_string(i) + "_OProj");

        const std::string mlp_prefix = "model.layers." + std::to_string(i) + ".mlp";
        linear_[layer_prefix + ".mlp.gate_up_fused"] = std::make_unique<FusedGateUpLinearLayer>(
            mlp_prefix,
            config,
            static_cast<uint32_t>(hidden_size_),
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(intermediate_size_),
            "Symbolic_Layer_" + std::to_string(i) + "_GateUp");

        linear_[layer_prefix + ".mlp.down_proj"] = std::make_unique<LinearLayer>(
            mlp_prefix + ".down_proj",
            config,
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(hidden_size_),
            "Symbolic_Layer_" + std::to_string(i) + "_DownProj");
    }

    activation_layer_ = std::make_unique<ActivationLayer>(config, "Symbolic_Activation");
    lm_head_ = std::make_unique<LMHeadLinearLayer>(
        "lm_head",
        config,
        static_cast<uint32_t>(hidden_size_),
        static_cast<uint32_t>(vocab_size_),
        "Symbolic_LMHead");

    rope_theta_ = model_config.value("rope_theta", 1000000.0f);
    rope_scale_ = 1.0f;
    if (model_config.contains("rope_scaling") && model_config["rope_scaling"].is_object()) {
        const auto rope_scaling = model_config["rope_scaling"];
        if (rope_scaling.contains("factor")) {
            rope_scale_ = rope_scaling["factor"].get<float>();
        }
        const std::string rope_type = rope_scaling.value(
            "type",
            rope_scaling.value("rope_type", std::string("")));
        if (rope_type == "mrope" && rope_scaling.contains("mrope_section")) {
            use_mrope_ = true;
            for (const auto& section : rope_scaling["mrope_section"]) {
                mrope_section_.push_back(section.get<int32_t>());
            }
            int32_t cumulative = 0;
            for (size_t i = 0; i < mrope_section_.size() && i < 3; ++i) {
                cumulative += mrope_section_[i] * 2;
                mrope_section_cumsum_host_[i] = cumulative;
            }
            CUDA_CHECK_THROW(
                cudaMalloc(&mrope_section_cumsum_gpu_, 3 * sizeof(int32_t)),
                "Failed to allocate layered model mrope_section_cumsum");
            CUDA_CHECK_THROW(
                cudaMemcpy(
                    mrope_section_cumsum_gpu_,
                    mrope_section_cumsum_host_,
                    3 * sizeof(int32_t),
                    cudaMemcpyHostToDevice),
                "Failed to copy layered model mrope_section_cumsum to GPU");
        }
    }

    WeightLoader& loader = WeightLoader::instance();
    const auto& prefill_weights = loader.get(ModelStage::Prefill);
    const auto& decode_weights = loader.get(ModelStage::Decode);

    embed_head_->load_weights(prefill_weights, decode_weights);
    lm_head_->load_weights(prefill_weights, decode_weights);
    for (int32_t i = 0; i < num_layers_; ++i) {
        const std::string input_norm_key = "layers." + std::to_string(i) + ".input_layernorm";
        layernorms_[input_norm_key]->load_weights(prefill_weights, decode_weights);
        const std::string post_norm_key = "layers." + std::to_string(i) + ".post_attention_layernorm";
        layernorms_[post_norm_key]->load_weights(prefill_weights, decode_weights);
    }
    layernorms_["final_norm"]->load_weights(prefill_weights, decode_weights);
    for (int32_t i = 0; i < num_layers_; ++i) {
        const std::string attn_key = "layers." + std::to_string(i) + ".attn";
        attentions_[attn_key]->load_weights(prefill_weights, decode_weights);

        const std::string layer_prefix = "layers." + std::to_string(i);
        linear_[layer_prefix + ".attn.qkv_fused"]->load_weights(prefill_weights, decode_weights);
        linear_[layer_prefix + ".attn.o_proj"]->load_weights(prefill_weights, decode_weights);
        linear_[layer_prefix + ".mlp.gate_up_fused"]->load_weights(prefill_weights, decode_weights);
        linear_[layer_prefix + ".mlp.down_proj"]->load_weights(prefill_weights, decode_weights);
    }
    activation_layer_->load_weights(prefill_weights, decode_weights);
}

void LayeredTransformerModel::prefill(const Context& context) {
    const auto& token_ids = context.tensors().at(ModelTensors::TOKEN_IDS);
    const int32_t seq_len = static_cast<int32_t>(token_ids.shape()[1]);
    forward_impl(context, seq_len, ModelStage::Prefill);
}

void LayeredTransformerModel::decode_step(const Context& context) {
    forward_impl(context, 1, ModelStage::Decode);
}

void LayeredTransformerModel::load_plan_from_config() {
    const auto& raw = engine_config_.raw();
    check<ConfigurationError>(
        raw.contains("_edgefm_internal") &&
            raw["_edgefm_internal"].is_object() &&
            raw["_edgefm_internal"].contains("execution_plan"),
        "LayeredTransformerModel requires _edgefm_internal.execution_plan");

    execution_plan_ = ExecutionPlan::from_json(
        raw["_edgefm_internal"]["execution_plan"]);
    check<ConfigurationError>(
        !execution_plan_.prefill_ops.empty(),
        "LayeredTransformerModel execution_plan must not be empty");
}

void LayeredTransformerModel::forward_impl(
    const Context& context,
    int32_t seq_len,
    ModelStage stage)
{
    auto& tensors = const_cast<std::unordered_map<std::string, Tensor>&>(context.tensors());
    cudaStream_t stream = context.stream();
    const Request* request = context.request();
    const size_t dtype_size = get_dtype_size(dtype_);
    const int32_t device_id = engine_config_.runtime_device_id();

    std::unordered_map<std::string, std::string> input_mapping = {
        {"token_ids", ModelTensors::TOKEN_IDS},
    };
    if (request->has_embedding()) {
        input_mapping.emplace("embeddings", ModelTensors::EMBEDDING);
        input_mapping.emplace("embed_token_id", ModelTensors::EMBED_TOKEN_ID);
    }
    auto embed_inputs = context.make_layer_inputs(input_mapping);
    auto embed_outputs = context.make_layer_outputs({{"output", ModelTensors::HIDDEN_STATES}});
    embed_head_->forward(embed_inputs, embed_outputs, stream, stage);

    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        const std::string layer_prefix = "layers." + std::to_string(layer_id);
        Tensor& norm_output = tensors[ModelTensors::NORM_OUTPUT];
        Tensor& hidden_states_tensor = tensors[ModelTensors::HIDDEN_STATES];
        tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(
            hidden_states_tensor.data_ptr(),
            {seq_len, hidden_size_},
            dtype_,
            Device::GPU,
            device_id);
        Tensor& hidden_2d = tensors[ModelTensors::HIDDEN_STATES_RESHAPE];
        Tensor& post_norm_output = tensors[ModelTensors::POST_NORM_OUTPUT];

        const std::string input_norm_key = layer_prefix + ".input_layernorm";
        if (layer_id == 0) {
            auto norm_inputs = context.make_layer_inputs({{"input", ModelTensors::HIDDEN_STATES_RESHAPE}});
            auto norm_outputs = context.make_layer_outputs({{"output", ModelTensors::NORM_OUTPUT}});
            layernorms_[input_norm_key]->forward(norm_inputs, norm_outputs, stream, stage);
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(
                    post_norm_output.data_ptr(),
                    hidden_states_tensor.data_ptr(),
                    seq_len * hidden_size_ * dtype_size,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "Failed to save layered model initial residual");
        } else {
            auto norm_inputs = context.make_layer_inputs({
                {"input", ModelTensors::NORM_OUTPUT},
                {"residual", ModelTensors::POST_NORM_OUTPUT},
            });
            auto norm_outputs = context.make_layer_outputs({{"output", ModelTensors::NORM_OUTPUT}});
            layernorms_[input_norm_key]->forward(norm_inputs, norm_outputs, stream, stage);
        }

        const int32_t q_dim = num_attention_heads_ * head_dim_;
        const int32_t k_dim = num_kv_heads_ * head_dim_;
        const int32_t v_dim = num_kv_heads_ * head_dim_;
        const int32_t qkv_total = q_dim + k_dim + v_dim;
        const size_t qkv_row_bytes = static_cast<size_t>(qkv_total) * dtype_size;
        const size_t q_row_bytes = static_cast<size_t>(q_dim) * dtype_size;
        const size_t k_row_bytes = static_cast<size_t>(k_dim) * dtype_size;
        const size_t v_row_bytes = static_cast<size_t>(v_dim) * dtype_size;

        Tensor& qkv_tensor = tensors[ModelTensors::QKV_PROJ_OUTPUT];
        Tensor& q_tensor = tensors[ModelTensors::Q_PROJ_OUTPUT];
        void* qkv_buf = qkv_tensor.data_ptr();
        void* q_buf = q_tensor.data_ptr();

        Tensor qkv_out = Tensor::view(qkv_buf, {seq_len, qkv_total}, dtype_, Device::GPU, device_id);
        linear_[layer_prefix + ".attn.qkv_fused"]->forward_fp16_bf16(norm_output, qkv_out, stream, stage);

        Tensor& k_write = tensors[ModelTensors::k_write_layer(layer_id)];
        Tensor& v_write = tensors[ModelTensors::v_write_layer(layer_id)];
        if (seq_len == 1) {
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(
                    k_write.data_ptr(),
                    static_cast<char*>(qkv_buf) + q_row_bytes,
                    k_row_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy layered model K to k_write");
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(
                    v_write.data_ptr(),
                    static_cast<char*>(qkv_buf) + q_row_bytes + k_row_bytes,
                    v_row_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy layered model V to v_write");
        } else {
            CUDA_CHECK_THROW(
                cudaMemcpy2DAsync(
                    q_buf,
                    q_row_bytes,
                    qkv_buf,
                    qkv_row_bytes,
                    q_row_bytes,
                    seq_len,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy layered model Q from qkv");
            CUDA_CHECK_THROW(
                cudaMemcpy2DAsync(
                    k_write.data_ptr(),
                    k_row_bytes,
                    static_cast<char*>(qkv_buf) + q_row_bytes,
                    qkv_row_bytes,
                    k_row_bytes,
                    seq_len,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy layered model K to k_write");
            CUDA_CHECK_THROW(
                cudaMemcpy2DAsync(
                    v_write.data_ptr(),
                    v_row_bytes,
                    static_cast<char*>(qkv_buf) + q_row_bytes + k_row_bytes,
                    qkv_row_bytes,
                    v_row_bytes,
                    seq_len,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy layered model V to v_write");
        }

        if (use_mrope_ && tensors.count(ModelTensors::POSITION_IDS)) {
            const int32_t* pos_ids = static_cast<const int32_t*>(
                tensors.at(ModelTensors::POSITION_IDS).data_ptr());
            const int32_t* cumsum = static_cast<const int32_t*>(mrope_section_cumsum_gpu_);
            AttentionLayer::apply_mrope(
                q_buf,
                k_write.data_ptr(),
                pos_ids,
                cumsum,
                seq_len,
                num_attention_heads_,
                num_kv_heads_,
                head_dim_,
                rope_theta_,
                rope_scale_,
                dtype_,
                stream);
        }

        Tensor q_attn = Tensor::view(
            q_buf,
            {seq_len, num_attention_heads_, head_dim_},
            dtype_,
            Device::GPU,
            device_id);
        Tensor& k_cache = tensors[ModelTensors::k_cache_layer(layer_id)];
        Tensor& v_cache = tensors[ModelTensors::v_cache_layer(layer_id)];
        std::unordered_map<std::string, Tensor> attn_inputs;
        attn_inputs.emplace("q", std::move(q_attn));
        attn_inputs.emplace(
            "k",
            Tensor::view(
                k_cache.data_ptr(),
                k_cache.shape(),
                k_cache.dtype(),
                std::get<0>(k_cache.device()),
                std::get<1>(k_cache.device())));
        attn_inputs.emplace(
            "v",
            Tensor::view(
                v_cache.data_ptr(),
                v_cache.shape(),
                v_cache.dtype(),
                std::get<0>(v_cache.device()),
                std::get<1>(v_cache.device())));
        if (tensors.count(ModelTensors::D_KV_LEN)) {
            const Tensor& dkv = tensors.at(ModelTensors::D_KV_LEN);
            attn_inputs.emplace(
                "d_kv_len",
                Tensor::view(
                    dkv.data_ptr(),
                    dkv.shape(),
                    dkv.dtype(),
                    std::get<0>(dkv.device()),
                    std::get<1>(dkv.device())));
        }
        auto attn_outputs = context.make_layer_outputs({{"o", ModelTensors::ATTENTION_OUTPUT}});
        attentions_[layer_prefix + ".attn"]->forward(attn_inputs, attn_outputs, stream, stage);

        Tensor& attn_output = tensors[ModelTensors::ATTENTION_OUTPUT];
        Tensor o_proj_input = Tensor::view(
            attn_output.data_ptr(),
            {seq_len, hidden_size_},
            dtype_,
            Device::GPU,
            device_id);
        Tensor o_proj_output = Tensor::view(
            hidden_states_tensor.data_ptr(),
            {seq_len, hidden_size_},
            dtype_,
            Device::GPU,
            device_id);
        linear_[layer_prefix + ".attn.o_proj"]->forward_fp16_bf16(
            o_proj_input,
            o_proj_output,
            stream,
            stage);

        tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(
            hidden_states_tensor.data_ptr(),
            {seq_len, hidden_size_},
            dtype_,
            Device::GPU,
            device_id);
        auto post_norm_inputs = context.make_layer_inputs({
            {"input", ModelTensors::HIDDEN_STATES_RESHAPE},
            {"residual", ModelTensors::POST_NORM_OUTPUT},
        });
        auto post_norm_outputs = context.make_layer_outputs({{"output", ModelTensors::HIDDEN_STATES_RESHAPE}});
        layernorms_[layer_prefix + ".post_attention_layernorm"]->forward(
            post_norm_inputs,
            post_norm_outputs,
            stream,
            stage);

        Tensor& mlp_activation_input = tensors[ModelTensors::MLP_ACTIVATION_INPUT];
        Tensor gate_up_flat = Tensor::view(
            mlp_activation_input.data_ptr(),
            {seq_len, 2 * intermediate_size_},
            dtype_,
            Device::GPU,
            device_id);
        linear_[layer_prefix + ".mlp.gate_up_fused"]->forward_fp16_bf16(
            hidden_2d,
            gate_up_flat,
            stream,
            stage);

        auto activation_inputs = context.make_layer_inputs({{"input", ModelTensors::MLP_ACTIVATION_INPUT}});
        auto activation_outputs = context.make_layer_outputs({{"output", ModelTensors::MLP_INTERMEDIATE}});
        activation_layer_->forward(activation_inputs, activation_outputs, stream, stage);

        Tensor& mlp_intermediate = tensors[ModelTensors::MLP_INTERMEDIATE];
        if (layer_id < num_layers_ - 1) {
            Tensor norm_output_2d = Tensor::view(
                norm_output.data_ptr(),
                {seq_len, hidden_size_},
                dtype_,
                Device::GPU,
                device_id);
            linear_[layer_prefix + ".mlp.down_proj"]->forward_fp16_bf16(
                mlp_intermediate,
                norm_output_2d,
                stream,
                stage);
        } else {
            linear_[layer_prefix + ".mlp.down_proj"]->forward_fp16_bf16(
                mlp_intermediate,
                hidden_2d,
                stream,
                stage);
        }
    }

    Tensor& hidden_states_final_ref = tensors[ModelTensors::HIDDEN_STATES];
    tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(
        hidden_states_final_ref.data_ptr(),
        {seq_len, hidden_size_},
        dtype_,
        Device::GPU,
        device_id);
    auto final_norm_inputs = context.make_layer_inputs({
        {"input", ModelTensors::HIDDEN_STATES_RESHAPE},
        {"residual", ModelTensors::POST_NORM_OUTPUT},
    });
    auto final_norm_outputs = context.make_layer_outputs({{"output", ModelTensors::HIDDEN_STATES_RESHAPE}});
    layernorms_["final_norm"]->forward(final_norm_inputs, final_norm_outputs, stream, stage);

    Tensor& hidden_states_final = tensors[ModelTensors::HIDDEN_STATES];
    Tensor& logits = tensors[ModelTensors::LOGITS];
    Tensor hidden_states_2d = Tensor::view(
        hidden_states_final.data_ptr(),
        {seq_len, hidden_size_},
        dtype_,
        Device::GPU,
        device_id);
    Tensor logits_2d = Tensor::view(
        logits.data_ptr(),
        {seq_len, vocab_size_},
        logits.dtype(),
        Device::GPU,
        device_id);
    lm_head_->forward_fp16_bf16(hidden_states_2d, logits_2d, stream, stage);
}

void LayeredTransformerModel::prepare_decode_position_ids(
    Context& context,
    Device device,
    int32_t device_id)
{
    if (!use_mrope_) {
        return;
    }
    const std::vector<int32_t>* last_pos = context.get_model_state("mrope_last_pos");
    if (last_pos == nullptr || last_pos->size() < 3) {
        return;
    }

    const int32_t generated = context.get_generated_tokens();
    int32_t decode_pos[3];
    for (int32_t dim = 0; dim < 3; ++dim) {
        decode_pos[dim] = (*last_pos)[dim] + generated;
    }

    cudaStream_t stream = context.stream();
    void* pos_ptr = StaticBufferManager::get_cache_buf(
        "layered_decode_position_ids",
        3 * sizeof(int32_t),
        device_id);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            pos_ptr,
            decode_pos,
            3 * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        "Failed to copy layered model decode position_ids to GPU");
    context.tensors()[ModelTensors::POSITION_IDS] = Tensor::view(
        pos_ptr,
        {3, 1},
        DType::Int32,
        device,
        device_id);
}

} // namespace edge_fm
