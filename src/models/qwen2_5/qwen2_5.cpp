#include "models/qwen2_5/qwen2_5.h"
#include "layers/attention.h"
#include "engine/cuda/scheduler.h"
#include "engine/cuda/kernels/decode_runtime_kernels.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/device/weight_loader.h"
#include "utils/check.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace edge_fm {

Qwen2_5::Qwen2_5(const EngineConfig& config) : Model(config)
{
    // ============================================================================
    // create layers
    // ============================================================================
    nlohmann::json model_config = engine_config_.prefill_model_config();
    intermediate_size_ = model_config.value("intermediate_size", 0);
    num_attention_heads_ = model_config.value("num_attention_heads", 32);
    num_kv_heads_ = model_config.value("num_key_value_heads", num_attention_heads_);
    head_dim_ = hidden_size_ / num_attention_heads_;
    // 检查 attention 是否支持模型 dtype
    check<ConfigurationError>(
        dtype_ == DType::Float16 || dtype_ == DType::BFloat16,
        "Qwen2_5 attention only supports Float16 or BFloat16, got torch_dtype from config");
    // embed_head
    embed_head_ = std::make_unique<EmbedHeadLayer>(config, "Embedding");
    // attention layers
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string key = "layers." + std::to_string(i) + ".attn";
        attentions_[key] = std::make_unique<AttentionLayer>(config, "Layer_" + std::to_string(i) + "_Attention");
    }
    // layernorm layers（必须指定 NormWeightType，否则 post_attention 会错误加载 input_layernorm 权重）
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string input_key = "layers." + std::to_string(i) + ".input_layernorm";
        layernorms_[input_key] = std::make_unique<RMSNormLayer>(static_cast<uint32_t>(i), NormWeightType::Input, config, "Layer_" + std::to_string(i) + "_InputNorm");
        
        std::string post_key = "layers." + std::to_string(i) + ".post_attention_layernorm";
        layernorms_[post_key] = std::make_unique<RMSNormLayer>(static_cast<uint32_t>(i), NormWeightType::PostAttention, config, "Layer_" + std::to_string(i) + "_PostNorm");
    }
    // final norm (use UINT32_MAX as special layer_id)
    layernorms_["final_norm"] = std::make_unique<RMSNormLayer>(UINT32_MAX, NormWeightType::Final, config, "FinalNorm");
    // QKV projection: FusedQKVLinear（合并 q_proj+k_proj+v_proj，计算等价，减少 kernel 调用）
    int32_t q_dim = num_attention_heads_ * head_dim_;
    int32_t k_dim = num_kv_heads_ * head_dim_;
    int32_t v_dim = num_kv_heads_ * head_dim_;
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string layer_prefix = "layers." + std::to_string(i);
        std::string hf_attn_prefix = "model.layers." + std::to_string(i) + ".self_attn";
        linear_[layer_prefix + ".attn.qkv_fused"] = std::make_unique<FusedQKVLinearLayer>(
            hf_attn_prefix,
            config,
            static_cast<uint32_t>(hidden_size_),
            static_cast<uint32_t>(q_dim),
            static_cast<uint32_t>(k_dim),
            static_cast<uint32_t>(v_dim),
            "Layer_" + std::to_string(i) + "_QKVLinear");

        std::string o_key = "layers." + std::to_string(i) + ".attn.o_proj";
        linear_[o_key] = std::make_unique<LinearLayer>(
            "model.layers." + std::to_string(i) + ".self_attn.o_proj",
            config,
            static_cast<uint32_t>(num_attention_heads_ * head_dim_),
            static_cast<uint32_t>(hidden_size_),
            "Layer_" + std::to_string(i) + "_OProj");
    }
    // MLP linear layers
    for (int32_t i = 0; i < num_layers_; ++i) {
        // Fused gate+up projection layer
        std::string gate_up_key = "layers." + std::to_string(i) + ".mlp.gate_up_fused";
        linear_[gate_up_key] = std::make_unique<FusedGateUpLinearLayer>(
            "model.layers." + std::to_string(i) + ".mlp",
            config,
            static_cast<uint32_t>(hidden_size_),
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(intermediate_size_),
            "Layer_" + std::to_string(i) + "_GateUp");
        
        std::string down_key = "layers." + std::to_string(i) + ".mlp.down_proj";
        linear_[down_key] = std::make_unique<LinearLayer>(
            "model.layers." + std::to_string(i) + ".mlp.down_proj", 
            config,
            static_cast<uint32_t>(intermediate_size_),
            static_cast<uint32_t>(hidden_size_),
            "Layer_" + std::to_string(i) + "_DownProj");
    }
    activation_layer_ = std::make_unique<ActivationLayer>(config, "Activation");
    lm_head_ = std::make_unique<LMHeadLinearLayer>(
        "lm_head",
        config,
        static_cast<uint32_t>(hidden_size_),
        static_cast<uint32_t>(vocab_size_),
        "LMHead");

    // M-RoPE configuration
    rope_theta_ = model_config.value("rope_theta", 1000000.0f);
    rope_scale_ = 1.0f;
    if (model_config.contains("rope_scaling") && model_config["rope_scaling"].is_object()) {
        auto rope_scaling = model_config["rope_scaling"];
        if (rope_scaling.contains("factor")) {
            rope_scale_ = rope_scaling["factor"].get<float>();
        }
        std::string rope_type = rope_scaling.value("type",
                                    rope_scaling.value("rope_type", std::string("")));
        if (rope_type == "mrope" && rope_scaling.contains("mrope_section")) {
            use_mrope_ = true;
            for (auto& s : rope_scaling["mrope_section"]) {
                mrope_section_.push_back(s.get<int32_t>());
            }
            int32_t cum = 0;
            for (size_t i = 0; i < mrope_section_.size() && i < 3; ++i) {
                cum += mrope_section_[i];
                mrope_section_cumsum_host_[i] = cum;
            }
            CUDA_CHECK_THROW(cudaMalloc(&mrope_section_cumsum_gpu_, 3 * sizeof(int32_t)),
                             "Failed to allocate mrope_section_cumsum");
            CUDA_CHECK_THROW(cudaMemcpy(mrope_section_cumsum_gpu_, mrope_section_cumsum_host_,
                                        3 * sizeof(int32_t), cudaMemcpyHostToDevice),
                             "Failed to copy mrope_section_cumsum to GPU");
        }
    }

    // ============================================================================
    // load weights
    // ============================================================================
    WeightLoader& loader = WeightLoader::instance();
    prefill_weights_ = loader.take_stage(ModelStage::Prefill);
    decode_weights_ = loader.take_stage_or_empty(ModelStage::Decode);
    const auto& prefill_weights = prefill_weights_;
    const auto& decode_weights = decode_weights_;
    
    // embed_head (load first, as lm_head depends on its embedding table)
    embed_head_->load_weights(prefill_weights, decode_weights);
    // LM head (gets tied weight from embed_head)
    lm_head_->load_weights(prefill_weights, decode_weights);
    // layernorm
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string input_norm_key = "layers." + std::to_string(i) + ".input_layernorm";
        layernorms_[input_norm_key]->load_weights(prefill_weights, decode_weights);
        
        std::string post_norm_key = "layers." + std::to_string(i) + ".post_attention_layernorm";
        layernorms_[post_norm_key]->load_weights(prefill_weights, decode_weights);
    }
    // final norm
    layernorms_["final_norm"]->load_weights(prefill_weights, decode_weights);
    // attention: no weights, but need to call load_weights for setting weights_loaded_ to true
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string attn_key = "layers." + std::to_string(i) + ".attn";
        attentions_[attn_key]->load_weights(prefill_weights, decode_weights);
    }
    // linear
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string layer_prefix = "layers." + std::to_string(i);
        linear_[layer_prefix + ".attn.qkv_fused"]->load_weights(prefill_weights, decode_weights);

        std::string o_key = "layers." + std::to_string(i) + ".attn.o_proj";
        linear_[o_key]->load_weights(prefill_weights, decode_weights);
        
        std::string gate_up_key = "layers." + std::to_string(i) + ".mlp.gate_up_fused";
        linear_[gate_up_key]->load_weights(prefill_weights, decode_weights);
        
        std::string down_key = "layers." + std::to_string(i) + ".mlp.down_proj";
        linear_[down_key]->load_weights(prefill_weights, decode_weights);
    }
    // activation
    activation_layer_->load_weights(prefill_weights, decode_weights);
}

Qwen2_5::~Qwen2_5() {
    if (mrope_section_cumsum_gpu_ != nullptr) {
        (void)cudaFree(mrope_section_cumsum_gpu_);
        mrope_section_cumsum_gpu_ = nullptr;
    }
}

void Qwen2_5::reset_operator_impl_caches() {
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
    for (auto& [key, layer] : layernorms_) {
        (void)key;
        layer->reset_operator_impl_cache();
    }
}

AttentionLayer* Qwen2_5::attention_layer(int32_t layer_id) const {
    const std::string key = "layers." + std::to_string(layer_id) + ".attn";
    auto it = attentions_.find(key);
    return it != attentions_.end() ? it->second.get() : nullptr;
}

LinearLayer* Qwen2_5::linear_layer(const std::string& key) const {
    auto it = linear_.find(key);
    return it != linear_.end() ? it->second.get() : nullptr;
}

FusedGateUpLinearLayer* Qwen2_5::fused_gate_up_layer(int32_t layer_id) const {
    const std::string key = "layers." + std::to_string(layer_id) + ".mlp.gate_up_fused";
    auto it = linear_.find(key);
    return it != linear_.end() ? dynamic_cast<FusedGateUpLinearLayer*>(it->second.get()) : nullptr;
}

void Qwen2_5::prefill(const Context& context) {
    // Use actual prefill seq_len from TOKEN_IDS (with prefix, engine passes only non-prefix tokens)
    const auto& token_ids = context.tensors().at(ModelTensors::TOKEN_IDS);
    int32_t seq_len = static_cast<int32_t>(token_ids.shape()[1]);
    forward_impl(context, seq_len, ModelStage::Prefill);
}

void Qwen2_5::decode_step(const Context& context) {
    forward_impl(context, 1, ModelStage::Decode);
}

void Qwen2_5::forward_prefill(
    int32_t seq_len,
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream)
{
    size_t dtype_size = get_dtype_size(dtype_);
    int32_t device_id = engine_config_.runtime_device_id();

    // 1. Embedding
    const Tensor& token_ids = inputs.at("token_ids");
    Tensor& embedding_out = outputs.at("embedding");
    embed_head_->forward_for_tokens(token_ids, embedding_out, stream);

    // 2. 构建 layer 的 input（embedding 的 2D 视图）
    void* embed_ptr = embedding_out.data_ptr();
    std::unordered_map<std::string, Tensor> layer_inputs;
    layer_inputs.emplace("input", Tensor::view(embed_ptr, {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));

    // 3. 分配所有层的 outputs，复用单组 layer 缓冲（除 hidden_states_out 外）
    int32_t q_dim = num_attention_heads_ * head_dim_;
    int32_t k_dim = num_kv_heads_ * head_dim_;
    int32_t v_dim = num_kv_heads_ * head_dim_;
    Tensor& last_decoder_out = outputs.at("last_decoder_output");
    std::unordered_map<std::string, Tensor> layer_outputs;
    layer_outputs.emplace("output", Tensor::view(last_decoder_out.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));

    void* norm_buf = MemoryPool::instance().allocate(seq_len * hidden_size_ * dtype_size, stream, device_id);
    int32_t qkv_total = q_dim + k_dim + v_dim;
    void* qkv_buf = MemoryPool::instance().allocate(seq_len * qkv_total * dtype_size, stream, device_id);
    void* q_buf = use_mrope_
        ? MemoryPool::instance().allocate(seq_len * q_dim * dtype_size, stream, device_id)
        : qkv_buf;
    void* k_buf = MemoryPool::instance().allocate(seq_len * k_dim * dtype_size, stream, device_id);
    void* v_buf = MemoryPool::instance().allocate(seq_len * v_dim * dtype_size, stream, device_id);
    void* attn_buf = MemoryPool::instance().allocate(seq_len * hidden_size_ * dtype_size, stream, device_id);
    void* post_norm_buf = MemoryPool::instance().allocate(seq_len * hidden_size_ * dtype_size, stream, device_id);
    void* gate_up_buf = MemoryPool::instance().allocate(seq_len * 2 * intermediate_size_ * dtype_size, stream, device_id);
    void* mlp_inter_buf = MemoryPool::instance().allocate(seq_len * intermediate_size_ * dtype_size, stream, device_id);
    void* layer_out_buf = MemoryPool::instance().allocate(seq_len * hidden_size_ * dtype_size, stream, device_id);

    std::vector<void*> hidden_bufs(num_layers_);
    for (int32_t i = 0; i < num_layers_; ++i) {
        hidden_bufs[i] = MemoryPool::instance().allocate(seq_len * hidden_size_ * dtype_size, stream, device_id);
    }
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string prefix = "layer" + std::to_string(i) + "_";
        layer_outputs.emplace(prefix + "norm_output", Tensor::view(norm_buf, {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "q_proj_output", Tensor::view(q_buf, {seq_len, q_dim}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "k_proj_output", Tensor::view(k_buf, {seq_len, k_dim}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "v_proj_output", Tensor::view(v_buf, {seq_len, k_dim}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "attention_output", Tensor::view(attn_buf, {seq_len, num_attention_heads_, head_dim_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "post_attn_norm_output", Tensor::view(post_norm_buf, {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "mlp_activation_input", Tensor::view(gate_up_buf, {seq_len, 2 * intermediate_size_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "mlp_intermediate", Tensor::view(mlp_inter_buf, {seq_len, intermediate_size_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "output", Tensor::view(layer_out_buf, {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
        layer_outputs.emplace(prefix + "hidden_states_out", Tensor::view(hidden_bufs[i], {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
    }

    // 3b. Decoder layers
    const Tensor& initial_input = layer_inputs.at("input");
    for (int32_t i = 0; i < num_layers_; ++i) {
        std::string prefix = "layer" + std::to_string(i) + "_";
        const Tensor& hidden_input = (i == 0) ? initial_input : layer_outputs.at("layer" + std::to_string(i - 1) + "_hidden_states_out");
        Tensor& hidden_out = layer_outputs.at(prefix + "hidden_states_out");

        Tensor& norm_output = layer_outputs.at(prefix + "norm_output");
        Tensor& attn_output = layer_outputs.at(prefix + "attention_output");
        Tensor& post_attn_norm_output = layer_outputs.at(prefix + "post_attn_norm_output");
        Tensor& mlp_activation_input = layer_outputs.at(prefix + "mlp_activation_input");
        Tensor& mlp_intermediate = layer_outputs.at(prefix + "mlp_intermediate");
        Tensor& layer_output = layer_outputs.at(prefix + "output");

        std::string layer_prefix = "layers." + std::to_string(i);
        auto [input_device, input_device_id] = hidden_input.device();
        auto [out_device, out_device_id] = norm_output.device();
        std::unordered_map<std::string, Tensor> norm_inputs;
        if (i > 0) {
            // Fused: 用上一层的 mlp_output 做 input，residual 为上一层的 (layer_input + attn_output)
            // FusedAddRMSNorm 会更新 residual = residual + input，即 prev_layer_hidden = full_residual
            const Tensor& prev_layer_output = layer_outputs.at("layer" + std::to_string(i - 1) + "_output");
            Tensor& prev_layer_hidden = layer_outputs.at("layer" + std::to_string(i - 1) + "_hidden_states_out");
            size_t copy_bytes = seq_len * hidden_size_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(norm_output.data_ptr(), prev_layer_output.data_ptr(),
                                              copy_bytes, cudaMemcpyDeviceToDevice, stream),
                             "copy prev mlp_output to norm_output");
            norm_inputs.emplace("input", Tensor::view(norm_output.data_ptr(), norm_output.shape(),
                norm_output.dtype(), out_device, out_device_id));
            norm_inputs.emplace("residual", Tensor::view(
                prev_layer_hidden.data_ptr(), prev_layer_hidden.shape(),
                prev_layer_hidden.dtype(), std::get<0>(prev_layer_hidden.device()), std::get<1>(prev_layer_hidden.device())));
        } else {
            norm_inputs.emplace("input", Tensor::view(
                const_cast<void*>(hidden_input.data_ptr()), hidden_input.shape(),
                hidden_input.dtype(), input_device, input_device_id));
        }
        std::unordered_map<std::string, Tensor> norm_outputs;
        norm_outputs.emplace("output", Tensor::view(
            norm_output.data_ptr(), norm_output.shape(),
            norm_output.dtype(), out_device, out_device_id));
        layernorms_[layer_prefix + ".input_layernorm"]->forward(norm_inputs, norm_outputs, stream, ModelStage::Prefill);

        // hidden_out = 本层输入（用于 post_attn residual）。layer 0: embedding；layer 1+: 上一层 full_residual
        // layer 1+ 时 prev_layer_hidden 已在 input_layernorm 中被更新为 full_residual，需在此复制
        if (i > 0) {
            Tensor& prev_layer_hidden = layer_outputs.at("layer" + std::to_string(i - 1) + "_hidden_states_out");
            CUDA_CHECK_THROW(cudaMemcpyAsync(hidden_out.data_ptr(), prev_layer_hidden.data_ptr(),
                                              seq_len * hidden_size_ * dtype_size,
                                              cudaMemcpyDeviceToDevice, stream),
                             "hidden_out = full_residual (after input_layernorm fused add)");
        } else {
            CUDA_CHECK_THROW(cudaMemcpyAsync(hidden_out.data_ptr(), hidden_input.data_ptr(),
                                              seq_len * hidden_size_ * dtype_size,
                                              cudaMemcpyDeviceToDevice, stream),
                             "hidden_out = embedding (layer 0 residual)");
        }

        int32_t device_id = std::get<1>(hidden_input.device());
        // FusedQKV: 一次 matmul 输出 [seq_len, q_dim+k_dim+v_dim]，再按 stride 拷贝到 q/k/v buffer 供 attention
        Tensor qkv_output = Tensor::view(qkv_buf, {seq_len, qkv_total}, dtype_, Device::GPU, device_id);
        linear_[layer_prefix + ".attn.qkv_fused"]->forward_fp16_bf16(norm_output, qkv_output, stream, ModelStage::Prefill);
        size_t qkv_row_bytes = static_cast<size_t>(qkv_total) * dtype_size;
        size_t q_row_bytes = static_cast<size_t>(q_dim) * dtype_size;
        size_t k_row_bytes = static_cast<size_t>(k_dim) * dtype_size;
        size_t v_row_bytes = static_cast<size_t>(v_dim) * dtype_size;
        void* q_attn_ptr = q_buf;
        uint32_t q_attn_stride_n = static_cast<uint32_t>(q_dim);
        if (use_mrope_) {
            CUDA_CHECK_THROW(cudaMemcpy2DAsync(q_buf, q_row_bytes, qkv_buf, qkv_row_bytes, q_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy Q from qkv");
        } else {
            q_attn_ptr = qkv_buf;
            q_attn_stride_n = static_cast<uint32_t>(qkv_total);
        }
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(k_buf, k_row_bytes, static_cast<char*>(qkv_buf) + q_row_bytes, qkv_row_bytes, k_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy K from qkv");
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(v_buf, v_row_bytes, static_cast<char*>(qkv_buf) + q_row_bytes + k_row_bytes, qkv_row_bytes, v_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy V from qkv");

        Tensor q_view = Tensor::view(q_attn_ptr, {seq_len, num_attention_heads_, head_dim_}, dtype_, Device::GPU, device_id);
        Tensor k_view = Tensor::view(k_buf, {seq_len, num_kv_heads_, head_dim_}, dtype_, Device::GPU, device_id);
        Tensor v_view = Tensor::view(v_buf, {seq_len, num_kv_heads_, head_dim_}, dtype_, Device::GPU, device_id);
        attentions_[layer_prefix + ".attn"]->forward_prefill(
            q_view,
            k_view,
            v_view,
            attn_output,
            true,
            stream,
            q_attn_stride_n,
            static_cast<uint32_t>(head_dim_));

        Tensor o_proj_in = Tensor::view(attn_output.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        Tensor o_proj_out = Tensor::view(attn_output.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        linear_[layer_prefix + ".attn.o_proj"]->forward_fp16_bf16(o_proj_in, o_proj_out, stream, ModelStage::Prefill);

        CUDA_CHECK_THROW(cudaMemcpyAsync(post_attn_norm_output.data_ptr(), attn_output.data_ptr(),
                                          seq_len * hidden_size_ * dtype_size,
                                          cudaMemcpyDeviceToDevice, stream),
                         "Failed to copy attn_output to post_attn_norm_output");

        auto [post_in_dev, post_in_dev_id] = post_attn_norm_output.device();
        std::unordered_map<std::string, Tensor> post_norm_inputs;
        post_norm_inputs.emplace("input", Tensor::view(
            post_attn_norm_output.data_ptr(), post_attn_norm_output.shape(),
            post_attn_norm_output.dtype(), post_in_dev, post_in_dev_id));
        // post_attn residual = 本层输入（attn 前的 hidden），FusedAddRMSNorm 会将其更新为 attn+residual
        post_norm_inputs.emplace("residual", Tensor::view(
            hidden_out.data_ptr(), hidden_out.shape(),
            hidden_out.dtype(), std::get<0>(hidden_out.device()), std::get<1>(hidden_out.device())));
        std::unordered_map<std::string, Tensor> post_norm_outputs;
        post_norm_outputs.emplace("output", Tensor::view(
            post_attn_norm_output.data_ptr(), post_attn_norm_output.shape(),
            post_attn_norm_output.dtype(), post_in_dev, post_in_dev_id));
        layernorms_[layer_prefix + ".post_attention_layernorm"]->forward(post_norm_inputs, post_norm_outputs, stream, ModelStage::Prefill);

        std::string gate_up_key = layer_prefix + ".mlp.gate_up_fused";
        bool prefill_swiglu_fused = false;
        if (auto* fused_gate_up = dynamic_cast<FusedGateUpLinearLayer*>(linear_[gate_up_key].get());
            fused_gate_up != nullptr) {
            prefill_swiglu_fused = fused_gate_up->try_forward_prefill_swiglu_fused(
                post_attn_norm_output, mlp_intermediate, stream);
        }
        if (!prefill_swiglu_fused) {
            Tensor gate_up_flat = Tensor::view(mlp_activation_input.data_ptr(), {seq_len, 2 * intermediate_size_}, dtype_, Device::GPU, device_id);
            linear_[gate_up_key]->forward_fp16_bf16(post_attn_norm_output, gate_up_flat, stream, ModelStage::Prefill);
            activation_layer_->forward_silu_and_mul_up_gate(
                mlp_activation_input, mlp_intermediate, stream, ModelStage::Prefill);
        }

        linear_[layer_prefix + ".mlp.down_proj"]->forward_fp16_bf16(mlp_intermediate, layer_output, stream, ModelStage::Prefill);
    }

    std::string last_prefix = "layer" + std::to_string(num_layers_ - 1) + "_";
    Tensor& last_hidden = layer_outputs.at(last_prefix + "hidden_states_out");
    Tensor& last_layer_output = layer_outputs.at(last_prefix + "output");

    // 4. Final norm: 最后一层无下一层，由 final_norm 做 FusedAddRMSNorm(input=mlp_output, residual=layer_input+attn_output)
    // FusedAddRMSNorm 要求 input==output，故先复制 mlp_output 到 final_norm_out，再 inplace 计算
    Tensor& final_norm_out = outputs.at("final_norm_output");
    CUDA_CHECK_THROW(cudaMemcpyAsync(final_norm_out.data_ptr(), last_layer_output.data_ptr(),
                                      seq_len * hidden_size_ * dtype_size,
                                      cudaMemcpyDeviceToDevice, stream),
                     "copy mlp_output to final_norm_out for fused");
    std::unordered_map<std::string, Tensor> fn_inputs;
    fn_inputs.emplace("input", Tensor::view(final_norm_out.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
    fn_inputs.emplace("residual", Tensor::view(last_hidden.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
    std::unordered_map<std::string, Tensor> fn_outputs;
    fn_outputs.emplace("output", Tensor::view(final_norm_out.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id));
    layernorms_["final_norm"]->forward(fn_inputs, fn_outputs, stream, ModelStage::Prefill);

    // last_decoder_output 需为 full residual（用于 dump 对比），FusedAddRMSNorm 已把 residual 更新为 input+residual
    CUDA_CHECK_THROW(cudaMemcpyAsync(last_decoder_out.data_ptr(), last_hidden.data_ptr(),
                                      seq_len * hidden_size_ * dtype_size,
                                      cudaMemcpyDeviceToDevice, stream),
                     "copy full residual to last_decoder_output");

    // 5. LM head（输出 Float32 供 sampler/对齐测试）
    Tensor& lm_head_out = outputs.at("lm_head_output");
    Tensor final_norm_view = Tensor::view(final_norm_out.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
    Tensor lm_head_view = Tensor::view(lm_head_out.data_ptr(), {seq_len, vocab_size_}, lm_head_out.dtype(), Device::GPU, device_id);
    lm_head_->forward_fp16_bf16(final_norm_view, lm_head_view, stream, ModelStage::Prefill);
}

void Qwen2_5::forward_impl(const Context& context, int32_t seq_len, ModelStage stage) {
    auto& tensors = const_cast<std::unordered_map<std::string, Tensor>&>(context.tensors());
    cudaStream_t stream = cuda_stream(context);
    const Request* request = context.request();
    size_t dtype_size = get_dtype_size(dtype_);
    int32_t device_id = engine_config_.runtime_device_id();

    // 1. Embedding: token_ids -> hidden_states
    std::unordered_map<std::string, std::string> input_mapping = {{"token_ids", ModelTensors::TOKEN_IDS}};
    if (request->has_embedding()) {
        input_mapping.emplace("embeddings", ModelTensors::EMBEDDING);
        input_mapping.emplace("embed_token_id", ModelTensors::EMBED_TOKEN_ID);
    }
    auto embed_inputs = context.make_layer_inputs(input_mapping);
    auto embed_outputs = context.make_layer_outputs({{"output", ModelTensors::HIDDEN_STATES}});
    embed_head_->forward(embed_inputs, embed_outputs, stream, stage);

    // 2. Transformer layers（HIDDEN_STATES 为 3D [1, seq_len, hidden_size]，layernorm/linear 需要 2D reshape）
    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        std::string layer_prefix = "layers." + std::to_string(layer_id);
        
        // Get shared tensors for this layer
        Tensor& norm_output = tensors[ModelTensors::NORM_OUTPUT];
        Tensor& hidden_states_tensor = tensors[ModelTensors::HIDDEN_STATES];
        tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(hidden_states_tensor.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        Tensor& hidden_2d = tensors[ModelTensors::HIDDEN_STATES_RESHAPE];
        Tensor& post_norm_output = tensors[ModelTensors::POST_NORM_OUTPUT];
        
        // Input LayerNorm
        std::string input_norm_key = layer_prefix + ".input_layernorm";
        if (layer_id == 0) {
            // Layer 0: simple RMSNorm (no residual), then save embedding as initial residual.
            layernorms_[input_norm_key]->forward_rmsnorm(hidden_2d, norm_output, stream, stage);
            CUDA_CHECK_THROW(cudaMemcpyAsync(post_norm_output.data_ptr(), hidden_states_tensor.data_ptr(),
                                             seq_len * hidden_size_ * dtype_size,
                                             cudaMemcpyDeviceToDevice, stream),
                             "Failed to save initial residual");
        } else {
            // Layer 1+: NORM_OUTPUT already contains previous layer's MLP output (down_proj wrote there).
            // FusedAddRMSNorm(inout=NORM_OUTPUT, residual=POST_NORM_OUTPUT) — no copy needed.
            layernorms_[input_norm_key]->forward_fused_add_rmsnorm(
                norm_output, post_norm_output, stream, stage);
        }
        
        // QKV projection: FusedQKVLinear，输出到 context 提供的 QKV_PROJ_OUTPUT
        int32_t q_dim = num_attention_heads_ * head_dim_;
        int32_t k_dim = num_kv_heads_ * head_dim_;
        int32_t v_dim = num_kv_heads_ * head_dim_;
        int32_t qkv_total = q_dim + k_dim + v_dim;
        size_t qkv_row_bytes = static_cast<size_t>(qkv_total) * dtype_size;
        size_t q_row_bytes = static_cast<size_t>(q_dim) * dtype_size;
        size_t k_row_bytes = static_cast<size_t>(k_dim) * dtype_size;
        size_t v_row_bytes = static_cast<size_t>(v_dim) * dtype_size;
        int32_t device_id = std::get<1>(norm_output.device());

        Tensor& qkv_tensor = tensors[ModelTensors::QKV_PROJ_OUTPUT];
        Tensor& q_tensor = tensors[ModelTensors::Q_PROJ_OUTPUT];
        void* qkv_buf = qkv_tensor.data_ptr();
        void* q_buf = q_tensor.data_ptr();
        uint32_t q_attn_stride_n = static_cast<uint32_t>(q_dim);
        uint32_t q_attn_stride_h = static_cast<uint32_t>(head_dim_);

        Tensor qkv_out = Tensor::view(qkv_buf, {seq_len, qkv_total}, dtype_, Device::GPU, device_id);
        linear_[layer_prefix + ".attn.qkv_fused"]->forward_fp16_bf16(norm_output, qkv_out, stream, stage);
        Tensor& k_write = tensors[ModelTensors::k_write_layer(layer_id)];
        Tensor& v_write = tensors[ModelTensors::v_write_layer(layer_id)];
        Tensor& k_cache = tensors[ModelTensors::k_cache_layer(layer_id)];
        Tensor& v_cache = tensors[ModelTensors::v_cache_layer(layer_id)];
        void* k_rotated_src = static_cast<char*>(qkv_buf) + q_row_bytes;
        void* v_src = static_cast<char*>(qkv_buf) + q_row_bytes + k_row_bytes;
        const uint32_t* d_kv_len = nullptr;
        if (seq_len == 1 && tensors.count(ModelTensors::D_KV_LEN)) {
            d_kv_len = static_cast<const uint32_t*>(tensors.at(ModelTensors::D_KV_LEN).data_ptr());
        }
        const bool use_fused_decode_mrope_write_kv =
            (seq_len == 1) && (d_kv_len != nullptr) && use_mrope_ &&
            tensors.count(ModelTensors::POSITION_IDS);
        const bool use_combined_decode_kv_copy =
            (seq_len == 1) && (d_kv_len != nullptr) && !use_mrope_;
        if (seq_len == 1) {
            if (use_combined_decode_kv_copy) {
                launch_copy_decode_kv_cache_slots(
                    k_rotated_src,
                    v_src,
                    k_cache.data_ptr(),
                    v_cache.data_ptr(),
                    static_cast<int>(k_dim),
                    static_cast<int>(v_dim),
                    dtype_,
                    d_kv_len,
                    stream);
                CUDA_CHECK_THROW(cudaGetLastError(), "copy K/V to decode cache slots");
            } else if (use_fused_decode_mrope_write_kv) {
                // Handled by the fused decode M-RoPE + KV write kernel below.
            } else if (d_kv_len != nullptr) {
                launch_copy_decode_cache_slot(
                    v_src,
                    v_cache.data_ptr(),
                    static_cast<int>(v_dim),
                    dtype_,
                    d_kv_len,
                    stream);
                CUDA_CHECK_THROW(cudaGetLastError(), "copy V to decode cache slot");
            } else {
                CUDA_CHECK_THROW(cudaMemcpyAsync(v_write.data_ptr(),
                    v_src,
                    v_row_bytes, cudaMemcpyDeviceToDevice, stream), "copy V to v_write");
            }
        } else {
            // Prefill: multi-row strided copy required
            if (use_mrope_) {
                CUDA_CHECK_THROW(cudaMemcpy2DAsync(q_buf, q_row_bytes, qkv_buf, qkv_row_bytes, q_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy Q from qkv");
            } else {
                q_buf = qkv_buf;
                q_attn_stride_n = static_cast<uint32_t>(qkv_total);
            }
            CUDA_CHECK_THROW(cudaMemcpy2DAsync(k_write.data_ptr(), k_row_bytes, static_cast<char*>(qkv_buf) + q_row_bytes, qkv_row_bytes, k_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy K to k_write");
            CUDA_CHECK_THROW(cudaMemcpy2DAsync(v_write.data_ptr(), v_row_bytes, v_src, qkv_row_bytes, v_row_bytes, seq_len, cudaMemcpyDeviceToDevice, stream), "copy V to v_write");
            k_rotated_src = k_write.data_ptr();
        }

        // M-RoPE: rotate Q in-place. For decode, rotate K in the static QKV buffer
        // first, then copy the rotated K into the dynamic cache write location.
        if (use_fused_decode_mrope_write_kv) {
            const int32_t* pos_ids = static_cast<const int32_t*>(
                tensors.at(ModelTensors::POSITION_IDS).data_ptr());
            const int32_t* cumsum = static_cast<const int32_t*>(mrope_section_cumsum_gpu_);
            launch_decode_mrope_apply_q_write_kv(
                q_buf,
                k_rotated_src,
                v_src,
                q_buf,
                k_cache.data_ptr(),
                v_cache.data_ptr(),
                num_attention_heads_,
                num_kv_heads_,
                head_dim_,
                pos_ids,
                cumsum,
                rope_theta_,
                rope_scale_,
                dtype_,
                d_kv_len,
                stream);
            CUDA_CHECK_THROW(cudaGetLastError(), "decode fused M-RoPE + KV write");
        } else if (use_mrope_ && tensors.count(ModelTensors::POSITION_IDS)) {
            const int32_t* pos_ids = static_cast<const int32_t*>(
                tensors.at(ModelTensors::POSITION_IDS).data_ptr());
            const int32_t* cumsum = static_cast<const int32_t*>(mrope_section_cumsum_gpu_);
            AttentionLayer::apply_mrope(q_buf, k_rotated_src,
                        pos_ids, cumsum,
                        seq_len, num_attention_heads_, num_kv_heads_, head_dim_,
                        rope_theta_, rope_scale_, dtype_, stream);
        }
        if (seq_len == 1) {
            if (use_combined_decode_kv_copy) {
                // K/V were already copied together directly from the fused QKV buffer.
            } else if (use_fused_decode_mrope_write_kv) {
                // K/V were already written by the fused decode M-RoPE path.
            } else if (d_kv_len != nullptr) {
                launch_copy_decode_cache_slot(
                    k_rotated_src,
                    k_cache.data_ptr(),
                    static_cast<int>(k_dim),
                    dtype_,
                    d_kv_len,
                    stream);
                CUDA_CHECK_THROW(cudaGetLastError(), "copy rotated K to decode cache slot");
            } else {
                CUDA_CHECK_THROW(cudaMemcpyAsync(k_write.data_ptr(),
                    k_rotated_src,
                    k_row_bytes, cudaMemcpyDeviceToDevice, stream), "copy rotated K to k_write");
            }
        }

        std::string attn_key = layer_prefix + ".attn";
        Tensor q_attn = Tensor::view(q_buf, {seq_len, num_attention_heads_, head_dim_}, dtype_, Device::GPU, device_id);
        Tensor& attn_output = tensors[ModelTensors::ATTENTION_OUTPUT];
        if (stage == ModelStage::Prefill) {
            attentions_[attn_key]->forward_prefill(
                q_attn,
                k_cache,
                v_cache,
                attn_output,
                true,
                stream,
                q_attn_stride_n,
                q_attn_stride_h);
        } else {
            uint32_t* decode_kv_len = nullptr;
            uint32_t max_kv_len = 0;
            auto dkv_it = tensors.find(ModelTensors::D_KV_LEN);
            if (dkv_it != tensors.end()) {
                decode_kv_len = static_cast<uint32_t*>(dkv_it->second.data_ptr());
                max_kv_len = static_cast<uint32_t>(k_cache.shape()[0]);
            }
            attentions_[attn_key]->forward_decode(
                q_attn, k_cache, v_cache, attn_output, stream, decode_kv_len, max_kv_len);
        }
        
        // Output projection: read from ATTENTION_OUTPUT, write to HIDDEN_STATES (avoids D2D copy)
        std::string o_key = layer_prefix + ".attn.o_proj";
        Tensor o_proj_input = Tensor::view(attn_output.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        Tensor o_proj_output = Tensor::view(hidden_states_tensor.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        linear_[o_key]->forward_fp16_bf16(o_proj_input, o_proj_output, stream, stage);
        
        // Post-attention LayerNorm with fused add+rmsnorm (HIDDEN_STATES already has o_proj output)
        tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(hidden_states_tensor.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
        std::string post_norm_key = layer_prefix + ".post_attention_layernorm";
        layernorms_[post_norm_key]->forward_fused_add_rmsnorm(
            hidden_2d, post_norm_output, stream, stage);
        
        // MLP 1: Fused gate+up projection
        std::string gate_up_key = layer_prefix + ".mlp.gate_up_fused";
        std::string down_key = layer_prefix + ".mlp.down_proj";
        Tensor& mlp_activation_input = tensors[ModelTensors::MLP_ACTIVATION_INPUT];
        Tensor gate_up_flat = Tensor::view(mlp_activation_input.data_ptr(), {seq_len, 2 * intermediate_size_}, dtype_, Device::GPU, device_id);
        Tensor& mlp_intermediate = tensors[ModelTensors::MLP_INTERMEDIATE];
        bool swiglu_fused = false;
        if (auto* fused_gate_up = dynamic_cast<FusedGateUpLinearLayer*>(linear_[gate_up_key].get());
            fused_gate_up != nullptr) {
            if (stage == ModelStage::Decode) {
                swiglu_fused = fused_gate_up->try_forward_decode_swiglu_fused(
                    hidden_2d, mlp_intermediate, stream);
            } else if (seq_len >= 64) {
                swiglu_fused = fused_gate_up->try_forward_prefill_swiglu_fused(
                    hidden_2d, mlp_intermediate, stream);
            }
        }
        if (!swiglu_fused) {
            linear_[gate_up_key]->forward_fp16_bf16(hidden_2d, gate_up_flat, stream, stage);
            activation_layer_->forward_silu_and_mul_up_gate(
                mlp_activation_input, mlp_intermediate, stream, stage);
        }
        // MLP 3: Down projection — write to NORM_OUTPUT (for next layer) or HIDDEN_STATES (last layer, for final_norm)
        if (layer_id < num_layers_ - 1) {
            Tensor norm_output_2d = Tensor::view(norm_output.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
            linear_[down_key]->forward_fp16_bf16(mlp_intermediate, norm_output_2d, stream, stage);
        } else {
            linear_[down_key]->forward_fp16_bf16(mlp_intermediate, hidden_2d, stream, stage);
        }
    }

    // 3. Final norm: hidden_states, _ = self.norm(hidden_states, residual)
    Tensor& hidden_states_final_ref = tensors[ModelTensors::HIDDEN_STATES];
    tensors[ModelTensors::HIDDEN_STATES_RESHAPE] = Tensor::view(hidden_states_final_ref.data_ptr(), {seq_len, hidden_size_}, dtype_, Device::GPU, device_id);
    Tensor& hidden_states_final_2d = tensors[ModelTensors::HIDDEN_STATES_RESHAPE];
    Tensor& final_residual = tensors[ModelTensors::POST_NORM_OUTPUT];
    layernorms_["final_norm"]->forward_fused_add_rmsnorm(
        hidden_states_final_2d, final_residual, stream, stage);
    
    // 4. LM head: decode uses the single-step hidden state, while prefill only
    // needs the final token's logits for sampling the first generated token.
    Tensor& hidden_states_final = tensors[ModelTensors::HIDDEN_STATES];
    Tensor& logits = tensors[ModelTensors::LOGITS];
    void* lm_head_input_ptr = hidden_states_final.data_ptr();
    int32_t lm_head_rows = seq_len;
    if (stage == ModelStage::Prefill) {
        lm_head_rows = 1;
        lm_head_input_ptr = static_cast<char*>(hidden_states_final.data_ptr()) +
            static_cast<size_t>(seq_len - 1) * static_cast<size_t>(hidden_size_) * dtype_size;
    }
    Tensor hidden_states_2d = Tensor::view(
        lm_head_input_ptr, {lm_head_rows, hidden_size_}, dtype_, Device::GPU, device_id);
    Tensor logits_2d = Tensor::view(
        logits.data_ptr(), {lm_head_rows, vocab_size_}, logits.dtype(), Device::GPU, device_id);
    lm_head_->forward_fp16_bf16(hidden_states_2d, logits_2d, stream, stage);
}

void Qwen2_5::prepare_decode_position_ids(Context& context, Device device, int32_t device_id) {
    if (!use_mrope_) return;
    const std::vector<int32_t>* last_pos = context.get_model_state("mrope_last_pos");
    if (last_pos == nullptr || last_pos->size() < 3) return;

    int32_t gen = context.get_generated_tokens();
    int32_t decode_pos[3];
    for (int d = 0; d < 3; ++d) {
        decode_pos[d] = (*last_pos)[d] + gen;
    }
    cudaStream_t stream = cuda_stream(context);
    auto& tensors = context.tensors();
    void* pos_ptr = nullptr;
    auto pos_it = tensors.find(ModelTensors::POSITION_IDS);
    if (pos_it != tensors.end()) {
        pos_ptr = pos_it->second.data_ptr();
    } else {
        pos_ptr = StaticBufferManager::get_cache_buf("decode_position_ids", 3 * sizeof(int32_t), device_id);
        tensors[ModelTensors::POSITION_IDS] = Tensor::view(pos_ptr, {3, 1}, DType::Int32, device, device_id);
    }
    CUDA_CHECK_THROW(cudaMemcpyAsync(pos_ptr, decode_pos, 3 * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream),
                     "Failed to copy decode position_ids to GPU");
}

void Qwen2_5::advance_decode_runtime_tensors(Context& context, cudaStream_t stream) {
    if (!use_mrope_) return;

    auto& tensors = context.tensors();
    auto it = tensors.find(ModelTensors::POSITION_IDS);
    if (it == tensors.end()) return;

    launch_increment_int32_triplet(
        static_cast<int32_t*>(it->second.data_ptr()), stream);
    CUDA_CHECK_THROW(cudaGetLastError(), "Failed to advance decode position_ids");
}

std::vector<int32_t> Qwen2_5::derive_mrope_last_pos(
    const int32_t* position_ids,
    int64_t total_len) const
{
    std::vector<int32_t> last_pos = Model::derive_mrope_last_pos(position_ids, total_len);
    if (use_mrope_ && hidden_size_ >= 3584) {
        for (int32_t& pos : last_pos) {
            pos += 1;
        }
    }
    return last_pos;
}

} // namespace edge_fm
