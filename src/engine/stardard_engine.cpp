#include "engine/stardard_engine.h"
#include "layers/sampler.h"
#include "models/model.h"
#include "utils/device/memory.h"
#include "utils/device/tuning_cache.h"
#include "engine/kv_manager.h"
#include "utils/device/cuda_utils.h"
#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace edge_fm {

void StandardEngine::warmup() {
    KVManagerStatus kv_status = kv_manager_->get_status();
    bool decode_dry_run_done = false;

    for (const auto& slot : kv_status.slots) {
        if (slot.prefix_token_ids.empty() || slot.prefix_size == 0) {
            continue;
        }

        Request prefix_request(slot.request_id, slot.prefix_token_ids);
        Response temp_response;

        Context context = scheduler_->create_context(prefix_request, &temp_response);
        context.set_kv_write_ptrs(context.get_kv_read_ptrs());

        prepare_tensors(ModelStage::Prefill, context);
        model_->prefill(context);

        cudaStream_t stream = context.stream();
        if (stream != nullptr) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to synchronize stream during warmup");
        }

        // Decode dry-run on first slot: warm up cuBLASLt m=1 descriptors without corrupting KV cache.
        // When CUDA graph is enabled, this also captures the decode graph for later replay.
        if (!decode_dry_run_done) {
            int32_t vocab_size = model_->vocab_size();
            auto& tensors = context.tensors();
            const Tensor& logits_prefill = tensors.at(ModelTensors::LOGITS);
            int32_t seq_len = static_cast<int32_t>(logits_prefill.shape()[0]);
            void* last_row_ptr = static_cast<char*>(logits_prefill.data_ptr())
                + (seq_len - 1) * static_cast<int64_t>(vocab_size) * static_cast<int64_t>(sizeof(float));
            Tensor last_logits = Tensor::view(last_row_ptr, {1, vocab_size}, DType::Float32, device_, device_id_);
            Tensor& token_out_tensor = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);
            std::unordered_map<std::string, Tensor> in, out;
            in.emplace("logits", std::move(last_logits));
            out.emplace("token_ids", std::move(token_out_tensor));
            sampler_->forward(in, out, stream, ModelStage::Prefill);

            context.advance_after_prefill(seq_len);

            prepare_tensors(ModelStage::Decode, context);

            // Redirect KV write to temp buffers to avoid corrupting real cache
            {
                int32_t num_layers = model_->num_layers();
                auto model_config = config_.prefill_model_config();
                int32_t num_attention_heads = model_config.value("num_attention_heads", 32);
                int32_t num_kv_heads = model_config.value("num_key_value_heads", num_attention_heads);
                int32_t head_dim = model_->hidden_size() / num_attention_heads;
                auto kv_dtype = dtype_from_string(config_.kvcache_dtype());
                size_t kv_dtype_size = get_dtype_size(kv_dtype);
                size_t token_stride = kv_manager_->get_token_stride();
                AttentionType attention_type = kv_manager_->get_attention_type();

                if (attention_type == AttentionType::MLA) {
                    size_t ctx_write_bytes = token_stride;
                    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
                        std::string key = "decode_warmup_ctx_L" + std::to_string(layer_id);
                        void* tmp_ptr = StaticBufferManager::get_cache_buf(key, ctx_write_bytes, device_id_);
                        tensors[ModelTensors::context_write_layer(layer_id)] = Tensor::view(
                            tmp_ptr, {1, static_cast<int64_t>(token_stride / kv_dtype_size)},
                            kv_dtype, device_, device_id_);
                    }
                } else {
                    size_t kv_write_bytes = static_cast<size_t>(num_kv_heads) * head_dim * kv_dtype_size;
                    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
                        std::string k_key = "decode_warmup_k_L" + std::to_string(layer_id);
                        std::string v_key = "decode_warmup_v_L" + std::to_string(layer_id);
                        void* k_tmp = StaticBufferManager::get_cache_buf(k_key, kv_write_bytes, device_id_);
                        void* v_tmp = StaticBufferManager::get_cache_buf(v_key, kv_write_bytes, device_id_);
                        tensors[ModelTensors::k_write_layer(layer_id)] = Tensor::view(
                            k_tmp, {1, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
                        tensors[ModelTensors::v_write_layer(layer_id)] = Tensor::view(
                            v_tmp, {1, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
                    }
                }
            }

            model_->decode_step(context);
            if (stream != nullptr) {
                CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync stream during decode warmup");
            }

            if (config_.use_cuda_graph()) {
                // cuBLAS may leave a sticky error from algorithm probing during
                // the dry-run. Clear it so graph capture starts from a clean state.
                (void)cudaGetLastError();
                int32_t nl = model_->num_layers();
                std::vector<void*> k_ptrs(nl), v_ptrs(nl);
                for (int32_t i = 0; i < nl; ++i) {
                    k_ptrs[i] = tensors.at(ModelTensors::k_write_layer(i)).data_ptr();
                    v_ptrs[i] = tensors.at(ModelTensors::v_write_layer(i)).data_ptr();
                }
                cuda_graph_manager_.capture_decode(stream, [&]() {
                    model_->decode_step(context);
                    Tensor logits_view = Tensor::view(
                        tensors.at(ModelTensors::LOGITS).data_ptr(),
                        tensors.at(ModelTensors::LOGITS).shape(),
                        tensors.at(ModelTensors::LOGITS).dtype(), device_, device_id_);
                    Tensor token_out_view = Tensor::view(
                        tensors.at(ModelTensors::SAMPLER_TOKEN_OUT).data_ptr(),
                        tensors.at(ModelTensors::SAMPLER_TOKEN_OUT).shape(),
                        tensors.at(ModelTensors::SAMPLER_TOKEN_OUT).dtype(), device_, device_id_);
                    std::unordered_map<std::string, Tensor> s_in, s_out;
                    s_in.emplace("logits", std::move(logits_view));
                    s_out.emplace("token_ids", std::move(token_out_view));
                    sampler_->forward(s_in, s_out, stream, ModelStage::Decode);
                }, k_ptrs, v_ptrs);
                if (stream != nullptr) {
                    CUDA_CHECK_THROW(cudaStreamSynchronize(stream),
                                     "Failed to sync stream after CUDA graph capture in warmup");
                }
            }

            decode_dry_run_done = true;
        }
    }

    CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to synchronize device after warmup");
}

void StandardEngine::tune() {
    const std::string model_key = config_.tuning_model_key();
    TuningCache::instance().begin_session(model_key);
    try {
        KVManagerStatus kv_status = kv_manager_->get_status();
        for (const auto& slot : kv_status.slots) {
            const int32_t available = slot.max_tokens - static_cast<int32_t>(slot.prefix_size);
            if (available <= 0) {
                continue;
            }
            const int32_t tune_seq_len = std::min(available, 16);
            std::vector<int32_t> token_ids = slot.prefix_token_ids;
            token_ids.insert(token_ids.end(), static_cast<size_t>(tune_seq_len), 0);
            Request request(slot.request_id, token_ids);
            request.set_ignore_stop_tokens(true);
            run_tuning_pass(request);
        }
        TuningCache::instance().end_session();
    } catch (...) {
        TuningCache::instance().end_session();
        throw;
    }
}

Response StandardEngine::generate(const Request& request) {
    Response response;
    Context context = scheduler_->create_context(request, &response);
    cudaStream_t stream = context.stream();
    int32_t vocab_size = model_->vocab_size();
    auto& tensors = context.tensors();

    // Build stop token set: model eos_token_ids + config stop_token_ids + request stop_token_ids
    // When request.ignore_stop_tokens() (e.g. alignment tests), use empty set to generate full steps
    std::unordered_set<int32_t> stop_tokens;
    if (!request.ignore_stop_tokens()) {
        for (int32_t id : config_.eos_token_ids()) stop_tokens.insert(id);
        for (int32_t id : config_.stop_token_ids()) stop_tokens.insert(id);
        for (int32_t id : request.stop_token_ids()) stop_tokens.insert(id);
    }

    prepare_tensors(ModelStage::Prefill, context);
    if (tensors.count(ModelTensors::RESPONSE_TOKENS_DEVICE) == 0) {
        return response;
    }
    Tensor& token_out_tensor = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);

    auto run_sampler = [this](Tensor& logits, Tensor& token_out, cudaStream_t s, ModelStage stage) {
        std::unordered_map<std::string, Tensor> in, out;
        in.emplace("logits", std::move(logits));
        out.emplace("token_ids", std::move(token_out));
        sampler_->forward(in, out, s, stage);
    };

    // Check the last sampled token against stop tokens.
    // Returns true if the token is a stop token.
    int32_t host_token_buf = 0;
    auto check_stop = [&](void* device_token_ptr) -> bool {
        if (stop_tokens.empty()) return false;
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            &host_token_buf, device_token_ptr, sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream), "Failed to copy sampled token to host");
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync for stop token check");
        return stop_tokens.count(host_token_buf) > 0;
    };

    model_->prefill(context);

    const Tensor& logits_prefill = tensors.at(ModelTensors::LOGITS);
    int32_t seq_len = static_cast<int32_t>(logits_prefill.shape()[0]);
    void* last_row_ptr = static_cast<char*>(logits_prefill.data_ptr())
        + (seq_len - 1) * static_cast<int64_t>(vocab_size) * static_cast<int64_t>(sizeof(float));
    Tensor last_logits = Tensor::view(last_row_ptr, {1, vocab_size}, DType::Float32, device_, device_id_);
    void* prefill_write_ptr = context.get_response_token_write_ptr();
    run_sampler(last_logits, token_out_tensor, stream, ModelStage::Prefill);

    if (check_stop(prefill_write_ptr)) {
        context.finish();
    }

    context.advance_after_prefill(seq_len);

    bool use_cuda_graph = config_.use_cuda_graph();

    while (!context.is_finished()) {
        prepare_tensors(ModelStage::Decode, context);
        void* decode_write_ptr = context.get_response_token_write_ptr();

        if (use_cuda_graph) {
            if (!cuda_graph_manager_.is_decode_captured()) {
                // Fallback: capture now if warmup didn't run (no prefix slots).
                int32_t nl = model_->num_layers();
                std::vector<void*> k_ptrs(nl), v_ptrs(nl);
                for (int32_t i = 0; i < nl; ++i) {
                    k_ptrs[i] = tensors.at(ModelTensors::k_write_layer(i)).data_ptr();
                    v_ptrs[i] = tensors.at(ModelTensors::v_write_layer(i)).data_ptr();
                }
                cuda_graph_manager_.capture_decode(stream, [&]() {
                    model_->decode_step(context);
                    Tensor& tok = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);
                    run_sampler(tensors.at(ModelTensors::LOGITS), tok, stream, ModelStage::Decode);
                }, k_ptrs, v_ptrs);
            }
            sync_decode_graph(context);
            cuda_graph_manager_.decode().launch(stream);
        } else {
            model_->decode_step(context);
            Tensor& token_out = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);
            run_sampler(tensors.at(ModelTensors::LOGITS), token_out, stream, ModelStage::Decode);
        }

        flush_sampled_token(decode_write_ptr, stream);

        if (check_stop(decode_write_ptr)) { ++context; context.finish(); break; }
        ++context;
    }

    int32_t num_generated = context.get_generated_tokens();
    std::vector<int32_t> host_tokens(static_cast<size_t>(num_generated));
    void* response_tokens_base = context.tensors().at(ModelTensors::RESPONSE_TOKENS_DEVICE).data_ptr();
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        host_tokens.data(), response_tokens_base,
        static_cast<size_t>(num_generated) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
        "Failed to copy response tokens to host");
    if (stream != nullptr) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync stream before returning response");
    }
    response.token_ids().swap(host_tokens);

    return response;
}

void StandardEngine::prepare_tensors(ModelStage stage, Context& context) {
    if (stage == ModelStage::Prefill) {
        prepare_prefill_tensors(context);
    } else {
        prepare_decode_tensors(context);
    }
}

void StandardEngine::prepare_kvcache_tensors(
    Context& context,
    int32_t num_layers,      // 模型层数
    int32_t num_kv_heads,    // KV head 数
    int32_t head_dim,        // 每个 head 的维度
    int32_t seq_len,         // 当前步处理的 token 数（prefill 为整段长度，decode 为 1）
    size_t prefix_size)      // slot 的 prefix 长度，仅用于 cache 读长度（decode 时） 
{
    auto& tensors = context.tensors();
    
    std::vector<void*> kv_read_ptrs = context.get_kv_read_ptrs();
    std::vector<void*> kv_write_ptrs = context.get_kv_write_ptrs();
    size_t token_stride = kv_manager_->get_token_stride();
    
    auto kv_dtype = dtype_from_string(config_.kvcache_dtype());
    
    // cache_kv_len = attention 读取的 KV 总长度
    // - Prefill (generated_tokens==0): 若 write_ptr 在 read_ptr 之后（prefix 已在 cache 中），
    //   则 cache_kv_len = prefix_size + seq_len；否则仅为 seq_len（warmup 或无 prefix）
    // - Decode: prefix_size + prefill_len + generated_tokens
    int32_t generated_tokens = context.get_generated_tokens();
    int32_t cache_kv_len;
    if (generated_tokens == 0) {
        bool has_prefix_in_cache = (!kv_write_ptrs.empty() && !kv_read_ptrs.empty()
                                    && kv_write_ptrs[0] != kv_read_ptrs[0]);
        cache_kv_len = (has_prefix_in_cache ? static_cast<int32_t>(prefix_size) : 0) + seq_len;
    } else {
        int32_t prefill_len = static_cast<int32_t>(context.request()->token_ids().size());
        if (prefix_size > 0 && prefill_len > static_cast<int32_t>(prefix_size)) {
            prefill_len = prefill_len - static_cast<int32_t>(prefix_size);
        }
        cache_kv_len = static_cast<int32_t>(prefix_size) + prefill_len + generated_tokens;
    }
    context.set_decode_cache_kv_len(cache_kv_len);

    AttentionType attention_type = kv_manager_->get_attention_type();

    if (attention_type == AttentionType::MLA) {
        int64_t stride_elems = static_cast<int64_t>(token_stride / get_dtype_size(kv_dtype));
        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            void* context_read_ptr = kv_read_ptrs[layer_id];

            tensors[ModelTensors::context_write_layer(layer_id)] = Tensor::view(
                kv_write_ptrs[layer_id], {seq_len, stride_elems}, kv_dtype, device_, device_id_);

            tensors[ModelTensors::context_cache_layer(layer_id)] = Tensor::view(
                context_read_ptr, {cache_kv_len, stride_elems}, kv_dtype, device_, device_id_);
        }
    } else {
        size_t k_size_per_token = token_stride / 2;
        int32_t max_tokens = 0;
        {
            KVManagerStatus kv_status = kv_manager_->get_status();
            int32_t req_id = context.request()->request_id();
            for (const auto& slot : kv_status.slots) {
                if (slot.request_id == req_id) { max_tokens = slot.max_tokens; break; }
            }
        }
        bool is_decode = (seq_len == 1 && generated_tokens > 0);
        int32_t cache_shape_len = is_decode ? max_tokens : cache_kv_len;

        if (is_decode) {
            void* d_kv_len_ptr = StaticBufferManager::get_cache_buf(
                "decode_d_kv_len", sizeof(uint32_t), device_id_);
            uint32_t kv_len_val = static_cast<uint32_t>(cache_kv_len);
            CUDA_CHECK_THROW(cudaMemcpyAsync(d_kv_len_ptr, &kv_len_val,
                sizeof(uint32_t), cudaMemcpyHostToDevice, context.stream()),
                "copy d_kv_len to device");
            tensors[ModelTensors::D_KV_LEN] = Tensor::view(
                d_kv_len_ptr, {1}, DType::Int32, device_, device_id_);
        }

        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            void* read_ptr = kv_read_ptrs[layer_id];
            void* write_ptr = kv_write_ptrs[layer_id];

            size_t offset_bytes = static_cast<uint8_t*>(write_ptr) - static_cast<uint8_t*>(read_ptr);
            void* k_write_ptr = write_ptr;
            void* v_write_ptr = static_cast<uint8_t*>(read_ptr) + static_cast<size_t>(max_tokens) * k_size_per_token + offset_bytes;
            tensors[ModelTensors::k_write_layer(layer_id)] = Tensor::view(
                k_write_ptr, {seq_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
            tensors[ModelTensors::v_write_layer(layer_id)] = Tensor::view(
                v_write_ptr, {seq_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);

            void* k_cache_ptr = read_ptr;
            void* v_cache_ptr = static_cast<uint8_t*>(read_ptr) + static_cast<size_t>(max_tokens) * k_size_per_token;
            tensors[ModelTensors::k_cache_layer(layer_id)] = Tensor::view(
                k_cache_ptr, {cache_shape_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
            tensors[ModelTensors::v_cache_layer(layer_id)] = Tensor::view(
                v_cache_ptr, {cache_shape_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
        }
    }
}

void StandardEngine::prepare_prefill_tensors(Context& context) {
    auto& tensors = context.tensors();
    cudaStream_t stream = context.stream();

    // 获取模型参数
    int32_t num_layers = model_->num_layers();
    int32_t hidden_size = model_->hidden_size();
    int32_t vocab_size = model_->vocab_size();
    
    // 获取请求信息：prefix 已在 warmup 写入，generate 时只 prefill 未匹配的 prompt 部分
    const Request* request = context.request();
    const std::vector<int32_t>& token_ids_vec = request->token_ids();
    
    KVManagerStatus kv_status = kv_manager_->get_status();
    size_t prefix_size = 0;
    int32_t max_generated_tokens = 0;
    int32_t request_id = request->request_id();
    for (const auto& slot : kv_status.slots) {
        if (slot.request_id == request_id) {
            prefix_size = slot.prefix_size;
            max_generated_tokens = slot.max_tokens - static_cast<int32_t>(prefix_size);
            break;
        }
    }
    
    // seq_len = 实际写入 KV 的 token 数（prompt 部分，不含 prefix）
    int32_t seq_len = static_cast<int32_t>(token_ids_vec.size());
    if (prefix_size > 0 && token_ids_vec.size() > prefix_size) {
        seq_len = static_cast<int32_t>(token_ids_vec.size() - prefix_size);
    }
    
    // 获取模型配置以计算 attention 参数
    auto model_config = config_.prefill_model_config();
    int32_t num_attention_heads = model_config.value("num_attention_heads", 32);
    int32_t num_kv_heads = model_config.value("num_key_value_heads", num_attention_heads);
    int32_t head_dim = hidden_size / num_attention_heads;
    
    prepare_kvcache_tensors(context, num_layers, num_kv_heads, head_dim, seq_len, prefix_size);
    
    // ==================== 构建输入 tensors ====================
    // 0. Token IDs tensor: [1, seq_len]（有 prefix 时只传 prompt 部分，避免重复写入 prefix）
    const int32_t* token_ids_src = token_ids_vec.data();
    if (prefix_size > 0 && token_ids_vec.size() > prefix_size) {
        token_ids_src = token_ids_vec.data() + prefix_size;
    }
    size_t token_ids_size = static_cast<size_t>(seq_len) * sizeof(int32_t);
    void* token_ids_ptr = MemoryPool::instance().allocate(token_ids_size, stream, device_id_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(token_ids_ptr, token_ids_src, token_ids_size, 
                                     cudaMemcpyHostToDevice, stream), 
                     "Failed to copy token_ids to GPU");
    
    tensors[ModelTensors::TOKEN_IDS] = Tensor::adopt(
        token_ids_ptr,
        {1, seq_len},
        DType::Int32,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 1'. 可选的自定义 embedding: [num_custom_embeddings, hidden_size] 与 embed_token_id
    if (request->has_embedding()) {
        const Tensor& emb = request->embedding();
        auto [src_device, src_device_id] = emb.device();
        tensors[ModelTensors::EMBEDDING] = Tensor::clone_from(
            emb.data_ptr(),
            emb.shape(),
            emb.dtype(),
            src_device, src_device_id,
            device_, device_id_,
            MemoryOwnership::OwnCudaPool,
            stream
        );
        embed_token_id_buf_ = request->embed_token_id();
        tensors[ModelTensors::EMBED_TOKEN_ID] = Tensor::view(
            &embed_token_id_buf_,
            {1},
            DType::Int32,
            Device::CPU,
            0
        );
    }

    // 1''. 可选的 M-RoPE position_ids: [3, seq_len]
    if (request->has_position_ids()) {
        const Tensor& pos = request->position_ids();
        auto [pos_device, pos_device_id] = pos.device();
        tensors[ModelTensors::POSITION_IDS] = Tensor::clone_from(
            pos.data_ptr(),
            pos.shape(),
            pos.dtype(),
            pos_device, pos_device_id,
            device_, device_id_,
            MemoryOwnership::OwnCudaPool,
            stream
        );
        // Extract max position per dimension for decode-phase M-RoPE computation.
        int64_t total_len = pos.shape()[1];
        size_t bytes = 3 * total_len * sizeof(int32_t);
        std::vector<int32_t> pos_cpu(3 * total_len);
        if (pos_device == Device::GPU) {
            CUDA_CHECK_THROW(cudaMemcpyAsync(pos_cpu.data(), pos.data_ptr(), bytes,
                                             cudaMemcpyDeviceToHost, stream),
                             "copy position_ids to CPU");
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "sync after position_ids copy");
        } else {
            std::memcpy(pos_cpu.data(), pos.data_ptr(), bytes);
        }
        std::vector<int32_t> mrope_last_pos(3);
        for (int d = 0; d < 3; ++d) {
            int32_t mx = 0;
            for (int64_t i = 0; i < total_len; ++i) {
                int32_t v = pos_cpu[d * total_len + i];
                if (v > mx) mx = v;
            }
            mrope_last_pos[d] = mx;
        }
        context.set_model_state("mrope_last_pos", std::move(mrope_last_pos));
    }
    
    DType model_dtype = model_->dtype();
    size_t model_dtype_size = get_dtype_size(model_dtype);
    size_t fp32_size = get_dtype_size(DType::Float32);
    // ==================== 构建临时激活值 tensors ====================
    // 1. Hidden states: [batch_size=1, seq_len, hidden_size]（embed 层要求 3D，dtype 与 embedding 一致）
    size_t hidden_states_size = seq_len * hidden_size * model_dtype_size;
    void* hidden_states_ptr = MemoryPool::instance().allocate(hidden_states_size, stream, device_id_);
    tensors[ModelTensors::HIDDEN_STATES] = Tensor::adopt(
        hidden_states_ptr,
        {1, seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 2. Fused QKV projection output: [seq_len, qkv_total_dim]（临时 buffer，MemoryPool）
    // Layout: [Q: num_attention_heads * q_head_dim, K: num_kv_heads * head_dim, V: num_kv_heads * head_dim]
    // 支持多种 attention：MLA 用 q_head_dim，GQA 等用 head_dim
    int32_t q_head_dim = (kv_manager_->get_attention_type() == AttentionType::MLA)
                             ? kv_manager_->get_qk_rope_head_dim()
                             : head_dim;
    int32_t q_dim = num_attention_heads * q_head_dim;
    int32_t k_dim = num_kv_heads * head_dim;
    int32_t v_dim = num_kv_heads * head_dim;
    int32_t qkv_total_dim = q_dim + k_dim + v_dim;
    size_t qkv_proj_size = seq_len * qkv_total_dim * model_dtype_size;
    void* qkv_proj_ptr = MemoryPool::instance().allocate(qkv_proj_size, stream, device_id_);
    tensors[ModelTensors::QKV_PROJ_OUTPUT] = Tensor::adopt(
        qkv_proj_ptr,
        {seq_len, qkv_total_dim},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    // 2.1. Q projection output: 单独分配 [seq_len, num_attention_heads, q_head_dim]，attention 需要连续 stride
    size_t q_size = seq_len * num_attention_heads * q_head_dim * model_dtype_size;
    void* q_ptr = MemoryPool::instance().allocate(q_size, stream, device_id_);
    tensors[ModelTensors::Q_PROJ_OUTPUT] = Tensor::adopt(
        q_ptr,
        {seq_len, num_attention_heads, q_head_dim},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    // K/V 直接写入 k_write/v_write，无需 K_PROJ_OUTPUT、V_PROJ_OUTPUT
    
    // 3. Attention output: [seq_len, hidden_size]
    size_t attn_output_size = seq_len * hidden_size * model_dtype_size;
    void* attn_output_ptr = MemoryPool::instance().allocate(attn_output_size, stream, device_id_);
    tensors[ModelTensors::ATTENTION_OUTPUT] = Tensor::adopt(
        attn_output_ptr,
        {seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 4. MLP intermediate: [seq_len, intermediate_size]
    int32_t intermediate_size = model_config.value("intermediate_size", hidden_size * 4);
    size_t mlp_intermediate_size = seq_len * intermediate_size * model_dtype_size;
    void* mlp_intermediate_ptr = MemoryPool::instance().allocate(mlp_intermediate_size, stream, device_id_);
    tensors[ModelTensors::MLP_INTERMEDIATE] = Tensor::adopt(
        mlp_intermediate_ptr,
        {seq_len, intermediate_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 5. Norm output: [seq_len, hidden_size] (for LayerNorm outputs)
    size_t norm_output_size = seq_len * hidden_size * model_dtype_size;
    void* norm_output_ptr = MemoryPool::instance().allocate(norm_output_size, stream, device_id_);
    tensors[ModelTensors::NORM_OUTPUT] = Tensor::adopt(
        norm_output_ptr,
        {seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 6. Post-attention LayerNorm output: [seq_len, hidden_size]
    size_t post_norm_size = seq_len * hidden_size * model_dtype_size;
    void* post_norm_ptr = MemoryPool::instance().allocate(post_norm_size, stream, device_id_);
    tensors[ModelTensors::POST_NORM_OUTPUT] = Tensor::adopt(
        post_norm_ptr,
        {seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 7. MLP up projection output: [seq_len, intermediate_size]
    size_t up_proj_size = seq_len * intermediate_size * model_dtype_size;
    void* up_proj_ptr = MemoryPool::instance().allocate(up_proj_size, stream, device_id_);
    tensors[ModelTensors::UP_PROJ_OUTPUT] = Tensor::adopt(
        up_proj_ptr,
        {seq_len, intermediate_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 8. MLP activation input: [seq_len, 2 * intermediate_size] (gate + up concatenated)
    size_t mlp_activation_input_size = seq_len * 2 * intermediate_size * model_dtype_size;
    void* mlp_activation_input_ptr = MemoryPool::instance().allocate(mlp_activation_input_size, stream, device_id_);
    tensors[ModelTensors::MLP_ACTIVATION_INPUT] = Tensor::adopt(
        mlp_activation_input_ptr,
        {seq_len, 2 * intermediate_size},
        model_dtype,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );
    
    // 9. Logits: [seq_len, vocab_size]
    size_t logits_size = seq_len * vocab_size * fp32_size;
    void* logits_ptr = MemoryPool::instance().allocate(logits_size, stream, device_id_);
    tensors[ModelTensors::LOGITS] = Tensor::adopt(
        logits_ptr,
        {seq_len, vocab_size},
        DType::Float32,
        device_,
        device_id_,
        MemoryOwnership::OwnCudaPool,
        stream
    );

    // 10. Sampler output 与 11. Response tokens：sampler 直接写入 response 缓冲当前写位置，无需单独缓冲与 D2D copy
    if (max_generated_tokens > 0) {
        void* response_tokens_ptr = MemoryPool::instance().allocate(
            static_cast<size_t>(max_generated_tokens) * sizeof(int32_t), stream, device_id_);
        tensors[ModelTensors::RESPONSE_TOKENS_DEVICE] = Tensor::adopt(
            response_tokens_ptr,
            {max_generated_tokens},
            DType::Int32,
            device_,
            device_id_,
            MemoryOwnership::OwnCudaPool,
            stream
        );
        context.set_response_tokens_base_ptr(response_tokens_ptr);
        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(
            context.get_response_token_write_ptr(),
            {1},
            DType::Int32,
            device_,
            device_id_
        );
    }
}

void StandardEngine::prepare_decode_tensors(Context& context) {
    auto& tensors = context.tensors();
    cudaStream_t stream = context.stream();

    // 获取模型参数
    int32_t num_layers = model_->num_layers();
    int32_t hidden_size = model_->hidden_size();
    int32_t vocab_size = model_->vocab_size();
    
    // Decode 阶段：每次处理 1 个 token
    int32_t seq_len = 1;

    // TOKEN_IDS: input token for this decode step（上一拍采样结果，从 device response 缓冲读取，与 operator++ 对齐）
    if (context.get_generated_tokens() >= 1) {
        void* last_token_src = context.get_response_token_read_ptr();
        void* token_ids_ptr = StaticBufferManager::get_cache_buf("decode_token_ids", sizeof(int32_t), device_id_);
        CUDA_CHECK_THROW(cudaMemcpyAsync(token_ids_ptr, last_token_src, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream), "Failed to copy decode token_ids from response buffer");
        tensors[ModelTensors::TOKEN_IDS] = Tensor::view(
            token_ids_ptr,
            {1, seq_len},
            DType::Int32,
            device_,
            device_id_
        );
    }
    
    // 获取模型配置以计算 attention 参数
    auto model_config = config_.prefill_model_config();
    int32_t num_attention_heads = model_config.value("num_attention_heads", 32);
    int32_t num_kv_heads = model_config.value("num_key_value_heads", num_attention_heads);
    int32_t head_dim = hidden_size / num_attention_heads;
    
    // prepare kvcache tensors(write buffers and read buffers)
    KVManagerStatus kv_status = kv_manager_->get_status();
    size_t prefix_size = 0;
    const Request* request = context.request();
    int32_t request_id = request->request_id();
    for (const auto& slot : kv_status.slots) {
        if (slot.request_id == request_id) {
            prefix_size = slot.prefix_size;
            break;
        }
    }
    prepare_kvcache_tensors(context, num_layers, num_kv_heads, head_dim, seq_len, prefix_size);

    model_->prepare_decode_position_ids(context, device_, device_id_);
    
    DType model_dtype = model_->dtype();
    size_t model_dtype_size = get_dtype_size(model_dtype);
    size_t fp32_size = get_dtype_size(DType::Float32);

    int32_t q_head_dim = (kv_manager_->get_attention_type() == AttentionType::MLA)
                             ? kv_manager_->get_qk_rope_head_dim()
                             : head_dim;
    int32_t q_dim = num_attention_heads * q_head_dim;
    int32_t k_dim = num_kv_heads * head_dim;
    int32_t v_dim = num_kv_heads * head_dim;
    int32_t qkv_total_dim = q_dim + k_dim + v_dim;
    int32_t intermediate_size = model_config.value("intermediate_size", hidden_size * 4);

    // ==================== Decode activation buffers (persistent, cached by StaticBufferManager) ====================
    void* hidden_states_ptr = StaticBufferManager::get_cache_buf("decode_hidden_states",   hidden_size * model_dtype_size, device_id_);
    void* qkv_proj_ptr      = StaticBufferManager::get_cache_buf("decode_qkv_proj",        qkv_total_dim * model_dtype_size, device_id_);
    void* attn_output_ptr   = StaticBufferManager::get_cache_buf("decode_attn_output",     hidden_size * model_dtype_size, device_id_);
    void* norm_output_ptr   = StaticBufferManager::get_cache_buf("decode_norm_output",     hidden_size * model_dtype_size, device_id_);
    void* post_norm_ptr     = StaticBufferManager::get_cache_buf("decode_post_norm",       hidden_size * model_dtype_size, device_id_);
    void* mlp_inter_ptr     = StaticBufferManager::get_cache_buf("decode_mlp_inter",       intermediate_size * model_dtype_size, device_id_);
    void* mlp_act_ptr       = StaticBufferManager::get_cache_buf("decode_mlp_act",         2 * intermediate_size * model_dtype_size, device_id_);
    void* logits_ptr        = StaticBufferManager::get_cache_buf("decode_logits",          vocab_size * fp32_size, device_id_);
    void* sampler_out_ptr   = StaticBufferManager::get_cache_buf("decode_sampler_staging", sizeof(int32_t), device_id_);

    tensors[ModelTensors::HIDDEN_STATES] = Tensor::view(hidden_states_ptr, {1, seq_len, hidden_size}, model_dtype, device_, device_id_);
    tensors[ModelTensors::QKV_PROJ_OUTPUT] = Tensor::view(qkv_proj_ptr, {seq_len, qkv_total_dim}, model_dtype, device_, device_id_);
    tensors[ModelTensors::Q_PROJ_OUTPUT] = Tensor::view(qkv_proj_ptr, {seq_len, num_attention_heads, q_head_dim}, model_dtype, device_, device_id_);
    tensors[ModelTensors::ATTENTION_OUTPUT] = Tensor::view(attn_output_ptr, {seq_len, num_attention_heads, head_dim}, model_dtype, device_, device_id_);
    tensors[ModelTensors::NORM_OUTPUT] = Tensor::view(norm_output_ptr, {seq_len, hidden_size}, model_dtype, device_, device_id_);
    tensors[ModelTensors::POST_NORM_OUTPUT] = Tensor::view(post_norm_ptr, {seq_len, hidden_size}, model_dtype, device_, device_id_);
    tensors[ModelTensors::MLP_INTERMEDIATE] = Tensor::view(mlp_inter_ptr, {seq_len, intermediate_size}, model_dtype, device_, device_id_);
    tensors[ModelTensors::MLP_ACTIVATION_INPUT] = Tensor::view(mlp_act_ptr, {seq_len, 2 * intermediate_size}, model_dtype, device_, device_id_);
    tensors[ModelTensors::LOGITS] = Tensor::view(logits_ptr, {seq_len, vocab_size}, DType::Float32, device_, device_id_);
    tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(sampler_out_ptr, {1}, DType::Int32, device_, device_id_);
}

void StandardEngine::flush_sampled_token(void* write_ptr, cudaStream_t stream) {
    void* staging_ptr = StaticBufferManager::get_cache_buf(
        "decode_sampler_staging", sizeof(int32_t), device_id_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(write_ptr, staging_ptr,
        sizeof(int32_t), cudaMemcpyDeviceToDevice, stream),
        "copy sampled token from staging to response buffer");
}

void StandardEngine::sync_decode_graph(Context& context) {
    if (!cuda_graph_manager_.has_decode_dynamic_nodes()) return;

    // Read K/V write pointers directly from the tensors that
    // prepare_kvcache_tensors already computed for the current step.
    auto& tensors = context.tensors();
    int32_t nl = model_->num_layers();
    std::vector<void*> k(nl), v(nl);
    for (int32_t i = 0; i < nl; ++i) {
        k[i] = tensors.at(ModelTensors::k_write_layer(i)).data_ptr();
        v[i] = tensors.at(ModelTensors::v_write_layer(i)).data_ptr();
    }
    cuda_graph_manager_.update_decode_nodes(k, v);
}

void StandardEngine::run_tuning_pass(const Request& request) {
    Response response;
    Context context = scheduler_->create_context(request, &response);
    cudaStream_t stream = context.stream();
    auto& tensors = context.tensors();
    const int32_t vocab_size = model_->vocab_size();

    prepare_tensors(ModelStage::Prefill, context);
    if (tensors.count(ModelTensors::RESPONSE_TOKENS_DEVICE) == 0) {
        return;
    }
    Tensor& token_out_tensor = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);
    auto run_sampler = [this](Tensor& logits, Tensor& token_out, cudaStream_t s, ModelStage stage) {
        std::unordered_map<std::string, Tensor> in, out;
        in.emplace("logits", std::move(logits));
        out.emplace("token_ids", std::move(token_out));
        sampler_->forward(in, out, s, stage);
    };

    model_->prefill(context);

    const Tensor& logits_prefill = tensors.at(ModelTensors::LOGITS);
    const int32_t seq_len = static_cast<int32_t>(logits_prefill.shape()[0]);
    void* last_row_ptr = static_cast<char*>(logits_prefill.data_ptr())
        + (seq_len - 1) * static_cast<int64_t>(vocab_size) * static_cast<int64_t>(sizeof(float));
    Tensor last_logits = Tensor::view(last_row_ptr, {1, vocab_size}, DType::Float32, device_, device_id_);
    run_sampler(last_logits, token_out_tensor, stream, ModelStage::Prefill);

    context.advance_after_prefill(seq_len);
    if (context.is_finished()) {
        if (stream != nullptr) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync tuning prefill stream");
        }
        return;
    }

    prepare_tensors(ModelStage::Decode, context);
    model_->decode_step(context);
    Tensor& decode_token_out = tensors.at(ModelTensors::SAMPLER_TOKEN_OUT);
    run_sampler(tensors.at(ModelTensors::LOGITS), decode_token_out, stream, ModelStage::Decode);

    if (stream != nullptr) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync tuning decode stream");
    }
}

} // namespace edge_fm
