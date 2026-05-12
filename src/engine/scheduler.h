#pragma once

#include <edge-fm/core.h>
#include "engine/kv_manager.h"
#include "utils/non_copyable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

using EngineStreamHandle = void*;

class Context {
public:
    Context(const Request* request,
            const std::vector<void*>& kv_cache_ptrs,
            const std::vector<void*>& kv_write_ptrs,
            Response* response,
            int32_t max_generated_tokens,
            size_t prefix_size,
            int32_t slot_max_tokens,
            int32_t token_stride,
            EngineStreamHandle stream_handle=nullptr
        ):
            request_(request),
            kv_cache_ptrs_(kv_cache_ptrs),
            kv_write_ptrs_(kv_write_ptrs),
            response_(response),
            max_generated_tokens_(max_generated_tokens),
            generated_tokens_(0),
            prefix_size_(prefix_size),
            slot_max_tokens_(slot_max_tokens),
            token_stride_(token_stride),
            stream_handle_(stream_handle),
            response_tokens_base_ptr_(nullptr),
            eos_detected_(false) { /* empty */ }
    Context(Context&&) noexcept = default;
    Context& operator=(Context&&) noexcept = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() = default;

    bool is_finished() const { return eos_detected_ || generated_tokens_ >= max_generated_tokens_; }
    void finish() { eos_detected_ = true; }
    bool eos_detected() const { return eos_detected_; }

    Context& operator++();
    /// prefill 后调用：generated_tokens += 1，kv_write_ptrs += seq_len（与 operator++ 语义统一，decode 循环仅用 ++context）
    void advance_after_prefill(int32_t seq_len);
    std::vector<void*> get_kv_read_ptrs() const;
    std::vector<void*> get_kv_write_ptrs() const;
    const std::vector<void*>& kv_read_ptrs_ref() const { return kv_cache_ptrs_; }
    const std::vector<void*>& kv_write_ptrs_ref() const { return kv_write_ptrs_; }
    void set_kv_write_ptrs(const std::vector<void*>& ptrs);
    
    const Request* request() const { return request_; }
    Response* response() { return response_; }
    const Response* response() const { return response_; }  
    EngineStreamHandle stream_handle() const { return stream_handle_; }
    
    int32_t get_generated_tokens() const { return generated_tokens_; }
    int32_t max_generated_tokens() const { return max_generated_tokens_; }
    void set_response_tokens_base_ptr(void* p);
    void* get_response_token_write_ptr() const;  // current write slot for sampler
    void* get_response_token_read_ptr() const;   // previous slot for decode TOKEN_IDS; only valid when get_generated_tokens() >= 1
    
    // Tensor maps for model forward pass
    std::unordered_map<std::string, Tensor>& tensors() { return tensors_; }
    const std::unordered_map<std::string, Tensor>& tensors() const { return tensors_; }
    
    /**
     * @brief 创建 layer 的 inputs map
     * 
     * 从 context.tensors() 中获取指定的 tensor，创建 view 并构建 inputs map。
     * 用于简化 model 层创建 layer inputs 的过程。
     * 
     * @param name_mapping Map from layer input name to tensor name in context.tensors()
     *                     e.g., {{"token_ids", "token_ids"}, {"input", "hidden_states"}}
     * @return Layer inputs map with Tensor views
     */
    std::unordered_map<std::string, Tensor> 
    make_layer_inputs(const std::unordered_map<std::string, std::string>& name_mapping) const;
    
    /**
     * @brief 创建 layer 的 outputs map
     * 
     * 从 context.tensors() 中获取指定的 tensor，创建 view 并构建 outputs map。
     * 用于简化 model 层创建 layer outputs 的过程。
     * 
     * @param name_mapping Map from layer output name to tensor name in context.tensors()
     *                     e.g., {{"output", "hidden_states"}}
     * @return Layer outputs map with Tensor views
     */
    std::unordered_map<std::string, Tensor> 
    make_layer_outputs(const std::unordered_map<std::string, std::string>& name_mapping) const;

    /// 模型特定的 per-request 状态（如 M-RoPE last_pos）
    void set_model_state(const std::string& key, std::vector<int32_t> value) {
        model_state_[key] = std::move(value);
    }
    const std::vector<int32_t>* get_model_state(const std::string& key) const {
        auto it = model_state_.find(key);
        return it != model_state_.end() ? &it->second : nullptr;
    }

    void set_decode_cache_kv_len(int32_t v) { decode_cache_kv_len_ = v; }
    int32_t decode_cache_kv_len() const { return decode_cache_kv_len_; }
    size_t prefix_size() const { return prefix_size_; }
    int32_t slot_max_tokens() const { return slot_max_tokens_; }
    bool decode_tensors_initialized() const { return decode_tensors_initialized_; }
    void mark_decode_tensors_initialized() { decode_tensors_initialized_ = true; }

private:
    const Request* request_;
    std::vector<void*> kv_cache_ptrs_;
    std::vector<void*> kv_write_ptrs_;
    Response* response_;
    int32_t max_generated_tokens_;
    int32_t generated_tokens_;
    size_t prefix_size_;
    int32_t slot_max_tokens_;
    int32_t token_stride_;
    EngineStreamHandle stream_handle_;
    void* response_tokens_base_ptr_;
    bool eos_detected_;

    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::vector<int32_t>> model_state_;

    int32_t decode_cache_kv_len_ = 0;
    bool decode_tensors_initialized_ = false;
};

class Scheduler : public NonCopyable {
public:
    explicit Scheduler(std::shared_ptr<KVManager> kv_manager,
                       int32_t max_new_tokens,
                       EngineStreamHandle stream_handle = nullptr);
    virtual ~Scheduler() = default;

    Context create_context(const Request& request, Response* response);

protected:
    void set_stream_handle(EngineStreamHandle stream_handle) { stream_handle_ = stream_handle; }

private:
    EngineStreamHandle stream_handle_ = nullptr;
    std::shared_ptr<KVManager> kv_manager_;
    int32_t max_new_tokens_ = 1;
};

} // namespace edge_fm
