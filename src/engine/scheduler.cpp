#include "engine/scheduler.h"
#include "engine/kv_manager.h"
#include "utils/check.h"
#include <algorithm>
#include <cassert>
#include <edge-fm/core.h>
#include <stdexcept>

namespace edge_fm {

Context& Context::operator++() {
    if (is_finished()) {
        return *this;
    }
    generated_tokens_++;
    
    // 更新 kv_write_ptrs_：前进一个token的kvcache位置
    for (auto& ptr : kv_write_ptrs_) {
        ptr = static_cast<uint8_t*>(ptr) + token_stride_;
    }
    
    return *this;
}

void Context::advance_after_prefill(int32_t seq_len) {
    if (seq_len <= 0) return;
    generated_tokens_++;
    size_t stride = static_cast<size_t>(seq_len) * static_cast<size_t>(token_stride_);
    for (auto& ptr : kv_write_ptrs_) {
        ptr = static_cast<uint8_t*>(ptr) + stride;
    }
}

std::vector<void*> Context::get_kv_read_ptrs() const {
    return kv_cache_ptrs_;
}

std::vector<void*> Context::get_kv_write_ptrs() const {
    // Note: always return kv_write_ptrs_, even if finished
    // The finished check is for safety, but should not return empty vector
    // as callers expect valid pointers for tensor creation
    return kv_write_ptrs_;
}

void Context::set_kv_write_ptrs(const std::vector<void*>& ptrs) {
    kv_write_ptrs_ = ptrs;
}

void Context::set_response_tokens_base_ptr(void* p) {
    response_tokens_base_ptr_ = p;
}

void* Context::get_response_token_write_ptr() const {
    return static_cast<char*>(response_tokens_base_ptr_) + static_cast<size_t>(generated_tokens_) * sizeof(int32_t);
}

void* Context::get_response_token_read_ptr() const {
    assert(generated_tokens_ >= 1 && "only valid in decode when at least one token generated");
    return static_cast<char*>(response_tokens_base_ptr_) + static_cast<size_t>(generated_tokens_ - 1) * sizeof(int32_t);
}

Context Scheduler::create_context(const Request& request, Response* response) {
    KVManagerStatus status = kv_manager_->get_status();
    
    int32_t request_id = request.request_id();
    const auto& request_token_ids = request.token_ids();
    
    // find the matched slot for the request
    const KVSlotStatus* matched_slot = nullptr;
    for (const auto& slot : status.slots) {
        if (slot.request_id == request_id) {
            matched_slot = &slot;
            break;
        }
    }
    
    check<InvalidRequestError>(matched_slot != nullptr, 
                               "Cannot find slot for request_id: " + std::to_string(request_id));
    
    // check if prefix_token_ids is matched
    if (!matched_slot->prefix_token_ids.empty()) {
        check<InvalidRequestError>(
            request_token_ids.size() >= matched_slot->prefix_token_ids.size(),
            "Request token_ids length (" + std::to_string(request_token_ids.size()) + 
            ") is less than prefix_token_ids length (" + 
            std::to_string(matched_slot->prefix_token_ids.size()) + 
            ") for request_id: " + std::to_string(request_id)
        );
        
        for (size_t i = 0; i < matched_slot->prefix_token_ids.size(); ++i) {
            check<InvalidRequestError>(
                request_token_ids[i] == matched_slot->prefix_token_ids[i],
                "Request token_ids prefix does not match prefix_token_ids for request_id: " + 
                std::to_string(request_id) + ". " +
                "Mismatch at position " + std::to_string(i) + ": " +
                "expected " + std::to_string(matched_slot->prefix_token_ids[i]) + 
                ", got " + std::to_string(request_token_ids[i])
            );
        }
    }
    
    int32_t matched_request_id = request_id;
    check<InvalidRequestError>(
        kv_manager_->is_request_valid(matched_request_id),
        "Invalid request_id: " + std::to_string(matched_request_id)
    );
    
    std::vector<void*> kv_cache_read_ptrs = kv_manager_->get_read_kvcache(matched_request_id);
    std::vector<void*> kv_cache_write_ptrs = kv_manager_->get_write_kvcache(matched_request_id);
    size_t kvcache_token_stride = kv_manager_->get_token_stride();
    
    // For MHA/GQA: token_stride is K+V combined, but write_ptr advances by K only
    // For MLA: token_stride is the full cache size per token
    size_t k_cache_token_stride;
    if (kv_manager_->get_attention_type() == AttentionType::MLA) {
        k_cache_token_stride = kvcache_token_stride;
    } else {
        // MHA/GQA/MQA: K cache uses half of the combined stride
        k_cache_token_stride = kvcache_token_stride / 2;
    }
    
    // max_generated_tokens: KV total = prefix + non_prefix_prefill + decode = token_ids.size() + decode
    // => decode <= max_tokens - token_ids.size(), plus 1 for the prefill sample
    int32_t max_generated_tokens = 0;
    for (const auto& slot : status.slots) {
        if (slot.request_id == matched_request_id) {
            int32_t seq_len = static_cast<int32_t>(request_token_ids.size());
            int32_t kv_slots_for_decode = slot.max_tokens - seq_len;
            max_generated_tokens = std::max(1, kv_slots_for_decode + 1);
            break;
        }
    }
    
    Context context(&request,
                    kv_cache_read_ptrs,
                    kv_cache_write_ptrs,
                    response,
                    max_generated_tokens,
                    matched_slot->prefix_size,
                    matched_slot->max_tokens,
                    static_cast<int32_t>(k_cache_token_stride),
                    stream_);

    // Context::tensors_ is populated with dozens of string-keyed entries during
    // prefill/decode. Reserve upfront to avoid repeated rehash on the hot path.
    const size_t per_layer_tensor_count =
        (kv_manager_->get_attention_type() == AttentionType::MLA) ? 2u : 4u;
    const size_t base_tensor_count = 24u;
    context.tensors().reserve(base_tensor_count + per_layer_tensor_count * kv_cache_read_ptrs.size());

    return context;
}

Scheduler::~Scheduler() {
    cudaStreamDestroy(stream_);
}

Scheduler::Scheduler(std::shared_ptr<KVManager> kv_manager): kv_manager_(kv_manager) {
    CUDA_CHECK_THROW_EX(cudaStreamCreate(&stream_), "Failed to create CUDA stream", DeviceError);
}

std::unordered_map<std::string, Tensor> Context::make_layer_inputs(
    const std::unordered_map<std::string, std::string>& name_mapping) const {
    std::unordered_map<std::string, Tensor> result;
    result.reserve(name_mapping.size());
    
    for (const auto& [layer_input_name, tensor_name] : name_mapping) {
        const Tensor& tensor = tensors_.at(tensor_name);
        auto device_info = tensor.device();
        result.emplace(layer_input_name, Tensor::view(
            tensor.data_ptr(),
            tensor.shape(),
            tensor.dtype(),
            std::get<0>(device_info),
            std::get<1>(device_info)
        ));
    }
    
    return result;
}

std::unordered_map<std::string, Tensor> Context::make_layer_outputs(
    const std::unordered_map<std::string, std::string>& name_mapping) const {
    std::unordered_map<std::string, Tensor> result;
    result.reserve(name_mapping.size());
    
    for (const auto& [layer_output_name, tensor_name] : name_mapping) {
        const Tensor& tensor = tensors_.at(tensor_name);
        auto device_info = tensor.device();
        result.emplace(layer_output_name, Tensor::view(
            tensor.data_ptr(),
            tensor.shape(),
            tensor.dtype(),
            std::get<0>(device_info),
            std::get<1>(device_info)
        ));
    }
    
    return result;
}

} // namespace edge_fm
