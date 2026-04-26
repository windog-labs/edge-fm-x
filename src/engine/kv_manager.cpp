#include "engine/kv_manager.h"
#include "engine/engine.h"
#include "utils/check.h"
#include <edge-fm/core.h>

#include <new>
#include <utility>

namespace edge_fm {

namespace {
    AttentionType attention_type_from_string(const std::string& type_str) {
        if (type_str == "mha") {
            return AttentionType::MHA;
        } else if (type_str == "gqa") {
            return AttentionType::GQA;
        } else if (type_str == "mla") {
            return AttentionType::MLA;
        } else {
            throw ConfigurationError("Unsupported attention_type: " + type_str + ". Expected 'mha', 'gqa', or 'mla'");
        }
    }

}

void* HostKVBufferAllocator::get_buffer(const std::string& name, size_t bytes, int32_t device_id) {
    (void)device_id;
    if (bytes == 0) {
        return nullptr;
    }

    Buffer& buffer = buffers_[name];
    if (buffer.size < bytes) {
        try {
            buffer.data = std::make_unique<uint8_t[]>(bytes);
            buffer.size = bytes;
        } catch (const std::bad_alloc&) {
            throw OutOfMemoryError("Failed to allocate host KV buffer: " + name);
        }
    }
    return buffer.data.get();
}

KVManager::KVManager(const EngineConfig& engine_config)
    : KVManager(engine_config, std::make_shared<HostKVBufferAllocator>())
{}

KVManager::KVManager(const EngineConfig& engine_config, std::shared_ptr<KVBufferAllocator> buffer_allocator)
    : buffer_allocator_(std::move(buffer_allocator))
{
    check<ConfigurationError>(buffer_allocator_ != nullptr, "KVManager requires a non-null buffer allocator");

    // =============================== parsing common config ===============================
    // 1. 从 runtime 配置读取设备信息
    device_ = buffer_allocator_->device();
    device_id_ = engine_config.runtime_device_id();

    // 2. 从 kvcache 配置读取 attention_type 和 dtype
    std::string attention_type_str = engine_config.kvcache_attention_type();
    attention_type_ = attention_type_from_string(attention_type_str);
    
    std::string dtype_str = engine_config.kvcache_dtype();
    dtype_ = dtype_from_string(dtype_str);

    // 3. 获取模型配置
    nlohmann::json model_config = engine_config.prefill_model_config();

    num_layers_ = model_config.value("num_hidden_layers", 0);
    check(num_layers_ != 0, "num_hidden_layers is required in model config.json");

    int32_t num_attention_heads = model_config.value("num_attention_heads", 0);
    check(num_attention_heads != 0, "num_attention_heads is required in model config.json");

    num_kv_heads_ = model_config.value("num_key_value_heads", num_attention_heads);
    
    int32_t hidden_size = model_config.value("hidden_size", 0);
    check(hidden_size != 0, "hidden_size is required in model config.json");

    head_dim_ = hidden_size / num_attention_heads;
    check(head_dim_ * num_attention_heads == hidden_size,
          "hidden_size must be divisible by num_attention_heads. "
          "Got hidden_size=" + std::to_string(hidden_size) + 
          ", num_attention_heads=" + std::to_string(num_attention_heads));

    // 5. 验证 attention_type 是否匹配
    if (attention_type_ == AttentionType::MLA) {
        // MLA 需要 kv_lora_rank 和 qk_rope_head_dim
        check(model_config.contains("kv_lora_rank"),
              "kv_lora_rank is required in model config.json for MLA attention_type");
        check(model_config.contains("qk_rope_head_dim"),
              "qk_rope_head_dim is required in model config.json for MLA attention_type");
        kv_lora_rank_ = model_config.value("kv_lora_rank", 0);
        qk_rope_head_dim_ = model_config.value("qk_rope_head_dim", 0);
        check(kv_lora_rank_ != 0 && qk_rope_head_dim_ != 0,
              "kv_lora_rank and qk_rope_head_dim must be non-zero for MLA attention_type");
    } else if (attention_type_ == AttentionType::MHA) {
        // MHA: num_attention_heads == num_key_value_heads
        check(num_attention_heads == num_kv_heads_,
              "For MHA attention_type, num_attention_heads (" + 
              std::to_string(num_attention_heads) + 
              ") must equal num_key_value_heads (" + 
              std::to_string(num_kv_heads_) + ")");
        kv_lora_rank_ = 0;
        qk_rope_head_dim_ = 0;
    } else if (attention_type_ == AttentionType::GQA) {
        // GQA: num_attention_heads > num_key_value_heads
        check(num_attention_heads > num_kv_heads_,
              "For GQA attention_type, num_attention_heads (" + 
              std::to_string(num_attention_heads) + 
              ") must be greater than num_key_value_heads (" + 
              std::to_string(num_kv_heads_) + ")");
        kv_lora_rank_ = 0;
        qk_rope_head_dim_ = 0;
    } else {
        throw ConfigurationError("Unsupported attention_type: " + attention_type_str + ". Expected 'mha', 'gqa', or 'mla'");
    }

    // =============================== parsing for slots ===============================
    // 从 kvcache 配置中读取 requests 并构建 slots
    nlohmann::json kvcache_config = engine_config.kvcache();
    if (!kvcache_config.contains("requests") || !kvcache_config["requests"].is_array()) {
        throw ConfigurationError("kvcache.requests is required and must be an array");
    }
    
    const auto& requests = kvcache_config["requests"];
    check(requests.size() > 0, "kvcache.requests array must not be empty");
    
    for (const auto& request : requests) {
        check(request.is_object(), "Each request in kvcache.requests must be an object");
        
        int32_t request_id = request.value("request_id", -1);
        check(request_id >= 0, "request_id is required and must be non-negative");
        
        check(slots_.find(request_id) == slots_.end(), 
              "Duplicate request_id found: " + std::to_string(request_id));
        
        int32_t max_tokens = request.value("max_tokens", 0);
        check(max_tokens > 0, "max_tokens is required and must be positive for request_id " + 
              std::to_string(request_id));
        
        std::vector<int32_t> prefix_token_ids;
        if (request.contains("prefix_token_ids") && request["prefix_token_ids"].is_array()) {
            for (const auto& token_id : request["prefix_token_ids"]) {
                if (token_id.is_number_integer()) {
                    prefix_token_ids.push_back(static_cast<int32_t>(token_id));
                }
            }
        }
        
        size_t prefix_size = prefix_token_ids.size();
        
        Slot slot;
        slot.request_id = request_id;
        slot.max_tokens = max_tokens;
        slot.prefix_size = prefix_size;
        slot.allocated_size = 0;  // 暂时设为0，后续分配时再设置
        slot.prefix_token_ids = std::move(prefix_token_ids);
        // kv_cache_read_ptrs && kv_cache_write_ptrs 暂时不构建，后续分配时再设置
        
        slots_[request_id] = std::move(slot);
    }

    // =============================== allocate kv cache ===============================
    if (attention_type_ == AttentionType::MHA || 
        attention_type_ == AttentionType::GQA || 
        attention_type_ == AttentionType::MQA) {
        allocate_common_kvcache();
    } else if (attention_type_ == AttentionType::MLA) {
        allocate_mla_kvcache();
    }
}

void KVManager::allocate_common_kvcache() {
    size_t kv_cache_per_token = get_token_stride();
    size_t k_stride = kv_cache_per_token / 2;
    
    for (auto& [request_id, slot] : slots_) {
        size_t kv_cache_size = kv_cache_per_token * slot.max_tokens;
        slot.kv_cache_read_ptrs.resize(num_layers_);
        slot.kv_cache_write_ptrs.resize(num_layers_);
        
        size_t prefix_offset = k_stride * slot.prefix_size;
        for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
            std::string buf_name = "kv_cache_request_" + std::to_string(request_id) + 
                                   "_layer_" + std::to_string(layer_id);
            void* kv_cache_ptr = buffer_allocator_->get_buffer(buf_name, kv_cache_size, device_id_);

            slot.kv_cache_read_ptrs[layer_id] = kv_cache_ptr;
            slot.kv_cache_write_ptrs[layer_id] = static_cast<uint8_t*>(kv_cache_ptr) + prefix_offset;
        }
        
        slot.allocated_size = kv_cache_size * num_layers_;
    }
}

void KVManager::allocate_mla_kvcache() {
    size_t kv_cache_per_token = get_token_stride();
    
    for (auto& [request_id, slot] : slots_) {
        size_t kv_cache_size = kv_cache_per_token * slot.max_tokens;
        slot.kv_cache_read_ptrs.resize(num_layers_);
        slot.kv_cache_write_ptrs.resize(num_layers_);
        
        size_t prefix_offset = kv_cache_per_token * slot.prefix_size;
        for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
            std::string buf_name = "kv_cache_request_" + std::to_string(request_id) + 
                                   "_layer_" + std::to_string(layer_id);
            void* kv_cache_ptr = buffer_allocator_->get_buffer(buf_name, kv_cache_size, device_id_);

            slot.kv_cache_read_ptrs[layer_id] = kv_cache_ptr;
            slot.kv_cache_write_ptrs[layer_id] = static_cast<uint8_t*>(kv_cache_ptr) + prefix_offset;
        }
        
        slot.allocated_size = kv_cache_size * num_layers_;
    }
}

KVManagerStatus KVManager::get_status() const {
    KVManagerStatus status;
    status.device = device_;
    status.device_id = device_id_;
    
    status.slots.reserve(slots_.size());
    for (const auto& [request_id, slot] : slots_) {
        KVSlotStatus slot_status;
        slot_status.request_id = slot.request_id;
        slot_status.prefix_token_ids = slot.prefix_token_ids;
        slot_status.prefix_size = slot.prefix_size;
        slot_status.max_tokens = slot.max_tokens;
        slot_status.allocated_size = slot.allocated_size;
        status.slots.push_back(slot_status);
    }
    
    return status;
}

bool KVManager::is_request_valid(int32_t request_id) const {
    return slots_.find(request_id) != slots_.end();
}

std::vector<void*> KVManager::get_read_kvcache(int32_t request_id) const {
    auto it = slots_.find(request_id);
    if (it == slots_.end()) {
        throw InvalidRequestError("Invalid request_id: " + std::to_string(request_id));
    }
    return it->second.kv_cache_read_ptrs;
}

std::vector<void*> KVManager::get_write_kvcache(int32_t request_id) const {
    auto it = slots_.find(request_id);
    if (it == slots_.end()) {
        throw InvalidRequestError("Invalid request_id: " + std::to_string(request_id));
    }
    return it->second.kv_cache_write_ptrs;
}

size_t KVManager::get_token_stride() const {
    size_t dtype_size = get_dtype_size(dtype_);

    if (attention_type_ == AttentionType::MLA) {
        return (kv_lora_rank_ + qk_rope_head_dim_) * dtype_size;
    } else { // MHA, GQA, MQA
        return 2 * num_kv_heads_ * head_dim_ * dtype_size;
    }
}

} // namespace edge_fm
