#pragma once

#include <edge-fm/core.h>
#include "utils/non_copyable.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace edge_fm {

class EngineConfig;

struct KVSlotStatus {
    int32_t request_id;
    std::vector<int32_t> prefix_token_ids;
    size_t prefix_size;
    int32_t max_tokens;
    size_t allocated_size;
};

struct KVManagerStatus {
    std::vector<KVSlotStatus> slots;
    Device device;              // 设备类型
    int32_t device_id;          // 设备 ID
};

enum class AttentionType {
    MHA,
    GQA,
    MQA,
    MLA,
};

class KVManager : public NonCopyable {
public:
    KVManager(const EngineConfig& engine_config);
    ~KVManager() = default;

    size_t get_token_stride() const;
    AttentionType get_attention_type() const { return attention_type_; }
    int32_t get_qk_rope_head_dim() const { return qk_rope_head_dim_; }
    std::vector<void*> get_read_kvcache(int32_t request_id) const;
    std::vector<void*> get_write_kvcache(int32_t request_id) const;

    KVManagerStatus get_status() const;
    bool is_request_valid(int32_t request_id) const;

private:
    void allocate_common_kvcache();  // MHA, GQA, etc
    void allocate_mla_kvcache();
    // TODO: more kvcache allocation

private:
    struct Slot {
        int32_t request_id;
        int32_t max_tokens;
        size_t prefix_size;
        size_t allocated_size;
        std::vector<int32_t> prefix_token_ids;
        
        std::vector<void*> kv_cache_read_ptrs;   // 每层的 KV cache 读指针
        std::vector<void*> kv_cache_write_ptrs;  // 每层的 KV cache 写指针(prefill后的第一个token的kvcache地址)
    };

    Device device_;
    int32_t device_id_;
    DType dtype_;
    std::unordered_map<int32_t, Slot> slots_;
    // common
    int32_t num_layers_;
    AttentionType attention_type_;
    // for MHA, GQA...
    int32_t head_dim_;
    int32_t num_kv_heads_;
    // for MLA
    int32_t kv_lora_rank_;
    int32_t qk_rope_head_dim_;
};

} // namespace edge_fm
