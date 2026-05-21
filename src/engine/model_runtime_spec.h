#pragma once

#include "engine/engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace edge_fm {

struct ModelRuntimeSpec {
    std::string model_name;
    int32_t num_layers = 0;
    int32_t hidden_size = 0;
    int32_t vocab_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    std::vector<bool> layer_uses_kv_cache;
    bool supports_decode_cuda_graph = true;

    bool uses_kv_cache(int32_t layer_id) const;
    std::vector<int32_t> kv_cache_layer_ids() const;
};

ModelRuntimeSpec resolve_model_runtime_spec(const EngineConfig& config);

} // namespace edge_fm
