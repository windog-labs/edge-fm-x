#include "engine/model_runtime_spec.h"

#include "utils/check.h"

#include <nlohmann/json.hpp>

namespace edge_fm {

namespace {

int32_t required_int(const nlohmann::json& json, const std::string& key) {
    const int32_t value = json.value(key, 0);
    check<ConfigurationError>(value != 0, key + " is required in model config.json");
    return value;
}

std::vector<bool> all_layers_use_kv_cache(int32_t num_layers) {
    return std::vector<bool>(static_cast<size_t>(num_layers), true);
}

std::vector<bool> qwen3_5_kv_cache_plan(const nlohmann::json& model_config, int32_t num_layers) {
    std::vector<bool> uses_kv(static_cast<size_t>(num_layers), false);
    if (model_config.contains("layer_types") && model_config["layer_types"].is_array()) {
        const auto& layer_types = model_config["layer_types"];
        check<ConfigurationError>(
            static_cast<int32_t>(layer_types.size()) == num_layers,
            "qwen3_5 layer_types size must equal num_hidden_layers");
        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            const std::string layer_type = layer_types[static_cast<size_t>(layer_id)].get<std::string>();
            uses_kv[static_cast<size_t>(layer_id)] = (layer_type == "full_attention");
        }
        return uses_kv;
    }

    const int32_t full_attention_interval = model_config.value("full_attention_interval", 4);
    check<ConfigurationError>(full_attention_interval > 0, "qwen3_5 full_attention_interval must be positive");
    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        uses_kv[static_cast<size_t>(layer_id)] = ((layer_id + 1) % full_attention_interval == 0);
    }
    return uses_kv;
}

} // namespace

bool ModelRuntimeSpec::uses_kv_cache(int32_t layer_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int32_t>(layer_uses_kv_cache.size())) {
        return false;
    }
    return layer_uses_kv_cache[static_cast<size_t>(layer_id)];
}

std::vector<int32_t> ModelRuntimeSpec::kv_cache_layer_ids() const {
    std::vector<int32_t> ids;
    for (int32_t layer_id = 0; layer_id < static_cast<int32_t>(layer_uses_kv_cache.size()); ++layer_id) {
        if (layer_uses_kv_cache[static_cast<size_t>(layer_id)]) {
            ids.push_back(layer_id);
        }
    }
    return ids;
}

ModelRuntimeSpec resolve_model_runtime_spec(const EngineConfig& config) {
    ModelRuntimeSpec spec;
    spec.model_name = config.resolved_model_name();

    const nlohmann::json model_config = config.prefill_model_config();
    spec.num_layers = required_int(model_config, "num_hidden_layers");
    spec.hidden_size = required_int(model_config, "hidden_size");
    spec.vocab_size = model_config.value("vocab_size", 0);
    spec.num_attention_heads = required_int(model_config, "num_attention_heads");
    spec.num_kv_heads = model_config.value("num_key_value_heads", spec.num_attention_heads);

    if (model_config.contains("head_dim") && !model_config["head_dim"].is_null()) {
        spec.head_dim = model_config["head_dim"].get<int32_t>();
        check<ConfigurationError>(spec.head_dim > 0, "head_dim must be positive in model config.json");
    } else {
        spec.head_dim = spec.hidden_size / spec.num_attention_heads;
        check<ConfigurationError>(
            spec.head_dim * spec.num_attention_heads == spec.hidden_size,
            "hidden_size must be divisible by num_attention_heads when head_dim is not configured. "
            "Got hidden_size=" + std::to_string(spec.hidden_size) +
            ", num_attention_heads=" + std::to_string(spec.num_attention_heads));
    }

    if (spec.model_name == "qwen3_5") {
        spec.layer_uses_kv_cache = qwen3_5_kv_cache_plan(model_config, spec.num_layers);
        spec.supports_decode_cuda_graph = false;
    } else {
        spec.layer_uses_kv_cache = all_layers_use_kv_cache(spec.num_layers);
        spec.supports_decode_cuda_graph = true;
    }

    return spec;
}

} // namespace edge_fm
