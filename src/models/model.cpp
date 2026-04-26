#include "models/model.h"
#include "engine/engine.h"
#include "engine/cuda/scheduler.h"
#include "utils/check.h"
#include <edge-fm/core.h>
#include <nlohmann/json.hpp>

#include "models/qwen2_5/qwen2_5.h"

namespace edge_fm {

Model::~Model() {}

void Model::prepare_decode_position_ids(Context& /*context*/, Device /*device*/, int32_t /*device_id*/) {
    // Default: no-op. Models with position-dependent decode (e.g. M-RoPE) override.
}

void Model::advance_decode_runtime_tensors(Context& /*context*/, cudaStream_t /*stream*/) {
    // Default: no-op. Models with decode-time runtime state can override.
}

bool Model::has_static_decode_runtime_tensors() const {
    return false;
}

void Model::reset_operator_impl_caches() {
    // Default: no operator-specific cache to clear.
}

bool Model::needs_separate_prefill_q_buffer() const {
    return false;
}

std::vector<int32_t> Model::derive_mrope_last_pos(
    const int32_t* position_ids,
    int64_t total_len) const
{
    int32_t global_max = 0;
    for (int64_t i = 0; i < 3 * total_len; ++i) {
        if (position_ids[i] > global_max) {
            global_max = position_ids[i];
        }
    }
    return std::vector<int32_t>(3, global_max);
}

Model::Model(const EngineConfig& config)
    : engine_config_(config)
    , num_layers_(0)
    , hidden_size_(0)
    , vocab_size_(0)
    , dtype_(DType::Float16)  // Default to Float16
    , model_loaded_(false)
{
    // 从 prefill_model_config 中读取模型参数
    nlohmann::json model_config = engine_config_.prefill_model_config();
    
    num_layers_ = model_config.value("num_hidden_layers", 0);
    check(num_layers_ != 0, "num_hidden_layers is required in model config.json");
    
    hidden_size_ = model_config.value("hidden_size", 0);
    check(hidden_size_ != 0, "hidden_size is required in model config.json");
    
    vocab_size_ = model_config.value("vocab_size", 0);
    check(vocab_size_ != 0, "vocab_size is required in model config.json");
    
    // 读取 torch_dtype 并转换为 DType
    if (model_config.contains("torch_dtype")) {
        std::string torch_dtype_str = model_config["torch_dtype"].get<std::string>();
        dtype_ = dtype_from_string(torch_dtype_str);
    }
}

std::unique_ptr<Model> Model::create(const EngineConfig& config) {
    const std::string resolved_name = config.resolved_model_name();
    if (resolved_name == "qwen2_5" || resolved_name == "qwen2_5_vl") {
        return std::make_unique<Qwen2_5>(config);
    }

    throw ConfigurationError(
        "Unsupported model_name: " + resolved_name +
        ". This build currently supports: qwen2_5, qwen2_5_vl");
}

} // namespace edge_fm
