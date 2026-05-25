#include "models/model.h"
#include "engine/engine.h"
#include "engine/tasks/token_generation/scheduler.h"
#include "utils/check.h"
#include <edge-fm/core.h>
#include <nlohmann/json.hpp>

#include "models/qwen2_5/qwen2_5.h"
#include "models/qwen3_5/qwen3_5.h"

namespace edge_fm {

Model::~Model() {}

void Model::prepare_decode_position_ids(Context& /*context*/, Device /*device*/, int32_t /*device_id*/) {
    // Default: no-op. Models with position-dependent decode (e.g. M-RoPE) override.
}

void Model::advance_decode_runtime_tensors(Context& /*context*/, cudaStream_t /*stream*/) {
    // Default: no-op. Models with decode-time runtime state can override.
}

void Model::backup_decode_runtime_tensors(Context& /*context*/, cudaStream_t /*stream*/) {
    // Default: no-op. Models with mutable recurrent decode state can override.
}

void Model::restore_decode_runtime_tensors(Context& /*context*/, cudaStream_t /*stream*/) {
    // Default: no-op. Models with mutable recurrent decode state can override.
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
    runtime_spec_ = resolve_model_runtime_spec(engine_config_);

    // 从 prefill_model_config 中读取模型参数
    nlohmann::json model_config = engine_config_.prefill_model_config();
    
    num_layers_ = runtime_spec_.num_layers;
    hidden_size_ = runtime_spec_.hidden_size;
    vocab_size_ = runtime_spec_.vocab_size;
    check<ConfigurationError>(vocab_size_ > 0, "vocab_size is required in model config.json");
    
    // 读取 torch_dtype 并转换为 DType
    if (model_config.contains("torch_dtype") || model_config.contains("dtype")) {
        std::string torch_dtype_str = model_config.value(
            "torch_dtype",
            model_config.value("dtype", std::string("float16")));
        dtype_ = dtype_from_string(torch_dtype_str);
    }
}

std::unique_ptr<Model> Model::create(const EngineConfig& config) {
    const std::string resolved_name = config.resolved_model_name();
    if (resolved_name == "qwen2_5" || resolved_name == "qwen2_5_vl") {
        return std::make_unique<Qwen2_5>(config);
    }
    if (resolved_name == "qwen3_5") {
        return std::make_unique<Qwen3_5>(config);
    }

    throw ConfigurationError(
        "Unsupported model_name: " + resolved_name +
        ". This build currently supports: qwen2_5, qwen2_5_vl, qwen3_5");
}

} // namespace edge_fm
