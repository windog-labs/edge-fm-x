#include "models/model.h"
#include "engine/engine.h"
#include "engine/scheduler.h"
#include "utils/check.h"
#include <edge-fm/core.h>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

#include "models/qwen2_5/qwen2_5.h"

namespace edge_fm {

Model::~Model() {}

void Model::prepare_decode_position_ids(Context& /*context*/, Device /*device*/, int32_t /*device_id*/) {
    // Default: no-op. Models with position-dependent decode (e.g. M-RoPE) override.
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
    const std::string& model_name = config.model_name();
    if (model_name.empty()) {
        throw ConfigurationError("model_name is required in configuration");
    }
    
    std::string model_name_lower = model_name;
    std::transform(model_name_lower.begin(), model_name_lower.end(), 
                   model_name_lower.begin(), [](unsigned char c) { return std::tolower(c); });
    
    if (model_name_lower == "qwen2.5" || 
        model_name_lower == "qwen2_5" || 
        model_name_lower == "qwen2-5") 
    {
        return std::make_unique<Qwen2_5>(config);
    } else {
        throw ConfigurationError("Unsupported model_name: " + model_name + ". Supported models: qwen2.5");
    }
}

} // namespace edge_fm

