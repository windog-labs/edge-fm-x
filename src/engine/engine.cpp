#include "engine/engine.h"
#include "layers/sampler.h"
#include "utils/device/cuda_utils.h"
#include <edge-fm/core.h>
#include <fstream>

namespace edge_fm {

// ==================================================== engine config ====================================================
namespace {
    inline std::filesystem::path find_default_config_file() {
        return std::filesystem::path(EDGE_FM_INSTALL_PREFIX) / "config" / "engine_default.json";
    }
    
    nlohmann::json load_json_file(const std::filesystem::path& file_path, const std::string& error_context = "") {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::string error_msg = "Cannot open " + (error_context.empty() ? "file" : error_context) + ": " + file_path.string();
            throw ConfigurationError(error_msg);
        }
        
        nlohmann::json json_data;
        try {
            file >> json_data;
        } catch (const nlohmann::json::parse_error& e) {
            std::string error_msg = "Failed to parse " + (error_context.empty() ? "file" : error_context) + ": " + std::string(e.what());
            throw ConfigurationError(error_msg);
        }
        
        return json_data;
    }
    
    void validate_model_path(const std::string& path, const std::string& path_name) {
        std::filesystem::path dir_path = path;
        if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
            throw ConfigurationError(path_name + " does not exist or is not a directory: " + path);
        }
        
        std::filesystem::path config_file = dir_path / "config.json";
        if (!std::filesystem::exists(config_file) || !std::filesystem::is_regular_file(config_file)) {
            throw ConfigurationError("config.json does not exist in " + path_name + ": " + path);
        }
    }

    // 当值为 null 或非对象时返回空对象，避免下游对 value() 的 null 调用触发 type_error.306
    nlohmann::json get_object_or_empty(const nlohmann::json& j, const std::string& key) {
        if (!j.contains(key)) {
            return nlohmann::json::object();
        }
        const auto& v = j[key];
        if (v.is_null() || !v.is_object()) {
            return nlohmann::json::object();
        }
        return v;
    }

    template<typename T>
    T safe_value(const nlohmann::json& j, const std::string& key, T default_val) {
        if (!j.is_object() || !j.contains(key)) return default_val;
        const auto& v = j[key];
        if (v.is_null()) return default_val;
        try {
            return v.get<T>();
        } catch (...) {
            return default_val;
        }
    }
}

EngineConfig::EngineConfig(const std::string& config_path) {
    // Load default configuration first
    std::filesystem::path default_config_path = find_default_config_file();
    nlohmann::json config = load_json_file(default_config_path, "default configuration file");
    
    // Load user configuration file and merge (user config takes precedence)
    nlohmann::json user_config = load_json_file(config_path, "configuration file");
    config.update(user_config);
    
    // Validate prefill_model_path
    std::string prefill_path = config.value("prefill_model_path", std::string(""));
    if (prefill_path.empty()) {
        throw ConfigurationError("prefill_model_path is required in configuration");
    }
    validate_model_path(prefill_path, "prefill_model_path");
    
    // Handle decode_model_path
    if (!config.contains("decode_model_path") || 
        config["decode_model_path"].is_null() || 
        config["decode_model_path"].get<std::string>().empty()) 
    {
        config["decode_model_path"] = prefill_path;
    } else {
        validate_model_path(config["decode_model_path"].get<std::string>(), "decode_model_path");
    }
    
    config_ = config;
}

std::string EngineConfig::model_name() const { return config_.value("model_name", std::string("")); }
std::string EngineConfig::prefill_model_path() const { return config_.value("prefill_model_path", std::string("")); }

std::string EngineConfig::decode_model_path() const {
    if (config_.contains("decode_model_path") && !config_["decode_model_path"].is_null()) {
        return config_["decode_model_path"].get<std::string>();
    }
    return "";
}

nlohmann::json EngineConfig::prefill_model_config() const {
    std::string model_path = prefill_model_path();
    if (model_path.empty()) {
        throw ConfigurationError("prefill_model_path is required in config");
    }
    
    std::filesystem::path config_path = model_path;
    config_path /= "config.json";
    
    nlohmann::json config = load_json_file(config_path, "config.json file");
    // Qwen2.5-VL 等 VLM 的 config 中 LLM 参数在 text_config 下
    if (config.contains("text_config") && config["text_config"].is_object()) {
        return config["text_config"];
    }
    return config;
}

nlohmann::json EngineConfig::decode_model_config() const {
    std::string model_path = decode_model_path();
    if (model_path.empty()) {
        throw ConfigurationError("decode_model_path is required in config");
    }
    
    std::filesystem::path config_path = model_path;
    config_path /= "config.json";
    
    nlohmann::json config = load_json_file(config_path, "config.json file");
    if (config.contains("text_config") && config["text_config"].is_object()) {
        return config["text_config"];
    }
    return config;
}

nlohmann::json EngineConfig::runtime() const { return get_object_or_empty(config_, "runtime"); }
nlohmann::json EngineConfig::speculative() const { return get_object_or_empty(config_, "speculative"); }
nlohmann::json EngineConfig::kvcache() const { return get_object_or_empty(config_, "kvcache"); }
nlohmann::json EngineConfig::sampling() const { return get_object_or_empty(config_, "sampling"); }
nlohmann::json EngineConfig::metrics() const { return get_object_or_empty(config_, "metrics"); }

std::string EngineConfig::runtime_device() const { return safe_value(runtime(), "device", std::string("cuda")); }
int32_t EngineConfig::runtime_device_id() const { return safe_value(runtime(), "device_id", 0); }
bool EngineConfig::use_cuda_graph() const { return safe_value(runtime(), "use_cuda_graph", false); }
std::string EngineConfig::kvcache_dtype() const { return safe_value(kvcache(), "dtype", std::string("fp16")); }
std::string EngineConfig::kvcache_attention_type() const { return safe_value(kvcache(), "attention_type", std::string("mha")); }
float EngineConfig::sampling_temperature() const { return safe_value(sampling(), "temperature", 1.0f); }
uint64_t EngineConfig::sampling_seed() const { return static_cast<uint64_t>(safe_value(sampling(), "seed", 0)); }

std::vector<int32_t> EngineConfig::eos_token_ids() const {
    std::vector<int32_t> ids;
    auto add_from_json = [&](const nlohmann::json& j, const std::string& key) {
        if (!j.is_object() || !j.contains(key) || j[key].is_null()) return;
        const auto& v = j[key];
        if (v.is_number_integer()) {
            ids.push_back(v.get<int32_t>());
        } else if (v.is_array()) {
            for (const auto& elem : v) {
                if (elem.is_number_integer()) ids.push_back(elem.get<int32_t>());
            }
        }
    };
    // Read from model config (supports both int and array formats)
    std::string model_path = prefill_model_path();
    if (!model_path.empty()) {
        std::filesystem::path config_path = std::filesystem::path(model_path) / "config.json";
        if (std::filesystem::exists(config_path)) {
            try {
                nlohmann::json model_cfg = load_json_file(config_path, "model config");
                add_from_json(model_cfg, "eos_token_id");
            } catch (...) { /* ignore parse errors */ }
        }
    }
    return ids;
}

std::vector<int32_t> EngineConfig::stop_token_ids() const {
    std::vector<int32_t> ids;
    auto samp = sampling();
    if (samp.is_object() && samp.contains("stop_token_ids") && samp["stop_token_ids"].is_array()) {
        for (const auto& elem : samp["stop_token_ids"]) {
            if (elem.is_number_integer()) ids.push_back(elem.get<int32_t>());
        }
    }
    return ids;
}

// ==================================================== engine ====================================================
Engine::Engine(const EngineConfig& config) : config_(config) {
    device_ = device_from_string(config_.runtime_device());
    device_id_ = config_.runtime_device_id();
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to set device for engine init");

    model_ = Model::create(config_);
    kv_manager_ = std::make_shared<KVManager>(config_);
    scheduler_ = std::make_unique<Scheduler>(kv_manager_);
    sampler_ = std::make_unique<SamplerLayer>(config_);
    sampler_->load_weights({}, {});
}

Engine::~Engine() = default;

} // namespace edge_fm