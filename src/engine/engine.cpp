#include "engine/engine_factory.h"
#include <edge-fm/core.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace edge_fm {

// ==================================================== engine config ====================================================
namespace {
    inline std::filesystem::path find_default_config_file() {
        if (const char* config_dir = std::getenv("EDGE_FM_CONFIG_DIR");
            config_dir != nullptr && *config_dir != '\0')
        {
            return std::filesystem::path(config_dir) / "engine_default.json";
        }
        if (const char* install_prefix = std::getenv("EDGE_FM_INSTALL_PREFIX");
            install_prefix != nullptr && *install_prefix != '\0')
        {
            return std::filesystem::path(install_prefix) / "config" / "engine_default.json";
        }
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

    std::string load_text_file(const std::filesystem::path& file_path, const std::string& error_context = "") {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::string error_msg = "Cannot open " + (error_context.empty() ? "file" : error_context) + ": " + file_path.string();
            throw ConfigurationError(error_msg);
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    nlohmann::json load_model_top_config_from_path(const std::string& model_path) {
        if (model_path.empty()) {
            throw ConfigurationError("model_path is required in config");
        }

        std::filesystem::path config_path = std::filesystem::path(model_path) / "config.json";
        return load_json_file(config_path, "config.json file");
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

    std::string fnv1a_hex(const std::string& data) {
        constexpr uint64_t kOffset = 14695981039346656037ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        uint64_t hash = kOffset;
        for (unsigned char c : data) {
            hash ^= static_cast<uint64_t>(c);
            hash *= kPrime;
        }
        std::ostringstream oss;
        oss << std::hex << hash;
        return oss.str();
    }

    std::string normalize_model_name(const std::string& raw_name) {
        std::string normalized;
        normalized.reserve(raw_name.size());
        for (unsigned char ch : raw_name) {
            if (std::isalnum(ch)) {
                normalized.push_back(static_cast<char>(std::tolower(ch)));
            } else if (ch == '_' || ch == '.' || ch == '-' || ch == ' ' || ch == '/') {
                normalized.push_back('_');
            }
        }

        if (normalized == "qwen2_5" || normalized == "qwen25" || normalized == "qwen2") {
            return "qwen2_5";
        }
        if (normalized == "qwen2_5_vl" || normalized == "qwen25_vl" || normalized == "qwen2_vl" ||
            normalized == "qwen25vl" || normalized == "qwen2_5vl" || normalized == "qwen2vl") {
            return "qwen2_5_vl";
        }
        if (normalized == "smolvla" || normalized == "smol_vla" ||
            normalized == "lerobot_smolvla" || normalized == "lerobot_smol_vla") {
            return "smolvla";
        }
        return normalized;
    }

    std::string normalize_hw_profile(const std::string& raw_profile) {
        std::string normalized;
        normalized.reserve(raw_profile.size());
        for (unsigned char ch : raw_profile) {
            if (std::isalnum(ch)) {
                normalized.push_back(static_cast<char>(std::tolower(ch)));
            } else if (ch == '_' || ch == '.' || ch == '-' || ch == ' ' || ch == '/') {
                normalized.push_back('_');
            }
        }
        return normalized;
    }

    std::string current_hardware_fingerprint(int32_t device_id) {
        return cuda_hardware_fingerprint(device_id);
    }

    std::string current_backend_fingerprint(const std::string& backend_target, int32_t device_id) {
        if (backend_target == "cuda") {
            return current_hardware_fingerprint(device_id);
        }
        return backend_target + "-toolchain";
    }
}

EngineConfig::EngineConfig(const std::string& config_path) {
    shared_state_ = std::make_shared<SharedState>();
    config_dir_ = std::filesystem::absolute(std::filesystem::path(config_path)).parent_path();

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
    nlohmann::json config = load_model_top_config_from_path(prefill_model_path());
    // Qwen2.5-VL 等 VLM 的 config 中 LLM 参数在 text_config 下
    if (config.contains("text_config") && config["text_config"].is_object()) {
        return config["text_config"];
    }
    return config;
}

nlohmann::json EngineConfig::decode_model_config() const {
    nlohmann::json config = load_model_top_config_from_path(decode_model_path());
    if (config.contains("text_config") && config["text_config"].is_object()) {
        return config["text_config"];
    }
    return config;
}

std::string EngineConfig::resolved_model_name() const {
    const std::string normalized = normalize_model_name(model_name());
    if (normalized == "qwen2_5" || normalized == "qwen2_5_vl" || normalized == "smolvla") {
        return normalized;
    }

    throw ConfigurationError(
        "Unsupported or missing model_name in engine config. "
        "Supported values include: Qwen2.5, qwen2_5, Qwen2.5-VL, qwen2_5_vl, SmolVLA, smolvla");
}

nlohmann::json EngineConfig::runtime() const { return get_object_or_empty(config_, "runtime"); }
nlohmann::json EngineConfig::speculative() const { return get_object_or_empty(config_, "speculative"); }
nlohmann::json EngineConfig::kvcache() const { return get_object_or_empty(config_, "kvcache"); }
nlohmann::json EngineConfig::sampling() const { return get_object_or_empty(config_, "sampling"); }
nlohmann::json EngineConfig::metrics() const { return get_object_or_empty(config_, "metrics"); }
nlohmann::json EngineConfig::tuning() const { return get_object_or_empty(config_, "tuning"); }
std::string EngineConfig::configured_operator_impl_table_path() const {
    const std::string raw_path = config_.value("operator_impl_table_path", std::string(""));
    if (raw_path.empty()) {
        return raw_path;
    }

    const std::filesystem::path path(raw_path);
    if (path.is_absolute()) {
        return path.string();
    }
    return (config_dir_ / path).lexically_normal().string();
}
std::string EngineConfig::operator_impl_table_path() const {
    if (shared_state_ != nullptr && !shared_state_->operator_impl_table_override_path.empty()) {
        return shared_state_->operator_impl_table_override_path;
    }
    return configured_operator_impl_table_path();
}
bool EngineConfig::has_operator_impl_table_override() const {
    return shared_state_ != nullptr && !shared_state_->operator_impl_table_override_path.empty();
}
void EngineConfig::set_operator_impl_table_override(const std::string& path) {
    if (shared_state_ == nullptr) {
        shared_state_ = std::make_shared<SharedState>();
    }
    shared_state_->operator_impl_table_override_path = path;
}
void EngineConfig::clear_operator_impl_table_override() {
    if (shared_state_ == nullptr) {
        return;
    }
    shared_state_->operator_impl_table_override_path.clear();
}
bool EngineConfig::tuning_enabled() const {
    return safe_value(tuning(), "enabled", false);
}
BackendTarget EngineConfig::backend_target_kind() const {
    if (config_.contains("_edgefm_internal") && config_["_edgefm_internal"].is_object()) {
        const std::string internal = safe_value(config_["_edgefm_internal"], "backend_target", std::string(""));
        if (!internal.empty()) {
            if (internal == "cuda" || internal == "gpu") {
                return BackendTarget::Cuda;
            }
            if (internal == "horizon") {
                return BackendTarget::Horizon;
            }
            throw ConfigurationError("Unsupported backend_target: " + internal);
        }
    }

    const std::string runtime_device_str = runtime_device();
    if (runtime_device_str == "cuda" || runtime_device_str == "gpu") {
        return BackendTarget::Cuda;
    }
    if (runtime_device_str == "horizon") {
        return BackendTarget::Horizon;
    }
    throw ConfigurationError("Unsupported backend_target/runtime.device: " + runtime_device_str);
}
std::string EngineConfig::backend_target() const {
    return backend_target_to_string(backend_target_kind());
}
std::string EngineConfig::runtime_hw_profile() const {
    return safe_value(runtime(), "hw_profile", std::string(""));
}
std::string EngineConfig::resolved_hw_profile() const {
    const std::string explicit_profile = normalize_hw_profile(runtime_hw_profile());
    if (!explicit_profile.empty()) {
        return explicit_profile;
    }

    const std::string backend = backend_target();
    if (backend == "cuda") {
        return cuda_hw_profile(runtime_device_id());
    }
    if (backend == "horizon") {
        return "horizon";
    }
    if (backend == "cpu") {
        return "cpu";
    }
    return normalize_hw_profile(backend);
}
bool EngineConfig::has_backend_artifact() const {
    return config_.contains("_edgefm_internal") &&
           config_["_edgefm_internal"].is_object() &&
           config_["_edgefm_internal"].contains("backend_artifact") &&
           config_["_edgefm_internal"]["backend_artifact"].is_object();
}
nlohmann::json EngineConfig::backend_artifact() const {
    if (!has_backend_artifact()) {
        return nlohmann::json::object();
    }
    return config_["_edgefm_internal"]["backend_artifact"];
}
std::string EngineConfig::backend_cache_key() const {
    const std::string operator_impl_table_material = operator_impl_table_path().empty()
        ? std::string("builtin-operator-impl-table-v1")
        : load_text_file(operator_impl_table_path(), "operator_impl_table file");
    if (!config_.contains("_edgefm_internal") || !config_["_edgefm_internal"].is_object()) {
        nlohmann::json normalized_engine_config = config_;
        normalized_engine_config.erase("_edgefm_internal");
        normalized_engine_config["model_name"] = resolved_model_name();
        normalized_engine_config["operator_impl_table_path"] = operator_impl_table_path();
        normalized_engine_config["runtime"]["hw_profile"] = resolved_hw_profile();
        const std::string backend = backend_target();
        const std::string hardware = current_backend_fingerprint(backend, runtime_device_id());
        return fnv1a_hex(
            backend + "|" +
            resolved_model_name() + "|" +
            resolved_hw_profile() + "|" +
            normalized_engine_config.dump() + "|" +
            prefill_model_config().dump() + "|" +
            operator_impl_table_material + "|" +
            hardware);
    }
    const auto& internal = config_["_edgefm_internal"];
    const std::string key = safe_value(internal, "backend_cache_key", std::string(""));
    if (!key.empty()) {
        return key;
    }

    nlohmann::json normalized_engine_config = config_;
    normalized_engine_config.erase("_edgefm_internal");
    normalized_engine_config["model_name"] = resolved_model_name();
    normalized_engine_config["operator_impl_table_path"] = operator_impl_table_path();
    normalized_engine_config["runtime"]["hw_profile"] = resolved_hw_profile();
    const std::string backend = backend_target();
    const std::string hardware = current_backend_fingerprint(backend, runtime_device_id());
    return fnv1a_hex(
        backend + "|" +
        resolved_model_name() + "|" +
        resolved_hw_profile() + "|" +
        normalized_engine_config.dump() + "|" +
        prefill_model_config().dump() + "|" +
        operator_impl_table_material + "|" +
        hardware);
}

std::string EngineConfig::runtime_device() const { return safe_value(runtime(), "device", std::string("cuda")); }
int32_t EngineConfig::runtime_device_id() const { return safe_value(runtime(), "device_id", 0); }
bool EngineConfig::use_cuda_graph() const { return safe_value(runtime(), "use_cuda_graph", false); }
std::string EngineConfig::kvcache_dtype() const { return safe_value(kvcache(), "dtype", std::string("fp16")); }
std::string EngineConfig::kvcache_attention_type() const { return safe_value(kvcache(), "attention_type", std::string("mha")); }
float EngineConfig::sampling_temperature() const { return safe_value(sampling(), "temperature", 1.0f); }
uint64_t EngineConfig::sampling_seed() const { return static_cast<uint64_t>(safe_value(sampling(), "seed", 0)); }
int32_t EngineConfig::sampling_max_new_tokens() const {
    return std::max<int32_t>(1, safe_value(sampling(), "max_new_tokens", 256));
}

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
Engine::Engine(const EngineConfig& config)
    : device_(Device::CPU)
    , device_id_(0)
    , config_(config)
{}

Engine::~Engine() = default;

TensorMap Engine::prefill(int32_t request_id, const TensorRefMap& inputs) {
    (void)request_id;
    (void)inputs;
    throw ConfigurationError(
        "Tensor prefill stage is not implemented by this backend");
}

TensorMap Engine::decode(int32_t request_id, const TensorRefMap& inputs) {
    (void)request_id;
    (void)inputs;
    throw ConfigurationError(
        "Tensor decode stage is not implemented by this backend");
}

} // namespace edge_fm
