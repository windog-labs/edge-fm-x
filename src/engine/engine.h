#pragma once

#include "utils/non_copyable.h"
#include "backends/backend_target.h"
#include <edge-fm/core.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

namespace edge_fm {

enum class ModelStage { Prefill, Decode, };

inline std::string model_stage_to_string(ModelStage stage) {
    switch (stage) {
        case ModelStage::Prefill:
            return "Prefill";
        case ModelStage::Decode:
            return "Decode";
        default:
            return "Unknown";
    }
}

class EngineConfig {
public:
    explicit EngineConfig(const std::string& config_path);
    ~EngineConfig() = default;
    
    std::string model_name() const;
    std::string prefill_model_path() const;
    std::string decode_model_path() const;
    nlohmann::json prefill_model_config() const;
    nlohmann::json decode_model_config() const;
    std::string resolved_model_name() const;
    
    nlohmann::json runtime() const;
    nlohmann::json speculative() const;
    nlohmann::json kvcache() const;
    nlohmann::json sampling() const;
    nlohmann::json metrics() const;
    nlohmann::json tuning() const;
    nlohmann::json compact_vocab() const;
    std::string task() const;
    nlohmann::json stages() const;
    nlohmann::json planner() const;
    const std::filesystem::path& config_dir() const noexcept { return config_dir_; }
    std::string configured_operator_impl_table_path() const;
    std::string operator_impl_table_path() const;
    bool has_operator_impl_table_override() const;
    void set_operator_impl_table_override(const std::string& path);
    void clear_operator_impl_table_override();
    bool tuning_enabled() const;

    // 安全访问器：当值为 null 或缺失时返回默认值，避免 type_error.306
    std::string runtime_device() const;
    int32_t runtime_device_id() const;
    std::string runtime_hw_profile() const;
    std::string resolved_hw_profile() const;
    bool use_cuda_graph() const;
    std::string kvcache_dtype() const;
    std::string kvcache_attention_type() const;
    float sampling_temperature() const;
    uint64_t sampling_seed() const;
    int32_t sampling_max_new_tokens() const;
    std::vector<int32_t> eos_token_ids() const;
    std::vector<int32_t> stop_token_ids() const;
    bool lm_head_top1_enabled() const;
    BackendTarget backend_target_kind() const;
    std::string backend_target() const;
    bool has_backend_artifact() const;
    nlohmann::json backend_artifact() const;
    std::string backend_cache_key() const;
    
    const nlohmann::json& raw() const noexcept { return config_; }

private:
    struct SharedState {
        std::string operator_impl_table_override_path;
    };

    std::filesystem::path config_dir_;
    nlohmann::json config_;
    std::shared_ptr<SharedState> shared_state_;
};

class Engine : public NonCopyable {
public:
    explicit Engine(const EngineConfig& config);

    virtual ~Engine() = 0;

    virtual void warmup() = 0;
    virtual void tune() = 0;
    virtual Response generate(const Request& request) = 0;
    virtual TensorMap plan(int32_t request_id, const TensorRefMap& inputs);
    virtual TensorMap run_stage(int32_t request_id, const std::string& stage_name, const TensorRefMap& inputs);
    virtual TensorMap prefill(int32_t request_id, const TensorRefMap& inputs);
    virtual TensorMap decode(int32_t request_id, const TensorRefMap& inputs);
    virtual std::unordered_map<std::string, double> get_last_generate_metrics() const = 0;
    virtual std::unordered_map<std::string, double> get_last_plan_metrics() const;
    virtual std::unordered_map<std::string, double> get_last_stage_metrics() const;

protected:
    Device device_;
    int32_t device_id_;
    EngineConfig config_;
};

} // namespace edge_fm
