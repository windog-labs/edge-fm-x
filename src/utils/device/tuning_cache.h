#pragma once

#include "engine/engine.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace edge_fm {

struct TuningRecord {
    std::string op_name;
    std::string backend;
    ModelStage stage = ModelStage::Prefill;
    int32_t m = -1;
    int32_t algo_index = -1;

    nlohmann::json to_json() const;
    static TuningRecord from_json(const nlohmann::json& json);
};

class TuningCache {
public:
    static TuningCache& instance();

    std::optional<TuningRecord> get_record(
        const std::string& model_key,
        const std::string& op_name,
        ModelStage stage,
        int32_t m);

    void set_record(const std::string& model_key, const TuningRecord& record);

    void begin_session(const std::string& model_key);
    void end_session();
    bool is_session_active_for(const std::string& model_key) const;

private:
    TuningCache() = default;

    nlohmann::json& load_model_cache_locked(const std::string& model_key);
    void flush_model_cache_locked(const std::string& model_key, const nlohmann::json& json);
    std::string make_record_key(
        const std::string& op_name,
        ModelStage stage,
        int32_t m) const;
    std::filesystem::path cache_path_for_model(const std::string& model_key) const;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, nlohmann::json> cache_;
    std::string active_model_key_;
};

} // namespace edge_fm
