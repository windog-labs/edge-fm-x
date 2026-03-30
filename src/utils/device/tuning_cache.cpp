#include "utils/device/tuning_cache.h"

#include <cstdlib>
#include <fstream>

namespace edge_fm {

namespace {

std::string stage_to_cache_key(ModelStage stage) {
    return stage == ModelStage::Prefill ? "prefill" : "decode";
}

ModelStage stage_from_cache_key(const std::string& key) {
    return key == "decode" ? ModelStage::Decode : ModelStage::Prefill;
}

} // namespace

nlohmann::json TuningRecord::to_json() const {
    return nlohmann::json{
        {"op_name", op_name},
        {"backend", backend},
        {"stage", stage_to_cache_key(stage)},
        {"m", m},
        {"algo_index", algo_index},
    };
}

TuningRecord TuningRecord::from_json(const nlohmann::json& json) {
    TuningRecord record;
    record.op_name = json.value("op_name", std::string(""));
    record.backend = json.value("backend", std::string(""));
    record.stage = stage_from_cache_key(json.value("stage", std::string("prefill")));
    record.m = json.value("m", -1);
    record.algo_index = json.value("algo_index", -1);
    return record;
}

TuningCache& TuningCache::instance() {
    static TuningCache cache;
    return cache;
}

std::optional<TuningRecord> TuningCache::get_record(
    const std::string& model_key,
    const std::string& op_name,
    ModelStage stage,
    int32_t m)
{
    if (model_key.empty()) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& json = load_model_cache_locked(model_key);
    const auto record_key = make_record_key(op_name, stage, m);
    if (!json.contains("records") || !json["records"].is_object()) {
        return std::nullopt;
    }
    const auto& records = json["records"];
    if (!records.contains(record_key) || !records[record_key].is_object()) {
        return std::nullopt;
    }
    return TuningRecord::from_json(records[record_key]);
}

void TuningCache::set_record(const std::string& model_key, const TuningRecord& record) {
    if (model_key.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& json = load_model_cache_locked(model_key);
    if (!json.contains("records") || !json["records"].is_object()) {
        json["records"] = nlohmann::json::object();
    }
    json["schema"] = "edgefm_tuning_v1";
    json["records"][make_record_key(record.op_name, record.stage, record.m)] = record.to_json();
    flush_model_cache_locked(model_key, json);
}

void TuningCache::begin_session(const std::string& model_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_model_key_ = model_key;
    (void)load_model_cache_locked(model_key);
}

void TuningCache::end_session() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_model_key_.clear();
}

bool TuningCache::is_session_active_for(const std::string& model_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !model_key.empty() && active_model_key_ == model_key;
}

nlohmann::json& TuningCache::load_model_cache_locked(const std::string& model_key) {
    auto it = cache_.find(model_key);
    if (it != cache_.end()) {
        return it->second;
    }

    auto& json = cache_[model_key];
    const auto path = cache_path_for_model(model_key);
    if (std::filesystem::exists(path)) {
        std::ifstream file(path);
        if (file) {
            try {
                file >> json;
            } catch (...) {
                json = nlohmann::json::object();
            }
        }
    }
    if (!json.is_object()) {
        json = nlohmann::json::object();
    }
    if (!json.contains("records") || !json["records"].is_object()) {
        json["records"] = nlohmann::json::object();
    }
    return json;
}

void TuningCache::flush_model_cache_locked(
    const std::string& model_key,
    const nlohmann::json& json)
{
    const auto path = cache_path_for_model(model_key);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        return;
    }
    file << json.dump(2);
}

std::string TuningCache::make_record_key(
    const std::string& op_name,
    ModelStage stage,
    int32_t m) const
{
    return stage_to_cache_key(stage) + "|" + std::to_string(m) + "|" + op_name;
}

std::filesystem::path TuningCache::cache_path_for_model(const std::string& model_key) const {
    const char* home = std::getenv("HOME");
    std::filesystem::path root;
    if (home != nullptr && *home != '\0') {
        root = std::filesystem::path(home) / ".cache" / "edge-fm" / "tuning";
    } else {
        root = std::filesystem::temp_directory_path() / "edge-fm" / "tuning";
    }
    return root / (model_key + ".json");
}

} // namespace edge_fm
