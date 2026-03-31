#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace edge_fm {

struct BackendArtifact {
    std::string backend;
    std::string artifact_type;
    std::string artifact_path;
    std::string manifest_path;
    nlohmann::json metadata = nlohmann::json::object();

    nlohmann::json to_json() const;
    static BackendArtifact from_json(const nlohmann::json& json);
};

class BackendArtifactCache {
public:
    static BackendArtifactCache& instance();

    std::optional<BackendArtifact> get_artifact(const std::string& model_key);
    void set_artifact(const std::string& model_key, const BackendArtifact& artifact);

    std::filesystem::path artifact_directory_for_model(const std::string& model_key) const;

private:
    BackendArtifactCache() = default;

    nlohmann::json& load_model_cache_locked(const std::string& model_key);
    void flush_model_cache_locked(const std::string& model_key, const nlohmann::json& json);
    std::filesystem::path cache_path_for_model(const std::string& model_key) const;
    std::filesystem::path cache_root() const;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, nlohmann::json> cache_;
};

} // namespace edge_fm
