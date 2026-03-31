#include "backends/backend_artifact_cache.h"

#include <cstdlib>
#include <fstream>

namespace edge_fm {

nlohmann::json BackendArtifact::to_json() const {
    return nlohmann::json{
        {"backend", backend},
        {"artifact_type", artifact_type},
        {"artifact_path", artifact_path},
        {"manifest_path", manifest_path},
        {"metadata", metadata},
    };
}

BackendArtifact BackendArtifact::from_json(const nlohmann::json& json) {
    BackendArtifact artifact;
    artifact.backend = json.value("backend", std::string(""));
    artifact.artifact_type = json.value("artifact_type", std::string(""));
    artifact.artifact_path = json.value("artifact_path", std::string(""));
    artifact.manifest_path = json.value("manifest_path", std::string(""));
    artifact.metadata = json.contains("metadata") && json["metadata"].is_object()
        ? json["metadata"]
        : nlohmann::json::object();
    return artifact;
}

BackendArtifactCache& BackendArtifactCache::instance() {
    static BackendArtifactCache cache;
    return cache;
}

std::optional<BackendArtifact> BackendArtifactCache::get_artifact(const std::string& model_key) {
    if (model_key.empty()) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& json = load_model_cache_locked(model_key);
    if (!json.contains("artifact") || !json["artifact"].is_object()) {
        return std::nullopt;
    }
    return BackendArtifact::from_json(json["artifact"]);
}

void BackendArtifactCache::set_artifact(const std::string& model_key, const BackendArtifact& artifact) {
    if (model_key.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& json = load_model_cache_locked(model_key);
    json["schema"] = "edgefm_backend_artifact_v1";
    json["artifact"] = artifact.to_json();
    flush_model_cache_locked(model_key, json);
}

std::filesystem::path BackendArtifactCache::artifact_directory_for_model(const std::string& model_key) const {
    return cache_root() / model_key;
}

nlohmann::json& BackendArtifactCache::load_model_cache_locked(const std::string& model_key) {
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
    return json;
}

void BackendArtifactCache::flush_model_cache_locked(
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

std::filesystem::path BackendArtifactCache::cache_path_for_model(const std::string& model_key) const {
    return artifact_directory_for_model(model_key) / "artifact.json";
}

std::filesystem::path BackendArtifactCache::cache_root() const {
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "edge-fm" / "backend_artifacts";
    }
    return std::filesystem::temp_directory_path() / "edge-fm" / "backend_artifacts";
}

} // namespace edge_fm
