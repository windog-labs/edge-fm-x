#pragma once

#include "engine/engine.h"
#include "backends/backend_artifact_cache.h"
#include "backends/runtime_backend.h"

#include <memory>
#include <optional>
#include <unordered_map>

namespace edge_fm {

class HorizonEngine : public Engine {
public:
    explicit HorizonEngine(const EngineConfig& config);
    ~HorizonEngine() override = default;

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;

private:
    std::optional<BackendArtifact> resolve_artifact() const;
    bool ensure_runtime_initialized(bool require_artifact);
    RuntimeInitParams make_runtime_init_params(const BackendArtifact& artifact) const;
    void log_runtime_io() const;

    std::unique_ptr<IRuntimeBackend> runtime_backend_;
    std::unordered_map<std::string, double> last_generate_metrics_{};
};

} // namespace edge_fm
