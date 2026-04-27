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
    TensorMap prefill(int32_t request_id, const TensorRefMap& inputs) override;
    TensorMap decode(int32_t request_id, const TensorRefMap& inputs) override;
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;

private:
    struct CachedStageTensors {
        std::vector<std::string> names;
        std::unordered_map<std::string, Tensor> tensors;
    };

    std::optional<BackendArtifact> resolve_artifact() const;
    bool ensure_runtime_initialized(bool require_artifact);
    IRuntimeBackend& ensure_smolvla_stage_runtime(const std::string& stage_name);
    RuntimeInitParams make_runtime_init_params(const BackendArtifact& artifact) const;
    RuntimeInitParams make_runtime_init_params_for_stage(const nlohmann::json& stage) const;
    void log_runtime_io() const;
    void log_runtime_io(const IRuntimeBackend& runtime, const std::string& label) const;
    const nlohmann::json& require_smolvla_stage(const std::string& stage_name) const;

    std::unique_ptr<IRuntimeBackend> runtime_backend_;
    std::unordered_map<std::string, std::unique_ptr<IRuntimeBackend>> smolvla_stage_backends_;
    std::unordered_map<int32_t, CachedStageTensors> stage_tensor_cache_;
    std::optional<BackendArtifact> smolvla_artifact_;
    std::unordered_map<std::string, double> last_generate_metrics_{};
};

} // namespace edge_fm
