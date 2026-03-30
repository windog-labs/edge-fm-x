#include "engine/horizon_engine.h"

#include "utils/backend_artifact_cache.h"
#include "utils/check.h"
#include "utils/horizon_module_emitter.h"

#include <fstream>

namespace edge_fm {

namespace {

BackendArtifact build_horizon_artifact(const EngineConfig& config) {
    const std::string model_key = config.backend_cache_key();
    const auto artifact_dir = BackendArtifactCache::instance().artifact_directory_for_model(model_key);

    BackendArtifact artifact;
    artifact.backend = "horizon";
    artifact.artifact_type = "hbm";
    artifact.artifact_path = (artifact_dir / "model.hbm").string();
    artifact.manifest_path = (artifact_dir / "compile_spec.json").string();
    artifact.metadata = {
        {"status", "external_compile_required"},
        {"backend_target", config.backend_target()},
        {"model_description_hash", config.model_description_hash()},
    };
    return artifact;
}

} // namespace

HorizonEngine::HorizonEngine(const EngineConfig& config)
    : Engine(config)
{}

void HorizonEngine::warmup() {
    // Whole-graph backends typically warm up after artifact generation.
}

void HorizonEngine::tune() {
    check<ConfigurationError>(
        config_.backend_target() == "horizon",
        "HorizonEngine requires backend_target=horizon");
    check<ConfigurationError>(
        config_.has_model_description(),
        "HorizonEngine requires _edgefm_internal.model_description");

    const std::string model_key = config_.backend_cache_key();
    check<ConfigurationError>(
        !model_key.empty(),
        "HorizonEngine requires _edgefm_internal.backend_cache_key");

    BackendArtifact artifact = build_horizon_artifact(config_);
    std::filesystem::create_directories(std::filesystem::path(artifact.manifest_path).parent_path());
    HorizonModuleExport module_export = emit_horizon_module(
        config_.model_description(),
        config_.prefill_model_config(),
        config_.prefill_model_path(),
        config_.decode_model_path(),
        std::filesystem::path(artifact.manifest_path).parent_path());
    artifact.metadata["generated_module"] = module_export.to_json();

    nlohmann::json compile_spec = {
        {"schema", "edgefm_horizon_compile_spec_v1"},
        {"backend", "horizon"},
        {"model_description", config_.model_description()},
        {"model_config", config_.prefill_model_config()},
        {"generated_module", module_export.to_json()},
        {"helper_script", "scripts/compile_horizon_from_spec.py"},
        {"engine_config", {
            {"runtime", config_.runtime()},
            {"kvcache", config_.kvcache()},
            {"sampling", config_.sampling()},
            {"prefill_model_path", config_.prefill_model_path()},
            {"decode_model_path", config_.decode_model_path()},
        }},
        {"compile_entry", {
            {"module_path", module_export.module_path},
            {"factory_function", module_export.factory_function},
            {"default_kwargs", {{"stage", "prefill"}}},
            {"suggested_artifact_path", artifact.artifact_path},
        }},
        {"artifact", artifact.to_json()},
    };

    std::ofstream output(artifact.manifest_path);
    check<ConfigurationError>(
        output.is_open(),
        "Failed to write Horizon compile spec: " + artifact.manifest_path);
    output << compile_spec.dump(2);

    BackendArtifactCache::instance().set_artifact(model_key, artifact);
}

Response HorizonEngine::generate(const Request& /*request*/) {
    std::optional<BackendArtifact> artifact = BackendArtifactCache::instance().get_artifact(config_.backend_cache_key());
    if (!artifact.has_value() && config_.has_backend_artifact()) {
        artifact = BackendArtifact::from_json(config_.backend_artifact());
    }

    if (!artifact.has_value()) {
        throw ConfigurationError(
            "Horizon backend requires tune() to generate a compile spec before generate(). "
            "Then run: python scripts/compile_horizon_from_spec.py <generated spec>");
    }

    if (!std::filesystem::exists(artifact->artifact_path)) {
        throw ConfigurationError(
            "Horizon HBM artifact not found: " + artifact->artifact_path +
            ". Compile it with helper script scripts/compile_horizon_from_spec.py. "
            "generated spec: " + artifact->manifest_path);
    }

    throw InternalError("Horizon runtime execution is not implemented in this build");
}

void HorizonEngine::prepare_tensors(ModelStage /*stage*/, Context& /*context*/) {
    throw InternalError("HorizonEngine does not use the CUDA tensor preparation path");
}

} // namespace edge_fm
