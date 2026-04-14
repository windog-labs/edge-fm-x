#include "engine/horizon_engine.h"

#include "backends/backend_artifact_cache.h"
#include "backends/horizon_module_emitter.h"
#include "operators/operator_impl_table.h"
#include "utils/check.h"

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
        {"model_name", config.resolved_model_name()},
        {"resolved_hw_profile", config.resolved_hw_profile()},
    };
    return artifact;
}

bool config_uses_mrope(const EngineConfig& config) {
    const nlohmann::json model_config = config.prefill_model_config();
    return model_config.contains("rope_scaling") &&
        model_config["rope_scaling"].is_object() &&
        model_config["rope_scaling"].value(
            "type",
            model_config["rope_scaling"].value("rope_type", std::string(""))) == "mrope";
}

nlohmann::json linear_operator_table_for_model(const EngineConfig& config) {
    nlohmann::json records = nlohmann::json::array();
    for (const auto& record : OperatorImplTable::instance().records_for_model(
             config.resolved_model_name(),
             config.resolved_hw_profile(),
             config.operator_impl_table_path(),
             "linear"))
    {
        records.push_back(record.to_json());
    }
    return records;
}

nlohmann::json build_graph_tuning(const EngineConfig& config) {
    return nlohmann::json{
        {"attention_type", config.kvcache_attention_type()},
        {"kv_cache", {
            {"dtype", config.kvcache_dtype()},
            {"layout", config.kvcache_attention_type()},
        }},
        {"uses_mrope", config_uses_mrope(config)},
        {"uses_embedding_injection", config.resolved_model_name() == "qwen2_5_vl"},
        {"linear_operator_table", linear_operator_table_for_model(config)},
        {"target_hw_constraints", {
            {"backend_target", config.backend_target()},
            {"runtime_device", config.runtime_device()},
            {"runtime_device_id", config.runtime_device_id()},
            {"resolved_hw_profile", config.resolved_hw_profile()},
        }},
    };
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
    const std::string resolved_model_name = config_.resolved_model_name();
    check<ConfigurationError>(
        resolved_model_name == "qwen2_5" || resolved_model_name == "qwen2_5_vl",
        "HorizonEngine currently supports qwen2_5 / qwen2_5_vl");

    const std::string model_key = config_.backend_cache_key();
    check<ConfigurationError>(
        !model_key.empty(),
        "HorizonEngine failed to resolve backend_cache_key");

    BackendArtifact artifact = build_horizon_artifact(config_);
    std::filesystem::create_directories(std::filesystem::path(artifact.manifest_path).parent_path());
    const nlohmann::json graph_tuning = build_graph_tuning(config_);
    HorizonModuleExport module_export = emit_horizon_module(
        resolved_model_name,
        config_.prefill_model_config(),
        config_.prefill_model_path(),
        config_.decode_model_path(),
        graph_tuning,
        std::filesystem::path(artifact.manifest_path).parent_path());
    artifact.metadata["generated_module"] = module_export.to_json();

    nlohmann::json compile_spec = {
        {"schema", "edgefm_horizon_compile_spec_v2"},
        {"backend", "horizon"},
        {"model_name", resolved_model_name},
        {"model_variant", resolved_model_name},
        {"model_config", config_.prefill_model_config()},
        {"graph_tuning", graph_tuning},
        {"generated_module", module_export.to_json()},
        {"helper_script", "scripts/horizon/compile_horizon_from_spec.py"},
        {"engine_config", {
            {"model_name", config_.model_name()},
            {"operator_impl_table_path", config_.operator_impl_table_path()},
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
            "Then run: python scripts/horizon/compile_horizon_from_spec.py <generated spec>");
    }

    if (!std::filesystem::exists(artifact->artifact_path)) {
        throw ConfigurationError(
            "Horizon HBM artifact not found: " + artifact->artifact_path +
            ". Compile it with helper script scripts/horizon/compile_horizon_from_spec.py. "
            "generated spec: " + artifact->manifest_path);
    }

    throw InternalError("Horizon runtime execution is not implemented in this build");
}

std::unordered_map<std::string, double> HorizonEngine::get_last_generate_metrics() const {
    return {};
}

void HorizonEngine::prepare_tensors(ModelStage /*stage*/, Context& /*context*/) {
    throw InternalError("HorizonEngine does not use the CUDA tensor preparation path");
}

} // namespace edge_fm
