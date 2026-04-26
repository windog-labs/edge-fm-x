#include "engine/horizon/engine.h"

#include "backends/horizon_module_emitter.h"
#include "operators/operator_impl_table.h"
#include "utils/check.h"
#include "utils/logging.h"

#include <fstream>
#include <sstream>

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

nlohmann::json build_horizon_rewrite_plan(const EngineConfig& config) {
    const std::string model_name = config.resolved_model_name();
    const bool is_smolvla = model_name == "smolvla";
    nlohmann::json plan = {
        {"schema", "edgefm_horizon_j6m_rewrite_plan_v1"},
        {"enabled", true},
        {"target", {
            {"backend", "horizon"},
            {"hardware", config.resolved_hw_profile()},
            {"quantization", "int16"},
            {"runtime_buffer_policy", "copy_in_copy_out"},
        }},
        {"model_name", model_name},
        {"helper_module", "scripts/horizon/j6m_rewrite.py"},
        {"operator_rewrites", nlohmann::json::array({
            {
                {"id", "full_int16_quantization_contract"},
                {"kind", "quantization_contract"},
                {"description", "Compile and calibrate with signed int16 activation/weight ranges."},
            },
            {
                {"id", "attention_mask_bool_to_bounded_bias"},
                {"kind", "mask"},
                {"description", "Keep masks boolean and use bounded negative mask fill to avoid int16 overflow."},
                {"bounded_mask_fill", -32760.0},
            },
            {
                {"id", "rope_explicit_sincos_fp32"},
                {"kind", "position_encoding"},
                {"description", "Use explicit fp32 sin/cos RoPE tensors before casting to model dtype."},
            },
            {
                {"id", "gelu_tanh_int16_safe_piecewise"},
                {"kind", "activation"},
                {"description", "Replace GELU tanh activations with a piecewise int16-safe equivalent."},
                {"gelu_input_clamp", 8.0},
            },
            {
                {"id", "multimodal_input_normalization_contract"},
                {"kind", "input_normalization"},
                {"description", "Calibration inputs must match visual/state/action normalization used by policy preprocessing."},
            },
        })},
        {"scale_diagnostics", {
            {"enabled", true},
            {"int16_range", {-32768, 32767}},
            {"fail_on_nonfinite", true},
        }},
    };

    if (is_smolvla) {
        plan["smolvla"] = {
            {"lerobot_root", "~/DATA/repos/public/lerobot"},
            {"num_steps", config.prefill_model_config().value("num_steps", 10)},
            {"chunk_size", config.prefill_model_config().value("chunk_size", 50)},
            {"max_action_dim", config.prefill_model_config().value("max_action_dim", 32)},
            {"source_files", {
                "src/lerobot/policies/smolvla/modeling_smolvla.py",
                "src/lerobot/policies/smolvla/smolvlm_with_expert.py",
                "src/lerobot/policies/smolvla/configuration_smolvla.py",
                "src/lerobot/policies/smolvla/processor_smolvla.py",
            }},
        };
        plan["operator_rewrites"].push_back({
            {"id", "smolvla_make_att_2d_masks_int16_cumsum"},
            {"kind", "mask"},
            {"source_symbols", {"make_att_2d_masks"}},
            {"description", "Use int16 cumsum and boolean pad masks for SmolVLA block-causal masks."},
        });
        plan["operator_rewrites"].push_back({
            {"id", "smolvla_eager_attention_bounded_mask_fill"},
            {"kind", "attention"},
            {"source_symbols", {"SmolVLMWithExpertModel.eager_attention_forward"}},
            {"description", "Replace torch.finfo(float32).min mask fill in SmolVLA attention."},
        });
        plan["operator_rewrites"].push_back({
            {"id", "smolvla_flow_matching_loop_bins"},
            {"kind", "flow_matching"},
            {"source_symbols", {"VLAFlowMatching.sample_actions", "VLAFlowMatching.denoise_step"}},
            {"description", "Export x_t/v_t/time per denoise step and calibrate each loop step separately."},
        });
    }

    return plan;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace

HorizonEngine::HorizonEngine(const EngineConfig& config)
    : Engine(config)
{}

void HorizonEngine::warmup() {
    if (ensure_runtime_initialized(false) && runtime_backend_ != nullptr) {
        if (!runtime_backend_->warmup(1)) {
            throw ConfigurationError(runtime_backend_->last_error());
        }
    }
}

void HorizonEngine::tune() {
    check<ConfigurationError>(
        config_.backend_target() == "horizon",
        "HorizonEngine requires backend_target=horizon");
    const std::string resolved_model_name = config_.resolved_model_name();
    check<ConfigurationError>(
        resolved_model_name == "qwen2_5" ||
        resolved_model_name == "qwen2_5_vl" ||
        resolved_model_name == "smolvla",
        "HorizonEngine currently supports qwen2_5 / qwen2_5_vl / smolvla");

    const std::string model_key = config_.backend_cache_key();
    check<ConfigurationError>(
        !model_key.empty(),
        "HorizonEngine failed to resolve backend_cache_key");

    BackendArtifact artifact = build_horizon_artifact(config_);
    std::filesystem::create_directories(std::filesystem::path(artifact.manifest_path).parent_path());
    const nlohmann::json graph_tuning = build_graph_tuning(config_);
    const nlohmann::json horizon_rewrite = build_horizon_rewrite_plan(config_);
    nlohmann::json graph_tuning_with_rewrite = graph_tuning;
    graph_tuning_with_rewrite["horizon_rewrite"] = horizon_rewrite;
    HorizonModuleExport module_export = emit_horizon_module(
        resolved_model_name,
        config_.prefill_model_config(),
        config_.prefill_model_path(),
        config_.decode_model_path(),
        graph_tuning_with_rewrite,
        std::filesystem::path(artifact.manifest_path).parent_path());
    artifact.metadata["generated_module"] = module_export.to_json();

    nlohmann::json compile_spec = {
        {"schema", "edgefm_horizon_compile_spec_v2"},
        {"backend", "horizon"},
        {"model_name", resolved_model_name},
        {"model_variant", resolved_model_name},
        {"model_config", config_.prefill_model_config()},
        {"graph_tuning", graph_tuning},
        {"horizon_rewrite", horizon_rewrite},
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

std::optional<BackendArtifact> HorizonEngine::resolve_artifact() const {
    std::optional<BackendArtifact> artifact = BackendArtifactCache::instance().get_artifact(config_.backend_cache_key());
    if (!artifact.has_value() && config_.has_backend_artifact()) {
        artifact = BackendArtifact::from_json(config_.backend_artifact());
    }
    return artifact;
}

RuntimeInitParams HorizonEngine::make_runtime_init_params(const BackendArtifact& artifact) const {
    RuntimeInitParams params;
    const std::filesystem::path artifact_path(artifact.artifact_path);
    params.program_path = artifact_path.parent_path().string();
    params.model_name = artifact_path.stem().string();
    params.model_path = artifact.artifact_path;
    params.max_batch_size = 1;
    return params;
}

bool HorizonEngine::ensure_runtime_initialized(bool require_artifact) {
    if (runtime_backend_ != nullptr) {
        return true;
    }

    std::optional<BackendArtifact> artifact = resolve_artifact();
    if (!artifact.has_value()) {
        if (!require_artifact) {
            return false;
        }
        throw ConfigurationError(
            "Horizon backend requires tune() to generate a compile spec before generate(). "
            "Then run: python scripts/horizon/compile_horizon_from_spec.py <generated spec>");
    }

    if (!std::filesystem::exists(artifact->artifact_path)) {
        if (!require_artifact) {
            return false;
        }
        throw ConfigurationError(
            "Horizon HBM artifact not found: " + artifact->artifact_path +
            ". Compile it with helper script scripts/horizon/compile_horizon_from_spec.py. "
            "generated spec: " + artifact->manifest_path);
    }

    runtime_backend_ = create_runtime_backend(BackendTarget::Horizon);
    const RuntimeInitParams params = make_runtime_init_params(*artifact);
    if (!runtime_backend_->init(params)) {
        const std::string error = runtime_backend_->last_error().empty()
            ? "Horizon runtime backend failed to initialize"
            : runtime_backend_->last_error();
        runtime_backend_.reset();
        if (!require_artifact) {
            throw ConfigurationError(error);
        }
        throw ConfigurationError(error);
    }
    log_runtime_io();
    return true;
}

void HorizonEngine::log_runtime_io() const {
    if (runtime_backend_ == nullptr) {
        return;
    }
    for (const auto& name : runtime_backend_->input_names()) {
        std::vector<int64_t> shape;
        const std::string shape_str = runtime_backend_->get_input_shape(name, &shape)
            ? shape_to_string(shape)
            : std::string("<unknown>");
        Logging::instance().log_info("Horizon input '{}' shape {}", name, shape_str);
    }
    for (const auto& name : runtime_backend_->output_names()) {
        std::vector<int64_t> shape;
        const std::string shape_str = runtime_backend_->get_output_shape(name, &shape)
            ? shape_to_string(shape)
            : std::string("<unknown>");
        Logging::instance().log_info("Horizon output '{}' shape {}", name, shape_str);
    }
}

Response HorizonEngine::generate(const Request& request) {
    if (request.token_ids().empty()) {
        throw InvalidRequestError("Horizon generate requires at least one token id");
    }

    last_generate_metrics_.clear();
    ensure_runtime_initialized(true);
    log_runtime_io();
    throw InternalError("Horizon generate I/O mapping is not implemented in this interface phase");
}

std::unordered_map<std::string, double> HorizonEngine::get_last_generate_metrics() const {
    return last_generate_metrics_;
}

} // namespace edge_fm
