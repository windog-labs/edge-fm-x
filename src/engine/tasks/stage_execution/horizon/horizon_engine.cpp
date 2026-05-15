#include "engine/tasks/stage_execution/horizon/horizon_engine.h"

#include "backends/horizon_module_emitter.h"
#include "operators/operator_impl_table.h"
#include "utils/check.h"
#include "utils/logging.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace edge_fm {

namespace {

nlohmann::json build_smolvla_stage_specs(
    const EngineConfig& config,
    const std::filesystem::path& artifact_dir);

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
    if (config.resolved_model_name() == "smolvla") {
        artifact.metadata["stages"] = build_smolvla_stage_specs(config, artifact_dir);
    }
    return artifact;
}

int32_t smolvla_num_layers(const nlohmann::json& model_config) {
    const int32_t explicit_layers = model_config.value("num_vlm_layers", 0);
    if (explicit_layers > 0) {
        return explicit_layers;
    }
    const int32_t hidden_layers = model_config.value("num_hidden_layers", 16);
    return std::max(1, std::min(hidden_layers, 16));
}

nlohmann::json smolvla_options(const EngineConfig& config) {
    const auto& raw = config.raw();
    if (raw.contains("smolvla") && raw["smolvla"].is_object()) {
        return raw["smolvla"];
    }
    return nlohmann::json::object();
}

nlohmann::json build_smolvla_stage_specs(
    const EngineConfig& config,
    const std::filesystem::path& artifact_dir)
{
    const nlohmann::json model_config = config.prefill_model_config();
    const nlohmann::json opts = smolvla_options(config);
    const int32_t hidden_size = model_config.value("hidden_size", 960);
    const int32_t num_attention_heads = model_config.value("num_attention_heads", 15);
    const int32_t num_kv_heads = model_config.value("num_key_value_heads", 5);
    const int32_t head_dim = model_config.value(
        "head_dim",
        num_attention_heads > 0 ? hidden_size / num_attention_heads : 64);
    const int32_t num_layers = opts.value("num_layers", smolvla_num_layers(model_config));
    const int32_t prefix_len = opts.value("prefix_len", 128);
    const int32_t suffix_len = opts.value("suffix_len", opts.value("chunk_size", 50));
    const int32_t expert_hidden_size = opts.value(
        "expert_hidden_size",
        std::max(1, static_cast<int32_t>(hidden_size * 3 / 4)));

    auto tensor_desc = [](const std::string& name,
                          std::initializer_list<int64_t> shape,
                          const std::string& dtype = "float32") {
        return nlohmann::json{
            {"name", name},
            {"shape", std::vector<int64_t>(shape)},
            {"dtype", dtype},
        };
    };

    nlohmann::json prefill_outputs = nlohmann::json::array();
    nlohmann::json denoise_inputs = nlohmann::json::array({
        tensor_desc("suffix_embeds", {1, suffix_len, expert_hidden_size}),
        tensor_desc("denoise_attention_mask", {1, suffix_len, prefix_len + suffix_len}, "uint8"),
        tensor_desc("suffix_position_ids", {1, suffix_len}, "int32"),
    });
    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        const std::string name = "prefix_kv_layer_" + std::to_string(layer_id);
        nlohmann::json desc = tensor_desc(name, {2, prefix_len, num_kv_heads, head_dim});
        prefill_outputs.push_back(desc);
        denoise_inputs.push_back(desc);
    }

    return nlohmann::json::array({
        {
            {"name", "prefill"},
            {"artifact_path", (artifact_dir / "smolvla_prefill.hbm").string()},
            {"factory_kwargs", {{"stage", "prefill"}}},
            {"inputs", nlohmann::json::array({
                tensor_desc("prefix_embeds", {1, prefix_len, hidden_size}),
                tensor_desc("prefix_attention_mask", {1, prefix_len, prefix_len}, "uint8"),
                tensor_desc("prefix_position_ids", {1, prefix_len}, "int32"),
            })},
            {"outputs", prefill_outputs},
            {"kv_layout", "packed_layer_kv_v1"},
        },
        {
            {"name", "decode"},
            {"artifact_path", (artifact_dir / "smolvla_decode.hbm").string()},
            {"factory_kwargs", {{"stage", "decode"}}},
            {"inputs", denoise_inputs},
            {"outputs", nlohmann::json::array({
                tensor_desc("expert_hidden", {1, suffix_len, expert_hidden_size}),
            })},
            {"kv_layout", "packed_layer_kv_v1"},
        },
    });
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
    nlohmann::json graph_tuning = {
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
    if (config.resolved_model_name() == "smolvla") {
        const nlohmann::json model_config = config.prefill_model_config();
        const nlohmann::json opts = smolvla_options(config);
        graph_tuning["smolvla_phase1"] = {
            {"enabled", true},
            {"scope", "llm_prefill_plus_action_expert"},
            {"excluded", {"vit", "embed_suffix", "action_out_proj"}},
            {"num_layers", opts.value("num_layers", smolvla_num_layers(model_config))},
            {"kv_layout", "packed_layer_kv_v1"},
        };
    }
    return graph_tuning;
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
        const nlohmann::json opts = smolvla_options(config);
        plan["smolvla"] = {
            {"lerobot_root", opts.value("lerobot_root", std::string("~/Repos/public/lerobot-v0.4.4"))},
            {"num_layers", opts.value("num_layers", smolvla_num_layers(config.prefill_model_config()))},
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

RuntimeDType runtime_dtype_from_tensor_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return RuntimeDType::Float32;
        case DType::Float16:
            return RuntimeDType::Float16;
        case DType::Int32:
            return RuntimeDType::Int32;
        case DType::Int8:
            return RuntimeDType::Int8;
        case DType::UInt8:
            return RuntimeDType::UInt8;
        default:
            throw InvalidRequestError("Unsupported tensor dtype for Horizon runtime I/O");
    }
}

DType tensor_dtype_from_runtime_dtype(RuntimeDType dtype) {
    switch (dtype) {
        case RuntimeDType::Float32:
            return DType::Float32;
        case RuntimeDType::Float16:
            return DType::Float16;
        case RuntimeDType::Int32:
            return DType::Int32;
        case RuntimeDType::Int8:
            return DType::Int8;
        case RuntimeDType::UInt8:
            return DType::UInt8;
    }
    throw InvalidRequestError("Unsupported Horizon runtime dtype");
}

void copy_tensor_to_runtime_input(
    const Tensor& src,
    const RuntimeTensorView& dst,
    const std::string& input_name)
{
    auto [src_device, src_device_id] = src.device();
    (void)src_device_id;
    check<InvalidRequestError>(
        src_device == Device::CPU,
        "Horizon tensor stage currently expects CPU input tensor for: " + input_name);
    check<InvalidRequestError>(
        runtime_dtype_from_tensor_dtype(src.dtype()) == dst.dtype,
        "Horizon tensor stage input dtype mismatch for: " + input_name);
    check<InvalidRequestError>(
        src.shape() == dst.shape,
        "Horizon tensor stage input shape mismatch for: " + input_name +
        ", expected " + shape_to_string(dst.shape) +
        ", got " + shape_to_string(src.shape()));
    copy_contiguous_to_runtime_buffer(src.data_ptr(), dst);
}

Tensor clone_runtime_output_to_cpu(const RuntimeTensorView& src) {
    return Tensor::clone_from(
        src.data,
        src.shape,
        tensor_dtype_from_runtime_dtype(src.dtype),
        Device::CPU,
        0,
        Device::CPU,
        0,
        MemoryOwnership::OwnCpuMalloc);
}

Tensor clone_tensor_to_cpu(const Tensor& src) {
    auto [src_device, src_device_id] = src.device();
    return Tensor::clone_from(
        src.data_ptr(),
        src.shape(),
        src.dtype(),
        src_device,
        src_device_id,
        Device::CPU,
        0,
        MemoryOwnership::OwnCpuMalloc);
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
    if (resolved_model_name == "smolvla" &&
        artifact.metadata.contains("stages") &&
        artifact.metadata["stages"].is_array())
    {
        compile_spec["stages"] = artifact.metadata["stages"];
        compile_spec["compile_entry"]["default_kwargs"] = {{"stage", "prefill"}};
    }

    std::ofstream output(artifact.manifest_path);
    check<ConfigurationError>(
        output.is_open(),
        "Failed to write Horizon compile spec: " + artifact.manifest_path);
    output << compile_spec.dump(2);
    Logging::instance().log_info("Horizon compile spec written: {}", artifact.manifest_path);

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

RuntimeInitParams HorizonEngine::make_runtime_init_params_for_stage(const nlohmann::json& stage) const {
    check<ConfigurationError>(
        stage.is_object(),
        "SmolVLA Horizon stage metadata must be an object");
    const std::string artifact_path_str = stage.value("artifact_path", std::string(""));
    check<ConfigurationError>(
        !artifact_path_str.empty(),
        "SmolVLA Horizon stage is missing artifact_path");
    const std::filesystem::path artifact_path(artifact_path_str);
    RuntimeInitParams params;
    params.program_path = artifact_path.parent_path().string();
    params.model_name = artifact_path.stem().string();
    params.model_path = artifact_path.string();
    params.max_batch_size = 1;
    if (stage.contains("inputs") && stage["inputs"].is_array()) {
        for (const auto& input : stage["inputs"]) {
            if (!input.is_object()) {
                continue;
            }
            const std::string name = input.value("name", std::string(""));
            if (name.empty() || !input.contains("shape") || !input["shape"].is_array()) {
                continue;
            }
            params.input_shape_overrides[name] = input["shape"].get<std::vector<int64_t>>();
        }
    }
    return params;
}

const nlohmann::json& HorizonEngine::require_stage(const std::string& stage_name) const {
    check<ConfigurationError>(
        stage_artifact_.has_value(),
        "Horizon stage artifact is not resolved");
    const nlohmann::json& metadata = stage_artifact_->metadata;
    check<ConfigurationError>(
        metadata.contains("stages") && metadata["stages"].is_array(),
        "Horizon artifact metadata must contain stages");
    for (const auto& stage : metadata["stages"]) {
        if (stage.is_object() && stage.value("name", std::string("")) == stage_name) {
            return stage;
        }
    }
    throw ConfigurationError("Horizon artifact is missing stage: " + stage_name);
}

IRuntimeBackend& HorizonEngine::ensure_stage_runtime(const std::string& stage_name) {
    auto existing = stage_backends_.find(stage_name);
    if (existing != stage_backends_.end() && existing->second != nullptr) {
        return *existing->second;
    }

    std::optional<BackendArtifact> artifact = resolve_artifact();
    check<ConfigurationError>(
        artifact.has_value(),
        "Horizon backend requires tune() to generate stage compile specs before running stage APIs");
    stage_artifact_ = artifact;

    const nlohmann::json& stage = require_stage(stage_name);
    const std::string artifact_path = stage.value("artifact_path", std::string(""));
    check<ConfigurationError>(
        std::filesystem::exists(artifact_path),
        "Horizon stage HBM artifact not found: " + artifact_path +
        ". Compile it with scripts/horizon/compile_horizon_from_spec.py");

    std::unique_ptr<IRuntimeBackend> runtime = create_runtime_backend(BackendTarget::Horizon);
    RuntimeInitParams params = make_runtime_init_params_for_stage(stage);
    if (!runtime->init(params)) {
        const std::string error = runtime->last_error().empty()
            ? "Horizon runtime backend failed to initialize stage: " + stage_name
            : runtime->last_error();
        throw ConfigurationError(error);
    }

    IRuntimeBackend* runtime_ptr = runtime.get();
    stage_backends_[stage_name] = std::move(runtime);
    log_runtime_io(*runtime_ptr, "Horizon " + stage_name);
    return *runtime_ptr;
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
    log_runtime_io(*runtime_backend_, "Horizon");
}

void HorizonEngine::log_runtime_io(const IRuntimeBackend& runtime, const std::string& label) const {
    for (const auto& name : runtime.input_names()) {
        std::vector<int64_t> shape;
        const std::string shape_str = runtime.get_input_shape(name, &shape)
            ? shape_to_string(shape)
            : std::string("<unknown>");
        Logging::instance().log_info("{} input '{}' shape {}", label, name, shape_str);
    }
    for (const auto& name : runtime.output_names()) {
        std::vector<int64_t> shape;
        const std::string shape_str = runtime.get_output_shape(name, &shape)
            ? shape_to_string(shape)
            : std::string("<unknown>");
        Logging::instance().log_info("{} output '{}' shape {}", label, name, shape_str);
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

TensorMap HorizonEngine::run_stage(int32_t request_id, const std::string& stage_name, const TensorRefMap& inputs) {
    check<InvalidRequestError>(
        request_id >= 0,
        "Horizon run_stage requires non-negative request_id");

    IRuntimeBackend& runtime = ensure_stage_runtime(stage_name);
    auto cache_it = stage_tensor_cache_.find(request_id);

    for (const auto& name : runtime.input_names()) {
        const Tensor* tensor = nullptr;
        auto input_it = inputs.find(name);
        if (input_it != inputs.end()) {
            tensor = input_it->second;
            check<InvalidRequestError>(
                tensor != nullptr,
                "Horizon decode input tensor must be non-null: " + name);
        } else if (cache_it != stage_tensor_cache_.end()) {
            auto cached_it = cache_it->second.tensors.find(name);
            if (cached_it != cache_it->second.tensors.end()) {
                tensor = &cached_it->second;
            }
        }

        check<InvalidRequestError>(
            tensor != nullptr,
            "Horizon stage '" + stage_name + "' missing required input tensor: " + name +
            ". Provide it explicitly or run a producer stage with the same request_id first.");

        RuntimeTensorView view;
        check<InvalidRequestError>(
            runtime.get_input_buffer(name, &view),
            "Horizon stage HBM is missing input buffer: " + name);
        copy_tensor_to_runtime_input(*tensor, view, name);
    }

    if (runtime.forward_sync() != 0) {
        throw InternalError(
            runtime.last_error().empty()
                ? "Horizon stage runtime failed: " + stage_name
                : runtime.last_error());
    }

    TensorMap outputs;
    CachedStageTensors& cache = stage_tensor_cache_[request_id];
    for (const auto& name : runtime.output_names()) {
        RuntimeTensorView output;
        check<InternalError>(
            runtime.get_output_buffer(name, &output),
            "Failed to read Horizon stage output: " + name);
        Tensor tensor = clone_runtime_output_to_cpu(output);
        if (std::find(cache.names.begin(), cache.names.end(), name) == cache.names.end()) {
            cache.names.push_back(name);
        }
        cache.tensors[name] = clone_tensor_to_cpu(tensor);
        outputs.emplace(name, std::move(tensor));
    }
    last_stage_metrics_ = {
        {"stage_outputs", static_cast<double>(outputs.size())},
    };
    return outputs;
}

TensorMap HorizonEngine::prefill(int32_t request_id, const TensorRefMap& inputs) {
    return run_stage(request_id, "prefill", inputs);
}

TensorMap HorizonEngine::decode(int32_t request_id, const TensorRefMap& inputs) {
    return run_stage(request_id, "decode", inputs);
}

std::unordered_map<std::string, double> HorizonEngine::get_last_generate_metrics() const {
    return last_generate_metrics_;
}

std::unordered_map<std::string, double> HorizonEngine::get_last_stage_metrics() const {
    return last_stage_metrics_;
}

} // namespace edge_fm
