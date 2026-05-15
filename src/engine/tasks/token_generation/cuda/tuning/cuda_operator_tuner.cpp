#include "engine/tasks/token_generation/cuda/tuning/cuda_operator_tuner.h"

#include "engine/engine.h"
#include "layers/attention.h"
#include "layers/linear.h"
#include "models/qwen2_5/qwen2_5.h"
#include "operators/operator_impl_table.h"
#include "utils/device/cuda_utils.h"
#include "utils/logging.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace edge_fm {

namespace {

constexpr const char* kTuningReportSchema = "edgefm_cuda_operator_tuning_report_v1";
constexpr const char* kTuningTableSchema = "edgefm_operator_impl_table_v1";
constexpr const char* kTuningVersion = "cuda_operator_tuning_v1";

constexpr const char* kAttentionPrefillEnvWarmup = "EDGE_FM_TUNING_ATTENTION_WARMUP";
constexpr const char* kAttentionPrefillEnvIters = "EDGE_FM_TUNING_ATTENTION_ITERS";
constexpr const char* kLinearEnvWarmup = "EDGE_FM_TUNING_LINEAR_WARMUP";
constexpr const char* kLinearEnvIters = "EDGE_FM_TUNING_LINEAR_ITERS";
constexpr const char* kPrefillListEnv = "EDGE_FM_TUNING_PREFILL_LIST";
constexpr const char* kKvLensEnv = "EDGE_FM_TUNING_KV_LENS";
constexpr const char* kReducedCandidatesEnv = "EDGE_FM_TUNING_REDUCED_CANDIDATES";
constexpr const char* kMaxLinearAlgoCandidatesEnv = "EDGE_FM_TUNING_MAX_LINEAR_ALGO_CANDIDATES";

struct ScopedCudaEvent {
    ScopedCudaEvent() {
        CUDA_CHECK_THROW(cudaEventCreate(&event_), "Failed to create CUDA event");
    }

    ~ScopedCudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    ScopedCudaEvent(const ScopedCudaEvent&) = delete;
    ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct DeviceBuffer {
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size_bytes)
        : size_bytes(size_bytes)
    {
        if (size_bytes > 0) {
            CUDA_CHECK_THROW(cudaMalloc(&ptr, size_bytes), "Failed to allocate CUDA buffer");
        }
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr(other.ptr)
        , size_bytes(other.size_bytes)
    {
        other.ptr = nullptr;
        other.size_bytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            size_bytes = other.size_bytes;
            other.ptr = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }

    void* ptr = nullptr;
    size_t size_bytes = 0;
};

struct TuningOptions {
    std::vector<int32_t> prefill_list{512, 1024, 2048};
    std::vector<int32_t> kv_lens{512, 1024, 2048};
    int32_t attention_warmup = 20;
    int32_t attention_iters = 100;
    int32_t linear_warmup = 10;
    int32_t linear_iters = 60;
    bool reduced_candidates = false;
    int32_t max_linear_algo_candidates = -1;
};

struct LinearDims {
    int32_t hidden = 0;
    int32_t intermediate = 0;
    int32_t kv = 0;
    int32_t vocab = 0;
};

struct OverridePathGuard {
    explicit OverridePathGuard(EngineConfig& config)
        : config(config)
        , had_override(config.has_operator_impl_table_override())
        , original_path(config.operator_impl_table_path())
    {}

    ~OverridePathGuard() {
        if (had_override) {
            config.set_operator_impl_table_override(original_path);
        } else {
            config.clear_operator_impl_table_override();
        }
    }

    EngineConfig& config;
    bool had_override = false;
    std::string original_path;
};

size_t num_elements(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

Tensor make_zero_gpu_tensor(const std::vector<int64_t>& shape, DType dtype, int32_t device_id) {
    const size_t bytes = num_elements(shape) * get_dtype_size(dtype);
    if (bytes == 0) {
        return Tensor();
    }
    CUDA_CHECK_THROW(cudaSetDevice(device_id), "Failed to set device for tuning tensor allocation");

    void* ptr = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&ptr, bytes), "Failed to allocate tuning tensor");
    CUDA_CHECK_THROW(cudaMemset(ptr, 0, bytes), "Failed to zero tuning tensor");
    return Tensor::adopt(ptr, shape, dtype, Device::GPU, device_id, MemoryOwnership::OwnCudaMalloc);
}

DeviceBuffer make_uint32_device_value(uint32_t value, int32_t device_id) {
    CUDA_CHECK_THROW(cudaSetDevice(device_id), "Failed to set device for tuning scalar allocation");
    DeviceBuffer buffer(sizeof(uint32_t));
    CUDA_CHECK_THROW(
        cudaMemcpy(buffer.ptr, &value, sizeof(uint32_t), cudaMemcpyHostToDevice),
        "Failed to upload tuning scalar");
    return buffer;
}

std::filesystem::path backend_cache_root() {
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "edge-fm" / "backend_artifacts";
    }
    return std::filesystem::temp_directory_path() / "edge-fm" / "backend_artifacts";
}

bool env_is_truthy(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

int32_t env_i32(const char* name, int32_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return static_cast<int32_t>(std::stoi(raw));
    } catch (...) {
        return default_value;
    }
}

std::vector<int32_t> parse_i32_csv(const char* raw, const std::vector<int32_t>& default_value) {
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    std::vector<int32_t> values;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        try {
            values.push_back(static_cast<int32_t>(std::stoi(item)));
        } catch (...) {
            return default_value;
        }
    }

    if (values.empty()) {
        return default_value;
    }
    return values;
}

TuningOptions load_tuning_options() {
    TuningOptions options;
    options.attention_warmup = env_i32(kAttentionPrefillEnvWarmup, options.attention_warmup);
    options.attention_iters = env_i32(kAttentionPrefillEnvIters, options.attention_iters);
    options.linear_warmup = env_i32(kLinearEnvWarmup, options.linear_warmup);
    options.linear_iters = env_i32(kLinearEnvIters, options.linear_iters);
    options.prefill_list = parse_i32_csv(std::getenv(kPrefillListEnv), options.prefill_list);
    options.kv_lens = parse_i32_csv(std::getenv(kKvLensEnv), options.kv_lens);
    options.reduced_candidates = env_is_truthy(kReducedCandidatesEnv, false);
    options.max_linear_algo_candidates = env_i32(kMaxLinearAlgoCandidatesEnv, -1);
    if (options.reduced_candidates && options.max_linear_algo_candidates <= 0) {
        options.max_linear_algo_candidates = 2;
    }

    auto normalize = [](std::vector<int32_t>& values) {
        values.erase(
            std::remove_if(values.begin(), values.end(), [](int32_t value) { return value <= 0; }),
            values.end());
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    };
    normalize(options.prefill_list);
    normalize(options.kv_lens);
    if (options.prefill_list.empty()) {
        options.prefill_list = {512, 1024, 2048};
    }
    if (options.kv_lens.empty()) {
        options.kv_lens = {512, 1024, 2048};
    }
    return options;
}

double median_cuda_ms(const std::function<void()>& fn, int32_t warmup, int32_t iters) {
    for (int32_t i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to sync after tuning warmup");

    ScopedCudaEvent start;
    ScopedCudaEvent end;
    std::vector<float> values;
    values.reserve(static_cast<size_t>(iters));
    for (int32_t i = 0; i < iters; ++i) {
        CUDA_CHECK_THROW(cudaEventRecord(start.get()), "Failed to record tuning start event");
        fn();
        CUDA_CHECK_THROW(cudaEventRecord(end.get()), "Failed to record tuning end event");
        CUDA_CHECK_THROW(cudaEventSynchronize(end.get()), "Failed to sync tuning end event");

        float elapsed_ms = 0.0f;
        CUDA_CHECK_THROW(
            cudaEventElapsedTime(&elapsed_ms, start.get(), end.get()),
            "Failed to measure tuning latency");
        values.push_back(elapsed_ms);
    }

    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (static_cast<double>(values[mid - 1]) + static_cast<double>(values[mid]));
    }
    return static_cast<double>(values[mid]);
}

std::string attention_shape_sig(const Qwen2_5& model) {
    return "num_qo_heads=" + std::to_string(model.num_attention_heads()) +
        "|num_kv_heads=" + std::to_string(model.num_kv_heads()) +
        "|head_dim=" + std::to_string(model.head_dim());
}

LinearDims linear_dims(const Qwen2_5& model) {
    LinearDims dims;
    dims.hidden = model.hidden_size();
    dims.intermediate = model.intermediate_size();
    dims.kv = model.num_kv_heads() * model.head_dim();
    dims.vocab = model.vocab_size();
    return dims;
}

std::string linear_shape_sig(const std::string& kind, int32_t m, DType dtype, const LinearDims& dims) {
    int32_t in_features = 0;
    int32_t out_features = 0;
    if (kind == "fused_qkv") {
        in_features = dims.hidden;
        out_features = dims.hidden + 2 * dims.kv;
    } else if (kind == "attention_output") {
        in_features = dims.hidden;
        out_features = dims.hidden;
    } else if (kind == "mlp_down") {
        in_features = dims.intermediate;
        out_features = dims.hidden;
    } else if (kind == "fused_gate_up") {
        in_features = dims.hidden;
        out_features = 2 * dims.intermediate;
    } else if (kind == "lm_head") {
        in_features = dims.hidden;
        out_features = dims.vocab;
    } else {
        throw ConfigurationError("Unsupported linear tuning kind: " + kind);
    }

    const int dtype_id = static_cast<int>(dtype);
    return "m=" + std::to_string(m) +
        "|input=" + std::to_string(dtype_id) +
        "|weight=" + std::to_string(dtype_id) +
        "|output=" + std::to_string(dtype_id) +
        "|in_features=" + std::to_string(in_features) +
        "|out_features=" + std::to_string(out_features);
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& payload) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output.is_open()) {
        throw InternalError("Failed to open tuning output file: " + path.string());
    }
    output << payload.dump(2) << "\n";
}

nlohmann::json read_json_file_if_exists(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return nlohmann::json::object();
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        return nlohmann::json::object();
    }

    try {
        nlohmann::json payload;
        input >> payload;
        return payload;
    } catch (...) {
        return nlohmann::json::object();
    }
}

bool is_valid_cached_report(const std::filesystem::path& report_path) {
    const nlohmann::json report = read_json_file_if_exists(report_path);
    return report.is_object() &&
        report.value("schema", std::string()) == kTuningReportSchema &&
        report.value("tuning_version", std::string()) == kTuningVersion;
}

bool is_cuda_tuning_table_path(const std::filesystem::path& path) {
    return path.filename() == "operator_impl_table.json" &&
        path.parent_path().filename() == "cuda_operator_tuning";
}

std::vector<OperatorImplRecord> builtin_records() {
    return OperatorImplTable::instance().all_records(std::string());
}

std::vector<OperatorImplRecord> strip_builtin_records(const std::vector<OperatorImplRecord>& records) {
    std::unordered_map<std::string, int32_t> builtin_counts;
    for (const auto& record : builtin_records()) {
        ++builtin_counts[record.to_json().dump()];
    }

    std::vector<OperatorImplRecord> stripped;
    stripped.reserve(records.size());
    for (const auto& record : records) {
        const std::string key = record.to_json().dump();
        auto it = builtin_counts.find(key);
        if (it != builtin_counts.end() && it->second > 0) {
            --it->second;
            continue;
        }
        stripped.push_back(record);
    }
    return stripped;
}

void write_operator_table(const std::filesystem::path& path, const std::vector<OperatorImplRecord>& records) {
    nlohmann::json payload = {
        {"schema", kTuningTableSchema},
        {"records", nlohmann::json::array()},
    };
    for (const auto& record : strip_builtin_records(records)) {
        payload["records"].push_back(record.to_json());
    }
    write_json_file(path, payload);
}

bool record_matches(
    const OperatorImplRecord& record,
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& op_kind,
    const std::string& layer_role,
    const std::string& stage,
    const std::string& shape_sig)
{
    return record.model_name == model_name &&
        record.hw_profile == hw_profile &&
        record.op_kind == op_kind &&
        record.layer_role == layer_role &&
        record.stage == stage &&
        record.shape_sig == shape_sig;
}

std::optional<nlohmann::json> find_impl_params(
    const std::vector<OperatorImplRecord>& records,
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& op_kind,
    const std::string& layer_role,
    const std::string& stage,
    const std::string& shape_sig)
{
    for (const auto& record : records) {
        if (record_matches(record, model_name, hw_profile, op_kind, layer_role, stage, shape_sig)) {
            return record.impl_params;
        }
    }
    return std::nullopt;
}

std::vector<OperatorImplRecord> upsert_attention_record(
    const std::vector<OperatorImplRecord>& records,
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& stage,
    const std::string& shape_sig,
    const std::string& impl_id,
    const nlohmann::json& impl_params)
{
    std::vector<OperatorImplRecord> updated;
    updated.reserve(records.size() + 1);
    for (const auto& record : records) {
        if (record_matches(record, model_name, hw_profile, "attention", std::string(), stage, shape_sig)) {
            continue;
        }
        updated.push_back(record);
    }

    if (!impl_params.empty()) {
        OperatorImplRecord record;
        record.model_name = model_name;
        record.hw_profile = hw_profile;
        record.op_kind = "attention";
        record.stage = stage;
        record.shape_sig = shape_sig;
        record.impl_id = impl_id;
        record.impl_params = impl_params;
        updated.push_back(std::move(record));
    }
    return updated;
}

std::vector<OperatorImplRecord> upsert_linear_record(
    const std::vector<OperatorImplRecord>& records,
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& layer_role,
    const std::string& stage,
    const std::string& shape_sig,
    const nlohmann::json& impl_params)
{
    std::vector<OperatorImplRecord> updated;
    updated.reserve(records.size() + 1);
    for (const auto& record : records) {
        if (record_matches(record, model_name, hw_profile, "linear", layer_role, stage, shape_sig)) {
            continue;
        }
        updated.push_back(record);
    }

    if (!impl_params.is_null()) {
        OperatorImplRecord record;
        record.model_name = model_name;
        record.hw_profile = hw_profile;
        record.op_kind = "linear";
        record.layer_role = layer_role;
        record.stage = stage;
        record.shape_sig = shape_sig;
        record.impl_id = "cublasLt";
        record.impl_params = impl_params;
        updated.push_back(std::move(record));
    }
    return updated;
}

std::vector<OperatorImplRecord> remove_generic_decode_attention_fallback(
    const std::vector<OperatorImplRecord>& records,
    const std::string& model_name,
    const std::string& hw_profile)
{
    std::vector<OperatorImplRecord> cleaned;
    cleaned.reserve(records.size());
    for (const auto& record : records) {
        if (record.model_name == model_name &&
            record.hw_profile == hw_profile &&
            record.op_kind == "attention" &&
            record.stage == "decode" &&
            record.shape_sig.empty() &&
            record.impl_id == "flashinfer_attention_decode_sm80_tuned")
        {
            continue;
        }
        cleaned.push_back(record);
    }
    return cleaned;
}

std::string attention_prefill_candidate_label(const nlohmann::json& impl_params) {
    if (impl_params.empty()) {
        return "baseline";
    }
    if (impl_params.contains("prefill_cta_tile_q")) {
        return "prefill_cta_tile_q=" + std::to_string(impl_params.value("prefill_cta_tile_q", 0));
    }
    return impl_params.dump();
}

std::string attention_decode_candidate_label(const nlohmann::json& impl_params) {
    if (impl_params.empty()) {
        return "baseline";
    }
    return impl_params.dump();
}

std::string linear_candidate_label(const nlohmann::json& impl_params) {
    if (impl_params.is_null()) {
        return "baseline";
    }
    if (impl_params.contains("algo_index")) {
        return "algo_index=" + std::to_string(impl_params.value("algo_index", -1));
    }
    return impl_params.dump();
}

std::vector<nlohmann::json> unique_json_candidates(const std::vector<nlohmann::json>& candidates) {
    std::vector<nlohmann::json> unique;
    std::unordered_map<std::string, bool> seen;
    for (const auto& candidate : candidates) {
        const std::string key = candidate.dump();
        if (seen.find(key) != seen.end()) {
            continue;
        }
        seen.emplace(key, true);
        unique.push_back(candidate);
    }
    return unique;
}

std::vector<nlohmann::json> attention_prefill_candidates(
    const std::optional<nlohmann::json>& existing,
    bool reduced_candidates)
{
    std::vector<nlohmann::json> candidates = {nlohmann::json::object()};
    if (existing.has_value()) {
        candidates.push_back(*existing);
    }
    candidates.push_back(nlohmann::json{{"prefill_cta_tile_q", 16}});
    candidates.push_back(nlohmann::json{{"prefill_cta_tile_q", 64}});
    if (!reduced_candidates) {
        candidates.push_back(nlohmann::json{{"prefill_cta_tile_q", 128}});
    }
    return unique_json_candidates(candidates);
}

std::vector<nlohmann::json> attention_decode_candidates(
    const std::optional<nlohmann::json>& existing,
    bool reduced_candidates)
{
    std::vector<nlohmann::json> candidates = {nlohmann::json::object()};
    if (existing.has_value()) {
        candidates.push_back(*existing);
    }

    const std::vector<nlohmann::json> families = reduced_candidates
        ? std::vector<nlohmann::json>{
              {
                  {"short_seq_bdz", 3},
                  {"long_seq_bdz", 4},
                  {"long_seq_threshold", 1536},
                  {"no_split_kv_threshold", 384},
                  {"min_chunk_size", 128},
                  {"chunk_alignment", 128},
                  {"chunk_candidates", nlohmann::json::array({128, 256, 512, 1024})},
              },
              {
                  {"short_seq_bdz", 4},
                  {"long_seq_bdz", 4},
                  {"long_seq_threshold", 1024},
                  {"no_split_kv_threshold", 256},
                  {"min_chunk_size", 64},
                  {"chunk_alignment", 64},
                  {"chunk_candidates", nlohmann::json::array({64, 128, 256, 512})},
              },
          }
        : std::vector<nlohmann::json>{
              {
                  {"short_seq_bdz", 3},
                  {"long_seq_bdz", 3},
                  {"long_seq_threshold", 1024},
                  {"no_split_kv_threshold", 192},
                  {"min_chunk_size", 64},
                  {"chunk_alignment", 64},
                  {"chunk_candidates", nlohmann::json::array({64, 128, 256, 512})},
              },
              {
                  {"short_seq_bdz", 3},
                  {"long_seq_bdz", 3},
                  {"long_seq_threshold", 1024},
                  {"no_split_kv_threshold", 256},
                  {"min_chunk_size", 128},
                  {"chunk_alignment", 128},
                  {"chunk_candidates", nlohmann::json::array({128, 256, 512, 1024})},
              },
              {
                  {"short_seq_bdz", 3},
                  {"long_seq_bdz", 4},
                  {"long_seq_threshold", 1536},
                  {"no_split_kv_threshold", 256},
                  {"min_chunk_size", 128},
                  {"chunk_alignment", 128},
                  {"chunk_candidates", nlohmann::json::array({128, 256, 512, 1024})},
              },
              {
                  {"short_seq_bdz", 4},
                  {"long_seq_bdz", 4},
                  {"long_seq_threshold", 1536},
                  {"no_split_kv_threshold", 384},
                  {"min_chunk_size", 128},
                  {"chunk_alignment", 128},
                  {"chunk_candidates", nlohmann::json::array({128, 256, 512, 1024})},
              },
              {
                  {"short_seq_bdz", 4},
                  {"long_seq_bdz", 4},
                  {"long_seq_threshold", 1024},
                  {"no_split_kv_threshold", 256},
                  {"min_chunk_size", 64},
                  {"chunk_alignment", 64},
                  {"chunk_candidates", nlohmann::json::array({64, 128, 256, 512})},
              },
          };

    candidates.insert(candidates.end(), families.begin(), families.end());
    return unique_json_candidates(candidates);
}

AttentionLayer* representative_attention_layer(Qwen2_5& model) {
    AttentionLayer* layer = model.attention_layer(0);
    if (layer == nullptr) {
        throw InternalError("Failed to resolve representative Qwen2.5 attention layer for tuning");
    }
    return layer;
}

LinearLayer* representative_linear_layer(Qwen2_5& model, const std::string& kind) {
    if (kind == "fused_qkv") {
        return model.linear_layer("layers.0.attn.qkv_fused");
    }
    if (kind == "attention_output") {
        return model.linear_layer("layers.0.attn.o_proj");
    }
    if (kind == "mlp_down") {
        return model.linear_layer("layers.0.mlp.down_proj");
    }
    if (kind == "fused_gate_up") {
        return model.linear_layer("layers.0.mlp.gate_up_fused");
    }
    if (kind == "lm_head") {
        return model.lm_head_layer();
    }
    throw ConfigurationError("Unsupported representative linear tuning kind: " + kind);
}

std::pair<std::vector<int64_t>, int32_t> linear_io_shape(
    const std::string& kind,
    int32_t m,
    const LinearDims& dims)
{
    if (kind == "mlp_down") {
        return {{m, dims.intermediate}, dims.hidden};
    }
    if (kind == "fused_qkv") {
        return {{m, dims.hidden}, dims.hidden + 2 * dims.kv};
    }
    if (kind == "attention_output") {
        return {{m, dims.hidden}, dims.hidden};
    }
    if (kind == "fused_gate_up") {
        return {{m, dims.hidden}, 2 * dims.intermediate};
    }
    if (kind == "lm_head") {
        return {{m, dims.hidden}, dims.vocab};
    }
    throw ConfigurationError("Unsupported linear IO shape kind: " + kind);
}

std::filesystem::path candidate_table_path(
    const std::filesystem::path& tuning_dir,
    const std::string& step_key)
{
    return tuning_dir / "candidates" / step_key / "operator_impl_table.json";
}

void activate_candidate_table(
    EngineConfig& config,
    Model& model,
    const std::filesystem::path& table_path,
    const std::vector<OperatorImplRecord>& records)
{
    write_operator_table(table_path, records);
    config.set_operator_impl_table_override(table_path.string());
    OperatorImplTable::instance().invalidate(table_path.string());
    model.reset_operator_impl_caches();
}

nlohmann::json benchmark_attention_prefill_candidate(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    const std::vector<OperatorImplRecord>& candidate_records,
    const std::string& step_key,
    const std::vector<int32_t>& seq_lens,
    int32_t warmup,
    int32_t iters)
{
    activate_candidate_table(config, model, candidate_table_path(tuning_dir, step_key), candidate_records);

    AttentionLayer* layer = representative_attention_layer(qwen);
    std::vector<nlohmann::json> seq_results;
    seq_results.reserve(seq_lens.size());
    double total_median_ms = 0.0;
    const DType dtype = qwen.dtype();
    const int32_t device_id = config.runtime_device_id();

    for (int32_t seq_len : seq_lens) {
        Tensor q = make_zero_gpu_tensor(
            {seq_len, qwen.num_attention_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor k = make_zero_gpu_tensor(
            {seq_len, qwen.num_kv_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor v = make_zero_gpu_tensor(
            {seq_len, qwen.num_kv_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor o = make_zero_gpu_tensor(
            {seq_len, qwen.num_attention_heads(), qwen.head_dim()},
            dtype,
            device_id);

        auto run = [&]() {
            layer->forward_prefill(q, k, v, o, true, nullptr);
        };
        run();
        CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to sync after prefill tuning dry-run");

        const double median_ms = median_cuda_ms(run, warmup, iters);
        total_median_ms += median_ms;
        seq_results.push_back({
            {"seq_len", seq_len},
            {"median_ms", median_ms},
        });
    }

    return {
        {"shape_sig", attention_shape_sig(qwen)},
        {"seq_results", seq_results},
        {"total_median_ms", total_median_ms},
        {"avg_median_ms", total_median_ms / static_cast<double>(seq_results.size())},
    };
}

nlohmann::json benchmark_attention_decode_candidate(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    const std::vector<OperatorImplRecord>& candidate_records,
    const std::string& step_key,
    const std::vector<int32_t>& kv_lens,
    int32_t warmup,
    int32_t iters)
{
    activate_candidate_table(config, model, candidate_table_path(tuning_dir, step_key), candidate_records);

    AttentionLayer* layer = representative_attention_layer(qwen);
    std::vector<nlohmann::json> kv_results;
    kv_results.reserve(kv_lens.size());
    double total_median_ms = 0.0;
    const DType dtype = qwen.dtype();
    const int32_t device_id = config.runtime_device_id();

    for (int32_t kv_len : kv_lens) {
        Tensor q = make_zero_gpu_tensor(
            {1, qwen.num_attention_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor k = make_zero_gpu_tensor(
            {kv_len, qwen.num_kv_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor v = make_zero_gpu_tensor(
            {kv_len, qwen.num_kv_heads(), qwen.head_dim()},
            dtype,
            device_id);
        Tensor o = make_zero_gpu_tensor(
            {1, qwen.num_attention_heads(), qwen.head_dim()},
            dtype,
            device_id);
        DeviceBuffer d_kv_len = make_uint32_device_value(static_cast<uint32_t>(kv_len), device_id);

        auto run = [&]() {
            layer->forward_decode(
                q, k, v, o, nullptr, static_cast<uint32_t*>(d_kv_len.ptr), static_cast<uint32_t>(kv_len));
        };
        run();
        CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to sync after decode tuning dry-run");

        const double median_ms = median_cuda_ms(run, warmup, iters);
        total_median_ms += median_ms;
        kv_results.push_back({
            {"kv_len", kv_len},
            {"median_ms", median_ms},
        });
    }

    return {
        {"shape_sig", attention_shape_sig(qwen)},
        {"kv_results", kv_results},
        {"total_median_ms", total_median_ms},
        {"avg_median_ms", total_median_ms / static_cast<double>(kv_results.size())},
    };
}

nlohmann::json benchmark_linear_candidate(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    const std::vector<OperatorImplRecord>& candidate_records,
    const std::string& step_key,
    const std::string& kind,
    ModelStage stage,
    int32_t m,
    int32_t warmup,
    int32_t iters)
{
    activate_candidate_table(config, model, candidate_table_path(tuning_dir, step_key), candidate_records);

    LinearLayer* layer = representative_linear_layer(qwen, kind);
    if (layer == nullptr) {
        throw InternalError("Failed to resolve representative linear layer for tuning: " + kind);
    }

    const LinearDims dims = linear_dims(qwen);
    const auto [input_shape, output_dim] = linear_io_shape(kind, m, dims);
    const DType dtype = qwen.dtype();
    const int32_t device_id = config.runtime_device_id();
    Tensor input = make_zero_gpu_tensor(input_shape, dtype, device_id);
    Tensor output = make_zero_gpu_tensor({input_shape[0], output_dim}, dtype, device_id);

    auto run = [&]() {
        layer->forward_fp16_bf16(input, output, nullptr, stage);
    };
    run();
    CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to sync after linear tuning dry-run");

    const nlohmann::json debug_info = layer->debug_cached_impl_info(stage, m);
    const double median_ms = median_cuda_ms(run, warmup, iters);

    return {
        {"shape_sig", linear_shape_sig(kind, m, dtype, dims)},
        {"median_ms", median_ms},
        {"debug", {
            {"selected_impl_id", debug_info.value("selected_impl_id", std::string())},
            {"selected_impl_params", debug_info.value("selected_impl_params", nlohmann::json::object())},
            {"heuristic_candidate_count", debug_info.value("heuristic_candidate_count", 0)},
            {"best_algo_index", debug_info.value("best_algo_index", -1)},
            {"workspace_bytes", debug_info.value("workspace_bytes", 0)},
            {"waves_count", debug_info.value("waves_count", 0.0)},
            {"selected_algo_config", debug_info.value("selected_algo_config", nlohmann::json::object())},
        }},
    };
}

nlohmann::json tune_attention_prefill_step(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    std::vector<OperatorImplRecord>& records,
    const TuningOptions& options,
    int32_t step_index,
    int32_t total_steps)
{
    const std::string model_name = config.resolved_model_name();
    const std::string hw_profile = config.resolved_hw_profile();
    const std::string shape_sig = attention_shape_sig(qwen);
    const std::optional<nlohmann::json> existing = find_impl_params(
        records, model_name, hw_profile, "attention", std::string(), "prefill", shape_sig);
    const std::vector<nlohmann::json> candidates = attention_prefill_candidates(existing, options.reduced_candidates);

    Logging::instance().log_info(
        "[tuning] step {}/{} attention prefill shape={} candidates={}",
        step_index,
        total_steps,
        shape_sig,
        candidates.size());

    std::vector<nlohmann::json> candidate_reports;
    candidate_reports.reserve(candidates.size());
    double baseline_ms = std::numeric_limits<double>::infinity();
    double best_ms = std::numeric_limits<double>::infinity();
    size_t best_index = 0;

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& impl_params = candidates[idx];
        Logging::instance().log_info(
            "[tuning] step {}/{} attention prefill candidate {}/{} {}",
            step_index,
            total_steps,
            idx + 1,
            candidates.size(),
            attention_prefill_candidate_label(impl_params));

        const auto candidate_records = upsert_attention_record(
            records,
            model_name,
            hw_profile,
            "prefill",
            shape_sig,
            "flashinfer_attention",
            impl_params);
        nlohmann::json report = benchmark_attention_prefill_candidate(
            config,
            model,
            qwen,
            tuning_dir,
            candidate_records,
            "attention_prefill_" + std::to_string(idx),
            options.prefill_list,
            options.attention_warmup,
            options.attention_iters);
        report["candidate_label"] = attention_prefill_candidate_label(impl_params);
        report["impl_params"] = impl_params;
        candidate_reports.push_back(report);

        const double total_median_ms = report.value("total_median_ms", std::numeric_limits<double>::infinity());
        if (idx == 0) {
            baseline_ms = total_median_ms;
        }
        if (total_median_ms < best_ms) {
            best_ms = total_median_ms;
            best_index = idx;
        }
    }

    records = upsert_attention_record(
        records,
        model_name,
        hw_profile,
        "prefill",
        shape_sig,
        "flashinfer_attention",
        candidate_reports[best_index]["impl_params"]);

    const double gain_pct = std::isfinite(baseline_ms) && baseline_ms > 0.0
        ? (baseline_ms - best_ms) / baseline_ms * 100.0
        : 0.0;
    Logging::instance().log_info(
        "[tuning] step {}/{} attention prefill best={} total_median_ms={:.6f} gain_vs_baseline={:.2f}%",
        step_index,
        total_steps,
        candidate_reports[best_index].value("candidate_label", std::string("baseline")),
        best_ms,
        gain_pct);

    return {
        {"step_index", step_index},
        {"op_kind", "attention"},
        {"stage", "prefill"},
        {"shape_sig", shape_sig},
        {"existing_impl_params", existing.has_value() ? *existing : nlohmann::json()},
        {"best", candidate_reports[best_index]},
        {"candidates", candidate_reports},
    };
}

nlohmann::json tune_attention_decode_step(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    std::vector<OperatorImplRecord>& records,
    const TuningOptions& options,
    int32_t step_index,
    int32_t total_steps)
{
    const std::string model_name = config.resolved_model_name();
    const std::string hw_profile = config.resolved_hw_profile();
    const std::string shape_sig = attention_shape_sig(qwen);
    const std::optional<nlohmann::json> existing = find_impl_params(
        records, model_name, hw_profile, "attention", std::string(), "decode", shape_sig);
    const std::vector<nlohmann::json> candidates = attention_decode_candidates(existing, options.reduced_candidates);

    Logging::instance().log_info(
        "[tuning] step {}/{} attention decode shape={} candidates={}",
        step_index,
        total_steps,
        shape_sig,
        candidates.size());

    std::vector<nlohmann::json> candidate_reports;
    candidate_reports.reserve(candidates.size());
    double baseline_ms = std::numeric_limits<double>::infinity();
    double best_ms = std::numeric_limits<double>::infinity();
    size_t best_index = 0;

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& impl_params = candidates[idx];
        Logging::instance().log_info(
            "[tuning] step {}/{} attention decode candidate {}/{} {}",
            step_index,
            total_steps,
            idx + 1,
            candidates.size(),
            attention_decode_candidate_label(impl_params));

        const auto candidate_records = upsert_attention_record(
            records,
            model_name,
            hw_profile,
            "decode",
            shape_sig,
            "flashinfer_attention_decode_sm80_tuned",
            impl_params);
        nlohmann::json report = benchmark_attention_decode_candidate(
            config,
            model,
            qwen,
            tuning_dir,
            candidate_records,
            "attention_decode_" + std::to_string(idx),
            options.kv_lens,
            options.attention_warmup,
            options.attention_iters);
        report["candidate_label"] = attention_decode_candidate_label(impl_params);
        report["impl_params"] = impl_params;
        candidate_reports.push_back(report);

        const double total_median_ms = report.value("total_median_ms", std::numeric_limits<double>::infinity());
        if (idx == 0) {
            baseline_ms = total_median_ms;
        }
        if (total_median_ms < best_ms) {
            best_ms = total_median_ms;
            best_index = idx;
        }
    }

    records = upsert_attention_record(
        records,
        model_name,
        hw_profile,
        "decode",
        shape_sig,
        "flashinfer_attention_decode_sm80_tuned",
        candidate_reports[best_index]["impl_params"]);
    records = remove_generic_decode_attention_fallback(records, model_name, hw_profile);

    const double gain_pct = std::isfinite(baseline_ms) && baseline_ms > 0.0
        ? (baseline_ms - best_ms) / baseline_ms * 100.0
        : 0.0;
    Logging::instance().log_info(
        "[tuning] step {}/{} attention decode best={} total_median_ms={:.6f} gain_vs_baseline={:.2f}%",
        step_index,
        total_steps,
        candidate_reports[best_index].value("candidate_label", std::string("baseline")),
        best_ms,
        gain_pct);

    return {
        {"step_index", step_index},
        {"op_kind", "attention"},
        {"stage", "decode"},
        {"shape_sig", shape_sig},
        {"existing_impl_params", existing.has_value() ? *existing : nlohmann::json()},
        {"best", candidate_reports[best_index]},
        {"candidates", candidate_reports},
    };
}

nlohmann::json tune_linear_step(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    std::vector<OperatorImplRecord>& records,
    const TuningOptions& options,
    const std::string& kind,
    ModelStage stage,
    int32_t m,
    int32_t step_index,
    int32_t total_steps)
{
    const std::string model_name = config.resolved_model_name();
    const std::string hw_profile = config.resolved_hw_profile();
    const std::string stage_key = stage == ModelStage::Decode ? "decode" : "prefill";
    const std::string layer_role = kind;
    const LinearDims dims = linear_dims(qwen);
    const std::string shape_sig = linear_shape_sig(kind, m, qwen.dtype(), dims);
    const std::optional<nlohmann::json> existing = find_impl_params(
        records, model_name, hw_profile, "linear", layer_role, stage_key, shape_sig);

    Logging::instance().log_info(
        "[tuning] step {}/{} linear kind={} stage={} shape={} m={}",
        step_index,
        total_steps,
        kind,
        stage_key,
        shape_sig,
        m);

    std::vector<nlohmann::json> candidate_reports;
    candidate_reports.reserve(8);

    const auto baseline_records = upsert_linear_record(
        records,
        model_name,
        hw_profile,
        layer_role,
        stage_key,
        shape_sig,
        nlohmann::json(nullptr));
    nlohmann::json baseline_report = benchmark_linear_candidate(
        config,
        model,
        qwen,
        tuning_dir,
        baseline_records,
        "linear_" + kind + "_" + stage_key + "_baseline",
        kind,
        stage,
        m,
        options.linear_warmup,
        options.linear_iters);
    baseline_report["candidate_label"] = "baseline";
    baseline_report["impl_params"] = nlohmann::json(nullptr);
    candidate_reports.push_back(baseline_report);

    int32_t heuristic_candidate_count = baseline_report["debug"].value("heuristic_candidate_count", 0);
    if (options.max_linear_algo_candidates > 0) {
        heuristic_candidate_count = std::min(heuristic_candidate_count, options.max_linear_algo_candidates);
    }

    for (int32_t algo_index = 0; algo_index < heuristic_candidate_count; ++algo_index) {
        Logging::instance().log_info(
            "[tuning] step {}/{} linear {} {} candidate {}/{} algo_index={}",
            step_index,
            total_steps,
            kind,
            stage_key,
            algo_index + 1,
            heuristic_candidate_count,
            algo_index);

        const nlohmann::json impl_params = {{"algo_index", algo_index}};
        const auto candidate_records = upsert_linear_record(
            records,
            model_name,
            hw_profile,
            layer_role,
            stage_key,
            shape_sig,
            impl_params);
        nlohmann::json report = benchmark_linear_candidate(
            config,
            model,
            qwen,
            tuning_dir,
            candidate_records,
            "linear_" + kind + "_" + stage_key + "_" + std::to_string(algo_index),
            kind,
            stage,
            m,
            options.linear_warmup,
            options.linear_iters);
        report["candidate_label"] = linear_candidate_label(impl_params);
        report["impl_params"] = impl_params;
        candidate_reports.push_back(report);
    }

    size_t best_index = 0;
    double best_ms = std::numeric_limits<double>::infinity();
    for (size_t idx = 0; idx < candidate_reports.size(); ++idx) {
        const double median_ms = candidate_reports[idx].value("median_ms", std::numeric_limits<double>::infinity());
        if (median_ms < best_ms) {
            best_ms = median_ms;
            best_index = idx;
        }
    }

    records = upsert_linear_record(
        records,
        model_name,
        hw_profile,
        layer_role,
        stage_key,
        shape_sig,
        candidate_reports[best_index]["impl_params"]);

    const double baseline_ms = candidate_reports.front().value("median_ms", std::numeric_limits<double>::infinity());
    const double gain_pct = std::isfinite(baseline_ms) && baseline_ms > 0.0
        ? (baseline_ms - best_ms) / baseline_ms * 100.0
        : 0.0;
    Logging::instance().log_info(
        "[tuning] step {}/{} linear kind={} stage={} best={} median_ms={:.6f} gain_vs_baseline={:.2f}%",
        step_index,
        total_steps,
        kind,
        stage_key,
        candidate_reports[best_index].value("candidate_label", std::string("baseline")),
        best_ms,
        gain_pct);

    return {
        {"step_index", step_index},
        {"op_kind", "linear"},
        {"layer_kind", kind},
        {"stage", stage_key},
        {"m", m},
        {"shape_sig", shape_sig},
        {"existing_impl_params", existing.has_value() ? *existing : nlohmann::json()},
        {"best", candidate_reports[best_index]},
        {"candidates", candidate_reports},
    };
}

std::vector<nlohmann::json> run_tuning_steps(
    EngineConfig& config,
    Model& model,
    Qwen2_5& qwen,
    const std::filesystem::path& tuning_dir,
    std::vector<OperatorImplRecord>& records,
    const TuningOptions& options)
{
    const int32_t total_steps = static_cast<int32_t>(2 + 5 + 4 * options.prefill_list.size());
    std::vector<nlohmann::json> steps;
    steps.reserve(static_cast<size_t>(total_steps));

    int32_t step_index = 1;
    steps.push_back(tune_attention_prefill_step(config, model, qwen, tuning_dir, records, options, step_index++, total_steps));
    steps.push_back(tune_attention_decode_step(config, model, qwen, tuning_dir, records, options, step_index++, total_steps));

    const std::vector<std::string> decode_kinds = {
        "fused_qkv",
        "attention_output",
        "mlp_down",
        "fused_gate_up",
        "lm_head",
    };
    for (const auto& kind : decode_kinds) {
        steps.push_back(tune_linear_step(
            config, model, qwen, tuning_dir, records, options, kind, ModelStage::Decode, 1, step_index++, total_steps));
    }

    const std::vector<std::string> prefill_kinds = {
        "fused_qkv",
        "attention_output",
        "mlp_down",
        "fused_gate_up",
    };
    for (int32_t m : options.prefill_list) {
        for (const auto& kind : prefill_kinds) {
            steps.push_back(tune_linear_step(
                config, model, qwen, tuning_dir, records, options, kind, ModelStage::Prefill, m, step_index++, total_steps));
        }
    }

    return steps;
}

nlohmann::json tuning_options_to_json(const TuningOptions& options) {
    return {
        {"prefill_list", options.prefill_list},
        {"kv_lens", options.kv_lens},
        {"attention_warmup", options.attention_warmup},
        {"attention_iters", options.attention_iters},
        {"linear_warmup", options.linear_warmup},
        {"linear_iters", options.linear_iters},
        {"reduced_candidates", options.reduced_candidates},
        {"max_linear_algo_candidates", options.max_linear_algo_candidates},
    };
}

} // namespace

CudaOperatorTuningResult tune_cuda_operator_table(EngineConfig& config, Model& model) {
    if (config.backend_target() != "cuda") {
        throw ConfigurationError("CUDA operator tuning requires backend_target=cuda");
    }

    auto* qwen = dynamic_cast<Qwen2_5*>(&model);
    if (qwen == nullptr) {
        throw ConfigurationError("CUDA operator tuning currently supports Qwen2.5 / Qwen2.5-VL only");
    }

    const std::string current_path = config.operator_impl_table_path();
    if (!current_path.empty()) {
        const std::filesystem::path current_table_path(current_path);
        if (is_cuda_tuning_table_path(current_table_path) &&
            std::filesystem::exists(current_table_path) &&
            is_valid_cached_report(current_table_path.parent_path() / "tuning_report.json"))
        {
            Logging::instance().log_info(
                "[tuning] reusing active tuned operator table {}",
                current_table_path.string());
            return {
                current_table_path.parent_path(),
                current_table_path,
                current_table_path.parent_path() / "tuning_report.json",
                true,
            };
        }
    }

    const std::filesystem::path tuning_dir =
        backend_cache_root() / config.backend_cache_key() / "cuda_operator_tuning";
    const std::filesystem::path table_path = tuning_dir / "operator_impl_table.json";
    const std::filesystem::path report_path = tuning_dir / "tuning_report.json";
    if (std::filesystem::exists(table_path) && is_valid_cached_report(report_path)) {
        Logging::instance().log_info("[tuning] cache hit: {}", table_path.string());
        return {tuning_dir, table_path, report_path, true};
    }

    OverridePathGuard override_guard(config);

    const TuningOptions options = load_tuning_options();
    std::vector<OperatorImplRecord> records = OperatorImplTable::instance().all_records(config.operator_impl_table_path());

    const auto start_time = std::chrono::steady_clock::now();
    const int32_t total_steps = static_cast<int32_t>(2 + 5 + 4 * options.prefill_list.size());
    Logging::instance().log_info(
        "[tuning] start model={} hw_profile={} cache_dir={} total_steps={}",
        config.resolved_model_name(),
        config.resolved_hw_profile(),
        tuning_dir.string(),
        total_steps);

    const std::vector<nlohmann::json> steps =
        run_tuning_steps(config, model, *qwen, tuning_dir, records, options);

    write_operator_table(table_path, records);
    OperatorImplTable::instance().invalidate(table_path.string());

    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

    nlohmann::json report = {
        {"schema", kTuningReportSchema},
        {"tuning_version", kTuningVersion},
        {"model_name", config.resolved_model_name()},
        {"hw_profile", config.resolved_hw_profile()},
        {"prefill_model_path", config.prefill_model_path()},
        {"configured_operator_impl_table_path", config.configured_operator_impl_table_path()},
        {"tuning_dir", tuning_dir.string()},
        {"operator_table_path", table_path.string()},
        {"options", tuning_options_to_json(options)},
        {"steps", steps},
        {"elapsed_ms", elapsed_ms},
    };
    write_json_file(report_path, report);

    Logging::instance().log_info(
        "[tuning] completed table={} report={} elapsed_ms={:.2f}",
        table_path.string(),
        report_path.string(),
        elapsed_ms);

    return {tuning_dir, table_path, report_path, false};
}

} // namespace edge_fm
