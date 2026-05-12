#include "models/qwen2_5/trt_linear_bridge.h"
#include "utils/logging.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(EDGE_FM_ENABLE_TRT_MLP_BRIDGE) && EDGE_FM_ENABLE_TRT_MLP_BRIDGE
#include <NvInfer.h>
#endif

namespace edge_fm {
namespace {

bool env_flag_enabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

} // namespace

#if defined(EDGE_FM_ENABLE_TRT_MLP_BRIDGE) && EDGE_FM_ENABLE_TRT_MLP_BRIDGE
namespace {

enum class LinearRole {
    Qkv,
    OProj,
};

enum class Fp16WeightMode {
    None,
    Qkv,
    OProj,
    Both,
};

const char* role_name(LinearRole role)
{
    return role == LinearRole::Qkv ? "qkv" : "oproj";
}

const char* fp16_weight_mode_name(Fp16WeightMode mode)
{
    switch (mode) {
        case Fp16WeightMode::Qkv:
            return "qkv";
        case Fp16WeightMode::OProj:
            return "oproj";
        case Fp16WeightMode::Both:
            return "both";
        case Fp16WeightMode::None:
        default:
            return "none";
    }
}

bool parse_role(const std::string& role, LinearRole* out)
{
    const std::string value = to_lower(role);
    if (value == "qkv" || value == "fused_qkv") {
        *out = LinearRole::Qkv;
        return true;
    }
    if (value == "oproj" || value == "o_proj" || value == "attention_output") {
        *out = LinearRole::OProj;
        return true;
    }
    return false;
}

bool role_list_enables(LinearRole role)
{
    const char* raw = std::getenv("EDGE_FM_TRT_LINEAR_ROLES");
    if (raw == nullptr || *raw == '\0') {
        return true;
    }
    std::string value = to_lower(raw);
    if (value == "both" || value == "all") {
        return true;
    }

    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(
            std::remove_if(token.begin(), token.end(),
                           [](unsigned char c) { return std::isspace(c) != 0; }),
            token.end());
        LinearRole parsed = LinearRole::Qkv;
        if (parse_role(token, &parsed) && parsed == role) {
            return true;
        }
    }
    return false;
}

Fp16WeightMode parse_fp16_weight_mode()
{
    const char* raw = std::getenv("EDGE_FM_TRT_LINEAR_FP16_WEIGHTS");
    if (raw == nullptr || *raw == '\0') {
        return Fp16WeightMode::None;
    }
    const std::string value = to_lower(raw);
    if (value == "qkv" || value == "fused_qkv") {
        return Fp16WeightMode::Qkv;
    }
    if (value == "oproj" || value == "o_proj" || value == "attention_output") {
        return Fp16WeightMode::OProj;
    }
    if (value == "both" || value == "all") {
        return Fp16WeightMode::Both;
    }
    return Fp16WeightMode::None;
}

bool uses_fp16_weight(Fp16WeightMode mode, LinearRole role)
{
    return mode == Fp16WeightMode::Both ||
        (mode == Fp16WeightMode::Qkv && role == LinearRole::Qkv) ||
        (mode == Fp16WeightMode::OProj && role == LinearRole::OProj);
}

std::string shape_key(int64_t m, int64_t in_features, int64_t out_features)
{
    return "m" + std::to_string(m) +
        "_k" + std::to_string(in_features) +
        "_n" + std::to_string(out_features);
}

std::string matrix_key(int64_t rows, int64_t cols)
{
    return "r" + std::to_string(rows) + "_c" + std::to_string(cols);
}

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            Logging::instance().log_warn("TensorRT linear bridge: {}", msg == nullptr ? "" : msg);
        }
    }
};

std::vector<char> read_binary_file(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return {};
    }
    const std::streamsize size = in.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<char> bytes(static_cast<size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!in.read(bytes.data(), size)) {
        return {};
    }
    return bytes;
}

bool dims_equal(const nvinfer1::Dims& dims, std::initializer_list<int64_t> expected)
{
    if (dims.nbDims != static_cast<int32_t>(expected.size())) {
        return false;
    }
    int32_t i = 0;
    for (int64_t value : expected) {
        if (dims.d[i++] != value) {
            return false;
        }
    }
    return true;
}

bool is_gpu_dtype_2d(const Tensor& tensor, DType dtype)
{
    const auto [device, _device_id] = tensor.device();
    (void)_device_id;
    return device == Device::GPU &&
        tensor.dtype() == dtype &&
        tensor.shape().size() == 2 &&
        tensor.data_ptr() != nullptr;
}

bool is_gpu_dtype_1d(const Tensor& tensor, DType dtype)
{
    const auto [device, _device_id] = tensor.device();
    (void)_device_id;
    return device == Device::GPU &&
        tensor.dtype() == dtype &&
        tensor.shape().size() == 1 &&
        tensor.data_ptr() != nullptr;
}

bool engine_has_tensor(const nvinfer1::ICudaEngine& engine, const char* name)
{
    const int32_t count = engine.getNbIOTensors();
    for (int32_t i = 0; i < count; ++i) {
        const char* tensor_name = engine.getIOTensorName(i);
        if (tensor_name != nullptr && std::string(tensor_name) == name) {
            return true;
        }
    }
    return false;
}

bool engine_io_matches(
    const nvinfer1::ICudaEngine& engine,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    bool fp16_weight,
    bool has_bias)
{
    using nvinfer1::DataType;
    if (engine_has_tensor(engine, "bias") != has_bias) {
        return false;
    }
    if (has_bias &&
        (engine.getTensorDataType("bias") != DataType::kBF16 ||
         !dims_equal(engine.getTensorShape("bias"), {out_features}))) {
        return false;
    }
    return engine.getTensorDataType("input") == DataType::kBF16 &&
        engine.getTensorDataType("weight") == (fp16_weight ? DataType::kHALF : DataType::kBF16) &&
        engine.getTensorDataType("output") == DataType::kBF16 &&
        dims_equal(engine.getTensorShape("input"), {m, in_features}) &&
        dims_equal(engine.getTensorShape("weight"), {out_features, in_features}) &&
        dims_equal(engine.getTensorShape("output"), {m, out_features});
}

std::string generic_engine_name(
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    bool fp16_weight,
    bool has_bias)
{
    return "trt_linear_edgefm_bf16_fp16compute_weight-" +
        std::string(fp16_weight ? "fp16" : "bf16") +
        (has_bias ? "_bias-bf16" : "") + "_" +
        shape_key(m, in_features, out_features) + ".engine";
}

std::filesystem::path find_engine_file(
    const std::filesystem::path& dir,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    bool fp16_weight,
    bool has_bias)
{
    const std::filesystem::path generic = dir / generic_engine_name(
        m, in_features, out_features, fp16_weight, has_bias);
    if (std::filesystem::exists(generic)) {
        return generic;
    }

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return {};
    }

    const std::string suffix = "_" + shape_key(m, in_features, out_features) + ".engine";
    const std::string dtype_token = fp16_weight ? "weight-fp16" : "weight-bf16";
    const std::string bias_token = "bias-bf16";
    std::vector<std::filesystem::path> matches;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("trt_linear") == std::string::npos ||
            name.find(dtype_token) == std::string::npos) {
            continue;
        }
        const bool name_has_bias = name.find(bias_token) != std::string::npos;
        if (name_has_bias != has_bias) {
            continue;
        }
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            matches.push_back(entry.path());
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches.empty() ? std::filesystem::path() : matches.front();
}

struct EngineBundle {
    std::filesystem::path path;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

struct CastBundle {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

struct PersistentWeightCopy {
    Tensor tensor;
    const void* source_ptr = nullptr;
};

} // namespace

struct TrtPrefillLinearBridge::Impl {
    explicit Impl(const EngineConfig& config)
        : enabled(env_flag_enabled("EDGE_FM_PREFILL_TRT_LINEAR")),
          fp16_weight_mode(parse_fp16_weight_mode()),
          device_id(config.runtime_device_id())
    {
        const char* raw_dir = std::getenv("EDGE_FM_TRT_LINEAR_ENGINE_DIR");
        if (raw_dir == nullptr || *raw_dir == '\0') {
            raw_dir = std::getenv("EDGE_FM_TRT_MLP_ENGINE_DIR");
        }
        if (raw_dir != nullptr && *raw_dir != '\0') {
            engine_dir = std::filesystem::path(raw_dir);
        }
        if (enabled && fp16_weight_mode != Fp16WeightMode::None) {
            Logging::instance().log_warn(
                "TensorRT linear bridge FP16 persistent weight mode '{}' is experimental/default-off; using native fallback on missing engines or copy failure",
                fp16_weight_mode_name(fp16_weight_mode));
        }
    }

    bool try_forward(
        const std::string& role_name_value,
        int32_t layer_id,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output,
        cudaStream_t stream)
    {
        if (!enabled) {
            return false;
        }

        LinearRole role = LinearRole::Qkv;
        if (!parse_role(role_name_value, &role) || !role_list_enables(role)) {
            return false;
        }
        if (engine_dir.empty()) {
            log_once("missing_engine_dir",
                     "EDGE_FM_PREFILL_TRT_LINEAR=1 but no TensorRT linear engine dir is set; using native linear path");
            return false;
        }
        if (!is_gpu_dtype_2d(input, DType::BFloat16) ||
            !is_gpu_dtype_2d(weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(output, DType::BFloat16)) {
            return false;
        }
        if (bias != nullptr && !is_gpu_dtype_1d(*bias, DType::BFloat16)) {
            return false;
        }

        const auto& input_shape = input.shape();
        const auto& weight_shape = weight.shape();
        const auto& output_shape = output.shape();
        const int64_t m = input_shape[0];
        const int64_t in_features = input_shape[1];
        if (m <= 0 || in_features <= 0 ||
            weight_shape[1] != in_features ||
            output_shape[0] != m ||
            output_shape[1] != weight_shape[0]) {
            return false;
        }
        const int64_t out_features = weight_shape[0];
        const bool has_bias = bias != nullptr;
        if (has_bias && bias->shape()[0] != out_features) {
            return false;
        }
        const bool fp16_weight = uses_fp16_weight(fp16_weight_mode, role);

        EngineBundle* bundle = get_bundle(role, m, in_features, out_features, fp16_weight, has_bias);
        if (bundle == nullptr || !ensure_context(*bundle)) {
            return false;
        }

        const Tensor* weight_bind = &weight;
        if (fp16_weight) {
            weight_bind = get_or_create_fp16_weight_copy(
                role, layer_id, weight, stream);
            if (weight_bind == nullptr) {
                return false;
            }
        }

        nvinfer1::IExecutionContext& context = *bundle->context;
        if (!context.setTensorAddress("input", input.data_ptr()) ||
            !context.setTensorAddress("weight", weight_bind->data_ptr()) ||
            !context.setTensorAddress("output", output.data_ptr())) {
            log_once("set_tensor_address_failed_" + bundle_key(role, m, in_features, out_features, fp16_weight, has_bias),
                     "TensorRT linear bridge failed to bind {} tensors for {}; using native linear path",
                     role_name(role),
                     shape_key(m, in_features, out_features));
            return false;
        }
        if (has_bias && !context.setTensorAddress("bias", bias->data_ptr())) {
            log_once("set_bias_address_failed_" + bundle_key(role, m, in_features, out_features, fp16_weight, has_bias),
                     "TensorRT linear bridge failed to bind {} bias tensor for {}; using native linear path",
                     role_name(role),
                     shape_key(m, in_features, out_features));
            return false;
        }
        if (!context.enqueueV3(stream)) {
            log_once("enqueue_failed_" + bundle_key(role, m, in_features, out_features, fp16_weight, has_bias),
                     "TensorRT linear bridge enqueueV3 failed for {} {}; using native linear path",
                     role_name(role),
                     shape_key(m, in_features, out_features));
            return false;
        }
        return true;
    }

    void reset_runtime_caches()
    {
        persistent_weight_copies.clear();
        cast_bundles.clear();
    }

    template <typename... Args>
    void log_once(const std::string& key, const char* fmt, Args&&... args)
    {
        if (logged_messages.insert(key).second) {
            Logging::instance().log_warn(fmt, std::forward<Args>(args)...);
        }
    }

    std::string bundle_key(
        LinearRole role,
        int64_t m,
        int64_t in_features,
        int64_t out_features,
        bool fp16_weight,
        bool has_bias) const
    {
        return std::string(role_name(role)) + "_" +
            (fp16_weight ? "fp16" : "bf16") + "_" +
            (has_bias ? "bias_" : "nobias_") +
            shape_key(m, in_features, out_features);
    }

    EngineBundle* get_bundle(
        LinearRole role,
        int64_t m,
        int64_t in_features,
        int64_t out_features,
        bool fp16_weight,
        bool has_bias)
    {
        const std::string key = bundle_key(role, m, in_features, out_features, fp16_weight, has_bias);
        if (missing_shapes.count(key) != 0) {
            return nullptr;
        }
        auto it = bundles.find(key);
        if (it != bundles.end()) {
            return it->second.get();
        }

        const std::filesystem::path engine_path = find_engine_file(
            engine_dir, m, in_features, out_features, fp16_weight, has_bias);
        if (engine_path.empty()) {
            missing_shapes.insert(key);
            log_once("missing_engine_" + key,
                     "TensorRT linear bridge engine for {} {} weight={} bias={} not found under {}; using native linear path",
                     role_name(role),
                     shape_key(m, in_features, out_features),
                     fp16_weight ? "fp16" : "bf16",
                     has_bias ? "bf16" : "none",
                     engine_dir.string());
            return nullptr;
        }

        std::vector<char> serialized = read_binary_file(engine_path);
        if (serialized.empty()) {
            missing_shapes.insert(key);
            log_once("read_engine_failed_" + key,
                     "TensorRT linear bridge failed to read {}; using native linear path",
                     engine_path.string());
            return nullptr;
        }

        auto bundle = std::make_unique<EngineBundle>();
        bundle->path = engine_path;
        bundle->runtime.reset(nvinfer1::createInferRuntime(logger));
        if (bundle->runtime == nullptr) {
            missing_shapes.insert(key);
            log_once("create_runtime_failed_" + key,
                     "TensorRT linear bridge createInferRuntime failed; using native linear path");
            return nullptr;
        }
        bundle->engine.reset(bundle->runtime->deserializeCudaEngine(
            serialized.data(), serialized.size()));
        if (bundle->engine == nullptr ||
            !engine_io_matches(*bundle->engine, m, in_features, out_features, fp16_weight, has_bias)) {
            missing_shapes.insert(key);
            log_once("engine_mismatch_" + key,
                     "TensorRT linear bridge engine {} does not match expected EdgeFM-layout IO for {} weight={} bias={}; using native linear path",
                     engine_path.string(),
                     shape_key(m, in_features, out_features),
                     fp16_weight ? "fp16" : "bf16",
                     has_bias ? "bf16" : "none");
            return nullptr;
        }

        Logging::instance().log_info(
            "TensorRT linear bridge loaded {} for {} {} weight={} bias={}",
            engine_path.string(),
            role_name(role),
            shape_key(m, in_features, out_features),
            fp16_weight ? "fp16" : "bf16",
            has_bias ? "bf16" : "none");
        EngineBundle* raw = bundle.get();
        bundles.emplace(key, std::move(bundle));
        return raw;
    }

    bool ensure_context(EngineBundle& bundle)
    {
        if (bundle.context != nullptr) {
            return true;
        }
        std::unique_ptr<nvinfer1::IExecutionContext> context(
            bundle.engine->createExecutionContext());
        if (context == nullptr) {
            log_once("create_context_failed_" + bundle.path.string(),
                     "TensorRT linear bridge failed to create execution context for {}; using native linear path",
                     bundle.path.string());
            return false;
        }
        bundle.context = std::move(context);
        return true;
    }

    CastBundle* get_cast_bundle(int64_t rows, int64_t cols)
    {
        const std::string key = matrix_key(rows, cols);
        auto it = cast_bundles.find(key);
        if (it != cast_bundles.end()) {
            return it->second.get();
        }

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(logger));
        if (builder == nullptr) {
            log_once("cast_builder_failed_" + key,
                     "TensorRT linear bridge failed to create BF16->FP16 cast builder for {}; using native linear path",
                     key);
            return nullptr;
        }
        const auto flags = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(flags));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        if (network == nullptr || config == nullptr) {
            log_once("cast_network_failed_" + key,
                     "TensorRT linear bridge failed to create BF16->FP16 cast network for {}; using native linear path",
                     key);
            return nullptr;
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64ULL * 1024ULL * 1024ULL);

        nvinfer1::Dims dims{};
        dims.nbDims = 2;
        dims.d[0] = rows;
        dims.d[1] = cols;
        nvinfer1::ITensor* input = network->addInput("input", nvinfer1::DataType::kBF16, dims);
        if (input == nullptr) {
            log_once("cast_input_failed_" + key,
                     "TensorRT linear bridge failed to add BF16->FP16 cast input for {}; using native linear path",
                     key);
            return nullptr;
        }
        nvinfer1::ICastLayer* cast = network->addCast(*input, nvinfer1::DataType::kHALF);
        if (cast == nullptr || cast->getOutput(0) == nullptr) {
            log_once("cast_layer_failed_" + key,
                     "TensorRT linear bridge failed to add BF16->FP16 cast layer for {}; using native linear path",
                     key);
            return nullptr;
        }
        nvinfer1::ITensor* output = cast->getOutput(0);
        output->setName("output");
        network->markOutput(*output);

        auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (serialized == nullptr || serialized->data() == nullptr || serialized->size() == 0) {
            log_once("cast_build_failed_" + key,
                     "TensorRT linear bridge failed to build BF16->FP16 cast engine for {}; using native linear path",
                     key);
            return nullptr;
        }

        auto bundle = std::make_unique<CastBundle>();
        bundle->runtime.reset(nvinfer1::createInferRuntime(logger));
        if (bundle->runtime == nullptr) {
            log_once("cast_runtime_failed_" + key,
                     "TensorRT linear bridge failed to create BF16->FP16 cast runtime for {}; using native linear path",
                     key);
            return nullptr;
        }
        bundle->engine.reset(bundle->runtime->deserializeCudaEngine(
            serialized->data(), serialized->size()));
        if (bundle->engine == nullptr ||
            bundle->engine->getTensorDataType("input") != nvinfer1::DataType::kBF16 ||
            bundle->engine->getTensorDataType("output") != nvinfer1::DataType::kHALF ||
            !dims_equal(bundle->engine->getTensorShape("input"), {rows, cols}) ||
            !dims_equal(bundle->engine->getTensorShape("output"), {rows, cols})) {
            log_once("cast_engine_mismatch_" + key,
                     "TensorRT linear bridge BF16->FP16 cast engine IO mismatch for {}; using native linear path",
                     key);
            return nullptr;
        }
        bundle->context.reset(bundle->engine->createExecutionContext());
        if (bundle->context == nullptr) {
            log_once("cast_context_failed_" + key,
                     "TensorRT linear bridge failed to create BF16->FP16 cast context for {}; using native linear path",
                     key);
            return nullptr;
        }

        Logging::instance().log_info(
            "TensorRT linear bridge built BF16->FP16 persistent weight cast engine for {}",
            key);
        CastBundle* raw = bundle.get();
        cast_bundles.emplace(key, std::move(bundle));
        return raw;
    }

    bool run_bf16_to_fp16_cast(const Tensor& source, Tensor& destination, cudaStream_t stream)
    {
        const auto& shape = source.shape();
        if (shape.size() != 2 || destination.shape() != shape ||
            !is_gpu_dtype_2d(source, DType::BFloat16) ||
            !is_gpu_dtype_2d(destination, DType::Float16)) {
            return false;
        }
        CastBundle* bundle = get_cast_bundle(shape[0], shape[1]);
        if (bundle == nullptr || bundle->context == nullptr) {
            return false;
        }
        if (!bundle->context->setTensorAddress("input", source.data_ptr()) ||
            !bundle->context->setTensorAddress("output", destination.data_ptr())) {
            log_once("cast_bind_failed_" + matrix_key(shape[0], shape[1]),
                     "TensorRT linear bridge failed to bind BF16->FP16 cast tensors for {}; using native linear path",
                     matrix_key(shape[0], shape[1]));
            return false;
        }
        if (!bundle->context->enqueueV3(stream)) {
            log_once("cast_enqueue_failed_" + matrix_key(shape[0], shape[1]),
                     "TensorRT linear bridge BF16->FP16 cast enqueue failed for {}; using native linear path",
                     matrix_key(shape[0], shape[1]));
            return false;
        }
        return true;
    }

    const Tensor* get_or_create_fp16_weight_copy(
        LinearRole role,
        int32_t layer_id,
        const Tensor& source,
        cudaStream_t stream)
    {
        const std::string key = std::string("layer") + std::to_string(layer_id) +
            "_" + role_name(role);
        auto existing = persistent_weight_copies.find(key);
        if (existing != persistent_weight_copies.end() &&
            existing->second.source_ptr == source.data_ptr()) {
            return &existing->second.tensor;
        }
        persistent_weight_copies.erase(key);

        const auto& shape = source.shape();
        if (shape.size() != 2) {
            return nullptr;
        }
        const size_t bytes = static_cast<size_t>(shape[0]) *
            static_cast<size_t>(shape[1]) * get_dtype_size(DType::Float16);
        void* data = nullptr;
        const cudaError_t alloc_status = cudaMalloc(&data, bytes);
        if (alloc_status != cudaSuccess) {
            log_once("fp16_weight_alloc_failed_" + std::string(role_name(role)),
                     "TensorRT linear bridge failed to allocate persistent FP16 {} weight copy ({} bytes): {}; using native linear path",
                     role_name(role),
                     bytes,
                     cudaGetErrorString(alloc_status));
            return nullptr;
        }

        Tensor fp16_tensor = Tensor::adopt(
            data,
            shape,
            DType::Float16,
            Device::GPU,
            device_id,
            MemoryOwnership::OwnCudaMalloc);
        if (!run_bf16_to_fp16_cast(source, fp16_tensor, stream)) {
            return nullptr;
        }

        PersistentWeightCopy copy;
        copy.tensor = std::move(fp16_tensor);
        copy.source_ptr = source.data_ptr();
        auto [inserted, _ok] = persistent_weight_copies.emplace(key, std::move(copy));
        (void)_ok;
        Logging::instance().log_info(
            "TensorRT linear bridge created persistent FP16 {} weight copy for layer {} ({} bytes)",
            role_name(role),
            layer_id,
            bytes);
        return &inserted->second.tensor;
    }

    bool enabled = false;
    Fp16WeightMode fp16_weight_mode = Fp16WeightMode::None;
    int32_t device_id = 0;
    std::filesystem::path engine_dir;
    TrtLogger logger;
    std::unordered_map<std::string, std::unique_ptr<EngineBundle>> bundles;
    std::unordered_map<std::string, std::unique_ptr<CastBundle>> cast_bundles;
    std::unordered_map<std::string, PersistentWeightCopy> persistent_weight_copies;
    std::unordered_set<std::string> missing_shapes;
    std::unordered_set<std::string> logged_messages;
};

#else

struct TrtPrefillLinearBridge::Impl {
    explicit Impl(const EngineConfig& config)
    {
        (void)config;
        if (env_flag_enabled("EDGE_FM_PREFILL_TRT_LINEAR")) {
            Logging::instance().log_warn(
                "EDGE_FM_PREFILL_TRT_LINEAR=1 but this build was configured without BUILD_TRT_MLP_BRIDGE=ON; using native linear path");
        }
    }

    bool try_forward(
        const std::string& role,
        int32_t layer_id,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output,
        cudaStream_t stream)
    {
        (void)role;
        (void)layer_id;
        (void)input;
        (void)weight;
        (void)bias;
        (void)output;
        (void)stream;
        return false;
    }

    void reset_runtime_caches() {}
};

#endif

TrtPrefillLinearBridge::TrtPrefillLinearBridge(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

TrtPrefillLinearBridge::~TrtPrefillLinearBridge() = default;

bool TrtPrefillLinearBridge::try_forward(
    const std::string& role,
    int32_t layer_id,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    cudaStream_t stream)
{
    return impl_->try_forward(role, layer_id, input, weight, bias, output, stream);
}

void TrtPrefillLinearBridge::reset_runtime_caches()
{
    impl_->reset_runtime_caches();
}

} // namespace edge_fm
