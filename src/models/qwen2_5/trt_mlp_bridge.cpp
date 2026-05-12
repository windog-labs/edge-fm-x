#include "models/qwen2_5/trt_mlp_bridge.h"
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

} // namespace

#if defined(EDGE_FM_ENABLE_TRT_MLP_BRIDGE) && EDGE_FM_ENABLE_TRT_MLP_BRIDGE
namespace {

std::string shape_key(int64_t m, int64_t hidden, int64_t intermediate)
{
    return "m" + std::to_string(m) +
        "_h" + std::to_string(hidden) +
        "_i" + std::to_string(intermediate);
}

std::string matrix_key(int64_t rows, int64_t cols)
{
    return "r" + std::to_string(rows) + "_c" + std::to_string(cols);
}

enum class Fp16WeightMode {
    None,
    GateUp,
    Down,
    Both,
};

const char* fp16_weight_mode_name(Fp16WeightMode mode)
{
    switch (mode) {
        case Fp16WeightMode::GateUp:
            return "gateup";
        case Fp16WeightMode::Down:
            return "down";
        case Fp16WeightMode::Both:
            return "both";
        case Fp16WeightMode::None:
        default:
            return "none";
    }
}

Fp16WeightMode parse_fp16_weight_mode()
{
    const char* raw = std::getenv("EDGE_FM_TRT_MLP_FP16_WEIGHTS");
    if (raw == nullptr || *raw == '\0') {
        return Fp16WeightMode::None;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "gateup" || value == "gate_up" || value == "gate-up") {
        return Fp16WeightMode::GateUp;
    }
    if (value == "down" || value == "downproj" || value == "down_proj" || value == "down-proj") {
        return Fp16WeightMode::Down;
    }
    if (value == "both" || value == "all") {
        return Fp16WeightMode::Both;
    }
    return Fp16WeightMode::None;
}

bool uses_fp16_gate_up(Fp16WeightMode mode)
{
    return mode == Fp16WeightMode::GateUp || mode == Fp16WeightMode::Both;
}

bool uses_fp16_down(Fp16WeightMode mode)
{
    return mode == Fp16WeightMode::Down || mode == Fp16WeightMode::Both;
}

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            Logging::instance().log_warn("TensorRT MLP bridge: {}", msg == nullptr ? "" : msg);
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

bool engine_io_matches(
    const nvinfer1::ICudaEngine& engine,
    int64_t m,
    int64_t hidden,
    int64_t intermediate,
    Fp16WeightMode fp16_weight_mode)
{
    using nvinfer1::DataType;
    const DataType gateup_dtype = uses_fp16_gate_up(fp16_weight_mode) ? DataType::kHALF : DataType::kBF16;
    const DataType down_dtype = uses_fp16_down(fp16_weight_mode) ? DataType::kHALF : DataType::kBF16;
    return engine.getTensorDataType("input") == DataType::kBF16 &&
        engine.getTensorDataType("gateup_weight") == gateup_dtype &&
        engine.getTensorDataType("down_weight") == down_dtype &&
        engine.getTensorDataType("output") == DataType::kBF16 &&
        dims_equal(engine.getTensorShape("input"), {m, hidden}) &&
        dims_equal(engine.getTensorShape("gateup_weight"), {2 * intermediate, hidden}) &&
        dims_equal(engine.getTensorShape("down_weight"), {hidden, intermediate}) &&
        dims_equal(engine.getTensorShape("output"), {m, hidden});
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

std::string generic_engine_name(
    int64_t m,
    int64_t hidden,
    int64_t intermediate,
    Fp16WeightMode fp16_weight_mode)
{
    std::string name = "trt_mlp_bf16_edgefm_fp16";
    if (fp16_weight_mode != Fp16WeightMode::None) {
        name += "_fp16weights-";
        name += fp16_weight_mode_name(fp16_weight_mode);
    }
    name += "_" + shape_key(m, hidden, intermediate) + ".engine";
    return name;
}

std::filesystem::path find_engine_file(
    const std::filesystem::path& dir,
    int64_t m,
    int64_t hidden,
    int64_t intermediate,
    Fp16WeightMode fp16_weight_mode)
{
    const std::filesystem::path generic = dir / generic_engine_name(m, hidden, intermediate, fp16_weight_mode);
    if (std::filesystem::exists(generic)) {
        return generic;
    }

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return {};
    }

    const std::string suffix = "_" + shape_key(m, hidden, intermediate) + ".engine";
    std::vector<std::filesystem::path> matches;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("edgefm_bf16_layout_fp16_compute") == std::string::npos) {
            continue;
        }
        const bool has_fp16_weight_suffix = name.find("fp16weights-") != std::string::npos;
        if (fp16_weight_mode == Fp16WeightMode::None) {
            if (has_fp16_weight_suffix) {
                continue;
            }
        } else {
            const std::string mode_token = std::string("fp16weights-") + fp16_weight_mode_name(fp16_weight_mode);
            if (name.find(mode_token) == std::string::npos) {
                continue;
            }
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

struct PersistentWeightCopy {
    Tensor tensor;
    const void* source_ptr = nullptr;
};

struct CastBundle {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

} // namespace

struct TrtPrefillMlpBridge::Impl {
    explicit Impl(const EngineConfig& config)
        : enabled(env_flag_enabled("EDGE_FM_PREFILL_TRT_MLP")),
          fp16_weight_mode(parse_fp16_weight_mode()),
          device_id(config.runtime_device_id())
    {
        const char* raw_dir = std::getenv("EDGE_FM_TRT_MLP_ENGINE_DIR");
        if (raw_dir != nullptr && *raw_dir != '\0') {
            engine_dir = std::filesystem::path(raw_dir);
        }
        if (enabled && fp16_weight_mode != Fp16WeightMode::None) {
            Logging::instance().log_warn(
                "TensorRT MLP bridge FP16 persistent weight mode '{}' is experimental/default-off; using native fallback on missing engines or copy failure",
                fp16_weight_mode_name(fp16_weight_mode));
        }
    }

    bool try_forward(
        int32_t layer_id,
        const Tensor& input,
        const Tensor& gate_up_weight,
        const Tensor& down_weight,
        Tensor& output,
        cudaStream_t stream)
    {
        if (!enabled) {
            return false;
        }
        if (fp16_weight_mode != Fp16WeightMode::None && fp16_weight_copy_disabled) {
            return false;
        }
        if (engine_dir.empty()) {
            log_once("missing_engine_dir",
                     "EDGE_FM_PREFILL_TRT_MLP=1 but EDGE_FM_TRT_MLP_ENGINE_DIR is not set; using native MLP path");
            return false;
        }
        if (!is_gpu_dtype_2d(input, DType::BFloat16) ||
            !is_gpu_dtype_2d(gate_up_weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(down_weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(output, DType::BFloat16)) {
            return false;
        }

        const auto& input_shape = input.shape();
        const auto& gate_up_shape = gate_up_weight.shape();
        const auto& down_shape = down_weight.shape();
        const auto& output_shape = output.shape();
        const int64_t m = input_shape[0];
        const int64_t hidden = input_shape[1];
        if (m <= 0 || hidden <= 0 ||
            gate_up_shape[0] % 2 != 0 ||
            gate_up_shape[1] != hidden ||
            down_shape[0] != hidden ||
            output_shape[0] != m ||
            output_shape[1] != hidden) {
            return false;
        }
        const int64_t intermediate = gate_up_shape[0] / 2;
        if (down_shape[1] != intermediate) {
            return false;
        }

        EngineBundle* bundle = get_bundle(m, hidden, intermediate);
        if (bundle == nullptr) {
            return false;
        }
        if (!ensure_context(*bundle)) {
            return false;
        }

        const Tensor* gate_up_bind = &gate_up_weight;
        const Tensor* down_bind = &down_weight;
        if (uses_fp16_gate_up(fp16_weight_mode)) {
            gate_up_bind = get_or_create_fp16_weight_copy(
                layer_id, "gateup", gate_up_weight, stream);
            if (gate_up_bind == nullptr) {
                return false;
            }
        }
        if (uses_fp16_down(fp16_weight_mode)) {
            down_bind = get_or_create_fp16_weight_copy(
                layer_id, "down", down_weight, stream);
            if (down_bind == nullptr) {
                return false;
            }
        }

        nvinfer1::IExecutionContext& context = *bundle->context;
        if (!context.setTensorAddress("input", input.data_ptr()) ||
            !context.setTensorAddress("gateup_weight", gate_up_bind->data_ptr()) ||
            !context.setTensorAddress("down_weight", down_bind->data_ptr()) ||
            !context.setTensorAddress("output", output.data_ptr())) {
            log_once("set_tensor_address_failed_" + shape_key(m, hidden, intermediate),
                     "TensorRT MLP bridge failed to bind tensors for {}; using native MLP path",
                     shape_key(m, hidden, intermediate));
            return false;
        }
        if (!context.enqueueV3(stream)) {
            log_once("enqueue_failed_" + shape_key(m, hidden, intermediate),
                     "TensorRT MLP bridge enqueueV3 failed for {}; using native MLP path",
                     shape_key(m, hidden, intermediate));
            return false;
        }
        return true;
    }

    void reset_runtime_caches()
    {
        persistent_weight_copies.clear();
        cast_bundles.clear();
        bundles.clear();
        fp16_weight_copy_disabled = false;
    }

    template <typename... Args>
    void log_once(const std::string& key, const char* fmt, Args&&... args)
    {
        if (logged_messages.insert(key).second) {
            Logging::instance().log_warn(fmt, std::forward<Args>(args)...);
        }
    }

    EngineBundle* get_bundle(int64_t m, int64_t hidden, int64_t intermediate)
    {
        const std::string key = shape_key(m, hidden, intermediate);
        if (missing_shapes.count(key) != 0) {
            return nullptr;
        }
        auto it = bundles.find(key);
        if (it != bundles.end()) {
            return it->second.get();
        }

        const std::filesystem::path engine_path = find_engine_file(
            engine_dir, m, hidden, intermediate, fp16_weight_mode);
        if (engine_path.empty()) {
            missing_shapes.insert(key);
            log_once("missing_engine_" + key,
                     "TensorRT MLP bridge engine for {} with fp16_weight_mode={} not found under {}; using native MLP path",
                     key,
                     fp16_weight_mode_name(fp16_weight_mode),
                     engine_dir.string());
            return nullptr;
        }

        std::vector<char> serialized = read_binary_file(engine_path);
        if (serialized.empty()) {
            missing_shapes.insert(key);
            log_once("read_engine_failed_" + key,
                     "TensorRT MLP bridge failed to read {}; using native MLP path",
                     engine_path.string());
            return nullptr;
        }

        auto bundle = std::make_unique<EngineBundle>();
        bundle->path = engine_path;
        bundle->runtime.reset(nvinfer1::createInferRuntime(logger));
        if (bundle->runtime == nullptr) {
            missing_shapes.insert(key);
            log_once("create_runtime_failed_" + key,
                     "TensorRT MLP bridge createInferRuntime failed; using native MLP path");
            return nullptr;
        }
        bundle->engine.reset(bundle->runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (bundle->engine == nullptr ||
            !engine_io_matches(*bundle->engine, m, hidden, intermediate, fp16_weight_mode)) {
            missing_shapes.insert(key);
            log_once("engine_mismatch_" + key,
                     "TensorRT MLP bridge engine {} does not match expected EdgeFM-layout IO for {} fp16_weight_mode={}; using native MLP path",
                     engine_path.string(),
                     key,
                     fp16_weight_mode_name(fp16_weight_mode));
            return nullptr;
        }

        Logging::instance().log_info(
            "TensorRT MLP bridge loaded {} for {} fp16_weight_mode={}",
            engine_path.string(),
            key,
            fp16_weight_mode_name(fp16_weight_mode));
        EngineBundle* raw = bundle.get();
        bundles.emplace(key, std::move(bundle));
        return raw;
    }

    bool ensure_context(EngineBundle& bundle)
    {
        if (bundle.context != nullptr) {
            return true;
        }
        std::unique_ptr<nvinfer1::IExecutionContext> context(bundle.engine->createExecutionContext());
        if (context == nullptr) {
            log_once("create_context_failed_" + bundle.path.string(),
                     "TensorRT MLP bridge failed to create execution context for {}; using native MLP path",
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

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        if (builder == nullptr) {
            log_once("cast_builder_failed_" + key,
                     "TensorRT MLP bridge failed to create BF16->FP16 cast builder for {}; using native MLP path",
                     key);
            return nullptr;
        }
        const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (network == nullptr || config == nullptr) {
            log_once("cast_network_failed_" + key,
                     "TensorRT MLP bridge failed to create BF16->FP16 cast network for {}; using native MLP path",
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
                     "TensorRT MLP bridge failed to add BF16->FP16 cast input for {}; using native MLP path",
                     key);
            return nullptr;
        }
        nvinfer1::ICastLayer* cast = network->addCast(*input, nvinfer1::DataType::kHALF);
        if (cast == nullptr || cast->getOutput(0) == nullptr) {
            log_once("cast_layer_failed_" + key,
                     "TensorRT MLP bridge failed to add BF16->FP16 cast layer for {}; using native MLP path",
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
                     "TensorRT MLP bridge failed to build BF16->FP16 cast engine for {}; using native MLP path",
                     key);
            return nullptr;
        }

        auto bundle = std::make_unique<CastBundle>();
        bundle->runtime.reset(nvinfer1::createInferRuntime(logger));
        if (bundle->runtime == nullptr) {
            log_once("cast_runtime_failed_" + key,
                     "TensorRT MLP bridge failed to create BF16->FP16 cast runtime for {}; using native MLP path",
                     key);
            return nullptr;
        }
        bundle->engine.reset(bundle->runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
        if (bundle->engine == nullptr ||
            bundle->engine->getTensorDataType("input") != nvinfer1::DataType::kBF16 ||
            bundle->engine->getTensorDataType("output") != nvinfer1::DataType::kHALF ||
            !dims_equal(bundle->engine->getTensorShape("input"), {rows, cols}) ||
            !dims_equal(bundle->engine->getTensorShape("output"), {rows, cols})) {
            log_once("cast_engine_mismatch_" + key,
                     "TensorRT MLP bridge BF16->FP16 cast engine IO mismatch for {}; using native MLP path",
                     key);
            return nullptr;
        }
        bundle->context.reset(bundle->engine->createExecutionContext());
        if (bundle->context == nullptr) {
            log_once("cast_context_failed_" + key,
                     "TensorRT MLP bridge failed to create BF16->FP16 cast context for {}; using native MLP path",
                     key);
            return nullptr;
        }

        Logging::instance().log_info(
            "TensorRT MLP bridge built BF16->FP16 persistent weight cast engine for {}",
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
                     "TensorRT MLP bridge failed to bind BF16->FP16 cast tensors for {}; using native MLP path",
                     matrix_key(shape[0], shape[1]));
            return false;
        }
        if (!bundle->context->enqueueV3(stream)) {
            log_once("cast_enqueue_failed_" + matrix_key(shape[0], shape[1]),
                     "TensorRT MLP bridge BF16->FP16 cast enqueue failed for {}; using native MLP path",
                     matrix_key(shape[0], shape[1]));
            return false;
        }
        return true;
    }

    const Tensor* get_or_create_fp16_weight_copy(
        int32_t layer_id,
        const std::string& role,
        const Tensor& source,
        cudaStream_t stream)
    {
        const std::string key = "layer" + std::to_string(layer_id) + "_" + role;
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
            log_once("fp16_weight_alloc_failed_" + role,
                     "TensorRT MLP bridge failed to allocate persistent FP16 {} weight copy ({} bytes): {}; using native MLP path",
                     role,
                     bytes,
                     cudaGetErrorString(alloc_status));
            disable_fp16_weight_copies_after_failure("alloc_failed_" + role);
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
            disable_fp16_weight_copies_after_failure("cast_failed_" + role);
            return nullptr;
        }

        PersistentWeightCopy copy;
        copy.tensor = std::move(fp16_tensor);
        copy.source_ptr = source.data_ptr();
        auto [inserted, _ok] = persistent_weight_copies.emplace(key, std::move(copy));
        Logging::instance().log_info(
            "TensorRT MLP bridge created persistent FP16 {} weight copy for layer {} ({} bytes)",
            role,
            layer_id,
            bytes);
        return &inserted->second.tensor;
    }

    void disable_fp16_weight_copies_after_failure(const std::string& reason)
    {
        if (fp16_weight_copy_disabled) {
            return;
        }
        fp16_weight_copy_disabled = true;
        log_once("fp16_weight_mode_disabled_" + reason,
                 "TensorRT MLP bridge disabled FP16 persistent weight mode for this engine after {}; using native MLP fallback",
                 reason);
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
    bool fp16_weight_copy_disabled = false;
};

#else

struct TrtPrefillMlpBridge::Impl {
    explicit Impl(const EngineConfig& config)
    {
        (void)config;
        if (env_flag_enabled("EDGE_FM_PREFILL_TRT_MLP")) {
            Logging::instance().log_warn(
                "EDGE_FM_PREFILL_TRT_MLP=1 but this build was configured without BUILD_TRT_MLP_BRIDGE=ON; using native MLP path");
        }
    }

    bool try_forward(
        int32_t layer_id,
        const Tensor& input,
        const Tensor& gate_up_weight,
        const Tensor& down_weight,
        Tensor& output,
        cudaStream_t stream)
    {
        (void)layer_id;
        (void)input;
        (void)gate_up_weight;
        (void)down_weight;
        (void)output;
        (void)stream;
        return false;
    }

    void reset_runtime_caches() {}
};

#endif

TrtPrefillMlpBridge::TrtPrefillMlpBridge(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

TrtPrefillMlpBridge::~TrtPrefillMlpBridge() = default;

bool TrtPrefillMlpBridge::try_forward(
    int32_t layer_id,
    const Tensor& input,
    const Tensor& gate_up_weight,
    const Tensor& down_weight,
    Tensor& output,
    cudaStream_t stream)
{
    return impl_->try_forward(layer_id, input, gate_up_weight, down_weight, output, stream);
}

void TrtPrefillMlpBridge::reset_runtime_caches()
{
    impl_->reset_runtime_caches();
}

} // namespace edge_fm
