#include "layers/linear.h"

#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

namespace edge_fm {
struct FusedGateUpLinearLayer::DecodeSwigluFusionState {
    bool available = false;
    std::string unavailable_reason;
    void* expert_offsets_device_ptr = nullptr;
    int sm_count = 0;
    int selected_kernel_config = 0;
    std::string selected_kernel_config_name;
};

FusedGateUpLinearLayer::FusedGateUpLinearLayer(
    const std::string& layer_prefix_base,
    const EngineConfig& engine_config,
    uint32_t in_features,
    uint32_t gate_out_features,
    uint32_t up_out_features,
    std::string layer_name)
    : LinearLayer(layer_prefix_base + ".gate_up_fused",
                  engine_config,
                  in_features,
                  gate_out_features + up_out_features,
                  std::move(layer_name)),
      in_features_(in_features),
      gate_out_features_(gate_out_features),
      up_out_features_(up_out_features),
      layer_prefix_base_(layer_prefix_base) {}

FusedGateUpLinearLayer::~FusedGateUpLinearLayer() = default;

namespace {

template <typename T>
struct CutlassScalarType;

template <typename T>
struct CutlassScalarType {};

template <>
struct CutlassScalarType<half> {
    using type = cutlass::half_t;
};

template <>
struct CutlassScalarType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

enum class DecodeSwigluKernelConfigId : uint8_t {
    Tile16x128x64Stage2 = 0,
    Tile16x256x64Stage2,
    Tile32x128x64Stage2,
    Tile64x128x64Stage2,
    Tile128x128x64Stage2,
    Tile16x128x64Stage3,
    Tile16x256x64Stage3,
};

struct DecodeSwigluKernelCandidate {
    DecodeSwigluKernelConfigId id;
    const char* name;
};

constexpr DecodeSwigluKernelConfigId kDefaultDecodeSwigluKernelConfig =
    DecodeSwigluKernelConfigId::Tile16x128x64Stage2;

constexpr std::array<DecodeSwigluKernelCandidate, 7> kDecodeSwigluKernelCandidates = {{
    {DecodeSwigluKernelConfigId::Tile16x128x64Stage2, "16x128x64_s2"},
    {DecodeSwigluKernelConfigId::Tile16x256x64Stage2, "16x256x64_s2"},
    {DecodeSwigluKernelConfigId::Tile32x128x64Stage2, "32x128x64_s2"},
    {DecodeSwigluKernelConfigId::Tile64x128x64Stage2, "64x128x64_s2"},
    {DecodeSwigluKernelConfigId::Tile128x128x64Stage2, "128x128x64_s2"},
    {DecodeSwigluKernelConfigId::Tile16x128x64Stage3, "16x128x64_s3"},
    {DecodeSwigluKernelConfigId::Tile16x256x64Stage3, "16x256x64_s3"},
}};

struct DecodeSwigluTuneKey {
    int sm = 0;
    int64_t output_features = 0;
    int64_t input_features = 0;
    DType dtype = DType::Float16;

    bool operator==(const DecodeSwigluTuneKey& other) const noexcept {
        return sm == other.sm && output_features == other.output_features &&
            input_features == other.input_features && dtype == other.dtype;
    }
};

struct DecodeSwigluTuneKeyHash {
    std::size_t operator()(const DecodeSwigluTuneKey& key) const noexcept {
        const std::size_t h1 = std::hash<int>{}(key.sm);
        const std::size_t h2 = std::hash<int64_t>{}(key.output_features);
        const std::size_t h3 = std::hash<int64_t>{}(key.input_features);
        const std::size_t h4 = std::hash<int>{}(static_cast<int>(key.dtype));
        return (((h1 * 1315423911u) ^ h2) * 2654435761u) ^ (h3 << 1) ^ (h4 << 3);
    }
};

std::unordered_map<DecodeSwigluTuneKey, DecodeSwigluKernelConfigId, DecodeSwigluTuneKeyHash>&
decode_swiglu_tune_cache() {
    static std::unordered_map<DecodeSwigluTuneKey, DecodeSwigluKernelConfigId, DecodeSwigluTuneKeyHash> cache;
    return cache;
}

std::mutex& decode_swiglu_tune_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

const char* decode_swiglu_kernel_config_name(DecodeSwigluKernelConfigId config_id) {
    for (const auto& candidate : kDecodeSwigluKernelCandidates) {
        if (candidate.id == config_id) {
            return candidate.name;
        }
    }
    return "unknown";
}

bool decode_swiglu_autotune_enabled() {
    const char* env = std::getenv("EDGE_FM_DECODE_SWIGLU_AUTOTUNE");
    if (env == nullptr || *env == '\0') {
        return true;
    }
    return std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "False";
}

bool try_get_decode_swiglu_kernel_override(DecodeSwigluKernelConfigId& config_id) {
    const char* env = std::getenv("EDGE_FM_DECODE_SWIGLU_CONFIG");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    const std::string value(env);
    for (const auto& candidate : kDecodeSwigluKernelCandidates) {
        if (value == candidate.name) {
            config_id = candidate.id;
            return true;
        }
    }
    return false;
}

int current_sm_version() {
    int device = -1;
    CUDA_CHECK_THROW(cudaGetDevice(&device), "FusedGateUpLinearLayer: failed to query CUDA device");

    int major = 0;
    int minor = 0;
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
        "FusedGateUpLinearLayer: failed to query compute capability major");
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
        "FusedGateUpLinearLayer: failed to query compute capability minor");
    return major * 10 + minor;
}

int current_sm_count() {
    int device = -1;
    CUDA_CHECK_THROW(cudaGetDevice(&device), "FusedGateUpLinearLayer: failed to query CUDA device");
    int sm_count = 0;
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
        "FusedGateUpLinearLayer: failed to query SM count");
    return sm_count;
}

template <typename T>
constexpr DType dtype_for_decode_swiglu_kernel() {
    if constexpr (std::is_same_v<T, half>) {
        return DType::Float16;
    } else {
        return DType::BFloat16;
    }
}

template <typename T, int TileM, int TileN, int TileK, int Stages>
void launch_decode_swiglu_fused_kernel_config(
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpLinearLayer::DecodeSwigluFusionState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream) {
    using CutlassT = typename CutlassScalarType<T>::type;
    using EpilogueTag = tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu;

    tensorrt_llm::kernels::cutlass_kernels_oss::sm80_generic_fused_moe_gemm_kernelLauncher<
        CutlassT,
        CutlassT,
        TileM,
        TileN,
        TileK,
        Stages,
        EpilogueTag>(
        reinterpret_cast<const CutlassT*>(input_ptr),
        reinterpret_cast<const CutlassT*>(weight_tensor->data_ptr()),
        bias_tensor ? reinterpret_cast<const CutlassT*>(bias_tensor->data_ptr()) : nullptr,
        true,
        reinterpret_cast<CutlassT*>(output_ptr),
        static_cast<const int64_t*>(state.expert_offsets_device_ptr),
        1,
        output_features,
        input_features,
        1,
        state.sm_count,
        stream,
        nullptr);
}

template <typename T>
void launch_decode_swiglu_fused_kernel(
    DecodeSwigluKernelConfigId config_id,
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpLinearLayer::DecodeSwigluFusionState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream) {
    switch (config_id) {
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 16, 128, 64, 2>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 16, 256, 64, 2>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile32x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 32, 128, 64, 2>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile64x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 64, 128, 64, 2>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile128x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 128, 128, 64, 2>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage3:
            launch_decode_swiglu_fused_kernel_config<T, 16, 128, 64, 3>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage3:
            launch_decode_swiglu_fused_kernel_config<T, 16, 256, 64, 3>(
                weight_tensor, bias_tensor, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
    }
}

template <typename T, int TileM, int TileN, int TileK, int Stages>
int query_decode_swiglu_kernel_occupancy_for_config() {
    using CutlassT = typename CutlassScalarType<T>::type;
    using EpilogueTag = tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu;

    int occupancy = 0;
    tensorrt_llm::kernels::cutlass_kernels_oss::sm80_generic_fused_moe_gemm_kernelLauncher<
        CutlassT,
        CutlassT,
        TileM,
        TileN,
        TileK,
        Stages,
        EpilogueTag>(
        nullptr,
        nullptr,
        nullptr,
        true,
        nullptr,
        nullptr,
        0,
        0,
        0,
        0,
        0,
        nullptr,
        &occupancy);
    return occupancy;
}

template <typename T>
int query_decode_swiglu_kernel_occupancy(DecodeSwigluKernelConfigId config_id) {
    switch (config_id) {
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage2:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 16, 128, 64, 2>();
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage2:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 16, 256, 64, 2>();
        case DecodeSwigluKernelConfigId::Tile32x128x64Stage2:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 32, 128, 64, 2>();
        case DecodeSwigluKernelConfigId::Tile64x128x64Stage2:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 64, 128, 64, 2>();
        case DecodeSwigluKernelConfigId::Tile128x128x64Stage2:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 128, 128, 64, 2>();
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage3:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 16, 128, 64, 3>();
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage3:
            return query_decode_swiglu_kernel_occupancy_for_config<T, 16, 256, 64, 3>();
    }
    return 0;
}

template <typename T>
float benchmark_decode_swiglu_kernel(
    DecodeSwigluKernelConfigId config_id,
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    int device_id,
    const FusedGateUpLinearLayer::DecodeSwigluFusionState& state) {
    CUDA_CHECK_THROW(cudaSetDevice(device_id), "FusedGateUpLinearLayer: failed to set CUDA device");

    const std::string tune_key = "decode_swiglu_fused.autotune." +
        std::to_string(static_cast<int>(dtype_for_decode_swiglu_kernel<T>())) + "." +
        std::to_string(input_features) + "." + std::to_string(output_features);
    void* input_buf = StaticBufferManager::get_cache_buf(
        tune_key + ".input", static_cast<size_t>(input_features) * sizeof(T), device_id);
    void* output_buf = StaticBufferManager::get_cache_buf(
        tune_key + ".output", static_cast<size_t>(output_features) * sizeof(T), device_id);

    cudaStream_t tune_stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    auto cleanup = [&]() {
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr) {
            cudaEventDestroy(stop_event);
        }
        if (tune_stream != nullptr) {
            cudaStreamDestroy(tune_stream);
        }
    };

    try {
        CUDA_CHECK_THROW(
            cudaStreamCreateWithFlags(&tune_stream, cudaStreamNonBlocking),
            "FusedGateUpLinearLayer: failed to create autotune stream");
        CUDA_CHECK_THROW(
            cudaEventCreate(&start_event),
            "FusedGateUpLinearLayer: failed to create autotune start event");
        CUDA_CHECK_THROW(
            cudaEventCreate(&stop_event),
            "FusedGateUpLinearLayer: failed to create autotune stop event");
        CUDA_CHECK_THROW(
            cudaMemsetAsync(input_buf, 0, static_cast<size_t>(input_features) * sizeof(T), tune_stream),
            "FusedGateUpLinearLayer: failed to initialize autotune input buffer");

        constexpr int kWarmupIters = 8;
        constexpr int kBenchmarkIters = 40;

        for (int iter = 0; iter < kWarmupIters; ++iter) {
            launch_decode_swiglu_fused_kernel<T>(
                config_id,
                weight_tensor,
                bias_tensor,
                output_features,
                input_features,
                state,
                static_cast<const T*>(input_buf),
                static_cast<T*>(output_buf),
                tune_stream);
        }

        CUDA_CHECK_THROW(
            cudaEventRecord(start_event, tune_stream),
            "FusedGateUpLinearLayer: failed to record autotune start event");
        for (int iter = 0; iter < kBenchmarkIters; ++iter) {
            launch_decode_swiglu_fused_kernel<T>(
                config_id,
                weight_tensor,
                bias_tensor,
                output_features,
                input_features,
                state,
                static_cast<const T*>(input_buf),
                static_cast<T*>(output_buf),
                tune_stream);
        }
        CUDA_CHECK_THROW(
            cudaEventRecord(stop_event, tune_stream),
            "FusedGateUpLinearLayer: failed to record autotune stop event");
        CUDA_CHECK_THROW(
            cudaEventSynchronize(stop_event),
            "FusedGateUpLinearLayer: failed to synchronize autotune stop event");

        float elapsed_ms = 0.0f;
        CUDA_CHECK_THROW(
            cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
            "FusedGateUpLinearLayer: failed to measure autotune elapsed time");

        cleanup();
        return elapsed_ms / static_cast<float>(kBenchmarkIters);
    } catch (...) {
        cleanup();
        throw;
    }
}

template <typename T>
DecodeSwigluKernelConfigId select_decode_swiglu_kernel_config(
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    int device_id,
    const FusedGateUpLinearLayer::DecodeSwigluFusionState& state) {
    DecodeSwigluKernelConfigId override_config = kDefaultDecodeSwigluKernelConfig;
    if (try_get_decode_swiglu_kernel_override(override_config)) {
        try {
            if (query_decode_swiglu_kernel_occupancy<T>(override_config) > 0) {
                return override_config;
            }
        } catch (const std::exception&) {
        }
        return kDefaultDecodeSwigluKernelConfig;
    }

    if (!decode_swiglu_autotune_enabled()) {
        return kDefaultDecodeSwigluKernelConfig;
    }

    const DecodeSwigluTuneKey cache_key{
        current_sm_version(),
        output_features,
        input_features,
        dtype_for_decode_swiglu_kernel<T>()};

    std::lock_guard<std::mutex> lock(decode_swiglu_tune_cache_mutex());
    auto& cache = decode_swiglu_tune_cache();
    auto it = cache.find(cache_key);
    if (it != cache.end()) {
        return it->second;
    }

    DecodeSwigluKernelConfigId best_config = kDefaultDecodeSwigluKernelConfig;
    float best_ms = std::numeric_limits<float>::infinity();
    for (const auto& candidate : kDecodeSwigluKernelCandidates) {
        int occupancy = 0;
        try {
            occupancy = query_decode_swiglu_kernel_occupancy<T>(candidate.id);
        } catch (const std::exception&) {
            continue;
        }
        if (occupancy <= 0) {
            continue;
        }

        try {
            const float candidate_ms = benchmark_decode_swiglu_kernel<T>(
                candidate.id,
                weight_tensor,
                bias_tensor,
                output_features,
                input_features,
                device_id,
                state);
            if (candidate_ms < best_ms) {
                best_ms = candidate_ms;
                best_config = candidate.id;
            }
        } catch (const std::exception&) {
        }
    }

    cache.emplace(cache_key, best_config);
    return best_config;
}

template <typename T>
void prepare_decode_swiglu_fusion_state_for_dtype(
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    int device_id,
    const std::string& layer_prefix,
    FusedGateUpLinearLayer::DecodeSwigluFusionState& state) {
    (void)device_id;
    const int sm = current_sm_version();
    if (sm < 80 || sm >= 90) {
        state.unavailable_reason =
            "decode fused SwiGLU path currently only targets sm80-sm89 devices";
        return;
    }

    if ((output_features % 64) != 0 || (input_features % 64) != 0) {
        state.unavailable_reason =
            "decode fused SwiGLU shape is unsupported by TRT-LLM SM80 fused MoE kernel";
        return;
    }

    CUDA_CHECK_THROW(cudaSetDevice(device_id), "FusedGateUpLinearLayer: failed to set CUDA device");

    const std::string buffer_name = layer_prefix + ".decode_swiglu_fused.expert_offsets";
    state.expert_offsets_device_ptr =
        StaticBufferManager::get_cache_buf(buffer_name, sizeof(int64_t), device_id);

    const std::array<int64_t, 1> expert_offsets = {1};
    CUDA_CHECK_THROW(
        cudaMemcpy(
            state.expert_offsets_device_ptr,
            expert_offsets.data(),
            sizeof(expert_offsets),
            cudaMemcpyHostToDevice),
        "FusedGateUpLinearLayer: failed to initialize decode expert offsets");
    state.sm_count = current_sm_count();
    state.available = true;
    state.selected_kernel_config = static_cast<int>(kDefaultDecodeSwigluKernelConfig);
    state.selected_kernel_config_name = decode_swiglu_kernel_config_name(kDefaultDecodeSwigluKernelConfig);

    try {
        const auto best_config = select_decode_swiglu_kernel_config<T>(
            weight_tensor,
            bias_tensor,
            output_features,
            input_features,
            device_id,
            state);
        state.selected_kernel_config = static_cast<int>(best_config);
        state.selected_kernel_config_name = decode_swiglu_kernel_config_name(best_config);
    } catch (const std::exception&) {
        state.selected_kernel_config = static_cast<int>(kDefaultDecodeSwigluKernelConfig);
        state.selected_kernel_config_name = decode_swiglu_kernel_config_name(kDefaultDecodeSwigluKernelConfig);
    }
}

template <typename T>
void run_decode_swiglu_fused_for_dtype(
    const Tensor* weight_tensor,
    const Tensor* bias_tensor,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpLinearLayer::DecodeSwigluFusionState& state,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    const auto config_id = static_cast<DecodeSwigluKernelConfigId>(state.selected_kernel_config);
    launch_decode_swiglu_fused_kernel<T>(
        config_id,
        weight_tensor,
        bias_tensor,
        output_features,
        input_features,
        state,
        static_cast<const T*>(input.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        stream);
}

} // namespace

void FusedGateUpLinearLayer::prepare_decode_swiglu_fusion_state() {
    decode_swiglu_fusion_state_ = std::make_unique<DecodeSwigluFusionState>();

    if (decode_weights_.quant_type_ != QuantType::FP16_BF16 || decode_weights_.weight_ == nullptr) {
        decode_swiglu_fusion_state_->unavailable_reason =
            "decode fused SwiGLU requires FP16/BF16 weights";
        return;
    }

    if (gate_out_features_ != up_out_features_) {
        decode_swiglu_fusion_state_->unavailable_reason =
            "decode fused SwiGLU requires gate/up output widths to match";
        return;
    }

    try {
        switch (decode_weights_.weight_->dtype()) {
            case DType::Float16:
                prepare_decode_swiglu_fusion_state_for_dtype<half>(
                    decode_weights_.weight_,
                    decode_weights_.bias_,
                    static_cast<int64_t>(up_out_features_),
                    static_cast<int64_t>(in_features_),
                    device_id_,
                    layer_prefix_,
                    *decode_swiglu_fusion_state_);
                break;
            case DType::BFloat16:
                prepare_decode_swiglu_fusion_state_for_dtype<__nv_bfloat16>(
                    decode_weights_.weight_,
                    decode_weights_.bias_,
                    static_cast<int64_t>(up_out_features_),
                    static_cast<int64_t>(in_features_),
                    device_id_,
                    layer_prefix_,
                    *decode_swiglu_fusion_state_);
                break;
            default:
                decode_swiglu_fusion_state_->unavailable_reason =
                    "decode fused SwiGLU requires FP16/BF16 weights";
                break;
        }
    } catch (const std::exception& ex) {
        decode_swiglu_fusion_state_->available = false;
        decode_swiglu_fusion_state_->unavailable_reason = ex.what();
    }
}

bool FusedGateUpLinearLayer::try_forward_decode_swiglu_fused(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    if (decode_swiglu_fusion_state_ == nullptr) {
        prepare_decode_swiglu_fusion_state();
    }
    if (decode_swiglu_fusion_state_ == nullptr || !decode_swiglu_fusion_state_->available) {
        return false;
    }

    check<InvalidRequestError>(
        input.shape().size() == 2 && input.shape()[0] == 1 &&
            input.shape()[1] == static_cast<int64_t>(in_features_),
        "FusedGateUpLinearLayer: decode fused SwiGLU expects input shape [1, in_features]");
    check<InvalidRequestError>(
        output.shape().size() == 2 && output.shape()[0] == 1 &&
            output.shape()[1] == static_cast<int64_t>(up_out_features_),
        "FusedGateUpLinearLayer: decode fused SwiGLU expects output shape [1, up_out_features]");
    check<InvalidRequestError>(
        input.dtype() == output.dtype() && input.dtype() == decode_weights_.weight_->dtype(),
        "FusedGateUpLinearLayer: decode fused SwiGLU requires input/output/weight dtypes to match");

    switch (input.dtype()) {
        case DType::Float16:
            run_decode_swiglu_fused_for_dtype<half>(
                decode_weights_.weight_,
                decode_weights_.bias_,
                static_cast<int64_t>(up_out_features_),
                static_cast<int64_t>(in_features_),
                *decode_swiglu_fusion_state_,
                input,
                output,
                stream);
            return true;
        case DType::BFloat16:
            run_decode_swiglu_fused_for_dtype<__nv_bfloat16>(
                decode_weights_.weight_,
                decode_weights_.bias_,
                static_cast<int64_t>(up_out_features_),
                static_cast<int64_t>(in_features_),
                *decode_swiglu_fusion_state_,
                input,
                output,
                stream);
            return true;
        default:
            return false;
    }
}

} // namespace edge_fm
