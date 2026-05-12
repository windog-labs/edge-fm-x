#include "operators/fused_gate_up_activation_op.h"

#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace edge_fm {

std::string FusedGateUpActivationOpContext::shape_sig() const {
    return "m=" + std::to_string(batch_rows) +
        "|input=" + std::to_string(static_cast<int>(input_dtype)) +
        "|weight=" + std::to_string(static_cast<int>(weight_dtype)) +
        "|output=" + std::to_string(static_cast<int>(output_dtype)) +
        "|in_features=" + std::to_string(input_features) +
        "|gate_out_features=" + std::to_string(gate_output_features) +
        "|up_out_features=" + std::to_string(up_output_features);
}

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
    int64_t batch_rows = 0;
    int64_t output_features = 0;
    int64_t input_features = 0;
    DType dtype = DType::Float16;

    bool operator==(const DecodeSwigluTuneKey& other) const noexcept {
        return sm == other.sm && batch_rows == other.batch_rows &&
            output_features == other.output_features &&
            input_features == other.input_features && dtype == other.dtype;
    }
};

struct DecodeSwigluTuneKeyHash {
    std::size_t operator()(const DecodeSwigluTuneKey& key) const noexcept {
        const std::size_t h1 = std::hash<int>{}(key.sm);
        const std::size_t h2 = std::hash<int64_t>{}(key.batch_rows);
        const std::size_t h3 = std::hash<int64_t>{}(key.output_features);
        const std::size_t h4 = std::hash<int64_t>{}(key.input_features);
        const std::size_t h5 = std::hash<int>{}(static_cast<int>(key.dtype));
        return ((((h1 * 1315423911u) ^ h2) * 2654435761u) ^ (h3 << 1)) ^ (h4 << 3) ^ (h5 << 5);
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

bool decode_swiglu_device_supported(int sm) {
    return sm == 80;
}

int sm_version_for_device(int device_id) {
    int major = 0;
    int minor = 0;
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
        "FusedGateUpActivationOp: failed to query compute capability major");
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
        "FusedGateUpActivationOp: failed to query compute capability minor");
    return major * 10 + minor;
}

int sm_count_for_device(int device_id) {
    int sm_count = 0;
    CUDA_CHECK_THROW(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
        "FusedGateUpActivationOp: failed to query SM count");
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
using DecodeSwigluGemmType = fused_moe::Fused_Moe_Kernel_sm80<
    typename CutlassScalarType<T>::type,
    typename CutlassScalarType<T>::type,
    typename CutlassScalarType<T>::type,
    TileM,
    TileN,
    TileK,
    Stages,
    fused_moe::EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>(true)>;

template <typename T, int TileM, int TileN, int TileK, int Stages>
int64_t query_decode_swiglu_tile_count(int64_t batch_rows, int64_t output_features) {
    using GemmType = DecodeSwigluGemmType<T, TileM, TileN, TileK, Stages>;
    const int64_t m_tiles = (batch_rows + GemmType::kMaxTileM - 1) / GemmType::kMaxTileM;
    const int64_t n_tiles = (output_features + GemmType::kTileN - 1) / GemmType::kTileN;
    return std::max<int64_t>(1, m_tiles * n_tiles);
}

template <typename T, int TileM, int TileN, int TileK, int Stages>
int query_decode_swiglu_threadblock_count(
    int64_t batch_rows,
    int64_t output_features,
    int sm_count,
    int occupancy) {
    const int64_t tile_count =
        query_decode_swiglu_tile_count<T, TileM, TileN, TileK, Stages>(batch_rows, output_features);
    const int64_t launch_limit = static_cast<int64_t>(sm_count) * static_cast<int64_t>(occupancy);
    return static_cast<int>(std::max<int64_t>(1, std::min(tile_count, launch_limit)));
}

template <typename T, int TileM, int TileN, int TileK, int Stages>
void launch_decode_swiglu_fused_kernel_config(
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpActivationOpState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream) {
    using CutlassT = typename CutlassScalarType<T>::type;
    using GemmType = DecodeSwigluGemmType<T, TileM, TileN, TileK, Stages>;
    using Arguments = typename GemmType::Arguments;

    const int threadblock_count = state.selected_threadblock_count > 0
        ? state.selected_threadblock_count
        : query_decode_swiglu_threadblock_count<T, TileM, TileN, TileK, Stages>(
              batch_rows,
              output_features,
              state.sm_count,
              std::max(1, state.selected_kernel_occupancy));

    if constexpr (GemmType::kSmemSize >= (48 << 10)) {
        CUDA_CHECK_THROW(
            cudaFuncSetAttribute(
                fused_moe::run_global<GemmType>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                GemmType::kSmemSize),
            "FusedGateUpActivationOp: failed to set fused decode SwiGLU dynamic shared memory size");
    }

    Arguments args{
        {const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(input_ptr)),
         const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(weight_tensor.data_ptr())),
         bias_tensor ? const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(bias_tensor->data_ptr())) : nullptr,
         reinterpret_cast<CutlassT*>(output_ptr),
         static_cast<const int64_t*>(state.expert_offsets_device_ptr),
         static_cast<int>(output_features),
         static_cast<int>(input_features),
         1,
         true},
        1,
        threadblock_count};
    auto params = GemmType::to_underlying_arguments(args);
    fused_moe::run_global<GemmType>
        <<<dim3(threadblock_count, 1, 1), dim3(GemmType::kThreadCount), GemmType::kSmemSize, stream>>>(params);
    CUDA_CHECK_THROW(cudaGetLastError(), "FusedGateUpActivationOp: failed to launch fused decode SwiGLU kernel");
}

template <typename T>
void launch_decode_swiglu_fused_kernel(
    DecodeSwigluKernelConfigId config_id,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpActivationOpState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream) {
    switch (config_id) {
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 16, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 16, 256, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile32x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 32, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile64x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 64, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile128x128x64Stage2:
            launch_decode_swiglu_fused_kernel_config<T, 128, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage3:
            launch_decode_swiglu_fused_kernel_config<T, 16, 128, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage3:
            launch_decode_swiglu_fused_kernel_config<T, 16, 256, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features, state, input_ptr, output_ptr, stream);
            return;
    }
}

template <typename T, int TileM, int TileN, int TileK, int Stages>
int query_decode_swiglu_kernel_occupancy_for_config() {
    using GemmType = DecodeSwigluGemmType<T, TileM, TileN, TileK, Stages>;
    return fused_moe::fused_gemm_maximum_active_blocks<GemmType>();
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
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    int device_id,
    const FusedGateUpActivationOpState& state,
    int occupancy,
    int threadblock_count) {
    CUDA_CHECK_THROW(cudaSetDevice(device_id), "FusedGateUpActivationOp: failed to set CUDA device");

    const std::string tune_key = "decode_swiglu_fused.autotune." +
        std::to_string(static_cast<int>(dtype_for_decode_swiglu_kernel<T>())) + "." +
        std::to_string(batch_rows) + "." +
        std::to_string(input_features) + "." + std::to_string(output_features);
    void* input_buf = StaticBufferManager::get_cache_buf(
        tune_key + ".input", static_cast<size_t>(batch_rows) * static_cast<size_t>(input_features) * sizeof(T), device_id);
    void* output_buf = StaticBufferManager::get_cache_buf(
        tune_key + ".output", static_cast<size_t>(batch_rows) * static_cast<size_t>(output_features) * sizeof(T), device_id);

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
            "FusedGateUpActivationOp: failed to create autotune stream");
        CUDA_CHECK_THROW(
            cudaEventCreate(&start_event),
            "FusedGateUpActivationOp: failed to create autotune start event");
        CUDA_CHECK_THROW(
            cudaEventCreate(&stop_event),
            "FusedGateUpActivationOp: failed to create autotune stop event");
        CUDA_CHECK_THROW(
            cudaMemsetAsync(
                input_buf,
                0,
                static_cast<size_t>(batch_rows) * static_cast<size_t>(input_features) * sizeof(T),
                tune_stream),
            "FusedGateUpActivationOp: failed to initialize autotune input buffer");

        constexpr int kWarmupIters = 8;
        constexpr int kBenchmarkIters = 40;

        for (int iter = 0; iter < kWarmupIters; ++iter) {
            FusedGateUpActivationOpState launch_state = state;
            launch_state.selected_kernel_occupancy = occupancy;
            launch_state.selected_threadblock_count = threadblock_count;
            launch_decode_swiglu_fused_kernel<T>(
                config_id,
                weight_tensor,
                bias_tensor,
                batch_rows,
                output_features,
                input_features,
                launch_state,
                static_cast<const T*>(input_buf),
                static_cast<T*>(output_buf),
                tune_stream);
        }

        CUDA_CHECK_THROW(
            cudaEventRecord(start_event, tune_stream),
            "FusedGateUpActivationOp: failed to record autotune start event");
        for (int iter = 0; iter < kBenchmarkIters; ++iter) {
            FusedGateUpActivationOpState launch_state = state;
            launch_state.selected_kernel_occupancy = occupancy;
            launch_state.selected_threadblock_count = threadblock_count;
            launch_decode_swiglu_fused_kernel<T>(
                config_id,
                weight_tensor,
                bias_tensor,
                batch_rows,
                output_features,
                input_features,
                launch_state,
                static_cast<const T*>(input_buf),
                static_cast<T*>(output_buf),
                tune_stream);
        }
        CUDA_CHECK_THROW(
            cudaEventRecord(stop_event, tune_stream),
            "FusedGateUpActivationOp: failed to record autotune stop event");
        CUDA_CHECK_THROW(
            cudaEventSynchronize(stop_event),
            "FusedGateUpActivationOp: failed to synchronize autotune stop event");

        float elapsed_ms = 0.0f;
        CUDA_CHECK_THROW(
            cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
            "FusedGateUpActivationOp: failed to measure autotune elapsed time");

        cleanup();
        return elapsed_ms / static_cast<float>(kBenchmarkIters);
    } catch (...) {
        cleanup();
        throw;
    }
}

template <typename T>
int query_decode_swiglu_threadblock_count(
    DecodeSwigluKernelConfigId config_id,
    int64_t batch_rows,
    int64_t output_features,
    int sm_count,
    int occupancy) {
    switch (config_id) {
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage2:
            return query_decode_swiglu_threadblock_count<T, 16, 128, 64, 2>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage2:
            return query_decode_swiglu_threadblock_count<T, 16, 256, 64, 2>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile32x128x64Stage2:
            return query_decode_swiglu_threadblock_count<T, 32, 128, 64, 2>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile64x128x64Stage2:
            return query_decode_swiglu_threadblock_count<T, 64, 128, 64, 2>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile128x128x64Stage2:
            return query_decode_swiglu_threadblock_count<T, 128, 128, 64, 2>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile16x128x64Stage3:
            return query_decode_swiglu_threadblock_count<T, 16, 128, 64, 3>(
                batch_rows, output_features, sm_count, occupancy);
        case DecodeSwigluKernelConfigId::Tile16x256x64Stage3:
            return query_decode_swiglu_threadblock_count<T, 16, 256, 64, 3>(
                batch_rows, output_features, sm_count, occupancy);
    }
    return 0;
}

template <typename T>
DecodeSwigluKernelConfigId select_decode_swiglu_kernel_config(
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    int device_id,
    const FusedGateUpActivationOpState& state) {
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
        sm_version_for_device(device_id),
        batch_rows,
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
        const int threadblock_count = query_decode_swiglu_threadblock_count<T>(
            candidate.id,
            batch_rows,
            output_features,
            state.sm_count,
            std::min(2, occupancy));

        try {
            const float candidate_ms = benchmark_decode_swiglu_kernel<T>(
                candidate.id,
                weight_tensor,
                bias_tensor,
                batch_rows,
                output_features,
                input_features,
                device_id,
                state,
                std::min(2, occupancy),
                threadblock_count);
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
void prepare_decode_swiglu_state_for_dtype(
    const FusedGateUpActivationOpContext& ctx,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    FusedGateUpActivationOpState& state) {
    const int sm = sm_version_for_device(ctx.device_id);
    if (!decode_swiglu_device_supported(sm)) {
        state.unavailable_reason =
            "decode fused SwiGLU path only enables TensorRT-LLM's SM80 kernel on SM80 by default";
        return;
    }

    if ((ctx.up_output_features % 64) != 0 || (ctx.input_features % 64) != 0) {
        state.unavailable_reason =
            "decode fused SwiGLU shape is unsupported by TRT-LLM SM80 fused MoE kernel";
        return;
    }

    CUDA_CHECK_THROW(cudaSetDevice(ctx.device_id), "FusedGateUpActivationOp: failed to set CUDA device");

    const std::string buffer_name = ctx.layer_prefix + ".swiglu_fused.total_tokens." +
        std::to_string(ctx.batch_rows);
    state.expert_offsets_device_ptr =
        StaticBufferManager::get_cache_buf(buffer_name, sizeof(int64_t), ctx.device_id);

    const std::array<int64_t, 1> expert_offsets = {ctx.batch_rows};
    CUDA_CHECK_THROW(
        cudaMemcpy(
            state.expert_offsets_device_ptr,
            expert_offsets.data(),
            sizeof(expert_offsets),
            cudaMemcpyHostToDevice),
        "FusedGateUpActivationOp: failed to initialize decode expert offsets");
    state.sm_count = sm_count_for_device(ctx.device_id);
    state.available = true;
    state.selected_kernel_config = static_cast<int>(kDefaultDecodeSwigluKernelConfig);
    state.selected_kernel_occupancy = 0;
    state.selected_threadblock_count = 0;
    state.selected_kernel_config_name = decode_swiglu_kernel_config_name(kDefaultDecodeSwigluKernelConfig);

    try {
        const auto best_config = select_decode_swiglu_kernel_config<T>(
            weight_tensor,
            bias_tensor,
            ctx.batch_rows,
            ctx.up_output_features,
            ctx.input_features,
            ctx.device_id,
            state);
        const int max_active_blocks = query_decode_swiglu_kernel_occupancy<T>(best_config);
        if (max_active_blocks <= 0) {
            state.available = false;
            state.unavailable_reason = "decode fused SwiGLU selected kernel has no legal occupancy";
            return;
        }
        state.selected_kernel_config = static_cast<int>(best_config);
        state.selected_kernel_occupancy = std::min(2, max_active_blocks);
        state.selected_threadblock_count = query_decode_swiglu_threadblock_count<T>(
            best_config,
            ctx.batch_rows,
            ctx.up_output_features,
            state.sm_count,
            state.selected_kernel_occupancy);
        state.selected_kernel_config_name = decode_swiglu_kernel_config_name(best_config);
    } catch (const std::exception&) {
        state.selected_kernel_config = static_cast<int>(kDefaultDecodeSwigluKernelConfig);
        state.selected_kernel_occupancy = 0;
        state.selected_threadblock_count = 0;
        state.selected_kernel_config_name = decode_swiglu_kernel_config_name(kDefaultDecodeSwigluKernelConfig);
    }
}

template <typename T>
void run_decode_swiglu_fused_for_dtype(
    const FusedGateUpActivationOpContext& ctx,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    const FusedGateUpActivationOpState& state,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream) {
    const auto config_id = static_cast<DecodeSwigluKernelConfigId>(state.selected_kernel_config);
    launch_decode_swiglu_fused_kernel<T>(
        config_id,
        weight_tensor,
        bias_tensor,
        ctx.batch_rows,
        ctx.up_output_features,
        ctx.input_features,
        state,
        static_cast<const T*>(input.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        stream);
}

class TrtLlmDecodeSwigluOp final : public FusedGateUpActivationOp {
public:
    std::string impl_id() const override { return "trtllm_decode_swiglu"; }

    bool supports(const FusedGateUpActivationOpContext& ctx) const override {
        if (ctx.batch_rows <= 0 || ctx.input_features <= 0 || ctx.gate_output_features <= 0 ||
            ctx.up_output_features <= 0) {
            return false;
        }
        if (ctx.gate_output_features != ctx.up_output_features) {
            return false;
        }
        if (ctx.input_dtype != ctx.output_dtype || ctx.input_dtype != ctx.weight_dtype) {
            return false;
        }
        if (ctx.input_dtype != DType::Float16 && ctx.input_dtype != DType::BFloat16) {
            return false;
        }
        if ((ctx.up_output_features % 64) != 0 || (ctx.input_features % 64) != 0) {
            return false;
        }

        try {
            const int sm = sm_version_for_device(ctx.device_id);
            // This path instantiates TensorRT-LLM's sm80 fused-MoE kernel. On
            // RTX 3060 (SM86) it is fast but not numerically equivalent to the
            // two-stage BF16 gate/up + SiLU path, which breaks greedy decode.
            return decode_swiglu_device_supported(sm);
        } catch (const std::exception&) {
            return false;
        }
    }

    void prepare(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        FusedGateUpActivationOpState& state) override
    {
        state = FusedGateUpActivationOpState{};
        state.initialized = true;

        const auto& weight_shape = weight.shape();
        if (weight_shape.size() != 2 ||
            weight_shape[0] != ctx.gate_output_features + ctx.up_output_features ||
            weight_shape[1] != ctx.input_features)
        {
            state.unavailable_reason = "decode fused SwiGLU weight shape does not match fused gate/up contract";
            return;
        }
        if (bias != nullptr) {
            const auto& bias_shape = bias->shape();
            if (bias_shape.size() != 1 ||
                bias_shape[0] != ctx.gate_output_features + ctx.up_output_features)
            {
                state.unavailable_reason = "decode fused SwiGLU bias shape does not match fused gate/up contract";
                return;
            }
        }
        if (!supports(ctx)) {
            state.unavailable_reason = "decode fused SwiGLU is unsupported for the current device or dtype";
            return;
        }

        try {
            switch (ctx.weight_dtype) {
                case DType::Float16:
                    prepare_decode_swiglu_state_for_dtype<half>(ctx, weight, bias, state);
                    break;
                case DType::BFloat16:
                    prepare_decode_swiglu_state_for_dtype<__nv_bfloat16>(ctx, weight, bias, state);
                    break;
                default:
                    state.unavailable_reason = "decode fused SwiGLU requires FP16/BF16 weights";
                    break;
            }
        } catch (const std::exception& ex) {
            state.available = false;
            state.unavailable_reason = ex.what();
        }
    }

    void run(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        const FusedGateUpActivationOpState& state,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream) override
    {
        if (!state.available) {
            throw InvalidRequestError(
                "decode fused SwiGLU fast path is unavailable: " + state.unavailable_reason);
        }

        switch (ctx.input_dtype) {
            case DType::Float16:
                run_decode_swiglu_fused_for_dtype<half>(ctx, weight, bias, state, input, output, stream);
                return;
            case DType::BFloat16:
                run_decode_swiglu_fused_for_dtype<__nv_bfloat16>(ctx, weight, bias, state, input, output, stream);
                return;
            default:
                throw InvalidRequestError("decode fused SwiGLU requires FP16/BF16 activations");
        }
    }
};

// ============================================================================
// CUTLASS-based Prefill Fused GateUp + SiLU + Mul operator
// Reuses TRT-LLM's Fused_Moe_Kernel_sm80 with num_experts=1 and prefill tile configs
// ============================================================================

enum class PrefillSwigluKernelConfigId : uint8_t {
    Tile64x128x64Stage3 = 0,
    Tile128x128x64Stage3,
    Tile128x256x64Stage3,
    Tile256x128x64Stage3,
    Tile32x128x64Stage2,
    Tile64x128x64Stage2,
    Tile128x128x64Stage2,
    Tile128x256x64Stage2,
};

struct PrefillSwigluKernelCandidate {
    PrefillSwigluKernelConfigId id;
    const char* name;
};

constexpr PrefillSwigluKernelConfigId kDefaultPrefillSwigluKernelConfig =
    PrefillSwigluKernelConfigId::Tile128x128x64Stage3;

constexpr std::array<PrefillSwigluKernelCandidate, 8> kPrefillSwigluKernelCandidates = {{
    {PrefillSwigluKernelConfigId::Tile64x128x64Stage3, "64x128x64_s3"},
    {PrefillSwigluKernelConfigId::Tile128x128x64Stage3, "128x128x64_s3"},
    {PrefillSwigluKernelConfigId::Tile128x256x64Stage3, "128x256x64_s3"},
    {PrefillSwigluKernelConfigId::Tile256x128x64Stage3, "256x128x64_s3"},
    {PrefillSwigluKernelConfigId::Tile32x128x64Stage2, "32x128x64_s2"},
    {PrefillSwigluKernelConfigId::Tile64x128x64Stage2, "64x128x64_s2"},
    {PrefillSwigluKernelConfigId::Tile128x128x64Stage2, "128x128x64_s2"},
    {PrefillSwigluKernelConfigId::Tile128x256x64Stage2, "128x256x64_s2"},
}};

const char* prefill_swiglu_kernel_config_name(PrefillSwigluKernelConfigId config_id) {
    for (const auto& candidate : kPrefillSwigluKernelCandidates) {
        if (candidate.id == config_id) {
            return candidate.name;
        }
    }
    return "unknown";
}

template <typename T, int MaxTileM, int TileN, int TileK, int Stages>
using PrefillSwigluGemmType = fused_moe::Fused_Moe_Kernel_sm80<
    typename CutlassScalarType<T>::type,
    typename CutlassScalarType<T>::type,
    typename CutlassScalarType<T>::type,
    MaxTileM,
    TileN,
    TileK,
    Stages,
    fused_moe::EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>(true)>;

template <typename T, int MaxTileM, int TileN, int TileK, int Stages>
int query_prefill_swiglu_occupancy_for_config() {
    using GemmType = PrefillSwigluGemmType<T, MaxTileM, TileN, TileK, Stages>;
    return fused_moe::fused_gemm_maximum_active_blocks<GemmType>();
}

template <typename T>
int query_prefill_swiglu_occupancy(PrefillSwigluKernelConfigId config_id) {
    switch (config_id) {
        case PrefillSwigluKernelConfigId::Tile64x128x64Stage3:
            return query_prefill_swiglu_occupancy_for_config<T, 64, 128, 64, 3>();
        case PrefillSwigluKernelConfigId::Tile128x128x64Stage3:
            return query_prefill_swiglu_occupancy_for_config<T, 128, 128, 64, 3>();
        case PrefillSwigluKernelConfigId::Tile128x256x64Stage3:
            return query_prefill_swiglu_occupancy_for_config<T, 128, 256, 64, 3>();
        case PrefillSwigluKernelConfigId::Tile256x128x64Stage3:
            return query_prefill_swiglu_occupancy_for_config<T, 256, 128, 64, 3>();
        case PrefillSwigluKernelConfigId::Tile32x128x64Stage2:
            return query_prefill_swiglu_occupancy_for_config<T, 32, 128, 64, 2>();
        case PrefillSwigluKernelConfigId::Tile64x128x64Stage2:
            return query_prefill_swiglu_occupancy_for_config<T, 64, 128, 64, 2>();
        case PrefillSwigluKernelConfigId::Tile128x128x64Stage2:
            return query_prefill_swiglu_occupancy_for_config<T, 128, 128, 64, 2>();
        case PrefillSwigluKernelConfigId::Tile128x256x64Stage2:
            return query_prefill_swiglu_occupancy_for_config<T, 128, 256, 64, 2>();
    }
    return -1;
}

template <typename T, int MaxTileM, int TileN, int TileK, int Stages>
void launch_prefill_swiglu_fused_kernel_config(
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpActivationOpState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream)
{
    using CutlassT = typename CutlassScalarType<T>::type;
    using GemmType = PrefillSwigluGemmType<T, MaxTileM, TileN, TileK, Stages>;
    using Arguments = typename GemmType::Arguments;

    const int occupancy = state.selected_kernel_occupancy > 0
        ? state.selected_kernel_occupancy
        : std::min(2, fused_moe::fused_gemm_maximum_active_blocks<GemmType>());

    const int threadblock_count = state.selected_threadblock_count > 0
        ? state.selected_threadblock_count
        : state.sm_count * occupancy;

    if constexpr (GemmType::kSmemSize >= (48 << 10)) {
        CUDA_CHECK_THROW(
            cudaFuncSetAttribute(
                fused_moe::run_global<GemmType>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                GemmType::kSmemSize),
            "FusedGateUpActivationOp: failed to set fused prefill SwiGLU dynamic shared memory size");
    }

    Arguments args{
        {const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(input_ptr)),
         const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(weight_tensor.data_ptr())),
         bias_tensor ? const_cast<CutlassT*>(reinterpret_cast<const CutlassT*>(bias_tensor->data_ptr())) : nullptr,
         reinterpret_cast<CutlassT*>(output_ptr),
         static_cast<const int64_t*>(state.expert_offsets_device_ptr),
         static_cast<int>(output_features),
         static_cast<int>(input_features),
         1,
         true},
        1,
        threadblock_count};
    auto params = GemmType::to_underlying_arguments(args);
    fused_moe::run_global<GemmType>
        <<<dim3(threadblock_count, 1, 1), dim3(GemmType::kThreadCount), GemmType::kSmemSize, stream>>>(params);
    CUDA_CHECK_THROW(cudaGetLastError(), "FusedGateUpActivationOp: failed to launch fused prefill SwiGLU kernel");
}

template <typename T>
void launch_prefill_swiglu_fused_kernel(
    PrefillSwigluKernelConfigId config_id,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    int64_t batch_rows,
    int64_t output_features,
    int64_t input_features,
    const FusedGateUpActivationOpState& state,
    const T* input_ptr,
    T* output_ptr,
    cudaStream_t stream)
{
    switch (config_id) {
        case PrefillSwigluKernelConfigId::Tile64x128x64Stage3:
            launch_prefill_swiglu_fused_kernel_config<T, 64, 128, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile128x128x64Stage3:
            launch_prefill_swiglu_fused_kernel_config<T, 128, 128, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile128x256x64Stage3:
            launch_prefill_swiglu_fused_kernel_config<T, 128, 256, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile256x128x64Stage3:
            launch_prefill_swiglu_fused_kernel_config<T, 256, 128, 64, 3>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile32x128x64Stage2:
            launch_prefill_swiglu_fused_kernel_config<T, 32, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile64x128x64Stage2:
            launch_prefill_swiglu_fused_kernel_config<T, 64, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile128x128x64Stage2:
            launch_prefill_swiglu_fused_kernel_config<T, 128, 128, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
        case PrefillSwigluKernelConfigId::Tile128x256x64Stage2:
            launch_prefill_swiglu_fused_kernel_config<T, 128, 256, 64, 2>(
                weight_tensor, bias_tensor, batch_rows, output_features, input_features,
                state, input_ptr, output_ptr, stream);
            return;
    }
}

template <typename T>
void prepare_prefill_swiglu_state_for_dtype(
    const FusedGateUpActivationOpContext& ctx,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    FusedGateUpActivationOpState& state)
{
    const int sm = sm_version_for_device(ctx.device_id);
    if (sm < 80) {
        state.unavailable_reason =
            "prefill fused SwiGLU path requires SM80+";
        return;
    }

    if ((ctx.up_output_features % 64) != 0 || (ctx.input_features % 64) != 0) {
        state.unavailable_reason =
            "prefill fused SwiGLU shape is unsupported by TRT-LLM SM80 fused MoE kernel";
        return;
    }

    CUDA_CHECK_THROW(cudaSetDevice(ctx.device_id), "FusedGateUpActivationOp: failed to set CUDA device");

    const std::string buffer_name = ctx.layer_prefix + ".prefill_swiglu_fused.total_tokens." +
        std::to_string(ctx.batch_rows);
    state.expert_offsets_device_ptr =
        StaticBufferManager::get_cache_buf(buffer_name, sizeof(int64_t), ctx.device_id);

    const std::array<int64_t, 1> expert_offsets = {ctx.batch_rows};
    CUDA_CHECK_THROW(
        cudaMemcpy(
            state.expert_offsets_device_ptr,
            expert_offsets.data(),
            sizeof(expert_offsets),
            cudaMemcpyHostToDevice),
        "FusedGateUpActivationOp: failed to initialize prefill expert offsets");

    state.sm_count = sm_count_for_device(ctx.device_id);
    state.available = true;

    auto config_throughput_score = [](PrefillSwigluKernelConfigId id) -> int {
        switch (id) {
            case PrefillSwigluKernelConfigId::Tile64x128x64Stage3:   return 64 * 128 * 3;
            case PrefillSwigluKernelConfigId::Tile128x128x64Stage3:  return 128 * 128 * 3;
            case PrefillSwigluKernelConfigId::Tile128x256x64Stage3:  return 128 * 256 * 3;
            case PrefillSwigluKernelConfigId::Tile256x128x64Stage3:  return 256 * 128 * 3;
            case PrefillSwigluKernelConfigId::Tile32x128x64Stage2:   return 32 * 128 * 2;
            case PrefillSwigluKernelConfigId::Tile64x128x64Stage2:   return 64 * 128 * 2;
            case PrefillSwigluKernelConfigId::Tile128x128x64Stage2:  return 128 * 128 * 2;
            case PrefillSwigluKernelConfigId::Tile128x256x64Stage2:  return 128 * 256 * 2;
        }
        return 0;
    };

    PrefillSwigluKernelConfigId best_config = kDefaultPrefillSwigluKernelConfig;
    int best_occupancy = -1;
    int best_score = 0;

    for (const auto& candidate : kPrefillSwigluKernelCandidates) {
        int occupancy = query_prefill_swiglu_occupancy<T>(candidate.id);
        if (occupancy <= 0) continue;
        int score = occupancy * config_throughput_score(candidate.id);
        if (best_occupancy <= 0 || score > best_score) {
            best_occupancy = occupancy;
            best_score = score;
            best_config = candidate.id;
        }
    }

    if (best_occupancy <= 0) {
        state.available = false;
        state.unavailable_reason = "prefill fused SwiGLU no kernel config has legal occupancy on this device";
        return;
    }

    state.selected_kernel_config = static_cast<int>(best_config);
    state.selected_kernel_occupancy = std::min(2, best_occupancy);
    state.selected_threadblock_count = state.sm_count * state.selected_kernel_occupancy;
    state.selected_kernel_config_name = prefill_swiglu_kernel_config_name(best_config);
}

template <typename T>
void run_prefill_swiglu_fused_for_dtype(
    const FusedGateUpActivationOpContext& ctx,
    const Tensor& weight_tensor,
    const Tensor* bias_tensor,
    const FusedGateUpActivationOpState& state,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    const auto config_id = static_cast<PrefillSwigluKernelConfigId>(state.selected_kernel_config);
    launch_prefill_swiglu_fused_kernel<T>(
        config_id,
        weight_tensor,
        bias_tensor,
        ctx.batch_rows,
        ctx.up_output_features,
        ctx.input_features,
        state,
        static_cast<const T*>(input.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        stream);
}

bool prefill_swiglu_fusion_enabled() {
    const char* env = std::getenv("EDGE_FM_PREFILL_SWIGLU_FUSION");
    if (env == nullptr || *env == '\0') {
        return false;
    }
    return std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "False";
}

class CutlassPrefillSwigluOp final : public FusedGateUpActivationOp {
public:
    std::string impl_id() const override { return "cutlass_prefill_swiglu"; }

    bool supports(const FusedGateUpActivationOpContext& ctx) const override {
        if (!prefill_swiglu_fusion_enabled()) {
            return false;
        }
        if (ctx.batch_rows < 64) {
            return false;
        }
        if (ctx.input_features <= 0 || ctx.gate_output_features <= 0 ||
            ctx.up_output_features <= 0) {
            return false;
        }
        if (ctx.gate_output_features != ctx.up_output_features) {
            return false;
        }
        if (ctx.input_dtype != ctx.output_dtype || ctx.input_dtype != ctx.weight_dtype) {
            return false;
        }
        if (ctx.input_dtype != DType::Float16 && ctx.input_dtype != DType::BFloat16) {
            return false;
        }
        if ((ctx.up_output_features % 64) != 0 || (ctx.input_features % 64) != 0) {
            return false;
        }
        try {
            const int sm = sm_version_for_device(ctx.device_id);
            return sm >= 80;
        } catch (const std::exception&) {
            return false;
        }
    }

    void prepare(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        FusedGateUpActivationOpState& state) override
    {
        state = FusedGateUpActivationOpState{};
        state.initialized = true;

        const auto& weight_shape = weight.shape();
        if (weight_shape.size() != 2 ||
            static_cast<int64_t>(weight_shape[0]) != ctx.gate_output_features + ctx.up_output_features ||
            static_cast<int64_t>(weight_shape[1]) != ctx.input_features)
        {
            state.unavailable_reason =
                "prefill fused SwiGLU weight shape does not match fused gate/up contract";
            return;
        }
        if (bias != nullptr) {
            const auto& bias_shape = bias->shape();
            if (bias_shape.size() != 1 ||
                bias_shape[0] != ctx.gate_output_features + ctx.up_output_features)
            {
                state.unavailable_reason = "prefill fused SwiGLU bias shape does not match fused gate/up contract";
                return;
            }
        }
        if (!supports(ctx)) {
            state.unavailable_reason =
                "prefill fused SwiGLU is unsupported for the current device/dtype/shape";
            return;
        }

        try {
            switch (ctx.weight_dtype) {
                case DType::Float16:
                    prepare_prefill_swiglu_state_for_dtype<half>(ctx, weight, bias, state);
                    break;
                case DType::BFloat16:
                    prepare_prefill_swiglu_state_for_dtype<__nv_bfloat16>(ctx, weight, bias, state);
                    break;
                default:
                    state.unavailable_reason = "prefill fused SwiGLU requires FP16/BF16 weights";
                    break;
            }
        } catch (const std::exception& ex) {
            state.available = false;
            state.unavailable_reason = ex.what();
        }
    }

    void run(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        const FusedGateUpActivationOpState& state,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream) override
    {
        if (!state.available) {
            throw InvalidRequestError(
                "prefill fused SwiGLU fast path is unavailable: " + state.unavailable_reason);
        }

        switch (ctx.input_dtype) {
            case DType::Float16:
                run_prefill_swiglu_fused_for_dtype<half>(ctx, weight, bias, state, input, output, stream);
                return;
            case DType::BFloat16:
                run_prefill_swiglu_fused_for_dtype<__nv_bfloat16>(ctx, weight, bias, state, input, output, stream);
                return;
            default:
                throw InvalidRequestError(
                    "prefill fused SwiGLU requires FP16/BF16 activations");
        }
    }
};

} // namespace

FusedGateUpActivationOpRegistry::FusedGateUpActivationOpRegistry() {
    impls_.emplace_back(std::make_unique<TrtLlmDecodeSwigluOp>());
    impls_.emplace_back(std::make_unique<CutlassPrefillSwigluOp>());
}

FusedGateUpActivationOpRegistry& FusedGateUpActivationOpRegistry::instance() {
    static FusedGateUpActivationOpRegistry registry;
    return registry;
}

FusedGateUpActivationOp* FusedGateUpActivationOpRegistry::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
}

FusedGateUpActivationOp* FusedGateUpActivationOpRegistry::default_impl(
    const FusedGateUpActivationOpContext& ctx) const
{
    for (const auto& impl : impls_) {
        if (impl->supports(ctx)) {
            return impl.get();
        }
    }
    return nullptr;
}

} // namespace edge_fm
