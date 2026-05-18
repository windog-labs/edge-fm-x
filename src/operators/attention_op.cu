#include "operators/attention_op.h"

#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#if defined(EDGE_FM_ENABLE_TRT_PLUGIN_OPS) && EDGE_FM_ENABLE_TRT_PLUGIN_OPS
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/fmhaParams_v2.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include <NvInferRuntime.h>
#endif
using namespace flashinfer;

namespace edge_fm {
namespace {

constexpr std::array<uint32_t, 7> kSupportedDecodeGroupSizes = {1U, 2U, 3U, 4U, 6U, 7U, 8U};

struct DecodeTunedPolicy {
    bool split_kv = false;
    uint32_t kv_chunk_size = 0;
    uint32_t bdz = 3;
};

struct DecodeTunedRuntimeConfig {
    uint32_t short_seq_bdz = 3;
    uint32_t long_seq_bdz = 4;
    uint32_t long_seq_threshold = 512;
    uint32_t no_split_kv_threshold = 192;
    uint32_t min_chunk_size = 32;
    uint32_t chunk_alignment = 32;
    std::array<uint32_t, 4> chunk_candidates = {32U, 64U, 128U, 256U};
};

struct PrefillTunedRuntimeConfig {
    bool enabled = false;
    uint32_t cta_tile_q = 0;
    uint32_t max_mma_kv_cap = 0;
    uint32_t short_qo_len_threshold = 0;
    uint32_t short_cta_tile_q = 0;
    uint32_t long_cta_tile_q = 0;
};

template <
    uint32_t NUM_QO_HEADS,
    uint32_t NUM_KV_HEADS,
    uint32_t HEAD_DIM,
    uint32_t VEC_SIZE,
    uint32_t TILE_SIZE_PER_BDX,
    uint32_t NUM_STAGES_SMEM,
    uint32_t SHORT_SEQ_BDZ,
    uint32_t LONG_SEQ_BDZ,
    uint32_t LONG_SEQ_THRESHOLD,
    uint32_t NO_SPLIT_KV_THRESHOLD,
    uint32_t MIN_CHUNK_SIZE,
    uint32_t CHUNK_ALIGNMENT,
    uint32_t CHUNK0,
    uint32_t CHUNK1,
    uint32_t CHUNK2,
    uint32_t CHUNK3>
struct DecodeTunedShape {
    static_assert(NUM_KV_HEADS > 0U && NUM_QO_HEADS % NUM_KV_HEADS == 0U, "Invalid decode tuned GQA shape");
    static_assert(VEC_SIZE > 0U && HEAD_DIM % VEC_SIZE == 0U, "Invalid decode tuned vectorization");

    static constexpr uint32_t kNumQoHeads = NUM_QO_HEADS;
    static constexpr uint32_t kNumKvHeads = NUM_KV_HEADS;
    static constexpr uint32_t kHeadDim = HEAD_DIM;
    static constexpr uint32_t kVecSize = VEC_SIZE;
    static constexpr uint32_t kGroupSize = NUM_QO_HEADS / NUM_KV_HEADS;
    static constexpr uint32_t kBdx = HEAD_DIM / VEC_SIZE;
    static constexpr uint32_t kTileSizePerBdx = TILE_SIZE_PER_BDX;
    static constexpr uint32_t kNumStagesSmem = NUM_STAGES_SMEM;
    static constexpr uint32_t kShortSeqBdz = SHORT_SEQ_BDZ;
    static constexpr uint32_t kLongSeqBdz = LONG_SEQ_BDZ;
    static constexpr uint32_t kLongSeqThreshold = LONG_SEQ_THRESHOLD;
    static constexpr uint32_t kNoSplitKvThreshold = NO_SPLIT_KV_THRESHOLD;
    static constexpr uint32_t kMinChunkSize = MIN_CHUNK_SIZE;
    static constexpr uint32_t kChunkAlignment = CHUNK_ALIGNMENT;

    static constexpr bool matches(const AttentionOpContext& ctx) {
        return ctx.num_qo_heads == kNumQoHeads &&
            ctx.num_kv_heads == kNumKvHeads &&
            ctx.head_dim == kHeadDim;
    }

    static constexpr std::array<uint32_t, 4> chunk_candidates() {
        return {CHUNK0, CHUNK1, CHUNK2, CHUNK3};
    }
};

using Qwen1P5DecodeTunedShape = DecodeTunedShape<
    12U, 2U, 128U, 8U, 1U, 2U,
    3U, 4U, 512U, 192U, 32U, 32U,
    32U, 64U, 128U, 256U>;

using Qwen0P5DecodeTunedShape = DecodeTunedShape<
    14U, 2U, 64U, 8U, 1U, 2U,
    3U, 4U, 1536U, 192U, 64U, 64U,
    64U, 128U, 256U, 512U>;

using Qwen3BDecodeTunedShape = DecodeTunedShape<
    16U, 2U, 128U, 8U, 1U, 2U,
    3U, 4U, 1024U, 192U, 32U, 32U,
    32U, 64U, 128U, 256U>;

using Qwen7BDecodeTunedShape = DecodeTunedShape<
    28U, 4U, 128U, 8U, 1U, 2U,
    3U, 4U, 1024U, 192U, 32U, 32U,
    32U, 64U, 128U, 256U>;

uint32_t round_up_u32(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

uint32_t get_u32_param_or(const nlohmann::json& json, const char* key, uint32_t default_value) {
    if (!json.is_object() || !json.contains(key)) {
        return default_value;
    }
    return json.at(key).get<uint32_t>();
}

bool get_bool_param_or(const nlohmann::json& json, const char* key, bool default_value) {
    if (!json.is_object() || !json.contains(key)) {
        return default_value;
    }
    return json.at(key).get<bool>();
}

bool is_supported_prefill_cta_tile_q(uint32_t value) {
    return value == 16U || value == 64U || value == 128U;
}

uint32_t validate_prefill_cta_tile_q(uint32_t value, const char* key) {
    check<ConfigurationError>(
        is_supported_prefill_cta_tile_q(value),
        std::string("flashinfer_attention prefill ") + key + " must be one of {16, 64, 128}");
    return value;
}

uint32_t validate_prefill_max_mma_kv_cap(uint32_t value) {
    check<ConfigurationError>(
        value == 0U || value == 1U || value == 2U || value == 4U || value == 8U,
        "flashinfer_attention prefill max_mma_kv_cap must be one of {0, 1, 2, 4, 8}");
    return value;
}

PrefillTunedRuntimeConfig resolve_prefill_tuned_runtime_config(const AttentionOpContext& ctx) {
    PrefillTunedRuntimeConfig config;
    if (!ctx.impl_params.is_object()) {
        return config;
    }

    config.cta_tile_q = get_u32_param_or(ctx.impl_params, "prefill_cta_tile_q", 0U);
    config.max_mma_kv_cap = validate_prefill_max_mma_kv_cap(
        get_u32_param_or(ctx.impl_params, "prefill_max_mma_kv_cap", 0U));
    config.short_qo_len_threshold =
        get_u32_param_or(ctx.impl_params, "prefill_short_qo_len_threshold", 0U);
    config.short_cta_tile_q =
        get_u32_param_or(ctx.impl_params, "prefill_short_cta_tile_q", 0U);
    config.long_cta_tile_q =
        get_u32_param_or(ctx.impl_params, "prefill_long_cta_tile_q", 0U);

    if (config.cta_tile_q != 0U) {
        config.cta_tile_q = validate_prefill_cta_tile_q(config.cta_tile_q, "cta_tile_q");
        config.enabled = true;
    }
    if (config.short_cta_tile_q != 0U) {
        config.short_cta_tile_q =
            validate_prefill_cta_tile_q(config.short_cta_tile_q, "short_cta_tile_q");
        config.enabled = true;
    }
    if (config.long_cta_tile_q != 0U) {
        config.long_cta_tile_q =
            validate_prefill_cta_tile_q(config.long_cta_tile_q, "long_cta_tile_q");
        config.enabled = true;
    }

    return config;
}

uint32_t choose_prefill_cta_tile_q(
    const AttentionOpContext& ctx,
    uint32_t qo_len,
    uint32_t head_dim)
{
    const PrefillTunedRuntimeConfig config = resolve_prefill_tuned_runtime_config(ctx);
    if (config.enabled) {
        if (config.cta_tile_q != 0U) {
            return config.cta_tile_q;
        }
        if (config.short_qo_len_threshold != 0U && qo_len <= config.short_qo_len_threshold) {
            if (config.short_cta_tile_q != 0U) {
                return config.short_cta_tile_q;
            }
        }
        if (config.long_cta_tile_q != 0U) {
            return config.long_cta_tile_q;
        }
        if (config.short_cta_tile_q != 0U) {
            return config.short_cta_tile_q;
        }
    }
    return FA2DetermineCtaTileQ(
        static_cast<int64_t>(qo_len) * static_cast<int64_t>(ctx.num_qo_heads / ctx.num_kv_heads),
        head_dim);
}

std::array<uint32_t, 4> get_chunk_candidates_or(
    const nlohmann::json& json,
    std::array<uint32_t, 4> default_value)
{
    if (!json.is_object() || !json.contains("chunk_candidates")) {
        return default_value;
    }
    const auto& value = json.at("chunk_candidates");
    check<ConfigurationError>(
        value.is_array() && value.size() == default_value.size(),
        "flashinfer_attention_decode_sm80_tuned expects impl_params.chunk_candidates to have exactly 4 entries");
    for (size_t i = 0; i < default_value.size(); ++i) {
        default_value[i] = value.at(i).get<uint32_t>();
    }
    return default_value;
}

std::array<uint32_t, 4> normalize_chunk_candidates(
    std::array<uint32_t, 4> chunk_candidates,
    uint32_t min_chunk_size,
    uint32_t chunk_alignment)
{
    uint32_t previous = round_up_u32(std::max(chunk_candidates[0], min_chunk_size), chunk_alignment);
    chunk_candidates[0] = previous;
    for (size_t i = 1; i < chunk_candidates.size(); ++i) {
        previous = round_up_u32(std::max(chunk_candidates[i], previous), chunk_alignment);
        chunk_candidates[i] = previous;
    }
    return chunk_candidates;
}

DecodeTunedRuntimeConfig normalize_decode_tuned_runtime_config(DecodeTunedRuntimeConfig config) {
    check<ConfigurationError>(config.short_seq_bdz >= 3U && config.short_seq_bdz <= 4U,
                              "flashinfer_attention_decode_sm80_tuned short_seq_bdz must be in [3, 4]");
    check<ConfigurationError>(config.long_seq_bdz >= 3U && config.long_seq_bdz <= 4U,
                              "flashinfer_attention_decode_sm80_tuned long_seq_bdz must be in [3, 4]");
    check<ConfigurationError>(config.min_chunk_size > 0U,
                              "flashinfer_attention_decode_sm80_tuned min_chunk_size must be > 0");
    check<ConfigurationError>(config.chunk_alignment > 0U,
                              "flashinfer_attention_decode_sm80_tuned chunk_alignment must be > 0");
    config.min_chunk_size = round_up_u32(config.min_chunk_size, config.chunk_alignment);
    config.chunk_candidates = normalize_chunk_candidates(
        config.chunk_candidates, config.min_chunk_size, config.chunk_alignment);
    return config;
}

template <typename Shape>
DecodeTunedRuntimeConfig resolve_decode_tuned_runtime_config(const AttentionOpContext& ctx) {
    DecodeTunedRuntimeConfig config;
    config.short_seq_bdz = Shape::kShortSeqBdz;
    config.long_seq_bdz = Shape::kLongSeqBdz;
    config.long_seq_threshold = Shape::kLongSeqThreshold;
    config.no_split_kv_threshold = Shape::kNoSplitKvThreshold;
    config.min_chunk_size = Shape::kMinChunkSize;
    config.chunk_alignment = Shape::kChunkAlignment;
    config.chunk_candidates = Shape::chunk_candidates();

    if (ctx.impl_params.is_object()) {
        config.short_seq_bdz = get_u32_param_or(ctx.impl_params, "short_seq_bdz", config.short_seq_bdz);
        config.long_seq_bdz = get_u32_param_or(ctx.impl_params, "long_seq_bdz", config.long_seq_bdz);
        config.long_seq_threshold =
            get_u32_param_or(ctx.impl_params, "long_seq_threshold", config.long_seq_threshold);
        config.no_split_kv_threshold =
            get_u32_param_or(ctx.impl_params, "no_split_kv_threshold", config.no_split_kv_threshold);
        config.min_chunk_size = get_u32_param_or(ctx.impl_params, "min_chunk_size", config.min_chunk_size);
        config.chunk_alignment = get_u32_param_or(ctx.impl_params, "chunk_alignment", config.chunk_alignment);
        config.chunk_candidates = get_chunk_candidates_or(ctx.impl_params, config.chunk_candidates);
    }

    return normalize_decode_tuned_runtime_config(config);
}

bool supports_flashinfer_decode_group_size(uint32_t group_size) {
    return std::find(kSupportedDecodeGroupSizes.begin(), kSupportedDecodeGroupSizes.end(), group_size) !=
        kSupportedDecodeGroupSizes.end();
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool can_use_flashinfer_decode_fast_path(const AttentionOpContext& ctx) {
    return ctx.num_kv_heads > 0 &&
        (ctx.num_qo_heads % ctx.num_kv_heads == 0) &&
        supports_flashinfer_decode_group_size(ctx.num_qo_heads / ctx.num_kv_heads);
}

__global__ void preapply_llama_rope_bf16_pair_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t input_stride_n,
    uint32_t input_stride_h,
    uint32_t output_stride_n,
    uint32_t output_stride_h,
    uint32_t pos_offset,
    float rope_rcp_scale,
    float rope_theta)
{
    const uint32_t half_dim = head_dim / 2U;
    const size_t total_pairs =
        static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) * static_cast<size_t>(half_dim);
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    const uint32_t dim = static_cast<uint32_t>(idx % half_dim);
    const size_t tmp = idx / half_dim;
    const uint32_t head = static_cast<uint32_t>(tmp % num_heads);
    const uint32_t token = static_cast<uint32_t>(tmp / num_heads);
    const uint32_t rope_pos = pos_offset + token;
    const size_t input_base =
        static_cast<size_t>(token) * static_cast<size_t>(input_stride_n) +
        static_cast<size_t>(head) * static_cast<size_t>(input_stride_h);
    const size_t output_base =
        static_cast<size_t>(token) * static_cast<size_t>(output_stride_n) +
        static_cast<size_t>(head) * static_cast<size_t>(output_stride_h);

    const float x0 = __bfloat162float(input[input_base + dim]);
    const float x1 = __bfloat162float(input[input_base + dim + half_dim]);
    const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
    const float angle = static_cast<float>(rope_pos) * rope_rcp_scale / powf(rope_theta, exponent);
    float sin_value;
    float cos_value;
    __sincosf(angle, &sin_value, &cos_value);

    output[output_base + dim] = __float2bfloat16(x0 * cos_value - x1 * sin_value);
    output[output_base + dim + half_dim] = __float2bfloat16(x1 * cos_value + x0 * sin_value);
}

void launch_preapply_llama_rope_bf16(
    const AttentionOpContext& ctx,
    const Tensor& input,
    Tensor& output,
    uint32_t num_heads,
    uint32_t input_stride_n,
    uint32_t input_stride_h,
    uint32_t output_stride_n,
    uint32_t output_stride_h,
    uint32_t pos_offset,
    cudaStream_t stream)
{
    const auto& shape = input.shape();
    const uint32_t seq_len = static_cast<uint32_t>(shape[0]);
    const uint32_t head_dim = ctx.head_dim;
    check<ConfigurationError>(
        head_dim % 2U == 0U,
        "flashinfer_attention_prefill_prerotate expects an even head_dim");
    constexpr int block = 256;
    const size_t total_pairs =
        static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim / 2U);
    const int grid = static_cast<int>((total_pairs + block - 1) / block);
    preapply_llama_rope_bf16_pair_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        seq_len,
        num_heads,
        head_dim,
        input_stride_n,
        input_stride_h,
        output_stride_n,
        output_stride_h,
        pos_offset,
        1.0f / ctx.rope_scale,
        ctx.rope_theta);
    CUDA_CHECK_THROW(cudaGetLastError(), "flashinfer_attention_prefill_prerotate RoPE kernel failed");
}

#if defined(EDGE_FM_ENABLE_TRT_PLUGIN_OPS) && EDGE_FM_ENABLE_TRT_PLUGIN_OPS

constexpr const char* kTrtFmhaPluginEnv = "EDGE_FM_PREFILL_TRT_FMHA_PLUGIN";
constexpr const char* kTrtFmhaPluginAllowBf16Fp16CastEnv =
    "EDGE_FM_PREFILL_TRT_FMHA_PLUGIN_ALLOW_BF16_FP16_CAST";
constexpr const char* kTrtFmhaPluginContiguousQKvEnv =
    "EDGE_FM_PREFILL_TRT_FMHA_PLUGIN_CONTIG_Q_KV";

int32_t sm_version() {
    const auto [major, minor] = GetCudaComputeCapability();
    return major * 10 + minor;
}

bool trt_context_fmha_plugin_enabled(const AttentionOpContext& ctx) {
    return env_flag_enabled(kTrtFmhaPluginEnv) ||
        get_bool_param_or(ctx.impl_params, "enabled", false);
}

bool trt_context_fmha_plugin_allow_bf16_fp16_cast(const AttentionOpContext& ctx) {
    return env_flag_enabled(kTrtFmhaPluginAllowBf16Fp16CastEnv) ||
        get_bool_param_or(ctx.impl_params, "allow_bf16_fp16_cast", false);
}

bool trt_context_fmha_plugin_use_contiguous_q_kv(const AttentionOpContext& ctx) {
    return env_flag_enabled(kTrtFmhaPluginContiguousQKvEnv) ||
        get_bool_param_or(ctx.impl_params, "contiguous_q_kv", false);
}

bool trt_context_fmha_plugin_dtype_supported(const AttentionOpContext& ctx) {
    if (ctx.dtype == DType::Float16) {
        return true;
    }
    if (ctx.dtype == DType::BFloat16) {
        return trt_context_fmha_plugin_allow_bf16_fp16_cast(ctx);
    }
    return false;
}

template <typename T>
__device__ __forceinline__ half to_half_device(T value);

template <>
__device__ __forceinline__ half to_half_device<half>(half value) {
    return value;
}

template <>
__device__ __forceinline__ half to_half_device<__nv_bfloat16>(__nv_bfloat16 value) {
    return __float2half(__bfloat162float(value));
}

template <typename T>
__device__ __forceinline__ float to_float_device(T value);

template <>
__device__ __forceinline__ float to_float_device<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float_device<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__global__ void pack_qkv_rope_and_seqlens_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    half* __restrict__ qkv_half,
    float* __restrict__ cos_sin,
    int32_t* __restrict__ cu_seqlens,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t k_stride_n,
    uint32_t k_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float rope_rcp_scale,
    float rope_theta)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t total_heads = num_q_heads + 2U * num_kv_heads;
    const size_t qkv_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(total_heads) * static_cast<size_t>(head_dim);
    const size_t rope_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim);

    if (idx == 0) {
        cu_seqlens[0] = 0;
        cu_seqlens[1] = static_cast<int32_t>(seq_len);
    }

    if (idx < qkv_elems) {
        const uint32_t dim = static_cast<uint32_t>(idx % head_dim);
        const uint32_t head = static_cast<uint32_t>((idx / head_dim) % total_heads);
        const uint32_t token = static_cast<uint32_t>(idx / (static_cast<size_t>(head_dim) * total_heads));

        T value;
        if (head < num_q_heads) {
            value = q[static_cast<size_t>(token) * q_stride_n + static_cast<size_t>(head) * q_stride_h + dim];
        } else if (head < num_q_heads + num_kv_heads) {
            const uint32_t kv_head = head - num_q_heads;
            value = k[static_cast<size_t>(token) * k_stride_n + static_cast<size_t>(kv_head) * k_stride_h + dim];
        } else {
            const uint32_t kv_head = head - num_q_heads - num_kv_heads;
            value = v[static_cast<size_t>(token) * kv_stride_n + static_cast<size_t>(kv_head) * kv_stride_h + dim];
        }
        qkv_half[idx] = to_half_device(value);
    }

    if (idx < rope_elems) {
        const uint32_t dim = static_cast<uint32_t>(idx % head_dim);
        const uint32_t pos = static_cast<uint32_t>(idx / head_dim);
        const uint32_t half_dim = head_dim / 2U;
        const uint32_t rotary_dim = (dim < half_dim) ? dim : dim - half_dim;
        const float exponent =
            2.0f * static_cast<float>(rotary_dim) / static_cast<float>(head_dim);
        const float angle =
            static_cast<float>(pos) * rope_rcp_scale / powf(rope_theta, exponent);
        float sin_value;
        float cos_value;
        __sincosf(angle, &sin_value, &cos_value);
        cos_sin[idx] = (dim < half_dim) ? cos_value : sin_value;
    }
}

__global__ void half_to_bf16_kernel(
    const half* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = __float2bfloat16(__half2float(input[idx]));
    }
}

template <typename T>
__device__ __forceinline__ half rope_to_half_device(
    const T* __restrict__ input,
    size_t base,
    uint32_t dim,
    uint32_t pos,
    uint32_t head_dim,
    float rope_rcp_scale,
    float rope_theta)
{
    const uint32_t half_dim = head_dim / 2U;
    const uint32_t pair_dim = (dim < half_dim) ? (dim + half_dim) : (dim - half_dim);
    const uint32_t rotary_dim = (dim < half_dim) ? dim : (dim - half_dim);
    const float x = static_cast<float>(input[base + dim]);
    const float pair = static_cast<float>(input[base + pair_dim]);
    const float exponent = 2.0f * static_cast<float>(rotary_dim) / static_cast<float>(head_dim);
    const float angle = static_cast<float>(pos) * rope_rcp_scale / powf(rope_theta, exponent);
    float sin_value;
    float cos_value;
    __sincosf(angle, &sin_value, &cos_value);
    const float value = (dim < half_dim)
        ? (x * cos_value - pair * sin_value)
        : (x * cos_value + pair * sin_value);
    return __float2half(value);
}

template <>
__device__ __forceinline__ half rope_to_half_device<half>(
    const half* __restrict__ input,
    size_t base,
    uint32_t dim,
    uint32_t pos,
    uint32_t head_dim,
    float rope_rcp_scale,
    float rope_theta)
{
    const uint32_t half_dim = head_dim / 2U;
    const uint32_t pair_dim = (dim < half_dim) ? (dim + half_dim) : (dim - half_dim);
    const uint32_t rotary_dim = (dim < half_dim) ? dim : (dim - half_dim);
    const float x = __half2float(input[base + dim]);
    const float pair = __half2float(input[base + pair_dim]);
    const float exponent = 2.0f * static_cast<float>(rotary_dim) / static_cast<float>(head_dim);
    const float angle = static_cast<float>(pos) * rope_rcp_scale / powf(rope_theta, exponent);
    float sin_value;
    float cos_value;
    __sincosf(angle, &sin_value, &cos_value);
    const float value = (dim < half_dim)
        ? (x * cos_value - pair * sin_value)
        : (x * cos_value + pair * sin_value);
    return __float2half(value);
}

template <>
__device__ __forceinline__ half rope_to_half_device<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ input,
    size_t base,
    uint32_t dim,
    uint32_t pos,
    uint32_t head_dim,
    float rope_rcp_scale,
    float rope_theta)
{
    const uint32_t half_dim = head_dim / 2U;
    const uint32_t pair_dim = (dim < half_dim) ? (dim + half_dim) : (dim - half_dim);
    const uint32_t rotary_dim = (dim < half_dim) ? dim : (dim - half_dim);
    const float x = __bfloat162float(input[base + dim]);
    const float pair = __bfloat162float(input[base + pair_dim]);
    const float exponent = 2.0f * static_cast<float>(rotary_dim) / static_cast<float>(head_dim);
    const float angle = static_cast<float>(pos) * rope_rcp_scale / powf(rope_theta, exponent);
    float sin_value;
    float cos_value;
    __sincosf(angle, &sin_value, &cos_value);
    const float value = (dim < half_dim)
        ? (x * cos_value - pair * sin_value)
        : (x * cos_value + pair * sin_value);
    return __float2half(value);
}

template <typename T>
__global__ void pack_contiguous_q_kv_rope_and_seqlens_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    half* __restrict__ q_half,
    half* __restrict__ kv_half,
    int32_t* __restrict__ cu_seqlens,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t k_stride_n,
    uint32_t k_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    uint32_t q_rope_pos_offset,
    uint32_t k_rope_pos_offset,
    bool k_already_prerotated,
    float rope_rcp_scale,
    float rope_theta)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t half_dim = head_dim / 2U;
    const size_t q_pair_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(num_q_heads) * static_cast<size_t>(half_dim);
    const size_t k_pair_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(num_kv_heads) * static_cast<size_t>(half_dim);
    const size_t v_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);

    if (idx == 0) {
        cu_seqlens[0] = 0;
        cu_seqlens[1] = static_cast<int32_t>(seq_len);
    }

    if (idx < q_pair_elems) {
        const uint32_t dim = static_cast<uint32_t>(idx % half_dim);
        const uint32_t head = static_cast<uint32_t>((idx / half_dim) % num_q_heads);
        const uint32_t token = static_cast<uint32_t>(idx / (static_cast<size_t>(half_dim) * num_q_heads));
        const size_t q_base =
            static_cast<size_t>(token) * static_cast<size_t>(q_stride_n) +
            static_cast<size_t>(head) * static_cast<size_t>(q_stride_h);
        const size_t out_base =
            (static_cast<size_t>(token) * static_cast<size_t>(num_q_heads) +
             static_cast<size_t>(head)) * static_cast<size_t>(head_dim);
        const float x0 = to_float_device(q[q_base + dim]);
        const float x1 = to_float_device(q[q_base + dim + half_dim]);
        const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
        const float angle =
            static_cast<float>(token + q_rope_pos_offset) * rope_rcp_scale / powf(rope_theta, exponent);
        float sin_value;
        float cos_value;
        __sincosf(angle, &sin_value, &cos_value);
        q_half[out_base + dim] = __float2half(x0 * cos_value - x1 * sin_value);
        q_half[out_base + dim + half_dim] = __float2half(x1 * cos_value + x0 * sin_value);
    }

    if (idx < k_pair_elems) {
        const uint32_t dim = static_cast<uint32_t>(idx % half_dim);
        const uint32_t head = static_cast<uint32_t>((idx / half_dim) % num_kv_heads);
        const uint32_t token = static_cast<uint32_t>(idx / (static_cast<size_t>(half_dim) * num_kv_heads));
        const size_t k_base =
            static_cast<size_t>(token) * static_cast<size_t>(k_stride_n) +
            static_cast<size_t>(head) * static_cast<size_t>(k_stride_h);
        const size_t out_base =
            ((static_cast<size_t>(token) * 2ULL) * static_cast<size_t>(num_kv_heads) +
             static_cast<size_t>(head)) * static_cast<size_t>(head_dim);
        if (k_already_prerotated) {
            kv_half[out_base + dim] = to_half_device(k[k_base + dim]);
            kv_half[out_base + dim + half_dim] = to_half_device(k[k_base + dim + half_dim]);
        } else {
            const float x0 = to_float_device(k[k_base + dim]);
            const float x1 = to_float_device(k[k_base + dim + half_dim]);
            const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
            const float angle =
                static_cast<float>(token + k_rope_pos_offset) * rope_rcp_scale / powf(rope_theta, exponent);
            float sin_value;
            float cos_value;
            __sincosf(angle, &sin_value, &cos_value);
            kv_half[out_base + dim] = __float2half(x0 * cos_value - x1 * sin_value);
            kv_half[out_base + dim + half_dim] = __float2half(x1 * cos_value + x0 * sin_value);
        }
    }

    if (idx < v_elems) {
        const uint32_t dim = static_cast<uint32_t>(idx % head_dim);
        const uint32_t head = static_cast<uint32_t>((idx / head_dim) % num_kv_heads);
        const uint32_t token = static_cast<uint32_t>(idx / (static_cast<size_t>(head_dim) * num_kv_heads));
        const size_t v_base =
            static_cast<size_t>(token) * static_cast<size_t>(kv_stride_n) +
            static_cast<size_t>(head) * static_cast<size_t>(kv_stride_h);
        const size_t out_base =
            ((static_cast<size_t>(token) * 2ULL + 1ULL) * static_cast<size_t>(num_kv_heads) +
             static_cast<size_t>(head)) * static_cast<size_t>(head_dim);
        kv_half[out_base + dim] = to_half_device(v[v_base + dim]);
    }
}

template <typename T>
__global__ void pack_contiguous_q_kv_rope_by_dim_and_seqlens_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    half* __restrict__ q_half,
    half* __restrict__ kv_half,
    int32_t* __restrict__ cu_seqlens,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t k_stride_n,
    uint32_t k_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    uint32_t q_rope_pos_offset,
    uint32_t k_rope_pos_offset,
    bool k_already_prerotated,
    float rope_rcp_scale,
    float rope_theta)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t half_dim = head_dim / 2U;
    const size_t work_items = static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim);

    if (idx == 0) {
        cu_seqlens[0] = 0;
        cu_seqlens[1] = static_cast<int32_t>(seq_len);
    }
    if (idx >= work_items) {
        return;
    }

    const uint32_t dim = static_cast<uint32_t>(idx % half_dim);
    const uint32_t token = static_cast<uint32_t>(idx / half_dim);
    const float exponent = 2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
    const float inv_freq = rope_rcp_scale / powf(rope_theta, exponent);

    float q_sin;
    float q_cos;
    __sincosf(static_cast<float>(token + q_rope_pos_offset) * inv_freq, &q_sin, &q_cos);
    float k_sin = q_sin;
    float k_cos = q_cos;
    if (k_rope_pos_offset != q_rope_pos_offset) {
        __sincosf(static_cast<float>(token + k_rope_pos_offset) * inv_freq, &k_sin, &k_cos);
    }

    const size_t q_token_base = static_cast<size_t>(token) * static_cast<size_t>(q_stride_n);
    const size_t q_out_token_base =
        static_cast<size_t>(token) * static_cast<size_t>(num_q_heads) * static_cast<size_t>(head_dim);
    for (uint32_t head = 0; head < num_q_heads; ++head) {
        const size_t q_base = q_token_base + static_cast<size_t>(head) * static_cast<size_t>(q_stride_h);
        const size_t out_base = q_out_token_base + static_cast<size_t>(head) * static_cast<size_t>(head_dim);
        const float x0 = to_float_device(q[q_base + dim]);
        const float x1 = to_float_device(q[q_base + dim + half_dim]);
        q_half[out_base + dim] = __float2half(x0 * q_cos - x1 * q_sin);
        q_half[out_base + dim + half_dim] = __float2half(x1 * q_cos + x0 * q_sin);
    }

    const size_t k_token_base = static_cast<size_t>(token) * static_cast<size_t>(k_stride_n);
    const size_t v_token_base = static_cast<size_t>(token) * static_cast<size_t>(kv_stride_n);
    const size_t kv_token_base =
        static_cast<size_t>(token) * 2ULL * static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
    const size_t v_out_token_base =
        (static_cast<size_t>(token) * 2ULL + 1ULL) *
        static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
    for (uint32_t head = 0; head < num_kv_heads; ++head) {
        const size_t k_base = k_token_base + static_cast<size_t>(head) * static_cast<size_t>(k_stride_h);
        const size_t k_out_base = kv_token_base + static_cast<size_t>(head) * static_cast<size_t>(head_dim);
        if (k_already_prerotated) {
            kv_half[k_out_base + dim] = to_half_device(k[k_base + dim]);
            kv_half[k_out_base + dim + half_dim] = to_half_device(k[k_base + dim + half_dim]);
        } else {
            const float x0 = to_float_device(k[k_base + dim]);
            const float x1 = to_float_device(k[k_base + dim + half_dim]);
            kv_half[k_out_base + dim] = __float2half(x0 * k_cos - x1 * k_sin);
            kv_half[k_out_base + dim + half_dim] = __float2half(x1 * k_cos + x0 * k_sin);
        }

        const size_t v_base = v_token_base + static_cast<size_t>(head) * static_cast<size_t>(kv_stride_h);
        const size_t v_out_base = v_out_token_base + static_cast<size_t>(head) * static_cast<size_t>(head_dim);
        kv_half[v_out_base + dim] = to_half_device(v[v_base + dim]);
        kv_half[v_out_base + dim + half_dim] = to_half_device(v[v_base + dim + half_dim]);
    }
}

template <typename T>
void launch_pack_qkv_rope_and_seqlens(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    half* qkv_half,
    float* cos_sin,
    int32_t* cu_seqlens,
    uint32_t seq_len,
    cudaStream_t stream)
{
    const uint32_t total_heads = ctx.num_qo_heads + 2U * ctx.num_kv_heads;
    const size_t qkv_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(total_heads) * static_cast<size_t>(ctx.head_dim);
    const size_t rope_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.head_dim);
    const size_t work_items = std::max(qkv_elems, rope_elems);
    constexpr int block = 256;
    const int grid = static_cast<int>((work_items + block - 1) / block);
    const uint32_t q_stride_n =
        (ctx.q_stride_n != 0U) ? ctx.q_stride_n : (ctx.num_qo_heads * ctx.head_dim);
    const uint32_t q_stride_h = (ctx.q_stride_h != 0U) ? ctx.q_stride_h : ctx.head_dim;
    const uint32_t kv_stride_n =
        (ctx.kv_stride_n != 0U) ? ctx.kv_stride_n : (ctx.num_kv_heads * ctx.head_dim);
    const uint32_t kv_stride_h = (ctx.kv_stride_h != 0U) ? ctx.kv_stride_h : ctx.head_dim;
    const uint32_t k_stride_n = (ctx.k_stride_n != 0U) ? ctx.k_stride_n : kv_stride_n;
    const uint32_t k_stride_h = (ctx.k_stride_h != 0U) ? ctx.k_stride_h : kv_stride_h;
    const float rope_rcp_scale = 1.0f / ctx.rope_scale;
    pack_qkv_rope_and_seqlens_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T*>(q.data_ptr()),
        static_cast<const T*>(k.data_ptr()),
        static_cast<const T*>(v.data_ptr()),
        qkv_half,
        cos_sin,
        cu_seqlens,
        seq_len,
        ctx.num_qo_heads,
        ctx.num_kv_heads,
        ctx.head_dim,
        q_stride_n,
        q_stride_h,
        k_stride_n,
        k_stride_h,
        kv_stride_n,
        kv_stride_h,
        rope_rcp_scale,
        ctx.rope_theta);
    CUDA_CHECK_THROW(cudaGetLastError(), "trt_context_fmha pack_qkv_rope kernel failed");
}

void launch_half_to_bf16(
    const half* input,
    __nv_bfloat16* output,
    size_t count,
    cudaStream_t stream)
{
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    half_to_bf16_kernel<<<grid, block, 0, stream>>>(input, output, count);
    CUDA_CHECK_THROW(cudaGetLastError(), "trt_context_fmha half_to_bf16 kernel failed");
}

template <typename T>
void launch_pack_contiguous_q_kv_rope_and_seqlens(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    half* q_half,
    half* kv_half,
    int32_t* cu_seqlens,
    uint32_t seq_len,
    cudaStream_t stream)
{
    const size_t q_pair_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.num_qo_heads) *
        static_cast<size_t>(ctx.head_dim / 2U);
    const size_t k_pair_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.num_kv_heads) *
        static_cast<size_t>(ctx.head_dim / 2U);
    const size_t v_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.num_kv_heads) * static_cast<size_t>(ctx.head_dim);
    const size_t work_items = std::max(q_pair_elems, std::max(k_pair_elems, v_elems));
    constexpr int block = 256;
    const int grid = static_cast<int>((work_items + block - 1) / block);
    const uint32_t q_stride_n =
        (ctx.q_stride_n != 0U) ? ctx.q_stride_n : (ctx.num_qo_heads * ctx.head_dim);
    const uint32_t q_stride_h = (ctx.q_stride_h != 0U) ? ctx.q_stride_h : ctx.head_dim;
    const uint32_t kv_stride_n =
        (ctx.kv_stride_n != 0U) ? ctx.kv_stride_n : (ctx.num_kv_heads * ctx.head_dim);
    const uint32_t kv_stride_h = (ctx.kv_stride_h != 0U) ? ctx.kv_stride_h : ctx.head_dim;
    const uint32_t k_stride_n = (ctx.k_stride_n != 0U) ? ctx.k_stride_n : kv_stride_n;
    const uint32_t k_stride_h = (ctx.k_stride_h != 0U) ? ctx.k_stride_h : kv_stride_h;
    const float rope_rcp_scale = 1.0f / ctx.rope_scale;
    if (get_bool_param_or(ctx.impl_params, "contiguous_q_kv_token_dim_pack", false)) {
        const size_t token_dim_work_items =
            static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.head_dim / 2U);
        const int token_dim_grid = static_cast<int>((token_dim_work_items + block - 1) / block);
        pack_contiguous_q_kv_rope_by_dim_and_seqlens_kernel<T><<<token_dim_grid, block, 0, stream>>>(
            static_cast<const T*>(q.data_ptr()),
            static_cast<const T*>(k.data_ptr()),
            static_cast<const T*>(v.data_ptr()),
            q_half,
            kv_half,
            cu_seqlens,
            seq_len,
            ctx.num_qo_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            q_stride_n,
            q_stride_h,
            k_stride_n,
            k_stride_h,
            kv_stride_n,
            kv_stride_h,
            ctx.q_rope_pos_offset,
            ctx.k_rope_pos_offset,
            ctx.k_already_prerotated,
            rope_rcp_scale,
            ctx.rope_theta);
    } else {
        pack_contiguous_q_kv_rope_and_seqlens_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T*>(q.data_ptr()),
        static_cast<const T*>(k.data_ptr()),
        static_cast<const T*>(v.data_ptr()),
        q_half,
        kv_half,
        cu_seqlens,
        seq_len,
        ctx.num_qo_heads,
        ctx.num_kv_heads,
        ctx.head_dim,
        q_stride_n,
        q_stride_h,
        k_stride_n,
        k_stride_h,
        kv_stride_n,
        kv_stride_h,
        ctx.q_rope_pos_offset,
        ctx.k_rope_pos_offset,
        ctx.k_already_prerotated,
        rope_rcp_scale,
        ctx.rope_theta);
    }
    CUDA_CHECK_THROW(cudaGetLastError(), "trt_context_fmha pack contiguous Q/KV RoPE kernel failed");
}

void forward_prefill_trt_context_fmha_contiguous_q_kv_impl(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream)
{
    const uint32_t seq_len = static_cast<uint32_t>(q.shape()[0]);
    const size_t q_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.num_qo_heads) * static_cast<size_t>(ctx.head_dim);
    const size_t kv_elems =
        static_cast<size_t>(seq_len) * 2ULL * static_cast<size_t>(ctx.num_kv_heads) *
        static_cast<size_t>(ctx.head_dim);
    const size_t out_elems = q_elems;

    half* q_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_contig_q_half", q_elems * sizeof(half), ctx.device_id));
    half* kv_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_contig_kv_half", kv_elems * sizeof(half), ctx.device_id));
    half* out_half = nullptr;
    if (ctx.dtype == DType::BFloat16) {
        out_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
            "trt_context_fmha_plugin_contig_out_half", out_elems * sizeof(half), ctx.device_id));
    } else {
        out_half = static_cast<half*>(o.data_ptr());
    }
    int32_t* cu_seqlens = static_cast<int32_t*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_contig_cu_seqlens", 2 * sizeof(int32_t), ctx.device_id));

    if (ctx.dtype == DType::BFloat16) {
        launch_pack_contiguous_q_kv_rope_and_seqlens<__nv_bfloat16>(
            ctx, q, k, v, q_half, kv_half, cu_seqlens, seq_len, stream);
    } else if (ctx.dtype == DType::Float16) {
        launch_pack_contiguous_q_kv_rope_and_seqlens<half>(
            ctx, q, k, v, q_half, kv_half, cu_seqlens, seq_len, stream);
    } else {
        throw ConfigurationError("trt_context_fmha_plugin_attention only supports Float16/BFloat16");
    }

    using namespace trt_edgellm;
    const int32_t sm = sm_version();
    check<ConfigurationError>(
        ContextFMHARunner::canImplement(static_cast<int32_t>(ctx.head_dim), sm, nvinfer1::DataType::kHALF),
        "trt_context_fmha_plugin_attention contiguous Q/KV path does not support this head_dim/SM");
    ContextFMHARunner::loadContextFMHAKernels(sm, nvinfer1::DataType::kHALF);
    ContextFMHARunner runner(
        nvinfer1::DataType::kHALF,
        1,
        static_cast<int32_t>(seq_len),
        static_cast<int32_t>(ctx.num_qo_heads),
        static_cast<int32_t>(ctx.num_kv_heads),
        static_cast<int32_t>(ctx.head_dim),
        sm,
        AttentionInputLayout::CONTIGUOUS_Q_KV);

    FusedMultiheadAttentionParamsV2 params{};
    runner.setupParams(params);
    params.q_ptr = q_half;
    params.kv_ptr = kv_half;
    params.o_ptr = out_half;
    params.cu_q_seqlens = cu_seqlens;
    params.cu_kv_seqlens = cu_seqlens;

    runner.dispatchFMHAKernel(params, stream);
    if (ctx.dtype == DType::BFloat16) {
        launch_half_to_bf16(out_half, static_cast<__nv_bfloat16*>(o.data_ptr()), out_elems, stream);
    }
}

void forward_prefill_trt_context_fmha_impl(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream)
{
    const uint32_t seq_len = static_cast<uint32_t>(q.shape()[0]);
    const uint32_t total_heads = ctx.num_qo_heads + 2U * ctx.num_kv_heads;
    const size_t qkv_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(total_heads) * static_cast<size_t>(ctx.head_dim);
    const size_t kv_cache_elems =
        2ULL * static_cast<size_t>(ctx.num_kv_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.head_dim);
    const size_t out_elems =
        static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.num_qo_heads) * static_cast<size_t>(ctx.head_dim);
    const size_t rope_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(ctx.head_dim);

    half* qkv_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_qkv_half", qkv_elems * sizeof(half), ctx.device_id));
    half* kv_cache_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_kv_cache_half", kv_cache_elems * sizeof(half), ctx.device_id));
    half* out_half = nullptr;
    if (ctx.dtype == DType::BFloat16) {
        out_half = static_cast<half*>(StaticBufferManager::get_cache_buf(
            "trt_context_fmha_plugin_out_half", out_elems * sizeof(half), ctx.device_id));
    } else {
        out_half = static_cast<half*>(o.data_ptr());
    }
    float* cos_sin = static_cast<float*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_rope", rope_elems * sizeof(float), ctx.device_id));
    int32_t* cu_seqlens = static_cast<int32_t*>(StaticBufferManager::get_cache_buf(
        "trt_context_fmha_plugin_cu_seqlens", 2 * sizeof(int32_t), ctx.device_id));

    if (ctx.dtype == DType::BFloat16) {
        launch_pack_qkv_rope_and_seqlens<__nv_bfloat16>(
            ctx, q, k, v, qkv_half, cos_sin, cu_seqlens, seq_len, stream);
    } else if (ctx.dtype == DType::Float16) {
        launch_pack_qkv_rope_and_seqlens<half>(
            ctx, q, k, v, qkv_half, cos_sin, cu_seqlens, seq_len, stream);
    } else {
        throw ConfigurationError("trt_context_fmha_plugin_attention only supports Float16/BFloat16");
    }

    using namespace trt_edgellm;
    rt::Tensor qkv_tensor(
        qkv_half,
        rt::Coords{1, seq_len, total_heads, ctx.head_dim},
        rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    rt::Tensor kv_cache_tensor(
        kv_cache_half,
        rt::Coords{1, 2, ctx.num_kv_heads, seq_len, ctx.head_dim},
        rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    rt::Tensor rope_tensor(
        cos_sin,
        rt::Coords{1, seq_len, ctx.head_dim},
        rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT);
    rt::Tensor empty_kv_scale;

    const int32_t sm = sm_version();
    check<ConfigurationError>(
        ContextFMHARunner::canImplement(static_cast<int32_t>(ctx.head_dim), sm, nvinfer1::DataType::kHALF),
        "trt_context_fmha_plugin_attention does not support this head_dim/SM");
    ContextFMHARunner::loadContextFMHAKernels(sm, nvinfer1::DataType::kHALF);
    ContextFMHARunner runner(
        nvinfer1::DataType::kHALF,
        1,
        static_cast<int32_t>(seq_len),
        static_cast<int32_t>(ctx.num_qo_heads),
        static_cast<int32_t>(ctx.num_kv_heads),
        static_cast<int32_t>(ctx.head_dim),
        sm,
        AttentionInputLayout::PACKED_QKV);

    FusedMultiheadAttentionParamsV2 params{};
    runner.setupParams(params);
    params.qkv_ptr = qkv_half;
    params.o_ptr = out_half;
    params.cu_q_seqlens = cu_seqlens;
    params.cu_kv_seqlens = cu_seqlens;

    kernel::launchApplyRopeWriteKVPackedQKV(rope_tensor, qkv_tensor, kv_cache_tensor, empty_kv_scale, stream);
    runner.dispatchFMHAKernel(params, stream);
    if (ctx.dtype == DType::BFloat16) {
        launch_half_to_bf16(out_half, static_cast<__nv_bfloat16*>(o.data_ptr()), out_elems, stream);
    }
}

#endif

template <typename Shape>
DecodeTunedPolicy choose_decode_tuned_policy(
    uint32_t grid_kv_len,
    int sm_count,
    const DecodeTunedRuntimeConfig& config)
{
    DecodeTunedPolicy policy;
    policy.bdz = (grid_kv_len >= config.long_seq_threshold) ? config.long_seq_bdz : config.short_seq_bdz;
    if (grid_kv_len <= config.no_split_kv_threshold) {
        policy.split_kv = false;
        policy.kv_chunk_size = grid_kv_len;
        return policy;
    }

    const uint32_t target_total_ctas = std::max<uint32_t>(static_cast<uint32_t>(sm_count), 1U);
    const uint32_t target_chunks =
        std::max<uint32_t>((target_total_ctas + Shape::kNumKvHeads - 1U) / Shape::kNumKvHeads, 1U);
    const uint32_t desired_chunk_size = round_up_u32(
        std::max<uint32_t>(ceil_div(grid_kv_len, target_chunks), config.min_chunk_size),
        config.chunk_alignment);

    const auto chunk_candidates = config.chunk_candidates;
    uint32_t kv_chunk_size = chunk_candidates.back();
    for (uint32_t candidate : chunk_candidates) {
        if (desired_chunk_size <= candidate) {
            kv_chunk_size = candidate;
            break;
        }
    }

    policy.split_kv = grid_kv_len > kv_chunk_size;
    policy.kv_chunk_size = policy.split_kv ? kv_chunk_size : grid_kv_len;
    return policy;
}

template <typename Shape, typename Params, typename AttentionVariant, PosEncodingMode POS_MODE, uint32_t BDZ>
cudaError_t launch_decode_tuned_kernel(
    Params params,
    typename Params::DTypeO* tmp,
    cudaStream_t stream,
    uint32_t grid_kv_len,
    const DecodeTunedPolicy& policy)
{
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;

    constexpr uint32_t smem_size =
        2U * Shape::kNumStagesSmem * Shape::kGroupSize * Shape::kTileSizePerBdx * BDZ *
            Shape::kHeadDim * sizeof(DTypeKV) +
        2U * Shape::kGroupSize * BDZ * sizeof(float);
    auto kernel =
        SingleDecodeWithKVCacheKernel<POS_MODE,
                                      Shape::kNumStagesSmem,
                                      Shape::kTileSizePerBdx,
                                      Shape::kVecSize,
                                      Shape::kBdx,
                                      Shape::kGroupSize,
                                      BDZ,
                                      AttentionVariant,
                                      Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nthrs(Shape::kBdx, Shape::kGroupSize, BDZ);
    if (!policy.split_kv || tmp == nullptr) {
        params.kv_chunk_size = grid_kv_len;
        dim3 nblks(1, params.num_kv_heads);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        return cudaSuccess;
    }

    const uint32_t num_chunks = ceil_div(grid_kv_len, policy.kv_chunk_size);
    DTypeO* o = params.o;
    float* lse = params.lse;
    params.o = tmp;
    params.lse = reinterpret_cast<float*>(tmp + num_chunks * params.num_qo_heads * Shape::kHeadDim);
    params.kv_chunk_size = policy.kv_chunk_size;

    dim3 nblks(num_chunks, params.num_kv_heads);
    void* args[] = {(void*)&params};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    if constexpr (AttentionVariant::use_softmax) {
        FLASHINFER_CUDA_CALL(
            MergeStates(tmp, params.lse, o, lse, num_chunks, 1, params.num_qo_heads, Shape::kHeadDim, stream));
    } else {
        FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, params.num_qo_heads, Shape::kHeadDim, stream));
    }
    return cudaSuccess;
}

template <
    uint32_t CTA_TILE_Q,
    uint32_t HEAD_DIM_QK,
    uint32_t HEAD_DIM_VO,
    PosEncodingMode POS_MODE,
    bool USE_FP16_QK_REDUCTION,
    MaskMode MASK_MODE,
    typename AttentionVariant,
    typename Params>
cudaError_t launch_prefill_tuned_cta_tile_q(
    Params params,
    typename Params::DTypeO* tmp,
    cudaStream_t stream,
    uint32_t max_mma_kv_cap)
{
    using DTypeQ = typename Params::DTypeQ;
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;

    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;
    const uint32_t num_qo_heads = params.num_qo_heads;
    const uint32_t num_kv_heads = params.num_kv_heads;
    const uint32_t group_size = num_qo_heads / num_kv_heads;

    constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
    constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half, float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    const int num_ctas_per_sm =
        max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                                (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
            ? 2
            : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
        ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));
    uint32_t max_num_mma_kv = min(max_num_mma_kv_smem, max_num_mma_kv_reg);
    if (max_mma_kv_cap != 0U) {
        max_num_mma_kv = min(max_num_mma_kv, max_mma_kv_cap);
    }

    DISPATCH_NUM_MMA_KV(max_num_mma_kv, NUM_MMA_KV, {
        using KTraits =
            KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                         NUM_WARPS_Q, NUM_WARPS_KV, POS_MODE, DTypeQ, DTypeKV, DTypeO,
                         DTypeQKAccum, typename Params::IdType, AttentionVariant>;
        if constexpr (KTraits::IsInvalid()) {
            std::ostringstream err_msg;
            err_msg << "FlashInfer prefill tuned CTA tile got invalid configuration: NUM_MMA_Q="
                    << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK
                    << " NUM_MMA_D_VO=" << NUM_MMA_D_VO << " NUM_MMA_KV=" << NUM_MMA_KV
                    << " NUM_WARPS_Q=" << NUM_WARPS_Q << " NUM_WARPS_KV=" << NUM_WARPS_KV;
            FLASHINFER_ERROR(err_msg.str());
        } else {
            constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
            auto kernel = SinglePrefillWithKVCacheKernel<KTraits, Params>;
            size_t smem_size = sizeof(typename KTraits::SharedStorage);
            FLASHINFER_CUDA_CALL(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

            int num_blocks_per_sm = 0;
            int num_sm = 0;
            FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm, kernel, num_threads, smem_size));

            const uint32_t max_num_kv_chunks =
                (num_blocks_per_sm * num_sm) / (num_kv_heads * ceil_div(qo_len * group_size, CTA_TILE_Q));
            uint32_t num_chunks;
            if (max_num_kv_chunks > 0) {
                const uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256U);
                num_chunks = ceil_div(kv_len, chunk_size);
            } else {
                num_chunks = 0;
            }

            if (num_chunks <= 1 || tmp == nullptr) {
                params.partition_kv = false;
                void* args[] = {(void*)&params};
                dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1, num_kv_heads);
                dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
                FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
            } else {
                params.partition_kv = true;
                float* tmp_lse = reinterpret_cast<float*>(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
                auto o = params.o;
                auto lse = params.lse;
                params.o = tmp;
                params.lse = tmp_lse;
                void* args[] = {(void*)&params};
                dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), num_chunks, num_kv_heads);
                dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
                FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
                if constexpr (AttentionVariant::use_softmax) {
                    FLASHINFER_CUDA_CALL(
                        MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
                } else {
                    FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
                }
            }
        }
    });

    return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
void forward_prefill_impl(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream)
{
    const uint32_t qo_len = static_cast<uint32_t>(q.shape()[0]);
    const uint32_t kv_len = static_cast<uint32_t>(k.shape()[0]);

    DTypeQ* q_data = static_cast<DTypeQ*>(q.data_ptr());
    DTypeKV* k_data = static_cast<DTypeKV*>(k.data_ptr());
    DTypeKV* v_data = static_cast<DTypeKV*>(v.data_ptr());
    DTypeO* o_data = static_cast<DTypeO*>(o.data_ptr());
    const uint32_t q_stride_n =
        (ctx.q_stride_n != 0U) ? ctx.q_stride_n : (ctx.num_qo_heads * ctx.head_dim);
    const uint32_t q_stride_h =
        (ctx.q_stride_h != 0U) ? ctx.q_stride_h : ctx.head_dim;
    const uint32_t kv_stride_n =
        (ctx.kv_stride_n != 0U) ? ctx.kv_stride_n : (ctx.num_kv_heads * ctx.head_dim);
    const uint32_t kv_stride_h =
        (ctx.kv_stride_h != 0U) ? ctx.kv_stride_h : ctx.head_dim;
    const uint32_t k_stride_n = (ctx.k_stride_n != 0U) ? ctx.k_stride_n : kv_stride_n;
    const uint32_t k_stride_h = (ctx.k_stride_h != 0U) ? ctx.k_stride_h : kv_stride_h;
    check<ConfigurationError>(
        k_stride_n == kv_stride_n && k_stride_h == kv_stride_h,
        "flashinfer_attention prefill requires K and V to share strides; "
        "use flashinfer_attention_prefill_prerotate for K-only strided input");

    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> prefill_params(
        q_data, k_data, v_data,
        nullptr, o_data, nullptr, nullptr,
        ctx.num_qo_heads, ctx.num_kv_heads,
        qo_len, kv_len,
        q_stride_n, q_stride_h,
        kv_stride_n, kv_stride_h,
        ctx.head_dim,
        -1, 0.0f,
        1.0f / sqrtf(static_cast<float>(ctx.head_dim)),
        ctx.rope_scale, ctx.rope_theta);

    void* tmp_ptr = StaticBufferManager::get_cache_buf(
        "single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, ctx.device_id);
    DTypeO* tmp = static_cast<DTypeO*>(tmp_ptr);

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    constexpr bool USE_FP16_QK_REDUCTION = false;
    const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
    const uint32_t tuned_cta_tile_q = choose_prefill_cta_tile_q(ctx, qo_len, ctx.head_dim);
    const uint32_t max_mma_kv_cap = resolve_prefill_tuned_runtime_config(ctx).max_mma_kv_cap;

    DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
        DISPATCH_HEAD_DIM(ctx.head_dim, HEAD_DIM, {
            cudaError_t err = cudaSuccess;
            switch (tuned_cta_tile_q) {
                case 16U:
                    err = launch_prefill_tuned_cta_tile_q<
                        16U, HEAD_DIM, HEAD_DIM, POS_MODE, USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
                        prefill_params, tmp, stream, max_mma_kv_cap);
                    break;
                case 64U:
                    err = launch_prefill_tuned_cta_tile_q<
                        64U, HEAD_DIM, HEAD_DIM, POS_MODE, USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
                        prefill_params, tmp, stream, max_mma_kv_cap);
                    break;
                case 128U:
                    err = launch_prefill_tuned_cta_tile_q<
                        128U, HEAD_DIM, HEAD_DIM, POS_MODE, USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
                        prefill_params, tmp, stream, max_mma_kv_cap);
                    break;
                default:
                    throw ConfigurationError(
                        "flashinfer_attention prefill got unsupported cta_tile_q=" +
                        std::to_string(tuned_cta_tile_q));
            }
            CUDA_CHECK_THROW(err, "SinglePrefillWithKVCacheDispatched failed");
        });
    });
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
void forward_decode_impl(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len)
{
    DTypeQ* q_data = static_cast<DTypeQ*>(q.data_ptr());
    DTypeKV* k_data = static_cast<DTypeKV*>(k.data_ptr());
    DTypeKV* v_data = static_cast<DTypeKV*>(v.data_ptr());
    DTypeO* o_data = static_cast<DTypeO*>(o.data_ptr());
    const uint32_t kv_len = static_cast<uint32_t>(k.shape()[0]);

    SingleDecodeParams<DTypeQ, DTypeKV, DTypeO> decode_params(
        q_data, k_data, v_data, o_data,
        nullptr, kv_len,
        ctx.num_qo_heads, ctx.num_kv_heads,
        QKVLayout::kNHD, ctx.head_dim,
        -1, 0.0f,
        1.0f / sqrtf(static_cast<float>(ctx.head_dim)),
        ctx.rope_scale, ctx.rope_theta);
    decode_params.d_kv_len = d_kv_len;
    decode_params.max_kv_len = max_kv_len;

    void* tmp_ptr = StaticBufferManager::get_cache_buf(
        "single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, ctx.device_id);
    DTypeO* tmp = static_cast<DTypeO*>(tmp_ptr);

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    DISPATCH_HEAD_DIM(ctx.head_dim, HEAD_DIM, {
        cudaError_t err = SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_MODE, AttentionVariant>(
            decode_params, tmp, stream);
        CUDA_CHECK_THROW(err, "SingleDecodeWithKVCacheDispatched failed");
    });
}

template <typename Shape, typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
void forward_decode_tuned_impl(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len)
{
    DTypeQ* q_data = static_cast<DTypeQ*>(q.data_ptr());
    DTypeKV* k_data = static_cast<DTypeKV*>(k.data_ptr());
    DTypeKV* v_data = static_cast<DTypeKV*>(v.data_ptr());
    DTypeO* o_data = static_cast<DTypeO*>(o.data_ptr());
    const uint32_t kv_len = static_cast<uint32_t>(k.shape()[0]);

    SingleDecodeParams<DTypeQ, DTypeKV, DTypeO> decode_params(
        q_data, k_data, v_data, o_data,
        nullptr, kv_len,
        ctx.num_qo_heads, ctx.num_kv_heads,
        QKVLayout::kNHD, ctx.head_dim,
        -1, 0.0f,
        1.0f / sqrtf(static_cast<float>(ctx.head_dim)),
        ctx.rope_scale, ctx.rope_theta);
    decode_params.d_kv_len = d_kv_len;
    decode_params.max_kv_len = max_kv_len;

    void* tmp_ptr = StaticBufferManager::get_cache_buf(
        "single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, ctx.device_id);
    DTypeO* tmp = static_cast<DTypeO*>(tmp_ptr);

    const uint32_t grid_kv_len = (max_kv_len > 0) ? max_kv_len : kv_len;
    const DecodeTunedRuntimeConfig tuned_config = resolve_decode_tuned_runtime_config<Shape>(ctx);
    const DecodeTunedPolicy policy =
        choose_decode_tuned_policy<Shape>(grid_kv_len, GetCudaMultiProcessorCount(), tuned_config);

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    const auto launch_shape_policy = [&]() -> cudaError_t {
        using Params = SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>;
        switch (policy.bdz) {
        case 4U:
            return launch_decode_tuned_kernel<Shape, Params, AttentionVariant, POS_MODE, 4U>(
                decode_params, tmp, stream, grid_kv_len, policy);
        case 3U:
            return launch_decode_tuned_kernel<Shape, Params, AttentionVariant, POS_MODE, 3U>(
                decode_params, tmp, stream, grid_kv_len, policy);
        }
        throw ConfigurationError("flashinfer_attention_decode_sm80_tuned got an invalid bdz policy");
    };
    cudaError_t err = launch_shape_policy();
    CUDA_CHECK_THROW(err, "launch_decode_tuned_kernel failed");
}

class FlashInferAttentionOp final : public AttentionOp {
public:
    std::string impl_id() const override { return "flashinfer_attention"; }

    bool supports(const AttentionOpContext& ctx) const override {
        return (ctx.dtype == DType::Float16 || ctx.dtype == DType::BFloat16) &&
            ctx.num_qo_heads > 0 && ctx.num_kv_heads > 0 && ctx.head_dim > 0;
    }

    void forward_prefill(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream) override
    {
        if (ctx.dtype == DType::BFloat16) {
            if (ctx.pos_encoding == AttentionPosEncoding::kNone) {
                forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kNone>(
                    ctx, q, k, v, o, causal, stream);
            } else {
                forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, causal, stream);
            }
            return;
        }

        if (ctx.dtype == DType::Float16) {
            if (ctx.pos_encoding == AttentionPosEncoding::kNone) {
                forward_prefill_impl<half, half, half, PosEncodingMode::kNone>(
                    ctx, q, k, v, o, causal, stream);
            } else {
                forward_prefill_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, causal, stream);
            }
            return;
        }

        throw ConfigurationError("attention operator only supports Float16 / BFloat16");
    }

    void forward_decode(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len) override
    {
        check<ConfigurationError>(
            can_use_flashinfer_decode_fast_path(ctx),
            "flashinfer_attention decode only supports GQA group sizes {1,2,3,4,6,7,8}; "
            "got num_qo_heads=" + std::to_string(ctx.num_qo_heads) +
                ", num_kv_heads=" + std::to_string(ctx.num_kv_heads));

        if (ctx.dtype == DType::BFloat16) {
            if (ctx.pos_encoding == AttentionPosEncoding::kNone) {
                forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kNone>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            } else {
                forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            }
            return;
        }

        if (ctx.dtype == DType::Float16) {
            if (ctx.pos_encoding == AttentionPosEncoding::kNone) {
                forward_decode_impl<half, half, half, PosEncodingMode::kNone>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            } else {
                forward_decode_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            }
            return;
        }

        throw ConfigurationError("attention operator only supports Float16 / BFloat16");
    }
};

class FlashInferAttentionPrefillPrerotateOp final : public AttentionOp {
public:
    std::string impl_id() const override { return "flashinfer_attention_prefill_prerotate"; }

    bool supports(const AttentionOpContext& ctx) const override {
        return ctx.dtype == DType::BFloat16 &&
            ctx.pos_encoding == AttentionPosEncoding::kRoPELlama &&
            ctx.num_qo_heads > 0 &&
            ctx.num_kv_heads > 0 &&
            ctx.head_dim > 0 &&
            (ctx.head_dim % 2U) == 0U;
    }

    void forward_prefill(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream) override
    {
        check<ConfigurationError>(
            supports(ctx),
            "flashinfer_attention_prefill_prerotate only supports BF16 Llama-RoPE prefill");

        const auto& q_shape = q.shape();
        const auto& k_shape = k.shape();
        const uint32_t qo_len = static_cast<uint32_t>(q_shape[0]);
        const uint32_t kv_len = static_cast<uint32_t>(k_shape[0]);
        const size_t q_elems =
            static_cast<size_t>(qo_len) * static_cast<size_t>(ctx.num_qo_heads) * static_cast<size_t>(ctx.head_dim);
        const uint32_t kv_input_stride_n =
            (ctx.kv_stride_n != 0U) ? ctx.kv_stride_n : (ctx.num_kv_heads * ctx.head_dim);
        const uint32_t kv_input_stride_h = (ctx.kv_stride_h != 0U) ? ctx.kv_stride_h : ctx.head_dim;
        const uint32_t k_input_stride_n =
            (ctx.k_stride_n != 0U) ? ctx.k_stride_n : kv_input_stride_n;
        const uint32_t k_input_stride_h = (ctx.k_stride_h != 0U) ? ctx.k_stride_h : kv_input_stride_h;
        const bool explicit_k_stride = ctx.k_stride_n != 0U || ctx.k_stride_h != 0U;
        const bool common_strided_kv =
            !explicit_k_stride &&
            (kv_input_stride_n != (ctx.num_kv_heads * ctx.head_dim) || kv_input_stride_h != ctx.head_dim);
        const uint32_t k_rot_stride_n =
            common_strided_kv ? kv_input_stride_n : (ctx.num_kv_heads * ctx.head_dim);
        const uint32_t k_rot_stride_h = common_strided_kv ? kv_input_stride_h : ctx.head_dim;
        const size_t k_alloc_elems =
            common_strided_kv
                ? static_cast<size_t>(kv_len) * static_cast<size_t>(k_rot_stride_n)
                : static_cast<size_t>(kv_len) * static_cast<size_t>(ctx.num_kv_heads) *
                    static_cast<size_t>(ctx.head_dim);

        auto* q_rot_ptr = static_cast<__nv_bfloat16*>(StaticBufferManager::get_cache_buf(
            "flashinfer_attention_prefill_prerotate_q", q_elems * sizeof(__nv_bfloat16), ctx.device_id));
        auto* k_rot_ptr = ctx.k_already_prerotated
            ? static_cast<__nv_bfloat16*>(const_cast<void*>(k.data_ptr()))
            : static_cast<__nv_bfloat16*>(StaticBufferManager::get_cache_buf(
                "flashinfer_attention_prefill_prerotate_k", k_alloc_elems * sizeof(__nv_bfloat16), ctx.device_id));
        Tensor q_rot = Tensor::view(
            q_rot_ptr,
            {static_cast<int64_t>(qo_len), static_cast<int64_t>(ctx.num_qo_heads), static_cast<int64_t>(ctx.head_dim)},
            DType::BFloat16,
            Device::GPU,
            ctx.device_id);
        Tensor k_rot = Tensor::view(
            k_rot_ptr,
            {static_cast<int64_t>(kv_len), static_cast<int64_t>(ctx.num_kv_heads), static_cast<int64_t>(ctx.head_dim)},
            DType::BFloat16,
            Device::GPU,
            ctx.device_id);

        const uint32_t q_stride_n =
            (ctx.q_stride_n != 0U) ? ctx.q_stride_n : (ctx.num_qo_heads * ctx.head_dim);
        const uint32_t q_stride_h = (ctx.q_stride_h != 0U) ? ctx.q_stride_h : ctx.head_dim;
        launch_preapply_llama_rope_bf16(
            ctx,
            q,
            q_rot,
            ctx.num_qo_heads,
            q_stride_n,
            q_stride_h,
            ctx.num_qo_heads * ctx.head_dim,
            ctx.head_dim,
            ctx.q_rope_pos_offset,
            stream);
        if (!ctx.k_already_prerotated) {
            launch_preapply_llama_rope_bf16(
                ctx,
                k,
                k_rot,
                ctx.num_kv_heads,
                k_input_stride_n,
                k_input_stride_h,
                k_rot_stride_n,
                k_rot_stride_h,
                ctx.k_rope_pos_offset,
                stream);
        }

        AttentionOpContext no_rope_ctx = ctx;
        no_rope_ctx.pos_encoding = AttentionPosEncoding::kNone;
        no_rope_ctx.q_stride_n = 0U;
        no_rope_ctx.q_stride_h = 0U;
        no_rope_ctx.k_stride_n = 0U;
        no_rope_ctx.k_stride_h = 0U;
        no_rope_ctx.kv_stride_n = (!ctx.k_already_prerotated && common_strided_kv) ? k_rot_stride_n : 0U;
        no_rope_ctx.kv_stride_h = (!ctx.k_already_prerotated && common_strided_kv) ? k_rot_stride_h : 0U;
        no_rope_ctx.k_already_prerotated = false;
        forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kNone>(
            no_rope_ctx, q_rot, k_rot, v, o, causal, stream);
    }

    void forward_decode(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len) override
    {
        forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
            ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
    }
};

#if defined(EDGE_FM_ENABLE_TRT_PLUGIN_OPS) && EDGE_FM_ENABLE_TRT_PLUGIN_OPS

class TrtContextFmhaPluginAttentionOp final : public AttentionOp {
public:
    std::string impl_id() const override { return "trt_context_fmha_plugin_attention"; }

    bool supports(const AttentionOpContext& ctx) const override {
        if (!trt_context_fmha_plugin_enabled(ctx)) {
            return false;
        }
        if (!trt_context_fmha_plugin_dtype_supported(ctx) ||
            ctx.pos_encoding != AttentionPosEncoding::kRoPELlama ||
            ctx.num_qo_heads == 0 || ctx.num_kv_heads == 0 || ctx.head_dim == 0) {
            return false;
        }
        const int32_t sm = sm_version();
        return trt_edgellm::ContextFMHARunner::canImplement(
            static_cast<int32_t>(ctx.head_dim), sm, nvinfer1::DataType::kHALF);
    }

    void forward_prefill(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream) override
    {
        if (can_use_plugin_prefill_shape(ctx, q, k, v, o, causal)) {
            if (trt_context_fmha_plugin_use_contiguous_q_kv(ctx)) {
                const auto& q_shape = q.shape();
                const uint32_t seq_len = static_cast<uint32_t>(q_shape[0]);
                const uint32_t min_seq_len =
                    get_u32_param_or(ctx.impl_params, "contiguous_q_kv_min_seq_len", 0U);
                if (seq_len < min_seq_len) {
                    forward_prefill_flashinfer(ctx, q, k, v, o, causal, stream);
                    return;
                }
                forward_prefill_trt_context_fmha_contiguous_q_kv_impl(ctx, q, k, v, o, stream);
                return;
            }
            forward_prefill_trt_context_fmha_impl(ctx, q, k, v, o, stream);
            return;
        }
        forward_prefill_flashinfer(ctx, q, k, v, o, causal, stream);
    }

    void forward_decode(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len) override
    {
        forward_decode_flashinfer(ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
    }

private:
    static bool can_use_plugin_prefill_shape(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        const Tensor& o,
        bool causal)
    {
        if (!causal || !supports_static(ctx)) {
            return false;
        }
        const auto& q_shape = q.shape();
        const auto& k_shape = k.shape();
        const auto& v_shape = v.shape();
        const auto& o_shape = o.shape();
        return q_shape.size() == 3 && k_shape.size() == 3 && v_shape.size() == 3 && o_shape.size() == 3 &&
            q_shape[0] > 1 &&
            q_shape[0] == k_shape[0] &&
            q_shape[0] == v_shape[0] &&
            o_shape[0] == q_shape[0] &&
            static_cast<uint32_t>(q_shape[1]) == ctx.num_qo_heads &&
            static_cast<uint32_t>(k_shape[1]) == ctx.num_kv_heads &&
            static_cast<uint32_t>(v_shape[1]) == ctx.num_kv_heads &&
            static_cast<uint32_t>(o_shape[1]) == ctx.num_qo_heads &&
            static_cast<uint32_t>(q_shape[2]) == ctx.head_dim &&
            static_cast<uint32_t>(k_shape[2]) == ctx.head_dim &&
            static_cast<uint32_t>(v_shape[2]) == ctx.head_dim &&
            static_cast<uint32_t>(o_shape[2]) == ctx.head_dim;
    }

    static bool supports_static(const AttentionOpContext& ctx) {
        return trt_context_fmha_plugin_dtype_supported(ctx) &&
            ctx.pos_encoding == AttentionPosEncoding::kRoPELlama &&
            ctx.num_qo_heads > 0 && ctx.num_kv_heads > 0 && ctx.head_dim > 0;
    }

    static void forward_prefill_flashinfer(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream)
    {
        if (ctx.dtype == DType::BFloat16 &&
            ctx.pos_encoding == AttentionPosEncoding::kRoPELlama &&
            (ctx.head_dim % 2U) == 0U) {
            FlashInferAttentionPrefillPrerotateOp prerotate_op;
            if (prerotate_op.supports(ctx)) {
                prerotate_op.forward_prefill(ctx, q, k, v, o, causal, stream);
                return;
            }
        }
        if (ctx.dtype == DType::BFloat16) {
            forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, causal, stream);
            return;
        }
        if (ctx.dtype == DType::Float16) {
            forward_prefill_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, causal, stream);
            return;
        }
        throw ConfigurationError("trt_context_fmha_plugin_attention only supports Float16/BFloat16");
    }

    static void forward_decode_flashinfer(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len)
    {
        check<ConfigurationError>(
            can_use_flashinfer_decode_fast_path(ctx),
            "trt_context_fmha_plugin_attention decode fallback only supports GQA group sizes {1,2,3,4,6,7,8}");
        if (ctx.dtype == DType::BFloat16) {
            forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        if (ctx.dtype == DType::Float16) {
            forward_decode_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        throw ConfigurationError("trt_context_fmha_plugin_attention only supports Float16/BFloat16");
    }
};

#endif

class FlashInferAttentionDecodeSm80TunedOp final : public AttentionOp {
public:
    std::string impl_id() const override { return "flashinfer_attention_decode_sm80_tuned"; }

    bool supports(const AttentionOpContext& ctx) const override {
        if ((ctx.dtype != DType::BFloat16 && ctx.dtype != DType::Float16) ||
            ctx.pos_encoding != AttentionPosEncoding::kRoPELlama) {
            return false;
        }
        const auto [major, minor] = GetCudaComputeCapability();
        (void)minor;
        return major == 8 &&
            (Qwen0P5DecodeTunedShape::matches(ctx) ||
             Qwen1P5DecodeTunedShape::matches(ctx) ||
             Qwen3BDecodeTunedShape::matches(ctx) ||
             Qwen7BDecodeTunedShape::matches(ctx));
    }

    void forward_prefill(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream) override
    {
        check<ConfigurationError>(supports(ctx), "flashinfer_attention_decode_sm80_tuned only supports Qwen2.5 FP16/BF16 RoPE decode/prefill path");
        if (ctx.dtype == DType::Float16) {
            throw ConfigurationError(
                "flashinfer_attention_decode_sm80_tuned FP16 support is decode-only; "
                "use flashinfer_attention for prefill");
        }
        forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
            ctx, q, k, v, o, causal, stream);
    }

    void forward_decode(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len) override
    {
        check<ConfigurationError>(supports(ctx), "flashinfer_attention_decode_sm80_tuned only supports Qwen2.5 FP16/BF16 RoPE decode path");
        if (Qwen0P5DecodeTunedShape::matches(ctx)) {
            if (ctx.dtype == DType::Float16) {
                forward_decode_tuned_impl<Qwen0P5DecodeTunedShape, half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
                return;
            }
            forward_decode_tuned_impl<Qwen0P5DecodeTunedShape, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        if (Qwen1P5DecodeTunedShape::matches(ctx)) {
            if (ctx.dtype == DType::Float16) {
                forward_decode_tuned_impl<Qwen1P5DecodeTunedShape, half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
                return;
            }
            forward_decode_tuned_impl<Qwen1P5DecodeTunedShape, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        if (Qwen3BDecodeTunedShape::matches(ctx)) {
            if (ctx.dtype == DType::Float16) {
                forward_decode_tuned_impl<Qwen3BDecodeTunedShape, half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
                return;
            }
            forward_decode_tuned_impl<Qwen3BDecodeTunedShape, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        if (Qwen7BDecodeTunedShape::matches(ctx)) {
            if (ctx.dtype == DType::Float16) {
                forward_decode_tuned_impl<Qwen7BDecodeTunedShape, half, half, half, PosEncodingMode::kRoPELlama>(
                    ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
                return;
            }
            forward_decode_tuned_impl<Qwen7BDecodeTunedShape, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
                ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
            return;
        }
        throw ConfigurationError("flashinfer_attention_decode_sm80_tuned got an unsupported Qwen2.5 decode shape");
    }
};

} // namespace

AttentionOpRegistry::AttentionOpRegistry() {
    impls_.emplace_back(std::make_unique<FlashInferAttentionOp>());
    impls_.emplace_back(std::make_unique<FlashInferAttentionPrefillPrerotateOp>());
#if defined(EDGE_FM_ENABLE_TRT_PLUGIN_OPS) && EDGE_FM_ENABLE_TRT_PLUGIN_OPS
    impls_.emplace_back(std::make_unique<TrtContextFmhaPluginAttentionOp>());
#endif
    impls_.emplace_back(std::make_unique<FlashInferAttentionDecodeSm80TunedOp>());
}

AttentionOpRegistry& AttentionOpRegistry::instance() {
    static AttentionOpRegistry registry;
    return registry;
}

AttentionOp* AttentionOpRegistry::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
}

AttentionOp* AttentionOpRegistry::default_impl(const AttentionOpContext& ctx) const {
    for (const auto& impl : impls_) {
        if (impl->supports(ctx)) {
            return impl.get();
        }
    }
    return nullptr;
}

void attention_forward_prefill(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream)
{
    AttentionOp* impl = AttentionOpRegistry::instance().default_impl(ctx);
    check<ConfigurationError>(impl != nullptr, "attention operator only supports Float16 / BFloat16");
    impl->forward_prefill(ctx, q, k, v, o, causal, stream);
}

void attention_forward_decode(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len)
{
    AttentionOp* impl = AttentionOpRegistry::instance().default_impl(ctx);
    check<ConfigurationError>(impl != nullptr, "attention operator only supports Float16 / BFloat16");
    impl->forward_decode(ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
}

} // namespace edge_fm
