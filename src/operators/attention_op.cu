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
#include <cmath>
#include <memory>
#include <string>
using namespace flashinfer;

namespace edge_fm {
namespace {

constexpr uint32_t kDecodeTunedNumQoHeads = 12U;
constexpr uint32_t kDecodeTunedNumKvHeads = 2U;
constexpr uint32_t kDecodeTunedHeadDim = 128U;
constexpr uint32_t kDecodeTunedGroupSize = kDecodeTunedNumQoHeads / kDecodeTunedNumKvHeads;
constexpr uint32_t kDecodeTunedVecSize = 8U;
constexpr uint32_t kDecodeTunedBdx = kDecodeTunedHeadDim / kDecodeTunedVecSize;
constexpr uint32_t kDecodeTunedTileSizePerBdx = 1U;
constexpr uint32_t kDecodeTunedNumStagesSmem = 2U;
constexpr std::array<uint32_t, 4> kDecodeChunkCandidates = {32U, 64U, 128U, 256U};

struct DecodeTunedPolicy {
    bool split_kv = false;
    uint32_t kv_chunk_size = 0;
    uint32_t bdz = 3;
};

uint32_t round_up_u32(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

DecodeTunedPolicy choose_decode_tuned_policy(uint32_t grid_kv_len, int sm_count) {
    DecodeTunedPolicy policy;
    policy.bdz = (grid_kv_len >= 512U) ? 4U : 3U;
    if (grid_kv_len <= 192U) {
        policy.split_kv = false;
        policy.kv_chunk_size = grid_kv_len;
        return policy;
    }

    const uint32_t target_total_ctas = std::max<uint32_t>(static_cast<uint32_t>(sm_count), 1U);
    const uint32_t target_chunks =
        std::max<uint32_t>((target_total_ctas + kDecodeTunedNumKvHeads - 1U) / kDecodeTunedNumKvHeads, 1U);
    const uint32_t desired_chunk_size = round_up_u32(
        std::max<uint32_t>(ceil_div(grid_kv_len, target_chunks), 32U), 32U);

    uint32_t kv_chunk_size = kDecodeChunkCandidates.back();
    for (uint32_t candidate : kDecodeChunkCandidates) {
        if (desired_chunk_size <= candidate) {
            kv_chunk_size = candidate;
            break;
        }
    }

    policy.split_kv = grid_kv_len > kv_chunk_size;
    policy.kv_chunk_size = policy.split_kv ? kv_chunk_size : grid_kv_len;
    return policy;
}

template <typename Params, typename AttentionVariant, PosEncodingMode POS_MODE, uint32_t BDZ>
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
        2U * kDecodeTunedNumStagesSmem * kDecodeTunedGroupSize * kDecodeTunedTileSizePerBdx * BDZ *
            kDecodeTunedHeadDim * sizeof(DTypeKV) +
        2U * kDecodeTunedGroupSize * BDZ * sizeof(float);
    auto kernel =
        SingleDecodeWithKVCacheKernel<POS_MODE,
                                      kDecodeTunedNumStagesSmem,
                                      kDecodeTunedTileSizePerBdx,
                                      kDecodeTunedVecSize,
                                      kDecodeTunedBdx,
                                      kDecodeTunedGroupSize,
                                      BDZ,
                                      AttentionVariant,
                                      Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nthrs(kDecodeTunedBdx, kDecodeTunedGroupSize, BDZ);
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
    params.lse = reinterpret_cast<float*>(tmp + num_chunks * params.num_qo_heads * kDecodeTunedHeadDim);
    params.kv_chunk_size = policy.kv_chunk_size;

    dim3 nblks(num_chunks, params.num_kv_heads);
    void* args[] = {(void*)&params};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    if constexpr (AttentionVariant::use_softmax) {
        FLASHINFER_CUDA_CALL(
            MergeStates(tmp, params.lse, o, lse, num_chunks, 1, params.num_qo_heads, kDecodeTunedHeadDim, stream));
    } else {
        FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, params.num_qo_heads, kDecodeTunedHeadDim, stream));
    }
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
    const uint32_t q_stride_n = ctx.num_qo_heads * ctx.head_dim;
    const uint32_t q_stride_h = ctx.head_dim;
    const uint32_t kv_stride_n = ctx.num_kv_heads * ctx.head_dim;
    const uint32_t kv_stride_h = ctx.head_dim;

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

    DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
        DISPATCH_HEAD_DIM(ctx.head_dim, HEAD_DIM, {
            cudaError_t err = SinglePrefillWithKVCacheDispatched<
                HEAD_DIM, HEAD_DIM, POS_MODE, USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant>(
                prefill_params, tmp, stream);
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

template <typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
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
    const DecodeTunedPolicy policy = choose_decode_tuned_policy(grid_kv_len, GetCudaMultiProcessorCount());

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    cudaError_t err = cudaSuccess;
    if (policy.bdz >= 4U) {
        err = launch_decode_tuned_kernel<SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>, AttentionVariant, POS_MODE, 4U>(
            decode_params, tmp, stream, grid_kv_len, policy);
    } else {
        err = launch_decode_tuned_kernel<SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>, AttentionVariant, POS_MODE, 3U>(
            decode_params, tmp, stream, grid_kv_len, policy);
    }
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

class FlashInferAttentionDecodeSm80TunedOp final : public AttentionOp {
public:
    std::string impl_id() const override { return "flashinfer_attention_decode_sm80_tuned"; }

    bool supports(const AttentionOpContext& ctx) const override {
        if (ctx.dtype != DType::BFloat16 ||
            ctx.pos_encoding != AttentionPosEncoding::kRoPELlama ||
            ctx.num_qo_heads != kDecodeTunedNumQoHeads ||
            ctx.num_kv_heads != kDecodeTunedNumKvHeads ||
            ctx.head_dim != kDecodeTunedHeadDim) {
            return false;
        }
        const auto [major, minor] = GetCudaComputeCapability();
        (void)minor;
        return major == 8;
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
        check<ConfigurationError>(supports(ctx), "flashinfer_attention_decode_sm80_tuned only supports Qwen2.5 BF16 RoPE decode/prefill path");
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
        check<ConfigurationError>(supports(ctx), "flashinfer_attention_decode_sm80_tuned only supports Qwen2.5 BF16 RoPE decode path");
        forward_decode_tuned_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PosEncodingMode::kRoPELlama>(
            ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
    }
};

} // namespace

AttentionOpRegistry::AttentionOpRegistry() {
    impls_.emplace_back(std::make_unique<FlashInferAttentionOp>());
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
