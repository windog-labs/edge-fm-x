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

#include <cmath>
using namespace flashinfer;

namespace edge_fm {
namespace {

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

} // namespace

void attention_forward_prefill(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream)
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

} // namespace edge_fm
