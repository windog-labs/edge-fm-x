#include "layers/attention.h"
#include "utils/device/nvtx.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/check.h"
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/utils.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

using namespace flashinfer;

namespace edge_fm {

namespace {

// ---- M-RoPE kernel (Multi-dimensional Rotary Position Embedding) ----------

template <typename DType>
__global__ void mrope_kernel(
    DType* __restrict__ data,               // [seq_len, num_heads, head_dim]
    const int32_t* __restrict__ position_ids,        // [3, seq_len]
    const int32_t* __restrict__ mrope_section_cumsum, // [3]
    int32_t seq_len,
    int32_t num_heads,
    int32_t head_dim,
    float rope_rcp_scale,
    float rope_rcp_theta)
{
    const int32_t half_dim = head_dim / 2;
    const int32_t token = blockIdx.x;
    const int32_t head  = blockIdx.y;
    const int32_t d     = threadIdx.x;
    if (d >= half_dim) return;

    const int32_t cum0 = mrope_section_cumsum[0];
    const int32_t cum1 = mrope_section_cumsum[1];

    auto get_section = [&](int32_t dim_idx) -> int32_t {
        if (dim_idx < cum0) return 0;
        if (dim_idx < cum1) return 1;
        return 2;
    };

    const int32_t d_lo = d;
    const int32_t d_hi = d + half_dim;

    const int32_t section_lo = get_section(d_lo);
    const int32_t section_hi = get_section(d_hi);

    const int32_t pos_lo = position_ids[section_lo * seq_len + token];
    const int32_t pos_hi = position_ids[section_hi * seq_len + token];

    const float inv_freq_d = 1.0f / powf(1.0f / rope_rcp_theta,
                                          static_cast<float>(2 * d) / static_cast<float>(head_dim));
    const float angle_lo = static_cast<float>(pos_lo) * inv_freq_d * rope_rcp_scale;
    const float angle_hi = static_cast<float>(pos_hi) * inv_freq_d * rope_rcp_scale;

    float cos_lo, sin_lo, cos_hi, sin_hi;
    __sincosf(angle_lo, &sin_lo, &cos_lo);
    __sincosf(angle_hi, &sin_hi, &cos_hi);

    const int32_t base = token * num_heads * head_dim + head * head_dim;
    const float val_lo = static_cast<float>(data[base + d_lo]);
    const float val_hi = static_cast<float>(data[base + d_hi]);

    data[base + d_lo] = static_cast<DType>(val_lo * cos_lo - val_hi * sin_lo);
    data[base + d_hi] = static_cast<DType>(val_hi * cos_hi + val_lo * sin_hi);
}

template <typename DType>
void apply_mrope_typed(
    void* q, void* k,
    const int32_t* position_ids,
    const int32_t* mrope_section_cumsum,
    int32_t seq_len,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float rope_theta,
    float rope_scale,
    cudaStream_t stream)
{
    const int32_t half_dim = head_dim / 2;
    const float rope_rcp_scale = 1.0f / rope_scale;
    const float rope_rcp_theta = 1.0f / rope_theta;

    {
        dim3 grid(seq_len, num_qo_heads);
        dim3 block(half_dim);
        mrope_kernel<DType><<<grid, block, 0, stream>>>(
            static_cast<DType*>(q),
            position_ids, mrope_section_cumsum,
            seq_len, num_qo_heads, head_dim,
            rope_rcp_scale, rope_rcp_theta);
    }
    {
        dim3 grid(seq_len, num_kv_heads);
        dim3 block(half_dim);
        mrope_kernel<DType><<<grid, block, 0, stream>>>(
            static_cast<DType*>(k),
            position_ids, mrope_section_cumsum,
            seq_len, num_kv_heads, head_dim,
            rope_rcp_scale, rope_rcp_theta);
    }
}

// ---- FlashInfer attention helpers -----------------------------------------

// 根据 dtype 选择 half 或 __nv_bfloat16，用于 FlashInfer 模板实例化
template <typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
void forward_prefill_impl(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float rope_scale, float rope_theta, bool causal,
    int32_t device_id, cudaStream_t stream)
{
    uint32_t qo_len = static_cast<uint32_t>(q.shape()[0]);
    uint32_t kv_len = static_cast<uint32_t>(k.shape()[0]);

    DTypeQ* q_data = static_cast<DTypeQ*>(q.data_ptr());
    DTypeKV* k_data = static_cast<DTypeKV*>(k.data_ptr());
    DTypeKV* v_data = static_cast<DTypeKV*>(v.data_ptr());
    DTypeO* o_data = static_cast<DTypeO*>(o.data_ptr());
    uint32_t q_stride_n = num_qo_heads * head_dim;
    uint32_t q_stride_h = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> prefill_params(
        q_data, k_data, v_data,
        nullptr, o_data, nullptr, nullptr,
        num_qo_heads, num_kv_heads,
        qo_len, kv_len,
        q_stride_n, q_stride_h,
        kv_stride_n, kv_stride_h,
        head_dim,
        -1, 0.0f,
        1.0f / sqrtf(static_cast<float>(head_dim)),
        rope_scale, rope_theta
    );

    void* tmp_ptr = StaticBufferManager::get_cache_buf("single_prefill_with_kv_cache_tmp",
                                                       32 * 1024 * 1024,
                                                       device_id);
    DTypeO* tmp = static_cast<DTypeO*>(tmp_ptr);

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    constexpr bool USE_FP16_QK_REDUCTION = false;
    MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;

    DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        cudaError_t err = SinglePrefillWithKVCacheDispatched<
            HEAD_DIM, HEAD_DIM, POS_MODE, USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant
        >(prefill_params, tmp, stream);
        CUDA_CHECK_THROW(err, "SinglePrefillWithKVCacheDispatched failed");
    });
    });
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, PosEncodingMode POS_MODE>
void forward_decode_impl(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float rope_scale, float rope_theta, int32_t device_id, cudaStream_t stream,
    uint32_t* d_kv_len, uint32_t max_kv_len)
{
    const auto& q_shape = q.shape();
    if (q_shape.size() != 3 || q_shape[0] != 1 ||
        static_cast<uint32_t>(q_shape[1]) != num_qo_heads ||
        static_cast<uint32_t>(q_shape[2]) != head_dim) {
        throw InvalidRequestError(
            "In decode mode, q must have shape [1, num_qo_heads, head_dim]. "
            "Got shape: [" + std::to_string(q_shape[0]) + ", " +
            std::to_string(q_shape[1]) + ", " + std::to_string(q_shape[2]) + "]");
    }
    const auto& k_shape = k.shape();
    const auto& v_shape = v.shape();
    uint32_t kv_len = static_cast<uint32_t>(k_shape[0]);
    if (k_shape.size() != 3 ||
        static_cast<uint32_t>(k_shape[1]) != num_kv_heads ||
        static_cast<uint32_t>(k_shape[2]) != head_dim) {
        throw InvalidRequestError("k must have shape [kv_len, num_kv_heads, head_dim]");
    }
    if (v_shape.size() != 3 ||
        static_cast<uint32_t>(v_shape[0]) != kv_len ||
        static_cast<uint32_t>(v_shape[1]) != num_kv_heads ||
        static_cast<uint32_t>(v_shape[2]) != head_dim) {
        throw InvalidRequestError("v must have shape [kv_len, num_kv_heads, head_dim]");
    }
    const auto& o_shape = o.shape();
    if (o_shape.size() != 3 || o_shape[0] != 1 ||
        static_cast<uint32_t>(o_shape[1]) != num_qo_heads ||
        static_cast<uint32_t>(o_shape[2]) != head_dim) {
        throw InvalidRequestError(
            "In decode mode, o must have shape [1, num_qo_heads, head_dim]. "
            "Got shape: [" + std::to_string(o_shape[0]) + ", " +
            std::to_string(o_shape[1]) + ", " + std::to_string(o_shape[2]) + "]");
    }

    DTypeQ* q_data = static_cast<DTypeQ*>(q.data_ptr());
    DTypeKV* k_data = static_cast<DTypeKV*>(k.data_ptr());
    DTypeKV* v_data = static_cast<DTypeKV*>(v.data_ptr());
    DTypeO* o_data = static_cast<DTypeO*>(o.data_ptr());

    SingleDecodeParams<DTypeQ, DTypeKV, DTypeO> decode_params(
        q_data, k_data, v_data, o_data,
        nullptr, kv_len,
        num_qo_heads, num_kv_heads,
        QKVLayout::kNHD, head_dim,
        -1, 0.0f,
        1.0f / sqrtf(static_cast<float>(head_dim)),
        rope_scale, rope_theta
    );
    decode_params.d_kv_len = d_kv_len;
    decode_params.max_kv_len = max_kv_len;

    void* tmp_ptr = StaticBufferManager::get_cache_buf("single_decode_with_kv_cache_tmp",
                                                      32 * 1024 * 1024,
                                                      device_id);
    DTypeO* tmp = static_cast<DTypeO*>(tmp_ptr);

    using AttentionVariant = DefaultAttention<false, false, false, false>;
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        cudaError_t err = SingleDecodeWithKVCacheDispatched<
            HEAD_DIM, POS_MODE, AttentionVariant
        >(decode_params, tmp, stream);
        CUDA_CHECK_THROW(err, "SingleDecodeWithKVCacheDispatched failed");
    });
}

} // anonymous namespace

AttentionLayer::AttentionLayer(const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name))
{
    // 加载 Prefill 阶段的模型配置
    nlohmann::json model_config = engine_config_.prefill_model_config();

    // 从 model_config 读取 torch_dtype，与模型一致
    std::string torch_dtype_str = model_config.value("torch_dtype", "float16");
    dtype_ = dtype_from_string(torch_dtype_str);
    check<ConfigurationError>(
        dtype_ == DType::Float16 || dtype_ == DType::BFloat16,
        "AttentionLayer only supports Float16 or BFloat16, got torch_dtype=" + torch_dtype_str);

    // 从 model_config 读取 attention 相关参数
    num_qo_heads_ = model_config.value("num_attention_heads", 32U);
    num_kv_heads_ = model_config.value("num_key_value_heads", num_qo_heads_);
    hidden_size_ = model_config.value("hidden_size", 4096U);
    head_dim_ = hidden_size_ / num_qo_heads_;
    
    // 验证 head_dim 计算是否正确
    check<ConfigurationError>(head_dim_ * num_qo_heads_ == hidden_size_,
                              "hidden_size must be divisible by num_attention_heads. "
                              "Got hidden_size=" + std::to_string(hidden_size_) + 
                              ", num_attention_heads=" + std::to_string(num_qo_heads_));
    
    rope_theta_ = model_config.value("rope_theta", 1000000.0f);
    rope_scale_ = 1.0f;
    rope_mode_ = RoPEMode::kRoPELlama;

    if (model_config.contains("rope_scaling") && model_config["rope_scaling"].is_object()) {
        auto rope_scaling = model_config["rope_scaling"];
        if (rope_scaling.contains("factor")) {
            rope_scale_ = rope_scaling["factor"].get<float>();
        }
        std::string rope_type = rope_scaling.value("type",
                                    rope_scaling.value("rope_type", std::string("")));
        if (rope_type == "mrope") {
            rope_mode_ = RoPEMode::kMRoPE;
        }
    }
}

void AttentionLayer::forward(const std::unordered_map<std::string, Tensor>& inputs,
                             std::unordered_map<std::string, Tensor>& outputs,
                             cudaStream_t stream,
                             ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("AttentionLayer is not initialized");
    }

    const auto& q_shape = inputs.at("q").shape();
    bool is_prefill = (q_shape[0] > 1);
    check<InvalidArgumentError>(is_prefill == (stage == ModelStage::Prefill), "AttentionLayer: stage mismatch");
    
    if (stage == ModelStage::Prefill) {
        forward_prefill(inputs.at("q"), inputs.at("k"), inputs.at("v"), outputs.at("o"), true, stream);
    } else {
        uint32_t* d_kv_len_ptr = nullptr;
        uint32_t max_kv_len = 0;
        auto it = inputs.find("d_kv_len");
        if (it != inputs.end()) {
            d_kv_len_ptr = static_cast<uint32_t*>(it->second.data_ptr());
            max_kv_len = static_cast<uint32_t>(inputs.at("k").shape()[0]);
        }
        forward_decode(inputs.at("q"), inputs.at("k"), inputs.at("v"), outputs.at("o"),
                       stream, d_kv_len_ptr, max_kv_len);
    }
}

void AttentionLayer::forward_prefill(const Tensor& q, 
                                     const Tensor& k, 
                                     const Tensor& v, 
                                     Tensor& o,
                                     bool causal,
                                     cudaStream_t stream) const 
{
    // M-RoPE mode: RoPE applied externally, use kNone inside FlashInfer
    bool use_none = (rope_mode_ == RoPEMode::kNone || rope_mode_ == RoPEMode::kMRoPE);

    if (dtype_ == DType::BFloat16) {
        if (use_none) {
            forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                 PosEncodingMode::kNone>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, causal, device_id_, stream);
        } else {
            forward_prefill_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                 PosEncodingMode::kRoPELlama>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, causal, device_id_, stream);
        }
    } else {
        if (use_none) {
            forward_prefill_impl<half, half, half, PosEncodingMode::kNone>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, causal, device_id_, stream);
        } else {
            forward_prefill_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, causal, device_id_, stream);
        }
    }
}

void AttentionLayer::forward_decode(const Tensor& q, 
                                    const Tensor& k, 
                                    const Tensor& v, 
                                    Tensor& o,
                                    cudaStream_t stream,
                                    uint32_t* d_kv_len,
                                    uint32_t max_kv_len) const 
{
    bool use_none = (rope_mode_ == RoPEMode::kNone || rope_mode_ == RoPEMode::kMRoPE);

    if (dtype_ == DType::BFloat16) {
        if (use_none) {
            forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                PosEncodingMode::kNone>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, device_id_, stream, d_kv_len, max_kv_len);
        } else {
            forward_decode_impl<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                PosEncodingMode::kRoPELlama>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, device_id_, stream, d_kv_len, max_kv_len);
        }
    } else {
        if (use_none) {
            forward_decode_impl<half, half, half, PosEncodingMode::kNone>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, device_id_, stream, d_kv_len, max_kv_len);
        } else {
            forward_decode_impl<half, half, half, PosEncodingMode::kRoPELlama>(
                q, k, v, o, num_qo_heads_, num_kv_heads_, head_dim_,
                rope_scale_, rope_theta_, device_id_, stream, d_kv_len, max_kv_len);
        }
    }
}

void AttentionLayer::apply_mrope(
    void* q, void* k,
    const int32_t* position_ids,
    const int32_t* mrope_section_cumsum,
    int32_t seq_len,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float rope_theta,
    float rope_scale,
    DType dtype,
    cudaStream_t stream)
{
    if (seq_len == 0) return;

    if (dtype == DType::BFloat16) {
        apply_mrope_typed<__nv_bfloat16>(
            q, k, position_ids, mrope_section_cumsum,
            seq_len, num_qo_heads, num_kv_heads, head_dim,
            rope_theta, rope_scale, stream);
    } else if (dtype == DType::Float16) {
        apply_mrope_typed<half>(
            q, k, position_ids, mrope_section_cumsum,
            seq_len, num_qo_heads, num_kv_heads, head_dim,
            rope_theta, rope_scale, stream);
    } else {
        throw ConfigurationError("apply_mrope only supports Float16 / BFloat16");
    }
}

} // namespace edge_fm
