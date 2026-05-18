#include "layers/attention.h"

#include "operators/attention_op.h"
#include "operators/operator_impl_table.h"
#include "utils/check.h"
#include "utils/device/nvtx.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <string>

namespace edge_fm {
namespace {

template <typename DType>
__global__ void mrope_kernel(
    DType* __restrict__ data,
    const int32_t* __restrict__ position_ids,
    const int32_t* __restrict__ mrope_section_cumsum,
    int32_t seq_len,
    int32_t num_heads,
    int32_t head_dim,
    float rope_rcp_scale,
    float rope_rcp_theta)
{
    const int32_t half_dim = head_dim / 2;
    const int32_t token = blockIdx.x;
    const int32_t head = blockIdx.y;
    const int32_t d = threadIdx.x;
    if (d >= half_dim) {
        return;
    }

    const int32_t cum0 = mrope_section_cumsum[0];
    const int32_t cum1 = mrope_section_cumsum[1];

    auto get_section = [&](int32_t dim_idx) -> int32_t {
        if (dim_idx < cum0) {
            return 0;
        }
        if (dim_idx < cum1) {
            return 1;
        }
        return 2;
    };

    const int32_t d_lo = d;
    const int32_t d_hi = d + half_dim;
    const int32_t section_lo = get_section(d_lo);
    const int32_t section_hi = section_lo;

    const int32_t pos_lo = position_ids[section_lo * seq_len + token];
    const int32_t pos_hi = position_ids[section_hi * seq_len + token];

    const float inv_freq_d = 1.0f / powf(
        1.0f / rope_rcp_theta, static_cast<float>(2 * d) / static_cast<float>(head_dim));
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
    void* q,
    void* k,
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
            position_ids,
            mrope_section_cumsum,
            seq_len,
            num_qo_heads,
            head_dim,
            rope_rcp_scale,
            rope_rcp_theta);
    }
    {
        dim3 grid(seq_len, num_kv_heads);
        dim3 block(half_dim);
        mrope_kernel<DType><<<grid, block, 0, stream>>>(
            static_cast<DType*>(k),
            position_ids,
            mrope_section_cumsum,
            seq_len,
            num_kv_heads,
            head_dim,
            rope_rcp_scale,
            rope_rcp_theta);
    }
}

void validate_tensor_device(const Tensor& tensor, Device device, int32_t device_id, const std::string& tensor_name) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    if (tensor_device != device || tensor_device_id != device_id) {
        throw DeviceError(tensor_name + " tensor must be on the same device as the layer.");
    }
}

void validate_tensor_dtype(const Tensor& tensor, DType dtype, const std::string& tensor_name) {
    if (tensor.dtype() != dtype) {
        throw ConfigurationError(
            tensor_name + " tensor dtype mismatch. Expected " + std::to_string(static_cast<int>(dtype)) +
            ", got " + std::to_string(static_cast<int>(tensor.dtype())));
    }
}

AttentionPosEncoding resolve_pos_encoding(RoPEMode rope_mode) {
    return (rope_mode == RoPEMode::kNone || rope_mode == RoPEMode::kMRoPE)
        ? AttentionPosEncoding::kNone
        : AttentionPosEncoding::kRoPELlama;
}

size_t stage_slot(ModelStage stage) {
    return stage == ModelStage::Decode ? 1U : 0U;
}

std::string stage_key(ModelStage stage) {
    return stage == ModelStage::Decode ? "decode" : "prefill";
}

std::string attention_shape_sig(uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim) {
    return "num_qo_heads=" + std::to_string(num_qo_heads) +
        "|num_kv_heads=" + std::to_string(num_kv_heads) +
        "|head_dim=" + std::to_string(head_dim);
}

std::string model_name_for_operator_resolution(const EngineConfig& engine_config) {
    try {
        return engine_config.resolved_model_name();
    } catch (const ConfigurationError&) {
        return std::string();
    }
}

} // namespace

AttentionLayer::AttentionLayer(const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name))
{
    nlohmann::json model_config = engine_config_.prefill_model_config();

    const std::string torch_dtype_str = model_config.value("torch_dtype", "float16");
    dtype_ = dtype_from_string(torch_dtype_str);
    check<ConfigurationError>(
        dtype_ == DType::Float16 || dtype_ == DType::BFloat16,
        "AttentionLayer only supports Float16 or BFloat16, got torch_dtype=" + torch_dtype_str);

    num_qo_heads_ = model_config.value("num_attention_heads", 32U);
    num_kv_heads_ = model_config.value("num_key_value_heads", num_qo_heads_);
    hidden_size_ = model_config.value("hidden_size", 4096U);
    head_dim_ = hidden_size_ / num_qo_heads_;

    check<ConfigurationError>(
        head_dim_ * num_qo_heads_ == hidden_size_,
        "hidden_size must be divisible by num_attention_heads. Got hidden_size=" +
            std::to_string(hidden_size_) + ", num_attention_heads=" + std::to_string(num_qo_heads_));

    rope_theta_ = model_config.value("rope_theta", 1000000.0f);
    rope_scale_ = 1.0f;
    rope_mode_ = RoPEMode::kRoPELlama;

    if (model_config.contains("rope_scaling") && model_config["rope_scaling"].is_object()) {
        const auto rope_scaling = model_config["rope_scaling"];
        if (rope_scaling.contains("factor")) {
            rope_scale_ = rope_scaling["factor"].get<float>();
        }
        const std::string rope_type = rope_scaling.value(
            "type", rope_scaling.value("rope_type", std::string("")));
        if (rope_type == "mrope") {
            rope_mode_ = RoPEMode::kMRoPE;
        }
    }
}

AttentionLayer::~AttentionLayer() = default;

void AttentionLayer::reset_operator_impl_cache() {
    selected_impl_ids_.fill(std::string());
    selected_impl_params_.fill(nlohmann::json::object());
    selected_impls_.fill(nullptr);
}

AttentionOp* AttentionLayer::resolve_impl(ModelStage stage) const {
    const size_t slot = stage_slot(stage);
    if (AttentionOp* impl = selected_impls_[slot]; impl != nullptr) {
        return impl;
    }

    auto& selected_impl_id = selected_impl_ids_[slot];
    if (!selected_impl_id.empty()) {
        if (AttentionOp* impl = AttentionOpRegistry::instance().find_impl_by_id(selected_impl_id); impl != nullptr) {
            selected_impls_[slot] = impl;
            return impl;
        }
        selected_impl_id.clear();
    }

    AttentionOpContext ctx;
    ctx.num_qo_heads = num_qo_heads_;
    ctx.num_kv_heads = num_kv_heads_;
    ctx.head_dim = head_dim_;
    ctx.rope_scale = rope_scale_;
    ctx.rope_theta = rope_theta_;
    ctx.dtype = dtype_;
    ctx.pos_encoding = resolve_pos_encoding(rope_mode_);
    ctx.device_id = device_id_;

    OperatorQuery query;
    query.op_kind = "attention";
    query.op_name = "attention";
    query.stage = stage_key(stage);
    query.shape_sig = attention_shape_sig(num_qo_heads_, num_kv_heads_, head_dim_);

    auto resolved = OperatorImplTable::instance().resolve(
        model_name_for_operator_resolution(engine_config_),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (AttentionOp* impl = AttentionOpRegistry::instance().find_impl_by_id(resolved->impl_id); impl != nullptr) {
            ctx.impl_params = resolved->impl_params;
            if (impl->supports(ctx)) {
                selected_impl_id = impl->impl_id();
                selected_impl_params_[slot] = resolved->impl_params;
                selected_impls_[slot] = impl;
                return impl;
            }
            // The impl table may contain a shape-tuned decode kernel for another
            // Qwen2.5 variant (for example the 1.5B path). When the selected impl
            // does not support the current attention shape, fall back to the
            // registry default instead of hard-failing the whole model.
            selected_impl_id.clear();
        } else {
            throw ConfigurationError(
                "AttentionLayer: operator_impl_table selected unknown impl '" + resolved->impl_id + "'");
        }
    }

    if (AttentionOp* impl = AttentionOpRegistry::instance().default_impl(ctx); impl != nullptr) {
        selected_impl_id = impl->impl_id();
        selected_impl_params_[slot] = nlohmann::json::object();
        selected_impls_[slot] = impl;
        return impl;
    }

    throw ConfigurationError("AttentionLayer: no supported implementation found");
}

void AttentionLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    ModelStage stage)
{
    NVTX::Range r(layer_name_);
    if (!is_initialized()) {
        throw InvalidRequestError("AttentionLayer is not initialized");
    }

    const auto& q_shape = inputs.at("q").shape();
    const bool is_prefill = (q_shape[0] > 1);
    check<InvalidArgumentError>(
        is_prefill == (stage == ModelStage::Prefill), "AttentionLayer: stage mismatch");

    if (stage == ModelStage::Prefill) {
        forward_prefill(inputs.at("q"), inputs.at("k"), inputs.at("v"), outputs.at("o"), true, stream);
        return;
    }

    uint32_t* d_kv_len_ptr = nullptr;
    uint32_t max_kv_len = 0;
    auto it = inputs.find("d_kv_len");
    if (it != inputs.end()) {
        d_kv_len_ptr = static_cast<uint32_t*>(it->second.data_ptr());
        max_kv_len = static_cast<uint32_t>(inputs.at("k").shape()[0]);
    }
    forward_decode(
        inputs.at("q"),
        inputs.at("k"),
        inputs.at("v"),
        outputs.at("o"),
        stream,
        d_kv_len_ptr,
        max_kv_len);
}

void AttentionLayer::forward_prefill(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    uint32_t k_stride_n,
    uint32_t k_stride_h,
    bool k_already_prerotated,
    uint32_t q_rope_pos_offset,
    uint32_t k_rope_pos_offset) const
{
    validate_tensor_device(q, device_, device_id_, "q");
    validate_tensor_device(k, device_, device_id_, "k");
    validate_tensor_device(v, device_, device_id_, "v");
    validate_tensor_device(o, device_, device_id_, "o");
    validate_tensor_dtype(q, dtype_, "q");
    validate_tensor_dtype(k, dtype_, "k");
    validate_tensor_dtype(v, dtype_, "v");
    validate_tensor_dtype(o, dtype_, "o");

    const auto& q_shape = q.shape();
    const auto& k_shape = k.shape();
    const auto& v_shape = v.shape();
    const auto& o_shape = o.shape();

    if (q_shape.size() != 3 ||
        static_cast<uint32_t>(q_shape[1]) != num_qo_heads_ ||
        static_cast<uint32_t>(q_shape[2]) != head_dim_) {
        throw InvalidRequestError("q must have shape [qo_len, num_qo_heads, head_dim]");
    }
    if (k_shape.size() != 3 ||
        static_cast<uint32_t>(k_shape[1]) != num_kv_heads_ ||
        static_cast<uint32_t>(k_shape[2]) != head_dim_) {
        throw InvalidRequestError("k must have shape [kv_len, num_kv_heads, head_dim]");
    }
    if (v_shape.size() != 3 ||
        v_shape[0] != k_shape[0] ||
        static_cast<uint32_t>(v_shape[1]) != num_kv_heads_ ||
        static_cast<uint32_t>(v_shape[2]) != head_dim_) {
        throw InvalidRequestError("v must have shape [kv_len, num_kv_heads, head_dim]");
    }
    if (o_shape.size() != 3 ||
        o_shape[0] != q_shape[0] ||
        static_cast<uint32_t>(o_shape[1]) != num_qo_heads_ ||
        static_cast<uint32_t>(o_shape[2]) != head_dim_) {
        throw InvalidRequestError("o must have shape [qo_len, num_qo_heads, head_dim]");
    }

    AttentionOpContext ctx;
    ctx.num_qo_heads = num_qo_heads_;
    ctx.num_kv_heads = num_kv_heads_;
    ctx.head_dim = head_dim_;
    ctx.q_stride_n = q_stride_n;
    ctx.q_stride_h = q_stride_h;
    ctx.k_stride_n = k_stride_n;
    ctx.k_stride_h = k_stride_h;
    ctx.kv_stride_n = kv_stride_n;
    ctx.kv_stride_h = kv_stride_h;
    ctx.rope_scale = rope_scale_;
    ctx.rope_theta = rope_theta_;
    ctx.q_rope_pos_offset = q_rope_pos_offset;
    ctx.k_rope_pos_offset = k_rope_pos_offset;
    ctx.dtype = dtype_;
    ctx.pos_encoding = resolve_pos_encoding(rope_mode_);
    ctx.k_already_prerotated = k_already_prerotated;
    ctx.device_id = device_id_;

    AttentionOp* impl = resolve_impl(ModelStage::Prefill);
    check<ConfigurationError>(
        !k_already_prerotated || impl->impl_id() == "flashinfer_attention_prefill_prerotate",
        "AttentionLayer: k_already_prerotated requires flashinfer_attention_prefill_prerotate");
    ctx.impl_params = selected_impl_params_[stage_slot(ModelStage::Prefill)];
    impl->forward_prefill(ctx, q, k, v, o, causal, stream);
}

void AttentionLayer::forward_decode(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len) const
{
    validate_tensor_device(q, device_, device_id_, "q");
    validate_tensor_device(k, device_, device_id_, "k");
    validate_tensor_device(v, device_, device_id_, "v");
    validate_tensor_device(o, device_, device_id_, "o");
    validate_tensor_dtype(q, dtype_, "q");
    validate_tensor_dtype(k, dtype_, "k");
    validate_tensor_dtype(v, dtype_, "v");
    validate_tensor_dtype(o, dtype_, "o");

    const auto& q_shape = q.shape();
    if (q_shape.size() != 3 || q_shape[0] != 1 ||
        static_cast<uint32_t>(q_shape[1]) != num_qo_heads_ ||
        static_cast<uint32_t>(q_shape[2]) != head_dim_) {
        throw InvalidRequestError(
            "In decode mode, q must have shape [1, num_qo_heads, head_dim]. Got shape: [" +
            std::to_string(q_shape[0]) + ", " + std::to_string(q_shape[1]) + ", " +
            std::to_string(q_shape[2]) + "]");
    }

    const auto& k_shape = k.shape();
    const auto& v_shape = v.shape();
    if (k_shape.size() != 3 ||
        static_cast<uint32_t>(k_shape[1]) != num_kv_heads_ ||
        static_cast<uint32_t>(k_shape[2]) != head_dim_) {
        throw InvalidRequestError("k must have shape [kv_len, num_kv_heads, head_dim]");
    }
    if (v_shape.size() != 3 ||
        v_shape[0] != k_shape[0] ||
        static_cast<uint32_t>(v_shape[1]) != num_kv_heads_ ||
        static_cast<uint32_t>(v_shape[2]) != head_dim_) {
        throw InvalidRequestError("v must have shape [kv_len, num_kv_heads, head_dim]");
    }

    const auto& o_shape = o.shape();
    if (o_shape.size() != 3 || o_shape[0] != 1 ||
        static_cast<uint32_t>(o_shape[1]) != num_qo_heads_ ||
        static_cast<uint32_t>(o_shape[2]) != head_dim_) {
        throw InvalidRequestError(
            "In decode mode, o must have shape [1, num_qo_heads, head_dim]. Got shape: [" +
            std::to_string(o_shape[0]) + ", " + std::to_string(o_shape[1]) + ", " +
            std::to_string(o_shape[2]) + "]");
    }

    AttentionOpContext ctx;
    ctx.num_qo_heads = num_qo_heads_;
    ctx.num_kv_heads = num_kv_heads_;
    ctx.head_dim = head_dim_;
    ctx.rope_scale = rope_scale_;
    ctx.rope_theta = rope_theta_;
    ctx.dtype = dtype_;
    ctx.pos_encoding = resolve_pos_encoding(rope_mode_);
    ctx.device_id = device_id_;

    AttentionOp* impl = resolve_impl(ModelStage::Decode);
    ctx.impl_params = selected_impl_params_[stage_slot(ModelStage::Decode)];
    impl->forward_decode(ctx, q, k, v, o, stream, d_kv_len, max_kv_len);
}

void AttentionLayer::apply_mrope(
    void* q,
    void* k,
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
    if (seq_len == 0) {
        return;
    }

    if (dtype == DType::BFloat16) {
        apply_mrope_typed<__nv_bfloat16>(
            q,
            k,
            position_ids,
            mrope_section_cumsum,
            seq_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_scale,
            stream);
        return;
    }

    if (dtype == DType::Float16) {
        apply_mrope_typed<half>(
            q,
            k,
            position_ids,
            mrope_section_cumsum,
            seq_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_scale,
            stream);
        return;
    }

    throw ConfigurationError("apply_mrope only supports Float16 / BFloat16");
}

} // namespace edge_fm
