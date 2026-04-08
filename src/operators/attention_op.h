#pragma once

#include <edge-fm/core.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace edge_fm {

enum class AttentionPosEncoding {
    kNone,
    kRoPELlama,
};

struct AttentionOpContext {
    uint32_t num_qo_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t q_stride_n = 0;
    uint32_t q_stride_h = 0;
    float rope_scale = 1.0f;
    float rope_theta = 1000000.0f;
    DType dtype = DType::Float16;
    AttentionPosEncoding pos_encoding = AttentionPosEncoding::kRoPELlama;
    int32_t device_id = 0;
    nlohmann::json impl_params = nlohmann::json::object();
};

class AttentionOp {
public:
    virtual ~AttentionOp() = default;

    virtual std::string impl_id() const = 0;
    virtual bool supports(const AttentionOpContext& ctx) const = 0;
    virtual void forward_prefill(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        bool causal,
        cudaStream_t stream) = 0;
    virtual void forward_decode(
        const AttentionOpContext& ctx,
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& o,
        cudaStream_t stream,
        uint32_t* d_kv_len,
        uint32_t max_kv_len) = 0;
};

class AttentionOpRegistry {
public:
    static AttentionOpRegistry& instance();

    AttentionOp* find_impl_by_id(const std::string& impl_id) const;
    AttentionOp* default_impl(const AttentionOpContext& ctx) const;

private:
    AttentionOpRegistry();

    std::vector<std::unique_ptr<AttentionOp>> impls_;
};

void attention_forward_prefill(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    bool causal,
    cudaStream_t stream);

void attention_forward_decode(
    const AttentionOpContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& o,
    cudaStream_t stream,
    uint32_t* d_kv_len,
    uint32_t max_kv_len);

} // namespace edge_fm
