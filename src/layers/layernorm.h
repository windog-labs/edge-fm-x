#pragma once
#include "operators/norm_op.h"
#include "layer.h"
#include <edge-fm/core.h>
#include <array>
#include <string>
#include <unordered_map>

namespace edge_fm {

/// RMSNorm 权重类型，用于 load_weights 时加载正确的权重名
enum class NormWeightType {
    Input,           ///< input_layernorm.weight
    PostAttention,    ///< post_attention_layernorm.weight
    Final,           ///< model.norm.weight (layer_id=UINT32_MAX)
};

class RMSNormLayer : public Layer {
public:
    explicit RMSNormLayer(uint32_t layer_id, NormWeightType weight_type, const EngineConfig& engine_config, std::string layer_name = "");
    ~RMSNormLayer() override;

    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;

    void reset_operator_impl_cache() override;

    void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) override;

    // RMSNorm 接口
    void forward_rmsnorm(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    );

    // Fused Add + RMSNorm 接口
    void forward_fused_add_rmsnorm(
        Tensor& inout,
        Tensor& residual,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    );

private:
    void forward_rmsnorm_impl(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream,
        ModelStage stage);

    void forward_fused_add_rmsnorm_impl(
        Tensor& inout,
        Tensor& residual,
        cudaStream_t stream,
        ModelStage stage);

    NormOp* resolve_impl(const RMSNormOpContext& ctx, ModelStage stage);

    // RMSNorm 参数
    uint32_t layer_id_;         ///< 层 ID
    NormWeightType weight_type_; ///< 权重类型（Input/PostAttention/Final），用于 load_weights
    uint32_t hidden_size_;      ///< 隐藏层大小
    float eps_;                 ///< 数值稳定性参数
    const Tensor* weight_;      ///< 权重张量指针（指向全局缓存中的权重或转换后的权重），形状为 [hidden_size]
    std::string layer_role_;
    std::array<std::string, 2> selected_impl_ids_ = {};
    std::array<NormOp*, 2> selected_impls_ = {nullptr, nullptr};
    std::array<RMSNormForwardFn, 2> selected_rms_norm_fns_ = {nullptr, nullptr};
    std::array<FusedAddRMSNormForwardFn, 2> selected_fused_add_rms_norm_fns_ = {nullptr, nullptr};
};

} // namespace edge_fm
