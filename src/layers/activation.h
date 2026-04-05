#pragma once
#include "layer.h"
#include <edge-fm/core.h>
#include <array>
#include <string>
#include <unordered_map>

namespace edge_fm {

class ActivationOp;
struct ActivationOpContext;

/**
 * @brief Activation layer for transformer MLP operations
 */
class ActivationLayer : public Layer {
public:
    explicit ActivationLayer(const EngineConfig& engine_config, std::string layer_name = "");
    ~ActivationLayer() override;

    void load_weights(
        [[maybe_unused]] const std::unordered_map<std::string, Tensor>& prefill_weights,
        [[maybe_unused]] const std::unordered_map<std::string, Tensor>& decode_weights
    ) override {
        weights_loaded_ = true;
    }

    void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) override;

    void forward_silu_and_mul(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    );

private:
    void forward_silu_and_mul_impl(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream,
        ModelStage stage);

    ActivationOp* resolve_impl(const ActivationOpContext& ctx, ModelStage stage);

    uint32_t hidden_size_;
    std::string activation_type_;
    std::string layer_role_;
    std::array<std::string, 2> selected_impl_ids_ = {};
    std::array<ActivationOp*, 2> selected_impls_ = {nullptr, nullptr};
};

} // namespace edge_fm
