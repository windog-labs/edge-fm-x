#pragma once
#include "layer.h"
#include <edge-fm/core.h>
#include <string>
#include <unordered_map>

namespace edge_fm {

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
        cudaStream_t stream = nullptr
    );

private:
    uint32_t hidden_size_;
    std::string activation_type_;
};

} // namespace edge_fm
