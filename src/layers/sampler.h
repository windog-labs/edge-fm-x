#pragma once
#include "layer.h"
#include <edge-fm/core.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace edge_fm {

class SamplerLayer : public Layer {
public:
    explicit SamplerLayer(const EngineConfig& engine_config);
    ~SamplerLayer() override = default;

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

    void forward_sampling(
        const Tensor &logits,
        Tensor& token_ids,
        cudaStream_t stream = nullptr
    );

private:
    uint32_t vocab_size_;        ///< 词汇表大小
    float temperature_;          ///< 温度参数
    uint64_t seed_;             ///< 随机数种子
};

} // namespace edge_fm