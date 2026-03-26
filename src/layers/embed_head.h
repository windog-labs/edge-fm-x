
#pragma once
#include "layer.h"
#include <edge-fm/core.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

class EmbedHeadLayer : public Layer {
public:
    explicit EmbedHeadLayer(const EngineConfig& engine_config, std::string layer_name = "");
    ~EmbedHeadLayer() override = default;

    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;

    void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) override;

    void forward_for_tokens(const Tensor& token_ids, 
                            Tensor& output, 
                            cudaStream_t stream = nullptr);

    void forward_for_embeddings(const Tensor& token_ids, 
                                const Tensor& embeddings, 
                                Tensor& output,
                                int32_t embed_token_id=-1,
                                cudaStream_t stream = nullptr);

    // Get embedding table pointer (for tied weights with LM head)
    const Tensor* embedding_table() const { return embedding_table_; }

private:
    uint32_t vocab_size_;           ///< 词汇表大小
    uint32_t hidden_size_;          ///< 隐藏层大小
    const Tensor* embedding_table_; ///< Embedding 权重表指针（指向全局缓存中的权重）
};

} // namespace edge_fm