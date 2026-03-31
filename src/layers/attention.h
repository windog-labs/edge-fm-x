#pragma once

#include "layer.h"
#include <edge-fm/core.h>
#include <string>
#include <unordered_map>

namespace edge_fm {

/// Position encoding mode for the attention layer.
enum class RoPEMode {
    kNone,       ///< No positional encoding (RoPE applied externally or not needed)
    kRoPELlama,  ///< Standard 1-D Llama-style RoPE (applied inside FlashInfer)
    kMRoPE       ///< Multi-dimensional RoPE (applied externally, FlashInfer uses kNone)
};

class AttentionLayer : public Layer {
public:
    explicit AttentionLayer(const EngineConfig& engine_config, std::string layer_name = "");
    ~AttentionLayer() override;

    void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) override;

    void load_weights(
        [[maybe_unused]] const std::unordered_map<std::string, Tensor>& prefill_weights,
        [[maybe_unused]] const std::unordered_map<std::string, Tensor>& decode_weights
    ) override {
        weights_loaded_ = true;
    }

    RoPEMode rope_mode() const { return rope_mode_; }

public:
    // public for test
    void forward_prefill(
        const Tensor& q,  // [qo_len, num_qo_heads, head_dim]
        const Tensor& k,  // [kv_len, num_kv_heads, head_dim]
        const Tensor& v,  // [kv_len, num_kv_heads, head_dim]
        Tensor& o,        // [qo_len, num_qo_heads, head_dim]
        bool causal,
        cudaStream_t stream) const;

    void forward_decode(
        const Tensor& q,  // [1, num_qo_heads, head_dim]
        const Tensor& k,  // [kv_len or max_kv_len, num_kv_heads, head_dim]
        const Tensor& v,  // [kv_len or max_kv_len, num_kv_heads, head_dim]
        Tensor& o,        // [1, num_qo_heads, head_dim]
        cudaStream_t stream,
        uint32_t* d_kv_len = nullptr,
        uint32_t max_kv_len = 0) const;

    /// Apply Multi-dimensional Rotary Position Embedding (M-RoPE) in-place.
    /// Splits head_dim into 3 sections (temporal / height / width), each using
    /// a different 1-D position from position_ids[section, token].
    /// @param q           [seq_len, num_qo_heads, head_dim], modified in-place
    /// @param k           [seq_len, num_kv_heads, head_dim], modified in-place
    /// @param position_ids [3, seq_len] Int32, device memory
    /// @param mrope_section_cumsum [3] Int32, device memory, cumsum of mrope_section*2
    static void apply_mrope(
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
        cudaStream_t stream);

private:
    uint32_t num_qo_heads_;
    uint32_t num_kv_heads_;
    uint32_t hidden_size_;
    uint32_t head_dim_;
    float rope_scale_;
    float rope_theta_;
    DType dtype_;
    RoPEMode rope_mode_;
};

} // namespace edge_fm
