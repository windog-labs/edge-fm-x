#pragma once
#include "models/model.h"
#include "layers/embed_head.h"
#include "layers/layernorm.h"
#include "layers/attention.h"
#include "layers/linear.h"
#include "layers/activation.h"
#include <cuda_runtime.h>
#include <unordered_map>

namespace edge_fm {

class Qwen2_5 : public Model {
public:
    explicit Qwen2_5(const EngineConfig& config);
    ~Qwen2_5() override;

    void prefill(const Context& context) override;
    void decode_step(const Context& context) override;
    void prepare_decode_position_ids(Context& context, Device device, int32_t device_id) override;
    void advance_decode_runtime_tensors(Context& context, cudaStream_t stream) override;
    bool has_static_decode_runtime_tensors() const override { return true; }
    bool needs_separate_prefill_q_buffer() const override { return use_mrope_; }

    /**
     * @brief 完整 prefill 接口（大规模对齐测试用）
     *
     * 执行完整 prefill：embed -> 所有 decoder layers -> final_norm -> lm_head。
     * 输出 4 个关键中间值用于与 transformers 对齐。
     *
     * @param seq_len 序列长度
     * @param inputs 输入: "token_ids" [1, seq_len] int32
     * @param outputs 输出（调用方预分配）:
     *               - "embedding" [1, seq_len, hidden_size]
     *               - "last_decoder_output" [seq_len, hidden_size]
     *               - "final_norm_output" [seq_len, hidden_size]
     *               - "lm_head_output" [seq_len, vocab_size]
     * @param stream CUDA stream
     */
    void forward_prefill(
        int32_t seq_len,
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr);

private:
    // Forward implementation shared by prefill and decode_step
    void forward_impl(const Context& context, int32_t seq_len, ModelStage stage);

    std::unique_ptr<EmbedHeadLayer> embed_head_;
    std::unordered_map<std::string, std::unique_ptr<RMSNormLayer>> layernorms_;
    std::unordered_map<std::string, std::unique_ptr<AttentionLayer>> attentions_;
    std::unordered_map<std::string, std::unique_ptr<LinearLayer>> linear_;
    std::unique_ptr<LMHeadLinearLayer> lm_head_;  // LM head (tied weights with embedding)
    std::unique_ptr<ActivationLayer> activation_layer_;

    int32_t intermediate_size_;
    int32_t num_attention_heads_;
    int32_t num_kv_heads_;
    int32_t head_dim_;

    // M-RoPE state (only active when rope_scaling.type == "mrope")
    bool use_mrope_ = false;
    float rope_theta_ = 1000000.0f;
    float rope_scale_ = 1.0f;
    std::vector<int32_t> mrope_section_;          // e.g. [16, 24, 24]
    int32_t mrope_section_cumsum_host_[3] = {};   // cumsum of section*2
    void* mrope_section_cumsum_gpu_ = nullptr;    // [3] Int32 on GPU
};

} // namespace edge_fm
