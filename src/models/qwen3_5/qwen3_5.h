#pragma once

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embed_head.h"
#include "layers/linear.h"
#include "models/model.h"

#include <edge-fm/core.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

class Qwen3_5 : public Model {
public:
    explicit Qwen3_5(const EngineConfig& config);
    ~Qwen3_5() override = default;

    void prefill(const Context& context) override;
    void decode_step(const Context& context) override;
    bool supports_decode_cuda_graph() const override { return false; }
    void reset_operator_impl_caches() override;

private:
    void forward_impl(const Context& context, int32_t seq_len, ModelStage stage);
    void run_full_attention_layer(
        const Context& context,
        int32_t layer_id,
        int32_t seq_len,
        const Tensor& norm_output,
        Tensor& mixer_output,
        Tensor& position_ids,
        cudaStream_t stream,
        ModelStage stage);
    void run_linear_attention_layer(
        const Context& context,
        int32_t layer_id,
        int32_t seq_len,
        const Tensor& norm_output,
        Tensor& mixer_output,
        cudaStream_t stream,
        ModelStage stage);
    void run_mlp(
        int32_t layer_id,
        int32_t seq_len,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream,
        ModelStage stage);
    Tensor make_position_ids(const Context& context, int32_t seq_len, cudaStream_t stream, ModelStage stage) const;
    Tensor workspace_tensor(
        const std::string& name,
        const std::vector<int64_t>& shape,
        DType dtype,
        int32_t device_id) const;
    bool is_full_attention_layer(int32_t layer_id) const;
    const Tensor& required_weight(const std::string& name) const;
    LinearLayer& linear_layer(const std::string& key) const;
    AttentionLayer& attention_layer(int32_t layer_id) const;

    std::unique_ptr<EmbedHeadLayer> embed_head_;
    std::unique_ptr<LMHeadLinearLayer> lm_head_;
    std::unique_ptr<ActivationLayer> activation_layer_;
    std::unordered_map<std::string, std::unique_ptr<LinearLayer>> linear_;
    std::unordered_map<int32_t, std::unique_ptr<AttentionLayer>> attentions_;

    std::unordered_map<std::string, Tensor> prefill_weights_;
    std::unordered_map<std::string, Tensor> decode_weights_;

    std::vector<std::string> layer_types_;
    int32_t intermediate_size_ = 0;
    int32_t num_attention_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t full_attention_dim_ = 0;
    int32_t linear_num_key_heads_ = 0;
    int32_t linear_num_value_heads_ = 0;
    int32_t linear_key_head_dim_ = 0;
    int32_t linear_value_head_dim_ = 0;
    int32_t linear_key_dim_ = 0;
    int32_t linear_value_dim_ = 0;
    int32_t linear_conv_kernel_dim_ = 0;
    int32_t linear_conv_dim_ = 0;
    float rms_norm_eps_ = 1e-6f;
    float rope_theta_ = 10000000.0f;
    int32_t rotary_dim_ = 0;
    std::vector<int32_t> mrope_section_;
};

} // namespace edge_fm
