#pragma once

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embed_head.h"
#include "layers/layernorm.h"
#include "layers/linear.h"
#include "models/model.h"
#include "models/model_description.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

class LayeredTransformerModel : public Model {
public:
    explicit LayeredTransformerModel(const EngineConfig& config);
    ~LayeredTransformerModel() override = default;

    void prefill(const Context& context) override;
    void decode_step(const Context& context) override;
    void prepare_decode_position_ids(Context& context, Device device, int32_t device_id) override;

    const ExecutionPlan& execution_plan() const noexcept { return execution_plan_; }

private:
    void forward_impl(const Context& context, int32_t seq_len, ModelStage stage);
    void load_plan_from_config();

    ExecutionPlan execution_plan_;

    std::unique_ptr<EmbedHeadLayer> embed_head_;
    std::unordered_map<std::string, std::unique_ptr<RMSNormLayer>> layernorms_;
    std::unordered_map<std::string, std::unique_ptr<AttentionLayer>> attentions_;
    std::unordered_map<std::string, std::unique_ptr<LinearLayer>> linear_;
    std::unique_ptr<LMHeadLinearLayer> lm_head_;
    std::unique_ptr<ActivationLayer> activation_layer_;

    int32_t intermediate_size_ = 0;
    int32_t num_attention_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;

    bool use_mrope_ = false;
    float rope_theta_ = 1000000.0f;
    float rope_scale_ = 1.0f;
    std::vector<int32_t> mrope_section_;
    int32_t mrope_section_cumsum_host_[3] = {};
    void* mrope_section_cumsum_gpu_ = nullptr;
};

} // namespace edge_fm
