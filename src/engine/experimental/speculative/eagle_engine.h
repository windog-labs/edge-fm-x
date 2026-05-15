#pragma once

#include "engine/engine.h"
#include "engine/tasks/token_generation/kv_manager.h"
#include "engine/tasks/token_generation/scheduler.h"
#include "models/model.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace edge_fm {

class EagleEngine : public Engine {
public:
    explicit EagleEngine(const EngineConfig& config);
    ~EagleEngine() override = default;

    void warmup() override;
    Response generate(const Request& request) override;
    void prepare_inouts(ModelStage stage, Context& context) override;

private:
    void prepare_prefill_tensors(Context& context);
    void prepare_decode_tensors(Context& context);
    void prepare_tree_attention_tensors(Context& context, int32_t num_draft_tokens);

    int32_t generate_draft_tokens(
        Context& context,
        std::vector<int32_t>& draft_tokens,
        Tensor& draft_probs
    );

    void verify_draft_tokens(
        Context& context,
        const std::vector<int32_t>& draft_tokens,
        Tensor& target_probs
    );

    int32_t generate_bonus_token(Context& context);

    std::unique_ptr<Model> draft_model_;
    std::shared_ptr<KVManager> draft_kv_manager_;
    std::unique_ptr<Scheduler> draft_scheduler_;
    
    int32_t num_steps_;
    int32_t eagle_topk_;
    int32_t num_draft_tokens_;
};

} // namespace edge_fm
