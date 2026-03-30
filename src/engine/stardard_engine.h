#pragma once

#include "engine.h"

namespace edge_fm {

class StandardEngine : public Engine {
public:
    explicit StandardEngine(const EngineConfig& config)
        : Engine(config)
    {
        initialize_standard_runtime();
    }
    ~StandardEngine() override = default;

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    void prepare_tensors(ModelStage stage, Context& context) override;

private:
    int32_t embed_token_id_buf_ = -1;

    void prepare_kvcache_tensors(
        Context& context,
        int32_t num_layers,
        int32_t num_kv_heads,
        int32_t head_dim,
        int32_t seq_len,
        size_t prefix_size
    );

    void prepare_prefill_tensors(Context& context);
    void prepare_decode_tensors(Context& context);

    /// Copy sampled token from staging buffer to the response array.
    void flush_sampled_token(void* write_ptr, cudaStream_t stream);

    /// Read per-layer K/V write pointers from the current step's tensors
    /// and push them into the decode graph's dynamic nodes.
    void sync_decode_graph(Context& context);

    void run_tuning_pass(const Request& request);
};

} // namespace edge_fm
