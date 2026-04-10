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
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;
    void prepare_tensors(ModelStage stage, Context& context) override;

private:
    struct PrefillReplayState {
        void* token_ids_ptr = nullptr;
        size_t token_ids_size_bytes = 0;
        void* response_tokens_ptr = nullptr;
        int32_t max_generated_tokens = 0;
        int32_t seq_len = 0;
        std::vector<void*> kv_read_ptrs;
        std::vector<void*> kv_write_ptrs;
    };

    static uint64_t prefill_graph_key(int32_t request_id, int32_t seq_len) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(request_id)) << 32)
             | static_cast<uint64_t>(static_cast<uint32_t>(seq_len));
    }

    int32_t embed_token_id_buf_ = -1;

    void run_sampler(const Tensor& logits,
                     const Tensor& token_out,
                     cudaStream_t stream,
                     ModelStage stage);
    bool try_run_prefill_cuda_graph(Context& context);
    void ensure_decode_graph_captured(Context& context);

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

    /// Copy the latest sampled token from the decode runtime buffer to the response array.
    void flush_sampled_token(const Context& context, void* write_ptr, cudaStream_t stream);

    /// Advance decode runtime buffers that stay at stable device addresses
    /// across graph replays (e.g. kv_len / position_ids).
    void advance_decode_runtime_state(Context& context, cudaStream_t stream);

    /// Read per-layer K/V write pointers from the current step's tensors
    /// and push them into the decode graph's dynamic nodes.
    void sync_decode_graph(Context& context);

    std::unordered_map<std::string, double> last_generate_metrics_{};
    std::unordered_map<uint64_t, PrefillReplayState> prefill_replay_states_{};
};

} // namespace edge_fm
