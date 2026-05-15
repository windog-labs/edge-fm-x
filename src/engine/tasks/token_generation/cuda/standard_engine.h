#pragma once

#include "engine/engine.h"
#include "engine/tasks/token_generation/compact_vocab.h"
#include "engine/tasks/token_generation/kv_manager.h"
#include "engine/tasks/token_generation/scheduler.h"
#include "layers/sampler.h"
#include "models/model.h"
#include "utils/device/cuda_graph.h"

#include <cuda_runtime.h>

namespace edge_fm {

/**
 * Manages captured CUDA graphs.
 *
 * Decode:  single CudaGraphRunner with capture-once / launch-many support.
 *          Dynamic KV-cache memcpy nodes are tracked internally and updated
 *          each step via integer handles.
 * Prefill: one CudaGraphRunner per (request_id, seq_len) bucket.
 */
class CudaGraphManager : public NonCopyable {
public:
    CudaGraphManager() = default;
    ~CudaGraphManager() = default;

    CudaGraphRunner& decode() { return decode_; }
    const CudaGraphRunner& decode() const { return decode_; }
    bool is_decode_captured() const { return decode_.is_captured(); }

    template <typename F>
    void capture_decode(cudaStream_t stream,
                        F&& capture_body,
                        const std::vector<void*>& k_write_ptrs,
                        const std::vector<void*>& v_write_ptrs)
    {
        decode_.begin_capture(stream);
        capture_body();
        decode_.end_capture(stream);

        int32_t n = static_cast<int32_t>(k_write_ptrs.size());
        k_memcpy_.resize(n);
        v_memcpy_.resize(n);
        bool has_dynamic_memcpy = false;
        for (int32_t i = 0; i < n; ++i) {
            k_memcpy_[i] = decode_.track_memcpy_node(k_write_ptrs[i]);
            v_memcpy_[i] = decode_.track_memcpy_node(v_write_ptrs[i]);
            has_dynamic_memcpy = has_dynamic_memcpy || k_memcpy_[i] >= 0 || v_memcpy_[i] >= 0;
        }
        if (!has_dynamic_memcpy) {
            k_memcpy_.clear();
            v_memcpy_.clear();
        }
    }

    void update_decode_nodes(const std::vector<void*>& next_k,
                             const std::vector<void*>& next_v)
    {
        int32_t n = static_cast<int32_t>(k_memcpy_.size());
        for (int32_t i = 0; i < n; ++i) {
            if (k_memcpy_[i] >= 0) decode_.update_memcpy_dst(k_memcpy_[i], next_k[i]);
            if (v_memcpy_[i] >= 0) decode_.update_memcpy_dst(v_memcpy_[i], next_v[i]);
        }
    }

    bool has_decode_dynamic_nodes() const { return !k_memcpy_.empty(); }

    CudaGraphRunner& prefill(int32_t request_id, int32_t seq_len) {
        return prefill_runners_[prefill_key(request_id, seq_len)];
    }

    CudaGraphRunner* find_prefill(int32_t request_id, int32_t seq_len) {
        auto it = prefill_runners_.find(prefill_key(request_id, seq_len));
        return it != prefill_runners_.end() ? &it->second : nullptr;
    }

    void reset() {
        decode_.reset();
        k_memcpy_.clear();
        v_memcpy_.clear();
        prefill_runners_.clear();
    }

private:
    static uint64_t prefill_key(int32_t request_id, int32_t seq_len) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(request_id)) << 32)
             | static_cast<uint64_t>(static_cast<uint32_t>(seq_len));
    }

    CudaGraphRunner decode_;
    std::unordered_map<uint64_t, CudaGraphRunner> prefill_runners_;
    std::vector<int> k_memcpy_;
    std::vector<int> v_memcpy_;
};

class StandardEngine : public Engine {
public:
    explicit StandardEngine(const EngineConfig& config);
    ~StandardEngine() override = default;

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;
    void prepare_tensors(ModelStage stage, Context& context);

private:
    struct PrefillReplayState {
        void* token_ids_ptr = nullptr;
        size_t token_ids_size_bytes = 0;
        void* response_tokens_ptr = nullptr;
        int32_t max_generated_tokens = 0;
        int32_t seq_len = 0;
        bool has_embedding = false;
        void* embedding_ptr = nullptr;
        size_t embedding_size_bytes = 0;
        int32_t embed_token_id = -1;
        bool has_position_ids = false;
        void* position_ids_ptr = nullptr;
        size_t position_ids_size_bytes = 0;
        std::vector<int32_t> mrope_last_pos;
        std::vector<void*> kv_read_ptrs;
        std::vector<void*> kv_write_ptrs;
    };

    static uint64_t prefill_graph_key(int32_t request_id, int32_t seq_len) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(request_id)) << 32)
             | static_cast<uint64_t>(static_cast<uint32_t>(seq_len));
    }

    int32_t embed_token_id_buf_ = -1;

    void initialize_standard_runtime();

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

    /// Finalize one decode token at the runtime boundary.
    void finalize_decode_token(Context& context,
                               void* write_ptr,
                               cudaStream_t stream,
                               bool advance_device_state);

    /// Advance decode runtime buffers that stay at stable device addresses
    /// across graph replays (e.g. kv_len / position_ids).
    void advance_decode_runtime_state(Context& context, cudaStream_t stream);

    /// Read per-layer K/V write pointers from the current step's tensors
    /// and push them into the decode graph's dynamic nodes.
    void sync_decode_graph(Context& context);

    CompactVocab compact_vocab_;
    std::unordered_map<std::string, double> last_generate_metrics_{};
    std::unordered_map<uint64_t, PrefillReplayState> prefill_replay_states_{};
    std::unique_ptr<Model> model_;
    std::shared_ptr<KVManager> kv_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<SamplerLayer> sampler_;
    CudaGraphManager cuda_graph_manager_;
    bool decode_graph_lm_head_top1_ = false;
};

} // namespace edge_fm
