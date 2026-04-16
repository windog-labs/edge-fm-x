#pragma once

#include "utils/non_copyable.h"
#include "utils/device/cuda_graph.h"
#include <edge-fm/core.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "engine/kv_manager.h"
#include "engine/scheduler.h"
#include "models/model.h"

namespace edge_fm {

class SamplerLayer;

enum class ModelStage { Prefill, Decode, };

inline std::string model_stage_to_string(ModelStage stage) {
    switch (stage) {
        case ModelStage::Prefill:
            return "Prefill";
        case ModelStage::Decode:
            return "Decode";
        default:
            return "Unknown";
    }
}

class EngineConfig {
public:
    explicit EngineConfig(const std::string& config_path);
    ~EngineConfig() = default;
    
    std::string model_name() const;
    std::string prefill_model_path() const;
    std::string decode_model_path() const;
    nlohmann::json prefill_model_config() const;
    nlohmann::json decode_model_config() const;
    std::string resolved_model_name() const;
    
    nlohmann::json runtime() const;
    nlohmann::json speculative() const;
    nlohmann::json kvcache() const;
    nlohmann::json sampling() const;
    nlohmann::json metrics() const;
    nlohmann::json tuning() const;
    std::string configured_operator_impl_table_path() const;
    std::string operator_impl_table_path() const;
    bool has_operator_impl_table_override() const;
    void set_operator_impl_table_override(const std::string& path);
    void clear_operator_impl_table_override();
    bool tuning_enabled() const;

    // 安全访问器：当值为 null 或缺失时返回默认值，避免 type_error.306
    std::string runtime_device() const;
    int32_t runtime_device_id() const;
    std::string runtime_hw_profile() const;
    std::string resolved_hw_profile() const;
    bool use_cuda_graph() const;
    std::string kvcache_dtype() const;
    std::string kvcache_attention_type() const;
    float sampling_temperature() const;
    uint64_t sampling_seed() const;
    std::vector<int32_t> eos_token_ids() const;
    std::vector<int32_t> stop_token_ids() const;
    std::string backend_target() const;
    bool has_backend_artifact() const;
    nlohmann::json backend_artifact() const;
    std::string backend_cache_key() const;
    
    const nlohmann::json& raw() const noexcept { return config_; }

private:
    struct SharedState {
        std::string operator_impl_table_override_path;
    };

    std::filesystem::path config_dir_;
    nlohmann::json config_;
    std::shared_ptr<SharedState> shared_state_;
};

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

    /// Capture the decode graph.  @p capture_body is a callable that
    /// submits all GPU work (decode_step + sampler) on @p stream.
    /// After capture the manager scans the graph and tracks every D2D
    /// memcpy node whose destination matches a per-layer K/V write pointer.
    template <typename F>
    void capture_decode(cudaStream_t stream, F&& capture_body,
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

    /// Update the decode graph's dynamic nodes so that the next launch
    /// writes K/V to the given per-layer addresses.
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

    // Per-layer dynamic-node handles for the decode graph.
    std::vector<int> k_memcpy_;
    std::vector<int> v_memcpy_;
};

class Engine : public NonCopyable {
public:
    explicit Engine(const EngineConfig& config);

    virtual ~Engine() = 0;

    virtual void warmup() = 0;
    virtual void tune() = 0;
    virtual Response generate(const Request& request) = 0;
    virtual std::unordered_map<std::string, double> get_last_generate_metrics() const = 0;
    virtual void prepare_tensors(ModelStage stage, Context& context) = 0;

    KVManagerStatus get_kv_status() const {
        if (kv_manager_ == nullptr) {
            throw InternalError("KVManager is not initialized for the active backend");
        }
        return kv_manager_->get_status();
    }

protected:
    void initialize_standard_runtime();

    Device device_;
    int32_t device_id_;
    EngineConfig config_;

    std::unique_ptr<Model> model_;
    std::shared_ptr<KVManager> kv_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<SamplerLayer> sampler_;
    CudaGraphManager cuda_graph_manager_;
};

} // namespace edge_fm
