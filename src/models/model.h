#pragma once

#include <edge-fm/core.h>
#include "utils/non_copyable.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

namespace edge_fm {

class Context;
class EngineConfig;

/**
 * @brief 标准模型 tensor 名称常量
 *
 * Engine 与 Model 约定的 tensor 名称。Engine 在 prepare_prefill_tensors /
 * prepare_decode_tensors 中按此创建 tensors，Model 在 prefill / decode_step 中按此访问。
 */
namespace ModelTensors {
    // ==================== 输入 tensors ====================
    constexpr const char* TOKEN_IDS = "token_ids";
    constexpr const char* EMBEDDING = "embedding";
    /// 可选：自定义 embedding 的起始 token ID（shape [1], Int32，仅当 has_embedding 时存在）
    constexpr const char* EMBED_TOKEN_ID = "embed_token_id";
    /// 可选：M-RoPE 3D position IDs（shape [3, seq_len], Int32，仅当 has_position_ids 时存在）
    constexpr const char* POSITION_IDS = "position_ids";

    // ==================== 中间激活值 tensors ====================
    constexpr const char* HIDDEN_STATES = "hidden_states";
    /// hidden_states 的 2D reshape [seq_len, hidden_size]，供 layernorm/linear 等需要 2D 的层使用
    constexpr const char* HIDDEN_STATES_RESHAPE = "hidden_states_reshape";
    constexpr const char* NORM_OUTPUT = "norm_output";
    constexpr const char* QKV_PROJ_OUTPUT = "qkv_proj_output";  // (legacy) Fused QKV [seq_len, q_dim+k_dim+v_dim]
    constexpr const char* Q_PROJ_OUTPUT = "q_proj_output";  // 独立 Q 输出 [seq_len, q_dim]
    constexpr const char* K_PROJ_OUTPUT = "k_proj_output";  // 独立 K 输出 [seq_len, k_dim]
    constexpr const char* V_PROJ_OUTPUT = "v_proj_output";  // 独立 V 输出 [seq_len, v_dim]
    constexpr const char* ATTENTION_OUTPUT = "attention_output";
    constexpr const char* MLP_INTERMEDIATE = "mlp_intermediate";
    constexpr const char* UP_PROJ_OUTPUT = "up_proj_output";
    constexpr const char* MLP_ACTIVATION_INPUT = "mlp_activation_input";  // [seq_len, 2 * intermediate_size] for up+gate concatenated
    constexpr const char* POST_NORM_OUTPUT = "post_norm_output";

    // ==================== 输出 tensors ====================
    constexpr const char* LOGITS = "logits";

    // ==================== Engine 内部 tensors（sampler / response 缓冲） ====================
    constexpr const char* SAMPLER_TOKEN_OUT = "sampler_token_out";
    constexpr const char* RESPONSE_TOKENS_DEVICE = "response_tokens_device";

    // ==================== CUDA graph support ====================
    constexpr const char* D_KV_LEN = "d_kv_len";  // device uint32_t, actual kv_len for attention

    // ==================== KV Cache tensors (per layer) ====================
    inline std::string k_write_layer(int32_t layer_id) { return "k_write_layer_" + std::to_string(layer_id); }
    inline std::string v_write_layer(int32_t layer_id) { return "v_write_layer_" + std::to_string(layer_id); }
    inline std::string k_cache_layer(int32_t layer_id) { return "k_cache_layer_" + std::to_string(layer_id); }
    inline std::string v_cache_layer(int32_t layer_id) { return "v_cache_layer_" + std::to_string(layer_id); }

    // ==================== Context tensors (MLA attention) ====================
    inline std::string context_write_layer(int32_t layer_id) { return "context_write_layer_" + std::to_string(layer_id); }
    inline std::string context_cache_layer(int32_t layer_id) { return "context_cache_layer_" + std::to_string(layer_id); }
}

class Model : public NonCopyable {
public:
    explicit Model(const EngineConfig& config);
    virtual ~Model() = 0;

    virtual void prefill(const Context& context) = 0;
    virtual void decode_step(const Context& context) = 0;

    /// 模型特定的 decode 阶段 position_ids 准备（如 M-RoPE）。默认空实现。
    virtual void prepare_decode_position_ids(Context& context, Device device, int32_t device_id);

    /// 模型特定的 decode 运行时状态推进。用于在每次 decode step 结束后
    /// 就地推进稳定地址缓冲（例如 M-RoPE position_ids）。
    virtual void advance_decode_runtime_tensors(Context& context, cudaStream_t stream);

    /// 当 decode graph steady-state 完全依赖稳定设备端 buffer 且不需要每步
    /// 重新构建 tensor 视图时返回 true，Engine 可跳过重复 prepare_decode_tensors。
    virtual bool has_static_decode_runtime_tensors() const;

    int32_t num_layers() const { return num_layers_; }
    int32_t hidden_size() const { return hidden_size_; }
    int32_t vocab_size() const { return vocab_size_; }
    DType dtype() const { return dtype_; }

    static std::unique_ptr<Model> create(const EngineConfig& config);

protected:
    const EngineConfig& engine_config_;

    int32_t num_layers_;
    int32_t hidden_size_;
    int32_t vocab_size_;
    DType dtype_;
    
    bool model_loaded_;
};

} // namespace edge_fm
