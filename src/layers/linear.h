
#pragma once
#include "layer.h"
#include <edge-fm/core.h>
#include "engine/engine.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cublasLt.h>

namespace edge_fm {

class LinearOpRegistry;
class LinearCublasLtImpl;
class LinearDecodeM1TiledImpl;
class LinearCutlassImpl;
class LinearCutileImpl;
class LinearAgentImpl;

class LinearLayer : public Layer {
public:
    explicit LinearLayer(const std::string& layer_prefix,
                         const EngineConfig& engine_config,
                         uint32_t in_features,
                         uint32_t out_features,
                         std::string layer_name = "");
    ~LinearLayer() override;

    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;

    void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) override;

    // Forward implementations for different quantization types (public for testing)
    void forward_fp16_bf16(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    );
    
    void forward_int4_groupwise(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    );

protected:
    friend class LinearOpRegistry;
    friend class LinearCublasLtImpl;
    friend class LinearDecodeM1TiledImpl;
    friend class LinearCutlassImpl;
    friend class LinearCutileImpl;
    friend class LinearAgentImpl;

    // Quantization type enumeration (protected for subclass access)
    enum class QuantType {
        FP16_BF16,      ///< FP16 or BFloat16 (standard)
        INT4_GROUPWISE, ///< INT4 group-wise quantized
        INT8,           ///< INT8 quantized (future)
        FP4,            ///< FP4 quantized (future)
    };
    // Weight set structure for different quantization types (protected for subclass access)
    struct WeightSet {
        QuantType quant_type_ = QuantType::FP16_BF16;
        const Tensor* weight_ = nullptr;         ///< Unified weight tensor (type determined by quant_type_)
        const Tensor* bias_ = nullptr;           ///< Optional bias tensor [out_features] (for FP16/BF16)
        const Tensor* packed_weight_ = nullptr;  ///< Optional packed decode-only weight tensor
        std::shared_ptr<Tensor> packed_weight_storage_;  ///< Owns packed_weight_ when present
        const Tensor* scaling_factors_ = nullptr; ///< Scaling factors (for INT4 group-wise: [in_features/group_size, out_features])
        uint32_t group_size_ = 128;              ///< Group size for group-wise quantization (e.g., 128 for INT4)
    };

    struct LinearShapeSignature {
        int32_t m = -1;
        DType input_dtype = DType::Float16;
        DType weight_dtype = DType::Float16;
        DType output_dtype = DType::Float16;
        uint32_t in_features = 0;
        uint32_t out_features = 0;

        std::string to_string() const;
    };

    struct LinearOpContext {
        std::string layer_prefix;
        std::string layer_role;
        ModelStage stage = ModelStage::Prefill;
        LinearShapeSignature shape;
        bool has_bias = false;
    };

    struct CachedDescriptors;

    class LinearImpl {
    public:
        virtual ~LinearImpl() = default;

        virtual std::string impl_id() const = 0;
        virtual bool supports(const LinearOpContext& ctx, const WeightSet& weight_set) const = 0;
        virtual void prepare(
            LinearLayer& owner,
            const LinearOpContext& ctx,
            const WeightSet& weight_set,
            const Tensor& input,
            Tensor& output,
            cudaStream_t stream,
            CachedDescriptors& cached) = 0;
        virtual void forward(
            LinearLayer& owner,
            const LinearOpContext& ctx,
            const WeightSet& weight_set,
            const Tensor& input,
            Tensor& output,
            cudaStream_t stream,
            CachedDescriptors& cached) = 0;
    };

    // Helper function to load a single WeightSet
    void load_weight_set(
        const std::unordered_map<std::string, Tensor>& weights,
        const std::string& weight_name_base,
        WeightSet& weight_set);

    void maybe_prepare_decode_m1_packed_weight(
        WeightSet& weight_set,
        const std::string& stage_tag);
    
    // cuBLASLt descriptor cache dimensions:
    // - Layer type: LinearLayer, FusedQKVLinearLayer, FusedGateUpLinearLayer, LMHeadLinearLayer
    // - Stage: Prefill vs Decode
    // - Prefill: multiple m (slot sizes) -> map m -> CachedDescriptors
    // - Decode: m=1 always -> single CachedDescriptors
    struct CachedDescriptors {
        cublasLtMatmulDesc_t matmul_desc_ = nullptr;
        cublasLtMatrixLayout_t Adesc_ = nullptr;
        cublasLtMatrixLayout_t Bdesc_ = nullptr;
        cublasLtMatrixLayout_t Cdesc_ = nullptr;
        cublasLtMatrixLayout_t Ddesc_ = nullptr;
        cublasLtMatmulHeuristicResult_t heuristic_ = {};
        bool has_algo_ = false;
        int cached_m_ = -1;  // Cached batch_size
        cudaDataType_t cached_input_type_ = CUDA_R_16F;
        cudaDataType_t cached_weight_type_ = CUDA_R_16F;
        cudaDataType_t cached_output_type_ = CUDA_R_16F;
        bool has_bias_ = false;
        std::vector<cublasLtMatmulHeuristicResult_t> heuristic_candidates_;
        int best_algo_index_ = -1;
        std::string selected_impl_id_;
        nlohmann::json selected_impl_params_ = nlohmann::json::object();
    };
    
    // Helper function to get or create descriptors
    void get_or_create_descriptors(
        int m,
        cudaDataType_t input_type,
        cudaDataType_t weight_type,
        cudaDataType_t output_type,
        const void* bias_ptr,
        CachedDescriptors& cached,
        cublasLtMatmulDesc_t& matmul_desc,
        cublasLtMatrixLayout_t& Adesc,
        cublasLtMatrixLayout_t& Bdesc,
        cublasLtMatrixLayout_t& Cdesc,
        cublasLtMatrixLayout_t& Ddesc
    );
    
    // Helper function to cleanup cached descriptors
    void cleanup_cached_descriptors(CachedDescriptors& cached);

    LinearImpl* find_impl_by_id(const std::string& impl_id) const;
    LinearImpl* resolve_impl(
        const LinearOpContext& ctx,
        const WeightSet& weight_set,
        CachedDescriptors& cached) const;
    
    // layer information
    uint32_t in_features_;
    uint32_t out_features_;
    std::string layer_prefix_;  ///< 层名称前缀（例如："model.layers.0.mlp.gate_proj"）
    std::string layer_role_;

    // Weight sets for prefill and decode stages
    WeightSet prefill_weights_;
    WeightSet decode_weights_;
    // CUBLASLt handle for FP16/BF16 operations with bias support
    cublasLtHandle_t cublaslt_handle_;
    // Cached descriptors: prefill per-m (different slot sizes), decode m=1 only
    std::unordered_map<int, CachedDescriptors> prefill_descriptors_map_;
    CachedDescriptors decode_descriptors_;
};

/**
 * @brief Fused QKV Linear Layer that combines Q, K, V projections into a single linear layer
 * 
 * This layer merges three separate linear projections (Q, K, V) into one for better performance.
 * The weights are concatenated along the output dimension: [in_features, q_out + k_out + v_out]
 */
class FusedQKVLinearLayer : public LinearLayer {
public:
    /**
     * @brief Constructor for FusedQKVLinearLayer
     * @param layer_prefix_base Base prefix for the layer (e.g., "model.layers.0.attn")
     * @param engine_config Engine configuration
     * @param in_features Input feature dimension
     * @param q_out_features Q projection output dimension
     * @param k_out_features K projection output dimension
     * @param v_out_features V projection output dimension
     */
    explicit FusedQKVLinearLayer(
        const std::string& layer_prefix_base,
        const EngineConfig& engine_config,
        uint32_t in_features,
        uint32_t q_out_features,
        uint32_t k_out_features,
        uint32_t v_out_features,
        std::string layer_name = "")
        : LinearLayer(layer_prefix_base + ".qkv_fused", engine_config, in_features, 
                      q_out_features + k_out_features + v_out_features, std::move(layer_name)),
          in_features_(in_features),
          q_out_features_(q_out_features),
          k_out_features_(k_out_features),
          v_out_features_(v_out_features),
          layer_prefix_base_(layer_prefix_base)
    {
        // Base class (LinearLayer) handles initialization
    }

    /**
     * @brief Load and merge Q, K, V weights into a single fused weight tensor
     */
    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;

private:
    // Input and output dimensions
    uint32_t in_features_;  // Store in_features since base class member is private
    uint32_t q_out_features_;
    uint32_t k_out_features_;
    uint32_t v_out_features_;
    
    // Base layer prefix (e.g., "model.layers.0.attn")
    std::string layer_prefix_base_;
    
    // Helper function to merge weights
    void merge_weights(
        const std::unordered_map<std::string, Tensor>& weights,
        Tensor& fused_weight,
        Tensor& fused_bias,
        cudaStream_t stream);
};

/**
 * @brief Fused Gate+Up Linear Layer for MLP
 * 
 * Merges gate_proj and up_proj into a single linear layer for better performance.
 * Output layout: [gate: gate_out_features, up: up_out_features, in_features]
 */
class FusedGateUpLinearLayer : public LinearLayer {
public:
    /**
     * @brief Constructor for FusedGateUpLinearLayer
     * @param layer_prefix_base Base prefix for the layer (e.g., "model.layers.0.mlp")
     * @param engine_config Engine configuration
     * @param in_features Input feature dimension
     * @param gate_out_features Gate projection output dimension
     * @param up_out_features Up projection output dimension
     */
    explicit FusedGateUpLinearLayer(
        const std::string& layer_prefix_base,
        const EngineConfig& engine_config,
        uint32_t in_features,
        uint32_t gate_out_features,
        uint32_t up_out_features,
        std::string layer_name = "")
        : LinearLayer(layer_prefix_base + ".gate_up_fused", engine_config, in_features, 
                      gate_out_features + up_out_features, std::move(layer_name)),
          in_features_(in_features),
          gate_out_features_(gate_out_features),
          up_out_features_(up_out_features),
          layer_prefix_base_(layer_prefix_base)
    {
        // Base class (LinearLayer) handles initialization
    }

    /**
     * @brief Load and merge gate, up weights into a single fused weight tensor
     */
    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;

private:
    // Input and output dimensions
    uint32_t in_features_;  // Store in_features since base class member is private
    uint32_t gate_out_features_;
    uint32_t up_out_features_;
    
    // Base layer prefix (e.g., "model.layers.0.mlp")
    std::string layer_prefix_base_;
    
    // Helper function to merge weights
    void merge_weights(
        const std::unordered_map<std::string, Tensor>& weights,
        Tensor& fused_weight,
        Tensor& fused_bias,
        cudaStream_t stream);
};

/**
 * @brief LM head linear layer
 *
 * Supports both:
 * - Separate lm_head.weight (e.g., Qwen2.5-7B, tie_word_embeddings=False)
 * - Tied weights with embedding table (e.g., model.embed_tokens.weight, tie_word_embeddings=True)
 *
 * Tries candidate weight names in order: lm_head.weight, model.lm_head.weight, model.embed_tokens.weight
 */
class LMHeadLinearLayer : public LinearLayer {
public:
    /**
     * @brief Constructor for LMHeadLinearLayer
     * @param layer_prefix Weight name prefix (e.g., "lm_head" -> tries "lm_head.weight")
     * @param engine_config Engine configuration
     * @param in_features Input feature dimension (hidden_size)
     * @param out_features Output feature dimension (vocab_size)
     */
    explicit LMHeadLinearLayer(
        const std::string& layer_prefix,
        const EngineConfig& engine_config,
        uint32_t in_features,
        uint32_t out_features,
        std::string layer_name = "");

    /**
     * @brief Load weights from lm_head.weight or model.embed_tokens.weight (tied)
     */
    void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) override;
};

} // namespace edge_fm
