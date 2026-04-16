#pragma once

#include <edge-fm/core.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "utils/non_copyable.h"
#include "engine/engine.h"
#include <cuda_runtime.h>

namespace edge_fm {

/**
 * @brief Base class for all neural network layers
 * 
 * This class provides the common interface for all layer implementations,
 * including weight loading, resource allocation, and forward computation.
 */
class Layer : public NonCopyable {
public:
    /**
     * @brief Construct a layer with the given configuration
     * 
     * @param engine_config Engine configuration object containing engine settings,
     *                      including runtime configuration (e.g., device type, device_id,
     *                      num_threads, use_cuda_graph), model paths, and other engine-specific
     *                      parameters.
     * 
     * @note The engine_config is a unified configuration object that consolidates runtime
     *       settings, model configuration, and other engine parameters. Derived classes
     *       should extract the relevant configuration sections they need from this object.
     * 
     * @throws ConfigurationError if required weights are missing or config is invalid
     * @throws DeviceError if device operations fail
     * @throws OutOfMemoryError if memory allocation fails
     */
    explicit Layer(const EngineConfig& engine_config, std::string layer_name = "");
    
    virtual ~Layer() = default;

    /**
     * @brief 检查模型是否正确初始化
     * 
     * @return true 如果模型已正确初始化（构造函数已调用且 load_weights 已被调用）
     * @return false 如果模型未初始化
     */
    inline bool is_initialized() const { return weights_loaded_; }

    /**
     * @brief Load weights for this layer
     * 
     * @param prefill_weights Map of weight names to tensors for prefill stage
     * @param decode_weights Map of weight names to tensors for decode stage
     * 
     * @note Each derived layer decides which weights to use from the provided maps.
     *       For example:
     *       - EmbedHeadLayer might only use prefill_weights (embedding is shared)
     *       - LinearLayer might use both prefill_weights and decode_weights (different quantization)
     *       - SamplerLayer might not use any weights
     *       The exact weight names depend on the layer type.
     * 
     * @throws ConfigurationError if required weights are missing
     * @throws DeviceError if weight loading fails
     */
    virtual void load_weights(
        const std::unordered_map<std::string, Tensor>& prefill_weights,
        const std::unordered_map<std::string, Tensor>& decode_weights
    ) = 0;

    /**
     * @brief Forward pass through the layer
     * 
     * @param inputs Map of input tensor names to tensors
     * @param outputs Map of output tensor names to tensors
     * @param stream CUDA stream to use for the forward pass (default: nullptr)
     *               If nullptr, the default stream will be used.
     *
     * @return Map of output tensor names to tensors
     * 
     * @note The input and output tensor names are layer-specific.
     *       For example, an Attention layer might expect "hidden_states" as input
     *       and produce "attention_output" as output.
     * 
     * @throws InvalidRequestError if input tensors are invalid or missing required inputs
     * @throws DeviceError if computation fails
     */
    virtual void forward(
        const std::unordered_map<std::string, Tensor>& inputs,
        std::unordered_map<std::string, Tensor>& outputs,
        cudaStream_t stream = nullptr,
        ModelStage stage = ModelStage::Prefill
    ) = 0;

    virtual void reset_operator_impl_cache() {}

protected:
    EngineConfig engine_config_;
    Device device_;     ///< 设备类型
    int32_t device_id_; ///< 设备 ID
    bool weights_loaded_ = false; ///< 权重是否已加载
    std::string layer_name_; ///< 层的唯一标识名（用于 NVTX 等，空时不做 profiling）
};

} // namespace edge_fm
