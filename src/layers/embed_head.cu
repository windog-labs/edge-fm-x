#include "layers/embed_head.h"
#include "utils/device/nvtx.h"
#include <nlohmann/json.hpp>
#include "utils/device/cuda_utils.h"
#include "utils/check.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include <algorithm>
#include <limits>

namespace edge_fm {

namespace {

// Helper function to convert float to dtype
template<typename dtype_t> __device__ __forceinline__ dtype_t float_to_dtype(float val);
// Specializations for float to __half and __nv_bfloat16
template<> __device__ __forceinline__ __half float_to_dtype<__half>(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 float_to_dtype<__nv_bfloat16>(float val) { return __float2bfloat16(val); }

template<typename dtype_t>
__global__ void embedding_lookup_kernel(
    const int32_t* __restrict__ input_ids,
    const dtype_t* __restrict__ embedding_table,
    dtype_t* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int32_t vocab_size,
    int64_t hidden_size)
{
    // Each warp handles one hidden state (one token's embedding)
    constexpr uint32_t warpSize = 32;
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;
    
    if (warpId >= batch_size * seq_len) {
        return;
    }
    
    uint32_t const batchIdx = warpId / seq_len;
    uint32_t const tokenIdx = warpId % seq_len;
    
    int32_t const tokenId = input_ids[batchIdx * seq_len + tokenIdx];
    bool const isValidToken = (tokenId >= 0 && tokenId < vocab_size);
    
    uint32_t const baseOutputIdx = warpId * hidden_size;
    
    for (uint32_t offset = laneId; offset < hidden_size; offset += warpSize) {
        dtype_t value;
        if (isValidToken) {
            uint32_t const embeddingOffset = tokenId * hidden_size + offset;
            value = embedding_table[embeddingOffset];
        } else {
            value = float_to_dtype<dtype_t>(0.0f);
        }
        
        uint32_t const outputIdx = baseOutputIdx + offset;
        output[outputIdx] = value;
    }
}

template<typename dtype_t>
__global__ void embedding_lookup_with_embedding_insertion_kernel(
    const int32_t* __restrict__ input_ids,
    const dtype_t* __restrict__ embedding_table,
    const dtype_t* __restrict__ custom_embeds,
    dtype_t* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int32_t vocab_size,
    int64_t hidden_size,
    int32_t embed_token_id,
    int64_t num_custom_embeddings)
{
    constexpr uint32_t warpSize = 32;
    
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;
    
    if (warpId >= batch_size * seq_len) {
        return;
    }
    
    uint32_t const batchIdx = warpId / seq_len;
    uint32_t const tokenIdx = warpId % seq_len;
    
    int32_t const tokenId = input_ids[batchIdx * seq_len + tokenIdx];
    
    // Check if this is a custom embedding token (similar to embeddingKernels.cu)
    // If tokenId >= embed_token_id, it's a custom embedding token
    // The index in custom_embeds is: tokenId - embed_token_id
    bool const isCustomEmbedToken = (embed_token_id >= 0) && 
                                    (tokenId >= embed_token_id) && 
                                    (tokenId < embed_token_id + static_cast<int32_t>(num_custom_embeddings));
    
    const dtype_t* sourceTable = nullptr;
    uint32_t baseEmbeddingOffset = 0;
    
    if (isCustomEmbedToken && custom_embeds != nullptr) {
        int32_t const customEmbedIdx = tokenId - embed_token_id;
        
        if (customEmbedIdx >= 0 && customEmbedIdx < static_cast<int32_t>(num_custom_embeddings)) {
            baseEmbeddingOffset = customEmbedIdx * hidden_size;
            sourceTable = custom_embeds;
        } else {
            sourceTable = nullptr;
        }
    } else {
        if (tokenId >= 0 && tokenId < vocab_size) {
            baseEmbeddingOffset = tokenId * hidden_size;
            sourceTable = embedding_table;
        } else {
            sourceTable = nullptr;
        }
    }
    
    uint32_t const baseOutputIdx = warpId * hidden_size;
    
    for (uint32_t offset = laneId; offset < hidden_size; offset += warpSize) {
        dtype_t value;
        if (sourceTable != nullptr) {
            uint32_t const embeddingOffset = baseEmbeddingOffset + offset;
            value = sourceTable[embeddingOffset];
        } else {
            value = float_to_dtype<dtype_t>(0.0f);
        }
        
        uint32_t const outputIdx = baseOutputIdx + offset;
        output[outputIdx] = value;
    }
}

template<typename dtype_t>
void launch_embedding_lookup_kernel(
    const int32_t* input_ids,
    const void* embedding_table,
    void* output,
    int64_t batch_size,
    int64_t seq_len,
    int32_t vocab_size,
    int64_t hidden_size,
    cudaStream_t stream,
    const void* custom_embeds = nullptr,
    int32_t embed_token_id = -1,
    int64_t num_custom_embeddings = 0)
{
    dim3 const threadsPerBlock(32, 4);
    uint32_t const gridSize = (batch_size * seq_len + 3) / 4;
    
    const dtype_t* embedding_table_typed = static_cast<const dtype_t*>(embedding_table);
    dtype_t* output_typed = static_cast<dtype_t*>(output);
    
    if (custom_embeds != nullptr && embed_token_id >= 0) {
        const dtype_t* custom_embeds_typed = static_cast<const dtype_t*>(custom_embeds);
        embedding_lookup_with_embedding_insertion_kernel<dtype_t><<<gridSize, threadsPerBlock, 0, stream>>>(
            input_ids, embedding_table_typed, custom_embeds_typed, output_typed,
            batch_size, seq_len, vocab_size, hidden_size, embed_token_id, num_custom_embeddings);
    } else {
        embedding_lookup_kernel<dtype_t><<<gridSize, threadsPerBlock, 0, stream>>>(
            input_ids, embedding_table_typed, output_typed, batch_size, seq_len, vocab_size, hidden_size);
    }
    CUDA_CHECK_THROW(cudaGetLastError(), "embedding_lookup_kernel launch failed");
}

} // anonymous namespace

EmbedHeadLayer::EmbedHeadLayer(const EngineConfig& engine_config, std::string layer_name)
    : Layer(engine_config, std::move(layer_name))
{
    auto model_config = engine_config_.prefill_model_config();
    vocab_size_ = model_config.value("vocab_size", 32000U);
    hidden_size_ = model_config.value("hidden_size", 4096U);
}

void EmbedHeadLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor>& decode_weights)
{
    // Embedding 通常 prefill 和 decode 共享，只从 prefill_weights 加载
    std::vector<std::string> possible_weights_ = {"model.embed_tokens.weight"};
    for (const auto& weight_name : possible_weights_) { 
        auto it = prefill_weights.find(weight_name);
        if (it != prefill_weights.end()) {
            embedding_table_ = &it->second;
            break;
        }
    }
    check(embedding_table_ != nullptr,
          "EmbedHeadLayer: missing 'model.embed_tokens.weight' in prefill_weights");
    
    const auto& shape = embedding_table_->shape();
    check(shape.size() == 2,
          "EmbedHeadLayer: embedding table must be 2D [vocab_size, hidden_size]. "
          "Got shape with " + std::to_string(shape.size()) + " dimensions");
    
    check(shape[0] == static_cast<int64_t>(vocab_size_),
          "EmbedHeadLayer: embedding table vocab_size mismatch. "
          "Expected " + std::to_string(vocab_size_) + 
          ", got " + std::to_string(shape[0]));
    
    check(shape[1] == static_cast<int64_t>(hidden_size_),
          "EmbedHeadLayer: embedding table hidden_size mismatch. "
          "Expected " + std::to_string(hidden_size_) + 
          ", got " + std::to_string(shape[1]));
    
    check(embedding_table_->dtype() == DType::Float16 || embedding_table_->dtype() == DType::BFloat16,
          "EmbedHeadLayer: embedding table dtype must be Float16 or BFloat16. "
          "Got dtype: " + std::to_string(static_cast<int>(embedding_table_->dtype())));
    
    weights_loaded_ = true;
}

void EmbedHeadLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    [[maybe_unused]] ModelStage stage)
{
    check<InvalidRequestError>(is_initialized(), "EmbedHeadLayer is not initialized");
    const auto& token_ids = inputs.at("token_ids");
    auto& output = outputs.at("output");
    
    // Check if embeddings are provided
    auto embeddings_it = inputs.find("embeddings");
    if (embeddings_it != inputs.end()) {
        const auto& embeddings = embeddings_it->second;
        int32_t embed_token_id = -1;
        auto embed_token_id_it = inputs.find("embed_token_id");
        if (embed_token_id_it != inputs.end()) {
            const Tensor& tid_tensor = embed_token_id_it->second;
            check<InvalidRequestError>(
                tid_tensor.shape().size() == 1 && tid_tensor.shape()[0] >= 1 && tid_tensor.dtype() == DType::Int32,
                "embed_token_id tensor must be 1D Int32 with at least one element");
            const int32_t* ptr = static_cast<const int32_t*>(tid_tensor.data_ptr());
            embed_token_id = ptr[0];
        }
        forward_for_embeddings(token_ids, embeddings, output, embed_token_id, stream);
    } else {
        forward_for_tokens(token_ids, output, stream);
    }
}

void EmbedHeadLayer::forward_for_tokens(
    const Tensor& token_ids,
    Tensor& output,
    cudaStream_t stream)
{
    NVTX::Range r(layer_name_);
    DType dtype = embedding_table_->dtype();
    check<ConfigurationError>(
        dtype == DType::Float16 || dtype == DType::BFloat16,
        "EmbedHeadLayer: embedding table dtype must be Float16 or BFloat16. "
        "Got dtype: " + std::to_string(static_cast<int>(dtype)));
    
    auto token_device = token_ids.device();
    check<DeviceError>(
        std::get<0>(token_device) == device_ && std::get<1>(token_device) == device_id_,
        "Input token_ids must be on the same device as the layer");
    
    auto output_device = output.device();
    check<DeviceError>(
        std::get<0>(output_device) == device_ && std::get<1>(output_device) == device_id_,
        "Output tensor must be on the same device as the layer");
    
    const auto& token_shape = token_ids.shape();
    check<InvalidRequestError>(
        token_shape.size() == 2,
        "Token IDs tensor must be 2D [batch_size, seq_len]. "
        "Got " + std::to_string(token_shape.size()) + "D");
    
    int64_t batch_size = token_shape[0];
    int64_t seq_len = token_shape[1];
    
    check<InvalidRequestError>(
        token_ids.dtype() == DType::Int32,
        "Token IDs dtype must be Int32. Got dtype: " + 
        std::to_string(static_cast<int>(token_ids.dtype())));
    
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(
        output_shape.size() == 3,
        "Output tensor must be 3D [batch_size, seq_len, hidden_size]. "
        "Got " + std::to_string(output_shape.size()) + "D");
    
    check<InvalidRequestError>(
        output_shape[0] == batch_size && output_shape[1] == seq_len && 
        output_shape[2] == static_cast<int64_t>(hidden_size_),
        "Output tensor shape mismatch. Expected [" + 
        std::to_string(batch_size) + ", " + std::to_string(seq_len) + ", " + 
        std::to_string(hidden_size_) + "], got [" +
        std::to_string(output_shape[0]) + ", " + std::to_string(output_shape[1]) + 
        ", " + std::to_string(output_shape[2]) + "]");
    
    check<InvalidRequestError>(
        output.dtype() == embedding_table_->dtype(),
        "Output dtype must match embedding table dtype. "
        "Expected " + std::to_string(static_cast<int>(embedding_table_->dtype())) +
        ", got " + std::to_string(static_cast<int>(output.dtype())));
    
    // Get data pointers
    const int32_t* token_ids_data = static_cast<const int32_t*>(token_ids.data_ptr());
    void* output_data_ptr = output.data_ptr();
    const void* embedding_table_data = embedding_table_->data_ptr();
    
    if (dtype == DType::Float16) {
        launch_embedding_lookup_kernel<__half>(
            token_ids_data, embedding_table_data, output_data_ptr,
            batch_size, seq_len, static_cast<int32_t>(vocab_size_),
            static_cast<int64_t>(hidden_size_), stream);
    } else {
        launch_embedding_lookup_kernel<__nv_bfloat16>(
            token_ids_data, embedding_table_data, output_data_ptr,
            batch_size, seq_len, static_cast<int32_t>(vocab_size_),
            static_cast<int64_t>(hidden_size_), stream);
    }
}

void EmbedHeadLayer::forward_for_embeddings(
    const Tensor& token_ids,
    const Tensor& embeddings,
    Tensor& output,
    int32_t embed_token_id,
    cudaStream_t stream)
{
    NVTX::Range r(layer_name_);
    DType dtype = embedding_table_->dtype();
    check<ConfigurationError>(
        dtype == DType::Float16 || dtype == DType::BFloat16,
        "EmbedHeadLayer: embedding table dtype must be Float16 or BFloat16. "
        "Got dtype: " + std::to_string(static_cast<int>(dtype)));
    
    auto token_device = token_ids.device();
    check<DeviceError>(
        std::get<0>(token_device) == device_ && std::get<1>(token_device) == device_id_,
        "Input token_ids must be on the same device as the layer");
    
    auto embedding_device = embeddings.device();
    check<DeviceError>(
        std::get<0>(embedding_device) == device_ && std::get<1>(embedding_device) == device_id_,
        "Input embeddings must be on the same device as the layer");
    
    auto output_device = output.device();
    check<DeviceError>(
        std::get<0>(output_device) == device_ && std::get<1>(output_device) == device_id_,
        "Output tensor must be on the same device as the layer");
    
    const auto& token_shape = token_ids.shape();
    check<InvalidRequestError>(
        token_shape.size() == 2,
        "Token IDs tensor must be 2D [batch_size, seq_len]. "
        "Got " + std::to_string(token_shape.size()) + "D");
    
    int64_t batch_size = token_shape[0];
    int64_t seq_len = token_shape[1];
    
    check<InvalidRequestError>(
        token_ids.dtype() == DType::Int32,
        "Token IDs dtype must be Int32. Got dtype: " + 
        std::to_string(static_cast<int>(token_ids.dtype())));
    
    const auto& embedding_shape = embeddings.shape();
    check<InvalidRequestError>(
        embedding_shape.size() == 2,
        "Embeddings tensor must be 2D [num_custom_embeddings, hidden_size]. "
        "Got " + std::to_string(embedding_shape.size()) + "D");
    
    int64_t num_custom_embeddings = embedding_shape[0];
    check<InvalidRequestError>(
        embedding_shape[1] == static_cast<int64_t>(hidden_size_),
        "Embeddings hidden_size mismatch. Expected " + std::to_string(hidden_size_) +
        ", got " + std::to_string(embedding_shape[1]));
    
    check<InvalidRequestError>(
        embeddings.dtype() == embedding_table_->dtype(),
        "Embeddings dtype must match embedding table dtype. "
        "Expected " + std::to_string(static_cast<int>(embedding_table_->dtype())) +
        ", got " + std::to_string(static_cast<int>(embeddings.dtype())));
    
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(
        output_shape.size() == 3,
        "Output tensor must be 3D [batch_size, seq_len, hidden_size]. "
        "Got " + std::to_string(output_shape.size()) + "D");
    
    check<InvalidRequestError>(
        output_shape[0] == batch_size && output_shape[1] == seq_len && 
        output_shape[2] == static_cast<int64_t>(hidden_size_),
        "Output tensor shape mismatch. Expected [" + 
        std::to_string(batch_size) + ", " + std::to_string(seq_len) + ", " + 
        std::to_string(hidden_size_) + "], got [" +
        std::to_string(output_shape[0]) + ", " + std::to_string(output_shape[1]) + 
        ", " + std::to_string(output_shape[2]) + "]");
    
    check<InvalidRequestError>(
        output.dtype() == embedding_table_->dtype(),
        "Output dtype must match embedding table dtype. "
        "Expected " + std::to_string(static_cast<int>(embedding_table_->dtype())) +
        ", got " + std::to_string(static_cast<int>(output.dtype())));
    
    // Validate embed_token_id
    // If embed_token_id >= 0, it's used to mark custom embedding tokens
    // Custom embedding tokens should be in range [embed_token_id, embed_token_id + num_custom_embeddings)
    if (embed_token_id >= 0) {
        check<InvalidRequestError>(
            embed_token_id + static_cast<int32_t>(num_custom_embeddings) <= std::numeric_limits<int32_t>::max(),
            "embed_token_id + num_custom_embeddings exceeds int32_t max value");
    }
    
    // Get data pointers
    const int32_t* token_ids_data = static_cast<const int32_t*>(token_ids.data_ptr());
    const void* embedding_table_data = embedding_table_->data_ptr();
    const void* embeddings_data = embeddings.data_ptr();
    void* output_data_ptr = output.data_ptr();
    
    if (dtype == DType::Float16) {
        launch_embedding_lookup_kernel<__half>(
            token_ids_data, embedding_table_data, output_data_ptr,
            batch_size, seq_len, static_cast<int32_t>(vocab_size_),
            static_cast<int64_t>(hidden_size_), stream,
            embeddings_data, embed_token_id, num_custom_embeddings);
    } else {
        launch_embedding_lookup_kernel<__nv_bfloat16>(
            token_ids_data, embedding_table_data, output_data_ptr,
            batch_size, seq_len, static_cast<int32_t>(vocab_size_),
            static_cast<int64_t>(hidden_size_), stream,
            embeddings_data, embed_token_id, num_custom_embeddings);
    }
}

} // namespace edge_fm