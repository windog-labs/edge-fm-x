#include "layers/sampler.h"
#include <flashinfer/sampling.cuh>
#include <flashinfer/utils.cuh>
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include <cub/cub.cuh>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdint>

using namespace flashinfer;

namespace edge_fm {
namespace sampler_trt {

struct TopK_2 {
    float value;
    int32_t index;
    __device__ __forceinline__ void init() { value = -FLT_MAX; index = -1; }
    __device__ __forceinline__ void insert(float elem, int32_t elemId) {
        if (elem > value || (elem == value && elemId < index)) {
            value = elem;
            index = elemId;
        }
    }
};

struct topk2MaxOpFunctor {
    __device__ __forceinline__ TopK_2 operator()(TopK_2 const& a, TopK_2 const& b) const {
        if (a.value > b.value) return a;
        if (a.value < b.value) return b;
        return a.index < b.index ? a : b;
    }
};

// TensorRT-Edge-LLM style topK=1 (greedy). Ported from third_party/TensorRT-Edge-LLM/cpp/sampler/sampling.cu
template <int32_t BLOCK_SIZE, int32_t BLOCKS_PER_BEAM>
__global__ void topKStage1Greedy(
    float const* __restrict__ logits,
    int32_t* topKTmpIdBuf,
    float* topKTmpValBuf,
    int32_t batchSize,
    int32_t vocabSize)
{
    typedef cub::BlockReduce<TopK_2, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t bid = static_cast<int32_t>(blockIdx.x);
    int32_t batchId = bid / BLOCKS_PER_BEAM;
    int32_t blockLane = bid % BLOCKS_PER_BEAM;

    if (batchId >= batchSize) return;

    int32_t rowOffset = batchId * vocabSize;
    int32_t tmpTopKBufIndex = batchId * BLOCKS_PER_BEAM + blockLane;

    TopK_2 partial;
    partial.init();
    for (int32_t elemId = tid + blockLane * BLOCK_SIZE; elemId < vocabSize; elemId += BLOCK_SIZE * BLOCKS_PER_BEAM) {
        int32_t index = elemId + rowOffset;
        partial.insert(logits[index], index);
    }
    TopK_2 total = BlockReduce(tempStorage).Reduce(partial, topk2MaxOpFunctor());

    if (tid == 0) {
        topKTmpIdBuf[tmpTopKBufIndex] = total.index;
        topKTmpValBuf[tmpTopKBufIndex] = total.value;
    }
}

template <int BLOCK_SIZE>
__global__ void topKStage2Greedy(
    int32_t const* __restrict__ topKTmpIdBuf,
    float* topKTmpValBuf,
    int32_t* __restrict__ selectedIndices,
    int32_t batchSize,
    int32_t vocabSize)
{
    constexpr int32_t BLOCKS_PER_BEAM = 8;
    constexpr int32_t size = BLOCKS_PER_BEAM;
    constexpr int32_t stride = BLOCKS_PER_BEAM;

    int32_t tid = static_cast<int32_t>(threadIdx.x);
    int32_t batchIdx = static_cast<int32_t>(blockIdx.x);
    if (batchIdx >= batchSize) return;

    int32_t base = batchIdx * stride;
    float* sVal = topKTmpValBuf + base;
    int32_t const* sId = topKTmpIdBuf + base;

    if (tid == 0) {
        TopK_2 best;
        best.init();
        for (int32_t i = 0; i < size; ++i) {
            float v = sVal[i];
            int32_t vocabIdx = (sId[i] >= 0) ? (sId[i] % vocabSize) : vocabSize;
            best.insert(v, vocabIdx);
        }
        int32_t outputId = best.index;
        if (outputId < 0 || outputId >= vocabSize) outputId = vocabSize - 1;
        selectedIndices[batchIdx] = outputId;
    }
}

inline size_t getTopK1WorkspaceSize(int32_t batchSize, int32_t vocabSize) {
    (void)vocabSize;
    auto align = [](size_t s) -> size_t {
        const size_t a = 256;
        return (s + a - 1) & ~(a - 1);
    };
    size_t total = 0;
    total += align(static_cast<size_t>(batchSize) * 8 * sizeof(int32_t));
    total += align(static_cast<size_t>(batchSize) * 8 * sizeof(float));
    return total;
}

}  // namespace sampler_trt

SamplerLayer::SamplerLayer(const EngineConfig& engine_config)
    : Layer(engine_config)
{
    nlohmann::json model_config = engine_config_.prefill_model_config();
    vocab_size_ = model_config.value("vocab_size", 32000U);
    
    temperature_ = engine_config_.sampling_temperature();
    seed_ = engine_config_.sampling_seed();
    
    if (temperature_ < 0.0f) {
        throw ConfigurationError("temperature must be non-negative. Got: " + std::to_string(temperature_));
    }
}

void SamplerLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    [[maybe_unused]] ModelStage stage) 
{
    const auto& logits = inputs.at("logits");
    auto& token_ids = outputs.at("token_ids");
    this->forward_sampling(logits, token_ids, stream);
}

void SamplerLayer::forward_sampling(
    const Tensor& logits,
    Tensor& token_ids,
    cudaStream_t stream)
{
    if (!is_initialized()) {
        throw InvalidRequestError("SamplerLayer is not initialized");
    }
    
    // 验证设备和设备 ID
    auto logits_device = logits.device();
    auto token_ids_device = token_ids.device();
    
    // 检查设备类型必须匹配
    if (std::get<0>(logits_device) != device_) {
        throw DeviceError("Logits tensor must be on the same device type as the layer. "
                         "Expected: " + std::to_string(static_cast<int>(device_)) + 
                         ", Got: " + std::to_string(static_cast<int>(std::get<0>(logits_device))));
    }
    if (std::get<0>(token_ids_device) != device_) {
        throw DeviceError("Token IDs tensor must be on the same device type as the layer. "
                         "Expected: " + std::to_string(static_cast<int>(device_)) + 
                         ", Got: " + std::to_string(static_cast<int>(std::get<0>(token_ids_device))));
    }
    
    if (device_ == Device::GPU && std::get<1>(logits_device) != device_id_) {
        throw DeviceError("Logits tensor must be on the same device ID as the layer. "
                          "Expected device_id: " + std::to_string(device_id_) + 
                          ", Got: " + std::to_string(std::get<1>(logits_device)));
    }
    if (device_ == Device::GPU && std::get<1>(token_ids_device) != device_id_) {
        throw DeviceError("Token IDs tensor must be on the same device ID as the layer. "
                          "Expected device_id: " + std::to_string(device_id_) + 
                          ", Got: " + std::to_string(std::get<1>(token_ids_device)));
    }
    
    // 验证输入形状: [batch_size, vocab_size]
    const auto& logits_shape = logits.shape();
    if (logits_shape.size() != 2) {
        throw InvalidRequestError("Logits tensor must be 2D [batch_size, vocab_size]. Got " + 
                                 std::to_string(logits_shape.size()) + "D");
    }
    
    uint32_t batch_size = static_cast<uint32_t>(logits_shape[0]);
    uint32_t vocab_size_input = static_cast<uint32_t>(logits_shape[1]);
    
    if (vocab_size_input != vocab_size_) {
        throw InvalidRequestError(
            "Vocab size mismatch. Expected " + std::to_string(vocab_size_) + 
            ", got " + std::to_string(vocab_size_input));
    }
    
    // 验证输出形状: [batch_size]
    const auto& token_ids_shape = token_ids.shape();
    if (token_ids_shape.size() != 1 || token_ids_shape[0] != static_cast<int64_t>(batch_size)) {
        throw InvalidRequestError(
            "Token IDs tensor shape mismatch. Expected [" + std::to_string(batch_size) + 
            "], got [" + std::to_string(token_ids_shape[0]) + "]");
    }
    
    // 验证输出 dtype 为 Int32
    if (token_ids.dtype() != DType::Int32) {
        throw InvalidRequestError(
            "Token IDs tensor dtype must be Int32. Got dtype: " + 
            std::to_string(static_cast<int>(token_ids.dtype())));
    }
    
    // 验证输入 dtype（只支持 Float32）
    DType logits_dtype = logits.dtype();
    if (logits_dtype != DType::Float32) {
        throw InvalidRequestError(
            "Logits dtype must be Float32. FlashInfer library only supports float32. Got dtype: " + 
            std::to_string(static_cast<int>(logits_dtype)));
    }
    
    float* logits_data = static_cast<float*>(const_cast<void*>(logits.data_ptr()));
    int32_t* token_ids_data = static_cast<int32_t*>(token_ids.data_ptr());
    
    // Greedy: multi-block two-stage reduction for all batch sizes.
    // Uses 8 blocks x 256 threads = 2048 threads for parallel argmax over vocab_size.
    // Workspace cached via StaticBufferManager to avoid 50+ cudaMallocAsync per generate.
    constexpr float GREEDY_THRESHOLD = 1e-6f;
    if (temperature_ < GREEDY_THRESHOLD) {
        size_t ws_size = sampler_trt::getTopK1WorkspaceSize(
            static_cast<int32_t>(batch_size), static_cast<int32_t>(vocab_size_));
        void* ws = StaticBufferManager::get_cache_buf("greedy_sampler_ws", ws_size, device_id_);
        auto align = [](size_t s) -> size_t { return (s + 255) & ~255ULL; };
        size_t off = 0;
        int32_t* topKTmpIdBuf = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + off);
        off += align(batch_size * 8 * sizeof(int32_t));
        float* topKTmpValBuf = reinterpret_cast<float*>(static_cast<char*>(ws) + off);

        constexpr int BLOCK_SIZE = 256;
        constexpr int BLOCKS_PER_BEAM = 8;
        sampler_trt::topKStage1Greedy<BLOCK_SIZE, BLOCKS_PER_BEAM><<<
            batch_size * BLOCKS_PER_BEAM, BLOCK_SIZE, 0, stream>>>(
            logits_data, topKTmpIdBuf, topKTmpValBuf,
            static_cast<int32_t>(batch_size), static_cast<int32_t>(vocab_size_));
        sampler_trt::topKStage2Greedy<BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
            topKTmpIdBuf, topKTmpValBuf, token_ids_data,
            static_cast<int32_t>(batch_size), static_cast<int32_t>(vocab_size_));
        CUDA_CHECK_THROW(cudaGetLastError(), "greedy sampling failed");
        return;
    }

    float effective_temperature = temperature_;
    
    constexpr uint32_t SMALL_BATCH_THRESHOLD = 128;
    constexpr uint32_t LARGE_VOCAB_THRESHOLD = 24576;
    constexpr uint32_t DEFAULT_SLICE_SIZE = 8192;
    constexpr size_t PARTIAL_SOFTMAX_RESULT_SIZE = 8;
    
    size_t workspace_size = 1024 * 1024;
    if (batch_size <= SMALL_BATCH_THRESHOLD && vocab_size_ >= LARGE_VOCAB_THRESHOLD) {
        uint32_t num_slices = (vocab_size_ + DEFAULT_SLICE_SIZE - 1) / DEFAULT_SLICE_SIZE;
        workspace_size = batch_size * num_slices * PARTIAL_SOFTMAX_RESULT_SIZE;
    }
    
    void* workspace = StaticBufferManager::get_cache_buf(
        "sampler_softmax_workspace", workspace_size, device_id_);
    size_t probs_size = batch_size * vocab_size_ * sizeof(float);
    float* probs_buffer = static_cast<float*>(StaticBufferManager::get_cache_buf(
        "sampler_probs_buffer", probs_size, device_id_));
    
    CUDA_CHECK_THROW(
        sampling::OnlineSoftmax<float>(
            logits_data,
            probs_buffer,
            batch_size,
            vocab_size_,
            nullptr,
            effective_temperature,
            workspace,
            workspace_size,
            false,
            stream
        ),
        "OnlineSoftmax failed"
    );
    
    cudaError_t err = sampling::SamplingFromProb<float, int32_t>(
        probs_buffer,
        token_ids_data,
        nullptr,
        batch_size,
        vocab_size_,
        false,
        seed_,
        0,
        stream
    );
    
    CUDA_CHECK_THROW(err, "Sampling failed");
}

} // namespace edge_fm
