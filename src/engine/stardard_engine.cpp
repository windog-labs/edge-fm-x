#include "engine/stardard_engine.h"
#include "layers/sampler.h"
#include "models/model.h"
#include "operators/operator_impl_table.h"
#include "utils/device/memory.h"
#include "engine/kv_manager.h"
#include "utils/device/decode_runtime_kernels.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/nvtx.h"
#include "utils/check.h"
#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace edge_fm {


namespace {

size_t tensor_nbytes(const Tensor& tensor) {
    size_t elements = 1;
    for (int64_t dim : tensor.shape()) {
        elements *= static_cast<size_t>(dim);
    }
    return elements * get_dtype_size(tensor.dtype());
}

Tensor make_tensor_view(const Tensor& tensor) {
    auto [device, device_id] = tensor.device();
    return Tensor::view(tensor.data_ptr(), tensor.shape(), tensor.dtype(), device, device_id);
}

Tensor last_token_logits_view(const Tensor& logits) {
    const auto& shape = logits.shape();
    int64_t row_width = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        row_width *= shape[i];
    }

    void* last_row_ptr = static_cast<char*>(logits.data_ptr())
        + (shape.front() - 1) * row_width * static_cast<int64_t>(get_dtype_size(logits.dtype()));
    auto [device, device_id] = logits.device();
    return Tensor::view(last_row_ptr, {1, row_width}, logits.dtype(), device, device_id);
}

int32_t prefill_token_count(const Context& context) {
    const Tensor& token_ids = context.tensors().at(ModelTensors::TOKEN_IDS);
    const auto& shape = token_ids.shape();
    if (shape.empty()) {
        throw InternalError("TOKEN_IDS tensor for prefill must have rank >= 1");
    }
    return static_cast<int32_t>(shape.back());
}

int32_t request_prefill_seq_len(const Context& context) {
    const Request* request = context.request();
    if (request == nullptr) {
        throw InternalError("Context request must not be null");
    }

    const auto& token_ids = request->token_ids();
    int32_t seq_len = static_cast<int32_t>(token_ids.size());
    size_t prefix_size = context.prefix_size();
    if (prefix_size > 0 && token_ids.size() > prefix_size) {
        seq_len -= static_cast<int32_t>(prefix_size);
    }
    return seq_len;
}

bool tensor_is_on_device(const Tensor& tensor, Device device, int32_t device_id) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    return tensor_device == device && tensor_device_id == device_id;
}

bool request_supports_prefill_cuda_graph(const Request& request, Device device, int32_t device_id) {
    if (request.has_embedding() && !tensor_is_on_device(request.embedding(), device, device_id)) {
        return false;
    }
    if (request.has_position_ids() && !tensor_is_on_device(request.position_ids(), device, device_id)) {
        return false;
    }
    return true;
}

struct DecodeRuntimeStateLayout {
    static constexpr size_t kTokenOffset = 0;
    static constexpr size_t kKvLenOffset = kTokenOffset + sizeof(int32_t);
    static constexpr size_t kPositionOffset = kKvLenOffset + sizeof(uint32_t);
    static constexpr size_t kPositionElems = 3;
    static constexpr size_t kBytes = kPositionOffset + kPositionElems * sizeof(int32_t);

    static void* base_ptr(int32_t device_id) {
        return StaticBufferManager::get_cache_buf("decode_runtime_state", kBytes, device_id);
    }

    static void* token_ids_ptr(void* base_ptr) {
        return static_cast<void*>(static_cast<uint8_t*>(base_ptr) + kTokenOffset);
    }

    static void* kv_len_ptr(void* base_ptr) {
        return static_cast<void*>(static_cast<uint8_t*>(base_ptr) + kKvLenOffset);
    }

    static void* position_ids_ptr(void* base_ptr) {
        return static_cast<void*>(static_cast<uint8_t*>(base_ptr) + kPositionOffset);
    }
};

struct DecodeWritePtrs {
    std::vector<void*> k;
    std::vector<void*> v;
};

DecodeWritePtrs collect_decode_write_ptrs_from_tensors(Context& context, int32_t num_layers) {
    auto& tensors = context.tensors();
    DecodeWritePtrs write_ptrs;
    write_ptrs.k.resize(num_layers);
    write_ptrs.v.resize(num_layers);
    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        write_ptrs.k[layer_id] = tensors.at(ModelTensors::k_write_layer(layer_id)).data_ptr();
        write_ptrs.v[layer_id] = tensors.at(ModelTensors::v_write_layer(layer_id)).data_ptr();
    }
    return write_ptrs;
}

DecodeWritePtrs collect_decode_write_ptrs_from_context(
    const Context& context,
    const KVManager& kv_manager,
    int32_t num_layers)
{
    DecodeWritePtrs write_ptrs;
    write_ptrs.k.resize(num_layers);
    write_ptrs.v.resize(num_layers);

    if (kv_manager.get_attention_type() == AttentionType::MLA) {
        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            write_ptrs.k[layer_id] = context.kv_write_ptrs_ref()[layer_id];
            write_ptrs.v[layer_id] = context.kv_write_ptrs_ref()[layer_id];
        }
        return write_ptrs;
    }

    const auto& kv_read_ptrs = context.kv_read_ptrs_ref();
    const auto& kv_write_ptrs = context.kv_write_ptrs_ref();
    const size_t token_stride = kv_manager.get_token_stride();
    const size_t k_size_per_token = token_stride / 2;
    const int32_t max_tokens = context.slot_max_tokens();

    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        void* read_ptr = kv_read_ptrs[layer_id];
        void* write_ptr = kv_write_ptrs[layer_id];
        size_t offset_bytes = static_cast<uint8_t*>(write_ptr) - static_cast<uint8_t*>(read_ptr);
        write_ptrs.k[layer_id] = write_ptr;
        write_ptrs.v[layer_id] = static_cast<uint8_t*>(read_ptr)
            + static_cast<size_t>(max_tokens) * k_size_per_token
            + offset_bytes;
    }
    return write_ptrs;
}

class ScopedKVWriteRedirect {
public:
    ScopedKVWriteRedirect(Context& context,
                          int32_t num_layers,
                          int32_t device_id,
                          const std::string& cache_key_prefix)
        : tensors_(context.tensors())
    {

        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            redirect(
                ModelTensors::k_write_layer(layer_id),
                cache_key_prefix + "_k_L" + std::to_string(layer_id),
                device_id);
            redirect(
                ModelTensors::v_write_layer(layer_id),
                cache_key_prefix + "_v_L" + std::to_string(layer_id),
                device_id);
        }
    }

    ~ScopedKVWriteRedirect() {
        restore();
    }

    void restore() {
        if (restored_) {
            return;
        }
        for (auto& saved : saved_) {
            tensors_[saved.name] = std::move(saved.tensor);
        }
        restored_ = true;
    }

private:
    struct SavedTensor {
        std::string name;
        Tensor tensor;
    };

    void redirect(const std::string& tensor_name,
                  const std::string& cache_key,
                  int32_t device_id)
    {
        const Tensor& original = tensors_.at(tensor_name);
        saved_.push_back(SavedTensor{tensor_name, make_tensor_view(original)});

        void* tmp_ptr = StaticBufferManager::get_cache_buf(cache_key, tensor_nbytes(original), device_id);
        auto [device, original_device_id] = original.device();
        tensors_[tensor_name] = Tensor::view(
            tmp_ptr, original.shape(), original.dtype(), device, original_device_id);
    }

    std::unordered_map<std::string, Tensor>& tensors_;
    std::vector<SavedTensor> saved_;
    bool restored_ = false;
};

class ScopedPrefillCaptureRedirect {
public:
    ScopedPrefillCaptureRedirect(Context& context,
                                 int32_t num_layers,
                                 int32_t device_id,
                                 const std::string& cache_key_prefix)
        : tensors_(context.tensors())
    {
        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            redirect_pair(
                ModelTensors::k_write_layer(layer_id),
                ModelTensors::k_cache_layer(layer_id),
                cache_key_prefix + "_k_L" + std::to_string(layer_id),
                device_id);
            redirect_pair(
                ModelTensors::v_write_layer(layer_id),
                ModelTensors::v_cache_layer(layer_id),
                cache_key_prefix + "_v_L" + std::to_string(layer_id),
                device_id);
        }
    }

    ~ScopedPrefillCaptureRedirect() {
        restore();
    }

    void restore() {
        if (restored_) {
            return;
        }
        for (auto& saved : saved_) {
            tensors_[saved.name] = std::move(saved.tensor);
        }
        restored_ = true;
    }

private:
    struct SavedTensor {
        std::string name;
        Tensor tensor;
    };

    void redirect_pair(const std::string& write_name,
                       const std::string& cache_name,
                       const std::string& buffer_name,
                       int32_t device_id)
    {
        const Tensor& write_tensor = tensors_.at(write_name);
        const Tensor& cache_tensor = tensors_.at(cache_name);

        check<InternalError>(
            tensor_nbytes(write_tensor) == tensor_nbytes(cache_tensor),
            "Prefill capture redirect requires write/cache tensor sizes to match");

        saved_.push_back(SavedTensor{write_name, make_tensor_view(write_tensor)});
        saved_.push_back(SavedTensor{cache_name, make_tensor_view(cache_tensor)});

        void* tmp_ptr = StaticBufferManager::get_cache_buf(
            buffer_name,
            tensor_nbytes(cache_tensor),
            device_id);

        auto [write_device, write_device_id] = write_tensor.device();
        auto [cache_device, cache_device_id] = cache_tensor.device();
        tensors_[write_name] = Tensor::view(
            tmp_ptr,
            write_tensor.shape(),
            write_tensor.dtype(),
            write_device,
            write_device_id);
        tensors_[cache_name] = Tensor::view(
            tmp_ptr,
            cache_tensor.shape(),
            cache_tensor.dtype(),
            cache_device,
            cache_device_id);
    }

    std::unordered_map<std::string, Tensor>& tensors_;
    std::vector<SavedTensor> saved_;
    bool restored_ = false;
};

class ScopedCudaEvent {
public:
    ScopedCudaEvent() {
        CUDA_CHECK_THROW(cudaEventCreate(&event_), "Failed to create CUDA event");
    }

    ~ScopedCudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, end), "Failed to query CUDA event elapsed time");
    return ms;
}

} // namespace

void StandardEngine::warmup() {
    KVManagerStatus kv_status = kv_manager_->get_status();
    bool need_decode_graph_capture = config_.use_cuda_graph() && !cuda_graph_manager_.is_decode_captured();

    for (const auto& slot : kv_status.slots) {
        if (slot.prefix_token_ids.empty() || slot.prefix_size == 0) {
            continue;
        }

        Request prefix_request(slot.request_id, slot.prefix_token_ids);
        Response temp_response;

        Context context = scheduler_->create_context(prefix_request, &temp_response);
        context.set_kv_write_ptrs(context.get_kv_read_ptrs());

        prepare_tensors(ModelStage::Prefill, context);
        model_->prefill(context);

        cudaStream_t stream = context.stream();
        if (stream != nullptr) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to synchronize stream during warmup");
        }

        if (!need_decode_graph_capture) {
            continue;
        }

        const Tensor& logits_prefill = context.tensors().at(ModelTensors::LOGITS);
        int32_t seq_len = prefill_token_count(context);
        run_sampler(
            last_token_logits_view(logits_prefill),
            context.tensors().at(ModelTensors::SAMPLER_TOKEN_OUT),
            stream,
            ModelStage::Prefill);
        context.advance_after_prefill(seq_len);
        prepare_tensors(ModelStage::Decode, context);
        ensure_decode_graph_captured(context);
        need_decode_graph_capture = false;
    }

    CUDA_CHECK_THROW(cudaDeviceSynchronize(), "Failed to synchronize device after warmup");
}

void StandardEngine::run_sampler(const Tensor& logits,
                                 const Tensor& token_out,
                                 cudaStream_t stream,
                                 ModelStage stage)
{
    auto [logits_device, logits_device_id] = logits.device();
    auto [token_device, token_device_id] = token_out.device();
    std::unordered_map<std::string, Tensor> in, out;
    in.emplace("logits", Tensor::view(
        logits.data_ptr(), logits.shape(), logits.dtype(), logits_device, logits_device_id));
    out.emplace("token_ids", Tensor::view(
        token_out.data_ptr(), token_out.shape(), token_out.dtype(), token_device, token_device_id));
    sampler_->forward(in, out, stream, stage);
}

bool StandardEngine::try_run_prefill_cuda_graph(Context& context) {
    if (!config_.use_cuda_graph()) {
        return false;
    }
    if (kv_manager_->get_attention_type() == AttentionType::MLA) {
        return false;
    }

    const Request* request = context.request();
    if (request == nullptr) {
        return false;
    }
    if (!request_supports_prefill_cuda_graph(*request, device_, device_id_)) {
        return false;
    }
    if (context.prefix_size() != 0) {
        return false;
    }

    auto& tensors = context.tensors();
    const int32_t seq_len = request_prefill_seq_len(context);
    if (seq_len <= 0) {
        return false;
    }

    const int32_t request_id = request->request_id();
    const uint64_t graph_key = prefill_graph_key(request_id, seq_len);
    cudaStream_t stream = context.stream();
    CudaGraphRunner& runner = cuda_graph_manager_.prefill(request_id, seq_len);

    auto invalidate_prefill_capture = [&]() {
        runner.reset();
        prefill_replay_states_.erase(graph_key);
    };

    if (runner.is_captured()) {
        auto replay_it = prefill_replay_states_.find(graph_key);
        if (replay_it == prefill_replay_states_.end()) {
            invalidate_prefill_capture();
        } else {
            const PrefillReplayState& replay_state = replay_it->second;
            bool kv_ptrs_match =
                replay_state.kv_read_ptrs == context.kv_read_ptrs_ref() &&
                replay_state.kv_write_ptrs == context.kv_write_ptrs_ref();
            bool request_inputs_match = replay_state.has_embedding == request->has_embedding() &&
                replay_state.has_position_ids == request->has_position_ids();
            if (request_inputs_match && replay_state.has_embedding) {
                const Tensor& embedding = request->embedding();
                request_inputs_match =
                    replay_state.embedding_ptr == embedding.data_ptr() &&
                    replay_state.embedding_size_bytes == tensor_nbytes(embedding) &&
                    replay_state.embed_token_id == request->embed_token_id();
            }
            if (request_inputs_match && replay_state.has_position_ids) {
                const Tensor& position_ids = request->position_ids();
                request_inputs_match =
                    replay_state.position_ids_ptr == position_ids.data_ptr() &&
                    replay_state.position_ids_size_bytes == tensor_nbytes(position_ids);
            }
            if (!kv_ptrs_match || !request_inputs_match) {
                invalidate_prefill_capture();
            } else {
                const int32_t* token_ids_src = request->token_ids().data();
                size_t prefix_size = context.prefix_size();
                if (prefix_size > 0 && request->token_ids().size() > prefix_size) {
                    token_ids_src += prefix_size;
                }

                if (replay_state.token_ids_size_bytes > 0) {
                    CUDA_CHECK_THROW(cudaMemcpyAsync(
                        replay_state.token_ids_ptr,
                        token_ids_src,
                        replay_state.token_ids_size_bytes,
                        cudaMemcpyHostToDevice,
                        stream),
                        "Failed to refresh prefill token_ids before CUDA graph replay");
                }

                tensors[ModelTensors::TOKEN_IDS] = Tensor::view(
                    replay_state.token_ids_ptr,
                    {1, seq_len},
                    DType::Int32,
                    device_,
                    device_id_);
                tensors[ModelTensors::RESPONSE_TOKENS_DEVICE] = Tensor::view(
                    replay_state.response_tokens_ptr,
                    {replay_state.max_generated_tokens},
                    DType::Int32,
                    device_,
                    device_id_);
                context.set_response_tokens_base_ptr(replay_state.response_tokens_ptr);
                tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(
                    context.get_response_token_write_ptr(),
                    {1},
                    DType::Int32,
                    device_,
                    device_id_);
                if (!replay_state.mrope_last_pos.empty()) {
                    context.set_model_state("mrope_last_pos", replay_state.mrope_last_pos);
                }

                runner.launch(stream);
                return true;
            }
        }
    }

    prepare_tensors(ModelStage::Prefill, context);
    if (tensors.count(ModelTensors::RESPONSE_TOKENS_DEVICE) == 0) {
        return false;
    }

    for (int32_t layer_id = 0; layer_id < model_->num_layers(); ++layer_id) {
        if (tensor_nbytes(tensors.at(ModelTensors::k_write_layer(layer_id))) !=
                tensor_nbytes(tensors.at(ModelTensors::k_cache_layer(layer_id))) ||
            tensor_nbytes(tensors.at(ModelTensors::v_write_layer(layer_id))) !=
                tensor_nbytes(tensors.at(ModelTensors::v_cache_layer(layer_id)))) {
            return false;
        }
    }

    PrefillReplayState replay_state;
    replay_state.token_ids_ptr = tensors.at(ModelTensors::TOKEN_IDS).data_ptr();
    replay_state.token_ids_size_bytes = tensor_nbytes(tensors.at(ModelTensors::TOKEN_IDS));
    replay_state.response_tokens_ptr = tensors.at(ModelTensors::RESPONSE_TOKENS_DEVICE).data_ptr();
    replay_state.max_generated_tokens = static_cast<int32_t>(
        tensors.at(ModelTensors::RESPONSE_TOKENS_DEVICE).shape().front());
    replay_state.seq_len = seq_len;
    replay_state.has_embedding = request->has_embedding();
    if (replay_state.has_embedding) {
        replay_state.embedding_ptr = request->embedding().data_ptr();
        replay_state.embedding_size_bytes = tensor_nbytes(request->embedding());
        replay_state.embed_token_id = request->embed_token_id();
    }
    replay_state.has_position_ids = request->has_position_ids();
    if (replay_state.has_position_ids) {
        replay_state.position_ids_ptr = request->position_ids().data_ptr();
        replay_state.position_ids_size_bytes = tensor_nbytes(request->position_ids());
        if (const std::vector<int32_t>* last_pos = context.get_model_state("mrope_last_pos");
            last_pos != nullptr) {
            replay_state.mrope_last_pos = *last_pos;
        }
    }
    replay_state.kv_read_ptrs = context.kv_read_ptrs_ref();
    replay_state.kv_write_ptrs = context.kv_write_ptrs_ref();

    {
        auto& saved_state = prefill_replay_states_[graph_key];
        saved_state = std::move(replay_state);
    }

    if (runner.is_captured()) {
        runner.launch(stream);
        return true;
    }

    {
        ScopedPrefillCaptureRedirect redirect_writes(
            context,
            model_->num_layers(),
            device_id_,
            "prefill_capture_warmup");

        Tensor sampler_out_saved = make_tensor_view(tensors.at(ModelTensors::SAMPLER_TOKEN_OUT));
        void* sampler_tmp_ptr = StaticBufferManager::get_cache_buf(
            "prefill_capture_sampler_out",
            tensor_nbytes(sampler_out_saved),
            device_id_);
        auto [sampler_device, sampler_device_id] = sampler_out_saved.device();
        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(
            sampler_tmp_ptr,
            sampler_out_saved.shape(),
            sampler_out_saved.dtype(),
            sampler_device,
            sampler_device_id);

        model_->prefill(context);
        run_sampler(
            last_token_logits_view(tensors.at(ModelTensors::LOGITS)),
            tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
            stream,
            ModelStage::Prefill);
        if (stream != nullptr) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream),
                             "Failed to sync stream before prefill CUDA graph capture");
        }

        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = std::move(sampler_out_saved);
    }

    (void)cudaGetLastError();

    runner.begin_capture(stream);
    model_->prefill(context);
    run_sampler(
        last_token_logits_view(tensors.at(ModelTensors::LOGITS)),
        tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
        stream,
        ModelStage::Prefill);
    runner.end_capture(stream);
    return true;
}

void StandardEngine::ensure_decode_graph_captured(Context& context) {
    if (!config_.use_cuda_graph() || cuda_graph_manager_.is_decode_captured()) {
        return;
    }

    if (kv_manager_->get_attention_type() == AttentionType::MLA) {
        throw InternalError("StandardEngine CUDA graph decode capture does not yet support MLA attention");
    }

    auto& tensors = context.tensors();
    cudaStream_t stream = context.stream();

    // Run one uncaptured decode into temporary KV buffers so lazy allocations
    // happen before capture and the real cache stays untouched. Sampler output
    // is redirected as well so the stable decode token buffer is not clobbered.
    {
        ScopedKVWriteRedirect redirect_writes(
            context,
            model_->num_layers(),
            device_id_,
            "decode_capture_warmup");

        Tensor sampler_out_saved = make_tensor_view(tensors.at(ModelTensors::SAMPLER_TOKEN_OUT));
        void* sampler_tmp_ptr = StaticBufferManager::get_cache_buf(
            "decode_capture_sampler_out",
            tensor_nbytes(sampler_out_saved),
            device_id_);
        auto [sampler_device, sampler_device_id] = sampler_out_saved.device();
        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(
            sampler_tmp_ptr,
            sampler_out_saved.shape(),
            sampler_out_saved.dtype(),
            sampler_device,
            sampler_device_id);

        model_->decode_step(context);
        run_sampler(
            tensors.at(ModelTensors::LOGITS),
            tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
            stream,
            ModelStage::Decode);
        if (stream != nullptr) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream),
                             "Failed to sync stream before CUDA graph capture");
        }

        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = std::move(sampler_out_saved);
    }

    (void)cudaGetLastError();

    DecodeWritePtrs write_ptrs = collect_decode_write_ptrs_from_tensors(context, model_->num_layers());
    cuda_graph_manager_.capture_decode(stream, [&]() {
        model_->decode_step(context);
        run_sampler(
            tensors.at(ModelTensors::LOGITS),
            tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
            stream,
            ModelStage::Decode);
        advance_decode_runtime_state(context, stream);
    }, write_ptrs.k, write_ptrs.v);

    // Stream capture executes the captured work, so restore the first decode
    // step's input state before the first graph replay.
    if (context.get_generated_tokens() >= 1) {
        void* token_ids_ptr = tensors.at(ModelTensors::TOKEN_IDS).data_ptr();
        void* last_token_src = context.get_response_token_read_ptr();
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            token_ids_ptr, last_token_src, sizeof(int32_t),
            cudaMemcpyDeviceToDevice, stream),
            "Failed to restore decode token_ids after CUDA graph capture");
    }
    if (tensors.count(ModelTensors::D_KV_LEN) > 0) {
        uint32_t kv_len_val = static_cast<uint32_t>(context.decode_cache_kv_len());
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            tensors.at(ModelTensors::D_KV_LEN).data_ptr(), &kv_len_val,
            sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
            "Failed to restore decode d_kv_len after CUDA graph capture");
    }
    model_->prepare_decode_position_ids(context, device_, device_id_);
}

void StandardEngine::tune() {

    (void)OperatorImplTable::instance().records_for_model(
        config_.resolved_model_name(),
        config_.resolved_hw_profile(),
        config_.operator_impl_table_path(),
        "linear");
}


Response StandardEngine::generate(const Request& request) {
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to set device for generate");

    last_generate_metrics_.clear();
    last_generate_metrics_ = {
        {"prefill_ms", 0.0},
        {"decode_ms", 0.0},
        {"total_stage_ms", 0.0},
        {"decode_step_avg_ms", 0.0},
        {"generated_tokens_total", 0.0},
        {"decode_steps", 0.0},
    };

    Response response;
    Context context = scheduler_->create_context(request, &response);
    cudaStream_t stream = context.stream();
    auto& tensors = context.tensors();
    NVTX::Range generate_range("EDGEFM_GENERATE", NVTXColor::WHITE);

    std::unique_ptr<ScopedCudaEvent> prefill_start_event;
    std::unique_ptr<ScopedCudaEvent> prefill_end_event;
    std::unique_ptr<ScopedCudaEvent> decode_start_event;
    std::unique_ptr<ScopedCudaEvent> decode_end_event;
    if (stream != nullptr) {
        prefill_start_event = std::make_unique<ScopedCudaEvent>();
        prefill_end_event = std::make_unique<ScopedCudaEvent>();
        decode_start_event = std::make_unique<ScopedCudaEvent>();
        decode_end_event = std::make_unique<ScopedCudaEvent>();
    }
    bool decode_started = false;

    // Build stop token set: model eos_token_ids + config stop_token_ids + request stop_token_ids
    // When request.ignore_stop_tokens() (e.g. alignment tests), use empty set to generate full steps
    std::unordered_set<int32_t> stop_tokens;
    if (!request.ignore_stop_tokens()) {
        for (int32_t id : config_.eos_token_ids()) stop_tokens.insert(id);
        for (int32_t id : config_.stop_token_ids()) stop_tokens.insert(id);
        for (int32_t id : request.stop_token_ids()) stop_tokens.insert(id);
    }

    // Check the last sampled token against stop tokens.
    // Returns true if the token is a stop token.
    int32_t host_token_buf = 0;
    auto check_stop = [&](void* device_token_ptr) -> bool {
        if (stop_tokens.empty()) return false;
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            &host_token_buf, device_token_ptr, sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream), "Failed to copy sampled token to host");
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync for stop token check");
        return stop_tokens.count(host_token_buf) > 0;
    };

    {
        NVTX::Range prefill_range("EDGEFM_PREFILL", NVTXColor::BLUE);
        if (prefill_start_event) {
            CUDA_CHECK_THROW(cudaEventRecord(prefill_start_event->get(), stream),
                             "Failed to record prefill start event");
        }

        const bool ran_prefill_graph = try_run_prefill_cuda_graph(context);
        if (tensors.count(ModelTensors::TOKEN_IDS) == 0) {
            prepare_tensors(ModelStage::Prefill, context);
        }
        if (tensors.count(ModelTensors::RESPONSE_TOKENS_DEVICE) == 0) {
            return response;
        }

        if (!ran_prefill_graph) {
            model_->prefill(context);
            run_sampler(
                last_token_logits_view(tensors.at(ModelTensors::LOGITS)),
                tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
                stream,
                ModelStage::Prefill);
        }

        int32_t seq_len = request_prefill_seq_len(context);
        void* prefill_write_ptr = context.get_response_token_write_ptr();
        if (check_stop(prefill_write_ptr)) {
            context.finish();
        }

        context.advance_after_prefill(seq_len);

        if (prefill_end_event) {
            CUDA_CHECK_THROW(cudaEventRecord(prefill_end_event->get(), stream),
                             "Failed to record prefill end event");
        }
    }

    {
        NVTX::Range decode_range("EDGEFM_GENERATION", NVTXColor::GREEN);
        while (!context.is_finished()) {
            if (!decode_started && decode_start_event) {
                CUDA_CHECK_THROW(cudaEventRecord(decode_start_event->get(), stream),
                                 "Failed to record decode start event");
                decode_started = true;
            }

            const bool skip_decode_prepare =
                config_.use_cuda_graph() &&
                cuda_graph_manager_.is_decode_captured() &&
                context.decode_tensors_initialized() &&
                model_->has_static_decode_runtime_tensors();
            if (!skip_decode_prepare) {
                prepare_tensors(ModelStage::Decode, context);
            }
            void* decode_write_ptr = context.get_response_token_write_ptr();

            if (config_.use_cuda_graph()) {
                ensure_decode_graph_captured(context);
                sync_decode_graph(context);
                cuda_graph_manager_.decode().launch(stream);
            } else {
                model_->decode_step(context);
                run_sampler(
                    tensors.at(ModelTensors::LOGITS),
                    tensors.at(ModelTensors::SAMPLER_TOKEN_OUT),
                    stream,
                    ModelStage::Decode);
                advance_decode_runtime_state(context, stream);
            }

            flush_sampled_token(context, decode_write_ptr, stream);

            if (check_stop(decode_write_ptr)) { ++context; context.finish(); break; }
            ++context;
        }

        if (decode_started && decode_end_event) {
            CUDA_CHECK_THROW(cudaEventRecord(decode_end_event->get(), stream),
                             "Failed to record decode end event");
        }
    }

    int32_t num_generated = context.get_generated_tokens();
    std::vector<int32_t> host_tokens(static_cast<size_t>(num_generated));
    void* response_tokens_base = context.tensors().at(ModelTensors::RESPONSE_TOKENS_DEVICE).data_ptr();
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        host_tokens.data(), response_tokens_base,
        static_cast<size_t>(num_generated) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
        "Failed to copy response tokens to host");
    if (stream != nullptr) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to sync stream before returning response");
    }
    response.token_ids().swap(host_tokens);

    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    if (prefill_start_event && prefill_end_event) {
        prefill_ms = static_cast<double>(elapsed_event_ms(prefill_start_event->get(), prefill_end_event->get()));
    }
    if (decode_started && decode_start_event && decode_end_event) {
        decode_ms = static_cast<double>(elapsed_event_ms(decode_start_event->get(), decode_end_event->get()));
    }
    const double decode_steps = static_cast<double>(std::max(0, num_generated - 1));
    last_generate_metrics_ = {
        {"prefill_ms", prefill_ms},
        {"decode_ms", decode_ms},
        {"total_stage_ms", prefill_ms + decode_ms},
        {"decode_step_avg_ms", decode_steps > 0.0 ? decode_ms / decode_steps : 0.0},
        {"generated_tokens_total", static_cast<double>(num_generated)},
        {"decode_steps", decode_steps},
    };

    return response;
}

std::unordered_map<std::string, double> StandardEngine::get_last_generate_metrics() const {
    return last_generate_metrics_;
}

void StandardEngine::prepare_tensors(ModelStage stage, Context& context) {

    if (stage == ModelStage::Prefill) {
        prepare_prefill_tensors(context);
    } else {
        prepare_decode_tensors(context);
    }
}

void StandardEngine::prepare_kvcache_tensors(
    Context& context,
    int32_t num_layers,      // 模型层数
    int32_t num_kv_heads,    // KV head 数
    int32_t head_dim,        // 每个 head 的维度
    int32_t seq_len,         // 当前步处理的 token 数（prefill 为整段长度，decode 为 1）
    size_t prefix_size)      // slot 的 prefix 长度，仅用于 cache 读长度（decode 时） 
{
    auto& tensors = context.tensors();
    const auto& kv_read_ptrs = context.kv_read_ptrs_ref();
    const auto& kv_write_ptrs = context.kv_write_ptrs_ref();
    size_t token_stride = kv_manager_->get_token_stride();
    
    auto kv_dtype = dtype_from_string(config_.kvcache_dtype());
    
    // cache_kv_len = attention 读取的 KV 总长度
    // - Prefill (generated_tokens==0): 若 write_ptr 在 read_ptr 之后（prefix 已在 cache 中），
    //   则 cache_kv_len = prefix_size + seq_len；否则仅为 seq_len（warmup 或无 prefix）
    // - Decode: prefix_size + prefill_len + generated_tokens
    int32_t generated_tokens = context.get_generated_tokens();
    int32_t cache_kv_len;
    if (generated_tokens == 0) {
        bool has_prefix_in_cache = (!kv_write_ptrs.empty() && !kv_read_ptrs.empty()
                                    && kv_write_ptrs[0] != kv_read_ptrs[0]);
        cache_kv_len = (has_prefix_in_cache ? static_cast<int32_t>(prefix_size) : 0) + seq_len;
    } else {
        int32_t prefill_len = static_cast<int32_t>(context.request()->token_ids().size());
        if (prefix_size > 0 && prefill_len > static_cast<int32_t>(prefix_size)) {
            prefill_len = prefill_len - static_cast<int32_t>(prefix_size);
        }
        cache_kv_len = static_cast<int32_t>(prefix_size) + prefill_len + generated_tokens;
    }
    context.set_decode_cache_kv_len(cache_kv_len);

    AttentionType attention_type = kv_manager_->get_attention_type();

    if (attention_type == AttentionType::MLA) {
        int64_t stride_elems = static_cast<int64_t>(token_stride / get_dtype_size(kv_dtype));
        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            void* context_read_ptr = kv_read_ptrs[layer_id];

            tensors[ModelTensors::context_write_layer(layer_id)] = Tensor::view(
                kv_write_ptrs[layer_id], {seq_len, stride_elems}, kv_dtype, device_, device_id_);

            tensors[ModelTensors::context_cache_layer(layer_id)] = Tensor::view(
                context_read_ptr, {cache_kv_len, stride_elems}, kv_dtype, device_, device_id_);
        }
    } else {
        size_t k_size_per_token = token_stride / 2;
        int32_t max_tokens = context.slot_max_tokens();
        bool is_decode = (seq_len == 1 && generated_tokens > 0);
        int32_t cache_shape_len = is_decode ? max_tokens : cache_kv_len;
        bool init_decode_static = is_decode && !context.decode_tensors_initialized();

        if (init_decode_static) {
            void* decode_state_ptr = DecodeRuntimeStateLayout::base_ptr(device_id_);
            void* d_kv_len_ptr = DecodeRuntimeStateLayout::kv_len_ptr(decode_state_ptr);
            if (generated_tokens <= 1) {
                uint32_t kv_len_val = static_cast<uint32_t>(cache_kv_len);
                CUDA_CHECK_THROW(cudaMemcpyAsync(d_kv_len_ptr, &kv_len_val,
                    sizeof(uint32_t), cudaMemcpyHostToDevice, context.stream()),
                    "copy d_kv_len to device");
            }
            tensors[ModelTensors::D_KV_LEN] = Tensor::view(
                d_kv_len_ptr, {1}, DType::Int32, device_, device_id_);
        }

        for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            void* read_ptr = kv_read_ptrs[layer_id];
            void* write_ptr = kv_write_ptrs[layer_id];

            size_t offset_bytes = static_cast<uint8_t*>(write_ptr) - static_cast<uint8_t*>(read_ptr);
            void* k_write_ptr = write_ptr;
            void* v_write_ptr = static_cast<uint8_t*>(read_ptr) + static_cast<size_t>(max_tokens) * k_size_per_token + offset_bytes;
            tensors[ModelTensors::k_write_layer(layer_id)] = Tensor::view(
                k_write_ptr, {seq_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
            tensors[ModelTensors::v_write_layer(layer_id)] = Tensor::view(
                v_write_ptr, {seq_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);

            if (!is_decode || init_decode_static) {
                void* k_cache_ptr = read_ptr;
                void* v_cache_ptr = static_cast<uint8_t*>(read_ptr) + static_cast<size_t>(max_tokens) * k_size_per_token;
                tensors[ModelTensors::k_cache_layer(layer_id)] = Tensor::view(
                    k_cache_ptr, {cache_shape_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
                tensors[ModelTensors::v_cache_layer(layer_id)] = Tensor::view(
                    v_cache_ptr, {cache_shape_len, num_kv_heads, head_dim}, kv_dtype, device_, device_id_);
            }
        }
    }
}

void StandardEngine::prepare_prefill_tensors(Context& context) {
    auto& tensors = context.tensors();
    cudaStream_t stream = context.stream();

    // 获取模型参数
    int32_t num_layers = model_->num_layers();
    int32_t hidden_size = model_->hidden_size();
    int32_t vocab_size = model_->vocab_size();
    
    // 获取请求信息：prefix 已在 warmup 写入，generate 时只 prefill 未匹配的 prompt 部分
    const Request* request = context.request();
    const std::vector<int32_t>& token_ids_vec = request->token_ids();
    
    size_t prefix_size = context.prefix_size();
    int32_t max_generated_tokens = context.slot_max_tokens() - static_cast<int32_t>(prefix_size);
    
    // seq_len = 实际写入 KV 的 token 数（prompt 部分，不含 prefix）
    int32_t seq_len = static_cast<int32_t>(token_ids_vec.size());
    if (prefix_size > 0 && token_ids_vec.size() > prefix_size) {
        seq_len = static_cast<int32_t>(token_ids_vec.size() - prefix_size);
    }
    
    // 获取模型配置以计算 attention 参数
    auto model_config = config_.prefill_model_config();
    int32_t num_attention_heads = model_config.value("num_attention_heads", 32);
    int32_t num_kv_heads = model_config.value("num_key_value_heads", num_attention_heads);
    int32_t head_dim = hidden_size / num_attention_heads;
    
    prepare_kvcache_tensors(context, num_layers, num_kv_heads, head_dim, seq_len, prefix_size);
    
    // ==================== 构建输入 tensors ====================
    // 0. Token IDs tensor: [1, seq_len]（有 prefix 时只传 prompt 部分，避免重复写入 prefix）
    const int32_t* token_ids_src = token_ids_vec.data();
    if (prefix_size > 0 && token_ids_vec.size() > prefix_size) {
        token_ids_src = token_ids_vec.data() + prefix_size;
    }
    size_t token_ids_size = static_cast<size_t>(seq_len) * sizeof(int32_t);
    void* token_ids_ptr = StaticBufferManager::get_cache_buf(
        "prefill_token_ids", token_ids_size, device_id_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(token_ids_ptr, token_ids_src, token_ids_size,
                                     cudaMemcpyHostToDevice, stream),
                     "Failed to copy token_ids to GPU");

    tensors[ModelTensors::TOKEN_IDS] = Tensor::view(
        token_ids_ptr,
        {1, seq_len},
        DType::Int32,
        device_,
        device_id_
    );
    
    // 1'. 可选的自定义 embedding: [num_custom_embeddings, hidden_size] 与 embed_token_id
    if (request->has_embedding()) {
        const Tensor& emb = request->embedding();
        auto [src_device, src_device_id] = emb.device();
        if (src_device == device_ && src_device_id == device_id_) {
            tensors[ModelTensors::EMBEDDING] = Tensor::view(
                emb.data_ptr(),
                emb.shape(),
                emb.dtype(),
                src_device,
                src_device_id
            );
        } else {
            tensors[ModelTensors::EMBEDDING] = Tensor::clone_from(
                emb.data_ptr(),
                emb.shape(),
                emb.dtype(),
                src_device, src_device_id,
                device_, device_id_,
                MemoryOwnership::OwnCudaPool,
                stream
            );
        }
        embed_token_id_buf_ = request->embed_token_id();
        tensors[ModelTensors::EMBED_TOKEN_ID] = Tensor::view(
            &embed_token_id_buf_,
            {1},
            DType::Int32,
            Device::CPU,
            0
        );
    }

    // 1''. 可选的 M-RoPE position_ids: [3, seq_len]
    if (request->has_position_ids()) {
        const Tensor& pos = request->position_ids();
        auto [pos_device, pos_device_id] = pos.device();
        if (pos_device == device_ && pos_device_id == device_id_) {
            tensors[ModelTensors::POSITION_IDS] = Tensor::view(
                pos.data_ptr(),
                pos.shape(),
                pos.dtype(),
                pos_device,
                pos_device_id
            );
        } else {
            tensors[ModelTensors::POSITION_IDS] = Tensor::clone_from(
                pos.data_ptr(),
                pos.shape(),
                pos.dtype(),
                pos_device, pos_device_id,
                device_, device_id_,
                MemoryOwnership::OwnCudaPool,
                stream
            );
        }

        if (request->has_mrope_last_pos()) {
            context.set_model_state("mrope_last_pos", request->mrope_last_pos());
        } else {
            // Fallback for older request producers that do not precompute M-RoPE state.
            int64_t total_len = pos.shape()[1];
            size_t bytes = 3 * total_len * sizeof(int32_t);
            std::vector<int32_t> pos_cpu(3 * total_len);
            if (pos_device == Device::GPU) {
                CUDA_CHECK_THROW(cudaMemcpyAsync(pos_cpu.data(), pos.data_ptr(), bytes,
                                                 cudaMemcpyDeviceToHost, stream),
                                 "copy position_ids to CPU");
                CUDA_CHECK_THROW(cudaStreamSynchronize(stream), "sync after position_ids copy");
            } else {
                std::memcpy(pos_cpu.data(), pos.data_ptr(), bytes);
            }
            std::vector<int32_t> mrope_last_pos(3);
            for (int d = 0; d < 3; ++d) {
                int32_t mx = 0;
                for (int64_t i = 0; i < total_len; ++i) {
                    int32_t v = pos_cpu[d * total_len + i];
                    if (v > mx) mx = v;
                }
                mrope_last_pos[d] = mx;
            }
            context.set_model_state("mrope_last_pos", std::move(mrope_last_pos));
        }
    }
    
    DType model_dtype = model_->dtype();
    size_t model_dtype_size = get_dtype_size(model_dtype);
    size_t fp32_size = get_dtype_size(DType::Float32);
    // ==================== 构建临时激活值 tensors ====================
    // 1. Hidden states: [batch_size=1, seq_len, hidden_size]（embed 层要求 3D，dtype 与 embedding 一致）
    size_t hidden_states_size = seq_len * hidden_size * model_dtype_size;
    void* hidden_states_ptr = StaticBufferManager::get_cache_buf(
        "prefill_hidden_states", hidden_states_size, device_id_);
    tensors[ModelTensors::HIDDEN_STATES] = Tensor::view(
        hidden_states_ptr,
        {1, seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_
    );
    
    // 2. Fused QKV projection output: [seq_len, qkv_total_dim]（临时 buffer，MemoryPool）
    // Layout: [Q: num_attention_heads * q_head_dim, K: num_kv_heads * head_dim, V: num_kv_heads * head_dim]
    // 支持多种 attention：MLA 用 q_head_dim，GQA 等用 head_dim
    int32_t q_head_dim = (kv_manager_->get_attention_type() == AttentionType::MLA)
                             ? kv_manager_->get_qk_rope_head_dim()
                             : head_dim;
    int32_t q_dim = num_attention_heads * q_head_dim;
    int32_t k_dim = num_kv_heads * head_dim;
    int32_t v_dim = num_kv_heads * head_dim;
    int32_t qkv_total_dim = q_dim + k_dim + v_dim;
    size_t qkv_proj_size = seq_len * qkv_total_dim * model_dtype_size;
    void* qkv_proj_ptr = StaticBufferManager::get_cache_buf(
        "prefill_qkv_proj", qkv_proj_size, device_id_);
    tensors[ModelTensors::QKV_PROJ_OUTPUT] = Tensor::view(
        qkv_proj_ptr,
        {seq_len, qkv_total_dim},
        model_dtype,
        device_,
        device_id_
    );
    // 2.1. Q projection output:
    // - 标准 LLM prefill 直接 alias 到 fused QKV 首段，避免额外缓冲
    // - M-RoPE 路径仍保留独立连续 Q 缓冲，供后续 in-place rotary 使用
    void* q_ptr = qkv_proj_ptr;
    if (model_->needs_separate_prefill_q_buffer()) {
        size_t q_size = seq_len * num_attention_heads * q_head_dim * model_dtype_size;
        q_ptr = StaticBufferManager::get_cache_buf(
            "prefill_q_proj", q_size, device_id_);
    }
    tensors[ModelTensors::Q_PROJ_OUTPUT] = Tensor::view(
        q_ptr,
        {seq_len, num_attention_heads, q_head_dim},
        model_dtype,
        device_,
        device_id_
    );
    // K/V 直接写入 k_write/v_write，无需 K_PROJ_OUTPUT、V_PROJ_OUTPUT
    
    // 3. Attention output: [seq_len, num_attention_heads, head_dim]
    size_t attn_output_size = seq_len * hidden_size * model_dtype_size;
    void* attn_output_ptr = StaticBufferManager::get_cache_buf(
        "prefill_attn_output", attn_output_size, device_id_);
    tensors[ModelTensors::ATTENTION_OUTPUT] = Tensor::view(
        attn_output_ptr,
        {seq_len, num_attention_heads, head_dim},
        model_dtype,
        device_,
        device_id_
    );
    
    // 4. MLP intermediate: [seq_len, intermediate_size]
    int32_t intermediate_size = model_config.value("intermediate_size", hidden_size * 4);
    size_t mlp_intermediate_size = seq_len * intermediate_size * model_dtype_size;
    void* mlp_intermediate_ptr = StaticBufferManager::get_cache_buf(
        "prefill_mlp_intermediate", mlp_intermediate_size, device_id_);
    tensors[ModelTensors::MLP_INTERMEDIATE] = Tensor::view(
        mlp_intermediate_ptr,
        {seq_len, intermediate_size},
        model_dtype,
        device_,
        device_id_
    );
    
    // 5. Norm output: [seq_len, hidden_size] (for LayerNorm outputs)
    size_t norm_output_size = seq_len * hidden_size * model_dtype_size;
    void* norm_output_ptr = StaticBufferManager::get_cache_buf(
        "prefill_norm_output", norm_output_size, device_id_);
    tensors[ModelTensors::NORM_OUTPUT] = Tensor::view(
        norm_output_ptr,
        {seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_
    );
    
    // 6. Post-attention LayerNorm output: [seq_len, hidden_size]
    size_t post_norm_size = seq_len * hidden_size * model_dtype_size;
    void* post_norm_ptr = StaticBufferManager::get_cache_buf(
        "prefill_post_norm", post_norm_size, device_id_);
    tensors[ModelTensors::POST_NORM_OUTPUT] = Tensor::view(
        post_norm_ptr,
        {seq_len, hidden_size},
        model_dtype,
        device_,
        device_id_
    );
    
    // 7. MLP activation input: [seq_len, 2 * intermediate_size] (gate + up concatenated)
    size_t mlp_activation_input_size = seq_len * 2 * intermediate_size * model_dtype_size;
    void* mlp_activation_input_ptr = StaticBufferManager::get_cache_buf(
        "prefill_mlp_activation_input", mlp_activation_input_size, device_id_);
    tensors[ModelTensors::MLP_ACTIVATION_INPUT] = Tensor::view(
        mlp_activation_input_ptr,
        {seq_len, 2 * intermediate_size},
        model_dtype,
        device_,
        device_id_
    );
    
    // 8. Prefill only samples the last token, so one logits row is enough.
    size_t logits_size = static_cast<size_t>(vocab_size) * fp32_size;
    void* logits_ptr = StaticBufferManager::get_cache_buf(
        "prefill_logits", logits_size, device_id_);
    tensors[ModelTensors::LOGITS] = Tensor::view(
        logits_ptr,
        {1, vocab_size},
        DType::Float32,
        device_,
        device_id_
    );

    // 9. Sampler output 与 10. Response tokens：sampler 直接写入 response 缓冲当前写位置，无需单独缓冲与 D2D copy
    if (max_generated_tokens > 0) {
        void* response_tokens_ptr = StaticBufferManager::get_cache_buf(
            "prefill_response_tokens",
            static_cast<size_t>(max_generated_tokens) * sizeof(int32_t),
            device_id_);
        tensors[ModelTensors::RESPONSE_TOKENS_DEVICE] = Tensor::view(
            response_tokens_ptr,
            {max_generated_tokens},
            DType::Int32,
            device_,
            device_id_
        );
        context.set_response_tokens_base_ptr(response_tokens_ptr);
        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(
            context.get_response_token_write_ptr(),
            {1},
            DType::Int32,
            device_,
            device_id_
        );
    }
}

void StandardEngine::prepare_decode_tensors(Context& context) {
    auto& tensors = context.tensors();
    cudaStream_t stream = context.stream();
    const bool init_static = !context.decode_tensors_initialized();

    // 获取模型参数
    int32_t num_layers = model_->num_layers();
    int32_t hidden_size = model_->hidden_size();
    int32_t vocab_size = model_->vocab_size();
    
    // Decode 阶段：每次处理 1 个 token
    int32_t seq_len = 1;

    void* decode_state_ptr = DecodeRuntimeStateLayout::base_ptr(device_id_);
    void* token_ids_ptr = DecodeRuntimeStateLayout::token_ids_ptr(decode_state_ptr);

    // 获取模型配置以计算 attention 参数
    auto model_config = config_.prefill_model_config();
    int32_t num_attention_heads = model_config.value("num_attention_heads", 32);
    int32_t num_kv_heads = model_config.value("num_key_value_heads", num_attention_heads);
    int32_t head_dim = hidden_size / num_attention_heads;
    
    // prepare kvcache tensors(write buffers and read buffers)
    prepare_kvcache_tensors(context, num_layers, num_kv_heads, head_dim, seq_len, context.prefix_size());

    DType model_dtype = model_->dtype();
    size_t model_dtype_size = get_dtype_size(model_dtype);
    size_t fp32_size = get_dtype_size(DType::Float32);

    int32_t q_head_dim = (kv_manager_->get_attention_type() == AttentionType::MLA)
                             ? kv_manager_->get_qk_rope_head_dim()
                             : head_dim;
    int32_t q_dim = num_attention_heads * q_head_dim;
    int32_t k_dim = num_kv_heads * head_dim;
    int32_t v_dim = num_kv_heads * head_dim;
    int32_t qkv_total_dim = q_dim + k_dim + v_dim;
    int32_t intermediate_size = model_config.value("intermediate_size", hidden_size * 4);

    // ==================== Decode activation buffers (persistent, cached by StaticBufferManager) ====================
    void* hidden_states_ptr = StaticBufferManager::get_cache_buf("decode_hidden_states",   hidden_size * model_dtype_size, device_id_);
    void* qkv_proj_ptr      = StaticBufferManager::get_cache_buf("decode_qkv_proj",        qkv_total_dim * model_dtype_size, device_id_);
    void* attn_output_ptr   = StaticBufferManager::get_cache_buf("decode_attn_output",     hidden_size * model_dtype_size, device_id_);
    void* norm_output_ptr   = StaticBufferManager::get_cache_buf("decode_norm_output",     hidden_size * model_dtype_size, device_id_);
    void* post_norm_ptr     = StaticBufferManager::get_cache_buf("decode_post_norm",       hidden_size * model_dtype_size, device_id_);
    void* mlp_inter_ptr     = StaticBufferManager::get_cache_buf("decode_mlp_inter",       intermediate_size * model_dtype_size, device_id_);
    void* mlp_act_ptr       = StaticBufferManager::get_cache_buf("decode_mlp_act",         2 * intermediate_size * model_dtype_size, device_id_);
    void* logits_ptr        = StaticBufferManager::get_cache_buf("decode_logits",          vocab_size * fp32_size, device_id_);
    void* sampler_out_ptr   = token_ids_ptr;

    if (init_static) {
        // TOKEN_IDS / SAMPLER_TOKEN_OUT share one stable device buffer. Seed it
        // once from the prefill sample, then keep updating it in-place.
        if (context.get_generated_tokens() >= 1) {
            if (context.get_generated_tokens() == 1) {
                void* last_token_src = context.get_response_token_read_ptr();
                CUDA_CHECK_THROW(cudaMemcpyAsync(
                    token_ids_ptr, last_token_src, sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream),
                    "Failed to seed decode token_ids from response buffer");
            }
            tensors[ModelTensors::TOKEN_IDS] = Tensor::view(
                token_ids_ptr,
                {1, seq_len},
                DType::Int32,
                device_,
                device_id_);
        }

        if (context.get_model_state("mrope_last_pos") != nullptr) {
            tensors[ModelTensors::POSITION_IDS] = Tensor::view(
                DecodeRuntimeStateLayout::position_ids_ptr(decode_state_ptr),
                {3, 1},
                DType::Int32,
                device_,
                device_id_);
        }

        if (context.get_generated_tokens() <= 1) {
            model_->prepare_decode_position_ids(context, device_, device_id_);
        }

        tensors[ModelTensors::HIDDEN_STATES] = Tensor::view(hidden_states_ptr, {1, seq_len, hidden_size}, model_dtype, device_, device_id_);
        tensors[ModelTensors::QKV_PROJ_OUTPUT] = Tensor::view(qkv_proj_ptr, {seq_len, qkv_total_dim}, model_dtype, device_, device_id_);
        tensors[ModelTensors::Q_PROJ_OUTPUT] = Tensor::view(qkv_proj_ptr, {seq_len, num_attention_heads, q_head_dim}, model_dtype, device_, device_id_);
        tensors[ModelTensors::ATTENTION_OUTPUT] = Tensor::view(attn_output_ptr, {seq_len, num_attention_heads, head_dim}, model_dtype, device_, device_id_);
        tensors[ModelTensors::NORM_OUTPUT] = Tensor::view(norm_output_ptr, {seq_len, hidden_size}, model_dtype, device_, device_id_);
        tensors[ModelTensors::POST_NORM_OUTPUT] = Tensor::view(post_norm_ptr, {seq_len, hidden_size}, model_dtype, device_, device_id_);
        tensors[ModelTensors::MLP_INTERMEDIATE] = Tensor::view(mlp_inter_ptr, {seq_len, intermediate_size}, model_dtype, device_, device_id_);
        tensors[ModelTensors::MLP_ACTIVATION_INPUT] = Tensor::view(mlp_act_ptr, {seq_len, 2 * intermediate_size}, model_dtype, device_, device_id_);
        tensors[ModelTensors::LOGITS] = Tensor::view(logits_ptr, {seq_len, vocab_size}, DType::Float32, device_, device_id_);
        tensors[ModelTensors::SAMPLER_TOKEN_OUT] = Tensor::view(sampler_out_ptr, {1}, DType::Int32, device_, device_id_);
        context.mark_decode_tensors_initialized();
    }
}

void StandardEngine::flush_sampled_token(const Context& context, void* write_ptr, cudaStream_t stream) {
    const void* sampled_ptr = context.tensors().at(ModelTensors::SAMPLER_TOKEN_OUT).data_ptr();
    CUDA_CHECK_THROW(cudaMemcpyAsync(write_ptr, sampled_ptr,
        sizeof(int32_t), cudaMemcpyDeviceToDevice, stream),
        "copy sampled token from decode runtime buffer to response buffer");
}

void StandardEngine::advance_decode_runtime_state(Context& context, cudaStream_t stream) {
    auto& tensors = context.tensors();
    auto kv_it = tensors.find(ModelTensors::D_KV_LEN);
    if (kv_it != tensors.end()) {
        launch_increment_uint32_scalar(
            static_cast<uint32_t*>(kv_it->second.data_ptr()), stream);
        CUDA_CHECK_THROW(cudaGetLastError(), "Failed to advance decode d_kv_len");
    }

    model_->advance_decode_runtime_tensors(context, stream);
}


void StandardEngine::sync_decode_graph(Context& context) {
    if (!cuda_graph_manager_.has_decode_dynamic_nodes()) return;

    DecodeWritePtrs write_ptrs = collect_decode_write_ptrs_from_context(
        context, *kv_manager_, model_->num_layers());
    cuda_graph_manager_.update_decode_nodes(write_ptrs.k, write_ptrs.v);
}

} // namespace edge_fm
