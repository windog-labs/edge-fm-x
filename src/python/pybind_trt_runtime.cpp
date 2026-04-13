/*
 * Python bindings for TensorRT-Edge-LLM runtime.
 * Enables in-process benchmarking without subprocess overhead.
 *
 * Requires: BUILD_TRT_EDGELLM_PYBIND=ON, TRT_PACKAGE_DIR, TensorRT-Edge-LLM built.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"

namespace py = pybind11;
namespace trt = trt_edgellm;
namespace rt = trt_edgellm::rt;

namespace {

std::string normalize_model_type_string(std::string value)
{
    value.erase(std::remove_if(value.begin(), value.end(),
                    [](unsigned char c) { return std::isspace(c); }),
        value.end());
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool config_indicates_llava_prepared_only(std::string const& multimodal_engine_dir)
{
    if (multimodal_engine_dir.empty()) {
        return false;
    }

    std::ifstream config_stream(multimodal_engine_dir + "/config.json");
    if (!config_stream.is_open()) {
        return false;
    }

    nlohmann::json config_json;
    try {
        config_json = nlohmann::json::parse(config_stream);
    } catch (...) {
        return false;
    }

    if (normalize_model_type_string(config_json.value("model_type", std::string{})) == "llava") {
        return true;
    }

    if (!config_json.contains("architectures") || !config_json["architectures"].is_array()) {
        return false;
    }
    for (auto const& architecture : config_json["architectures"]) {
        if (!architecture.is_string()) {
            continue;
        }
        if (normalize_model_type_string(architecture.get<std::string>()).find("llava") != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

class TrtEdgeLlmRuntime {
public:
    TrtEdgeLlmRuntime(const std::string& engine_dir,
                      const std::string& multimodal_engine_dir,
                      int device_id)
    {
        // Load plugin before creating runtime (TensorRT needs it for engine)
        m_plugin_handle = trt::loadEdgellmPluginLib();
        if (!m_plugin_handle) {
            throw std::runtime_error(
                "Failed to load TensorRT Edge-LLM plugin. Set EDGELLM_PLUGIN_PATH.");
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
        }

        err = cudaStreamCreate(&m_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err));
        }

        std::unordered_map<std::string, std::string> lora_weights_map;
        bool const llava_prepared_only = config_indicates_llava_prepared_only(multimodal_engine_dir);
        m_runtime = std::make_unique<rt::LLMInferenceRuntime>(
            engine_dir, llava_prepared_only ? std::string{} : multimodal_engine_dir, lora_weights_map, m_stream);
        if (llava_prepared_only) {
            m_runtime->setPreparedExternalMultimodalOnly(true);
        }

        if (!m_runtime->captureDecodingCUDAGraph(m_stream)) {
            // Non-fatal: proceed without CUDA graph
        }
    }

    ~TrtEdgeLlmRuntime()
    {
        if (m_stream) {
            cudaStreamDestroy(m_stream);
        }
    }

    std::pair<std::vector<std::vector<int32_t>>, std::vector<std::string>> generate(
        const std::string& prompt,
        int64_t max_generate_length,
        float temperature = 0.0f,
        float top_p = 1.0f,
        int64_t top_k = 1,
        bool ignore_stop_tokens = false)
    {
        return generate_impl(prompt, {}, max_generate_length, temperature, top_p, top_k, ignore_stop_tokens);
    }

    std::pair<std::vector<std::vector<int32_t>>, std::vector<std::string>> generate_from_token_ids(
        std::vector<int32_t> token_ids,
        int64_t max_generate_length,
        float temperature = 0.0f,
        float top_p = 1.0f,
        int64_t top_k = 1,
        bool ignore_stop_tokens = false)
    {
        return generate_impl("", std::move(token_ids), max_generate_length, temperature, top_p, top_k, ignore_stop_tokens);
    }

    void prepare_multimodal_from_token_ids(std::vector<int32_t> token_ids,
        py::array_t<float, py::array::c_style | py::array::forcecast> image_embeddings,
        std::vector<std::vector<int64_t>> image_grid_thw)
    {
        py::buffer_info info = image_embeddings.request();
        if (info.ndim != 2) {
            throw std::runtime_error("image_embeddings must be a 2D array [num_image_tokens, hidden_size]");
        }

        int64_t const num_image_tokens = static_cast<int64_t>(info.shape[0]);
        int64_t const hidden_size = static_cast<int64_t>(info.shape[1]);
        float const* src = static_cast<float const*>(info.ptr);
        std::vector<half> host_half(static_cast<size_t>(num_image_tokens * hidden_size));
        std::transform(src, src + host_half.size(), host_half.begin(), [](float x) { return __float2half_rn(x); });

        m_prepared_multimodal_embeddings = rt::Tensor(
            {num_image_tokens, hidden_size}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
            "TrtEdgeLlmRuntime::prepared_multimodal_embeddings");
        cudaError_t err = cudaMemcpyAsync(m_prepared_multimodal_embeddings.rawPointer(), host_half.data(),
            host_half.size() * sizeof(half), cudaMemcpyHostToDevice, m_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpyAsync for image_embeddings failed: ") + cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(m_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamSynchronize after image_embeddings copy failed: ") + cudaGetErrorString(err));
        }

        std::vector<std::vector<int32_t>> batched_input_ids{token_ids};
        if (!m_runtime->prepareExternalMultimodalInputs(
                batched_input_ids, m_prepared_multimodal_embeddings, image_grid_thw, m_stream)) {
            throw std::runtime_error("TRT prepareExternalMultimodalInputs failed");
        }

        m_prepared_multimodal_token_ids = std::move(token_ids);
        m_has_prepared_multimodal = true;
    }

    std::pair<std::vector<std::vector<int32_t>>, std::vector<std::string>> generate_from_prepared_multimodal(
        int64_t max_generate_length,
        float temperature = 0.0f,
        float top_p = 1.0f,
        int64_t top_k = 1,
        bool ignore_stop_tokens = false)
    {
        if (!m_has_prepared_multimodal) {
            throw std::runtime_error("No prepared multimodal inputs. Call prepare_multimodal_from_token_ids() first.");
        }
        return generate_impl(
            "", m_prepared_multimodal_token_ids, max_generate_length, temperature, top_p, top_k, ignore_stop_tokens);
    }

    std::unordered_map<std::string, double> last_generate_metrics() const
    {
        return m_last_metrics;
    }

private:
    std::pair<std::vector<std::vector<int32_t>>, std::vector<std::string>> generate_impl(
        const std::string& prompt,
        std::vector<int32_t> token_ids,
        int64_t max_generate_length,
        float temperature,
        float top_p,
        int64_t top_k,
        bool ignore_stop_tokens)
    {
        m_last_metrics.clear();
        trt::gTimer.reset();
        bool const prev_profiling_enabled = trt::getProfilingEnabled();
        trt::setProfilingEnabled(true);

        rt::LLMGenerationRequest request;
        request.maxGenerateLength = max_generate_length;
        request.temperature = temperature;
        request.topP = top_p;
        request.topK = top_k;
        request.ignoreStopTokens = ignore_stop_tokens;

        if (!token_ids.empty())
        {
            request.inputTokenIds.push_back(std::move(token_ids));
        }
        else
        {
            rt::Message::MessageContent msg_content;
            msg_content.type = "text";
            msg_content.content = prompt;
            rt::Message msg;
            msg.role = "user";
            msg.contents.push_back(std::move(msg_content));
            rt::LLMGenerationRequest::Request req;
            req.messages.push_back(std::move(msg));
            request.requests.push_back(std::move(req));
            request.applyChatTemplate = true;
            request.addGenerationPrompt = true;
        }

        rt::LLMGenerationResponse response;
        bool ok = false;
        try
        {
            ok = m_runtime->handleRequest(request, response, m_stream);
        }
        catch (...)
        {
            trt::setProfilingEnabled(prev_profiling_enabled);
            throw;
        }
        trt::setProfilingEnabled(prev_profiling_enabled);
        if (!ok) {
            throw std::runtime_error("TRT handleRequest failed");
        }

        auto prefill_timing = trt::gTimer.getTimingData(trt::metrics::StageNames::kLLM_PREFILL);
        auto generation_timing = trt::gTimer.getTimingData(trt::metrics::StageNames::kLLM_GENERATION);
        double const prefill_ms = prefill_timing ? static_cast<double>(prefill_timing->getTotalGpuTimeMs()) : 0.0;
        double const decode_ms = generation_timing ? static_cast<double>(generation_timing->getTotalGpuTimeMs()) : 0.0;

        int32_t generated_tokens_total = 0;
        for (auto const& ids : response.outputIds)
        {
            generated_tokens_total += static_cast<int32_t>(ids.size());
        }
        int32_t const decode_steps = std::max(0, generated_tokens_total - static_cast<int32_t>(response.outputIds.size()));
        m_last_metrics = {
            {"prefill_ms", prefill_ms},
            {"decode_ms", decode_ms},
            {"total_stage_ms", prefill_ms + decode_ms},
            {"decode_step_avg_ms", decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0},
            {"generated_tokens_total", static_cast<double>(generated_tokens_total)},
            {"decode_steps", static_cast<double>(decode_steps)},
        };

        return {response.outputIds, response.outputTexts};
    }

private:
    std::unique_ptr<void, trt::DlDeleter> m_plugin_handle;
    cudaStream_t m_stream = nullptr;
    std::unique_ptr<rt::LLMInferenceRuntime> m_runtime;
    rt::Tensor m_prepared_multimodal_embeddings;
    std::vector<int32_t> m_prepared_multimodal_token_ids;
    bool m_has_prepared_multimodal = false;
    std::unordered_map<std::string, double> m_last_metrics;
};

PYBIND11_MODULE(edge_fm_trt, m)
{
    m.doc() = "TensorRT-Edge-LLM runtime Python bindings for in-process inference";

    py::class_<TrtEdgeLlmRuntime>(m, "TrtEdgeLlmRuntime")
        .def(py::init<const std::string&, const std::string&, int>(),
             py::arg("engine_dir"),
             py::arg("multimodal_engine_dir") = "",
             py::arg("device_id") = 0)
        .def("generate",
             &TrtEdgeLlmRuntime::generate,
             py::arg("prompt"),
             py::arg("max_generate_length"),
             py::arg("temperature") = 0.0f,
             py::arg("top_p") = 1.0f,
             py::arg("top_k") = 1,
             py::arg("ignore_stop_tokens") = false,
             R"doc(
Generate tokens from a text prompt.

Args:
    prompt: Input text prompt
    max_generate_length: Maximum number of tokens to generate
    temperature: Sampling temperature (0 = greedy)
    top_p: Top-p (nucleus) sampling
    top_k: Top-k sampling

Returns:
    Tuple of (output_ids, output_texts) for each request in the batch.
)doc")
        .def("generate_from_token_ids",
             &TrtEdgeLlmRuntime::generate_from_token_ids,
             py::arg("token_ids"),
             py::arg("max_generate_length"),
             py::arg("temperature") = 0.0f,
             py::arg("top_p") = 1.0f,
             py::arg("top_k") = 1,
             py::arg("ignore_stop_tokens") = false,
             R"doc(
Generate tokens from pre-tokenized input token IDs (skip chat template / tokenize).
Ensures same prefill as Edge-FM for fair benchmarking.

Args:
    token_ids: List of input token IDs (e.g. from dump token_ids.npy)
    max_generate_length: Maximum number of tokens to generate
    temperature: Sampling temperature (0 = greedy)
    top_p: Top-p (nucleus) sampling
    top_k: Top-k sampling

Returns:
    Tuple of (output_ids, output_texts) for each request in the batch.
)doc")
        .def("prepare_multimodal_from_token_ids",
             &TrtEdgeLlmRuntime::prepare_multimodal_from_token_ids,
             py::arg("token_ids"),
             py::arg("image_embeddings"),
             py::arg("image_grid_thw"),
             R"doc(
Prepare a multimodal request from precomputed image embeddings and token IDs.

This path is intended for fair VLM runtime benchmarking where image embeddings
and prompt token IDs are already prepared outside the timed region. For Qwen-VL
style models, pass `image_grid_thw`; for Llava prepared benchmarking, an empty
list is accepted.
)doc")
        .def("generate_from_prepared_multimodal",
             &TrtEdgeLlmRuntime::generate_from_prepared_multimodal,
             py::arg("max_generate_length"),
             py::arg("temperature") = 0.0f,
             py::arg("top_p") = 1.0f,
             py::arg("top_k") = 1,
             py::arg("ignore_stop_tokens") = false,
             R"doc(
Generate tokens using the most recently prepared multimodal inputs.
)doc")
        .def("last_generate_metrics",
             &TrtEdgeLlmRuntime::last_generate_metrics,
             R"doc(
Return stage timing from the most recent generation run.

Keys include:
    prefill_ms
    decode_ms
    total_stage_ms
    decode_step_avg_ms
    generated_tokens_total
    decode_steps
)doc");
}
