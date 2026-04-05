/*
 * Python bindings for TensorRT-Edge-LLM runtime.
 * Enables in-process benchmarking without subprocess overhead.
 *
 * Requires: BUILD_TRT_EDGELLM_PYBIND=ON, TRT_PACKAGE_DIR, TensorRT-Edge-LLM built.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
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
        m_runtime = std::make_unique<rt::LLMInferenceRuntime>(
            engine_dir, multimodal_engine_dir, lora_weights_map, m_stream);

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
