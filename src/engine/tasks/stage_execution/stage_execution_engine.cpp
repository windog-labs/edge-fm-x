#include "engine/tasks/stage_execution/stage_execution_engine.h"

#include <edge-fm/core.h>

#include <chrono>

namespace edge_fm {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

StageExecutionEngine::StageExecutionEngine(const EngineConfig& config)
    : Engine(config)
    , stage_runtime_(config)
{}

void StageExecutionEngine::warmup() {}

void StageExecutionEngine::tune() {}

Response StageExecutionEngine::generate(const Request& request) {
    (void)request;
    throw ConfigurationError("generate() requires task=token_generation; this engine is task=stage_execution");
}

TensorMap StageExecutionEngine::run_stage(
    int32_t request_id,
    const std::string& stage_name,
    const TensorRefMap& inputs)
{
    auto start = std::chrono::steady_clock::now();
    TensorRefMap cached = state_manager_.refs(request_id);
    TensorMap outputs = stage_runtime_.run(request_id, stage_name, inputs, cached);
    state_manager_.put_all(request_id, outputs);
    auto end = std::chrono::steady_clock::now();
    last_stage_metrics_ = {
        {"stage_ms", elapsed_ms(start, end)},
        {"stage_outputs", static_cast<double>(outputs.size())},
    };
    return outputs;
}

std::unordered_map<std::string, double> StageExecutionEngine::get_last_generate_metrics() const {
    return {};
}

std::unordered_map<std::string, double> StageExecutionEngine::get_last_stage_metrics() const {
    return last_stage_metrics_;
}

} // namespace edge_fm
