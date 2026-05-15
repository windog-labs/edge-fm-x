#include "engine/tasks/trajectory_planning/trajectory_planner_engine.h"

#include <edge-fm/core.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace edge_fm {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Tensor clone_output_as_trajectory(const TensorMap& outputs, const std::string& output_name) {
    auto it = outputs.find(output_name);
    if (it == outputs.end()) {
        throw InternalError("Planner stage did not produce required output tensor: " + output_name);
    }
    return planner::clone_tensor_to_cpu(it->second);
}

std::vector<float> read_cpu_float_tensor(const Tensor& tensor, const std::string& name) {
    const float* data = planner::require_cpu_float32(tensor, name);
    const int64_t n = planner::tensor_numel(tensor.shape());
    return std::vector<float>(data, data + n);
}

Tensor add_scaled(const Tensor& lhs, const Tensor& rhs, float scale) {
    if (lhs.shape() != rhs.shape()) {
        throw InvalidRequestError("Euler flow state and velocity shapes must match");
    }
    std::vector<float> lhs_values = read_cpu_float_tensor(lhs, "planner state");
    std::vector<float> rhs_values = read_cpu_float_tensor(rhs, "planner velocity");
    for (size_t i = 0; i < lhs_values.size(); ++i) {
        lhs_values[i] += scale * rhs_values[i];
    }
    return planner::make_cpu_float32_tensor(lhs.shape(), lhs_values);
}

Tensor make_initial_state_from_config(const nlohmann::json& planner_config) {
    if (!planner_config.contains("trajectory_shape") || !planner_config["trajectory_shape"].is_array()) {
        const std::string state_name = planner_config.value("state_tensor", std::string("current_actions"));
        throw InvalidRequestError(
            "iterative_denoise planner requires input state tensor or planner.trajectory_shape: " + state_name);
    }
    const std::vector<int64_t> shape = planner_config["trajectory_shape"].get<std::vector<int64_t>>();
    const int64_t count = planner::tensor_numel(shape);
    const float sigma = planner_config.value("noise_sigma", 1.0f);
    const uint32_t seed = static_cast<uint32_t>(planner_config.value("seed", 0));

    std::vector<float> values(static_cast<size_t>(count), 0.0f);
    if (sigma != 0.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, sigma);
        for (float& value : values) {
            value = normal(rng);
        }
    }
    return planner::make_cpu_float32_tensor(shape, values);
}

float timestep_for_step(const nlohmann::json& planner_config, int32_t step, int32_t num_steps) {
    const float start = planner_config.value("timestep_start", 0.0f);
    const float end = planner_config.value("timestep_end", 1.0f);
    if (num_steps <= 1) {
        return start;
    }
    const float alpha = static_cast<float>(step) / static_cast<float>(num_steps - 1);
    return start + alpha * (end - start);
}

std::string normalize_planner_string(const std::string& raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        } else if (ch == '_' || ch == '-' || ch == '.' || ch == ' ' || ch == '/') {
            normalized.push_back('_');
        }
    }
    return normalized;
}

std::string resolve_planner_kind(const nlohmann::json& planner_config) {
    std::string kind = normalize_planner_string(planner_config.value("kind", std::string("")));
    const std::string method = normalize_planner_string(planner_config.value("method", std::string("")));
    if (kind.empty()) {
        if (method == "scoring" || method == "score") {
            kind = "candidate_scoring";
        } else if (method == "flow" ||
                   method == "flow_matching" ||
                   method == "diffusion" ||
                   method == "diffusion_policy")
        {
            kind = "iterative_denoise";
        } else {
            kind = "single_stage";
        }
    }

    if (kind == "single" || kind == "stage" || kind == "plan") {
        return "single_stage";
    }
    if (kind == "scoring" || kind == "score" || kind == "candidate_score") {
        return "candidate_scoring";
    }
    if (kind == "flow" ||
        kind == "flow_matching" ||
        kind == "diffusion" ||
        kind == "diffusion_policy" ||
        kind == "denoise" ||
        kind == "iterative")
    {
        return "iterative_denoise";
    }
    return kind;
}

Tensor select_argmax_trajectory(const Tensor& candidates, const Tensor& scores, Tensor* selected_index_out) {
    planner::require_cpu_float32(candidates, "candidate_trajectories");
    const float* score_values = planner::require_cpu_float32(scores, "candidate_scores");
    const std::vector<int64_t>& candidate_shape = candidates.shape();
    const std::vector<int64_t>& score_shape = scores.shape();
    if (candidate_shape.size() < 3 || score_shape.size() != 2) {
        throw InvalidRequestError("candidate scoring expects candidates [B,K,...] and scores [B,K]");
    }
    const int64_t batch = candidate_shape[0];
    const int64_t candidates_per_batch = candidate_shape[1];
    if (score_shape[0] != batch || score_shape[1] != candidates_per_batch) {
        throw InvalidRequestError("candidate_scores shape must match candidate_trajectories [B,K]");
    }
    if (batch <= 0 || candidates_per_batch <= 0) {
        throw InvalidRequestError("candidate scoring expects at least one batch and at least one candidate");
    }

    int64_t per_candidate = 1;
    std::vector<int64_t> trajectory_shape;
    trajectory_shape.reserve(candidate_shape.size() - 1);
    trajectory_shape.push_back(batch);
    for (size_t i = 2; i < candidate_shape.size(); ++i) {
        per_candidate *= candidate_shape[i];
        trajectory_shape.push_back(candidate_shape[i]);
    }

    const float* candidate_values = planner::require_cpu_float32(candidates, "candidate_trajectories");
    std::vector<float> selected_values(static_cast<size_t>(batch * per_candidate));
    std::vector<int32_t> selected_indices(static_cast<size_t>(batch), 0);
    for (int64_t b = 0; b < batch; ++b) {
        int64_t best = 0;
        float best_score = score_values[b * candidates_per_batch];
        for (int64_t k = 1; k < candidates_per_batch; ++k) {
            const float score = score_values[b * candidates_per_batch + k];
            if (score > best_score) {
                best_score = score;
                best = k;
            }
        }
        selected_indices[static_cast<size_t>(b)] = static_cast<int32_t>(best);
        const size_t src_offset = static_cast<size_t>((b * candidates_per_batch + best) * per_candidate);
        const size_t dst_offset = static_cast<size_t>(b * per_candidate);
        std::copy(
            candidate_values + static_cast<std::ptrdiff_t>(src_offset),
            candidate_values + static_cast<std::ptrdiff_t>(src_offset + static_cast<size_t>(per_candidate)),
            selected_values.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }

    *selected_index_out = planner::make_cpu_int32_tensor({batch}, selected_indices);
    return planner::make_cpu_float32_tensor(trajectory_shape, selected_values);
}

} // namespace

TrajectoryPlannerEngine::TrajectoryPlannerEngine(const EngineConfig& config)
    : Engine(config)
    , stage_runtime_(config)
{}

void TrajectoryPlannerEngine::warmup() {}

void TrajectoryPlannerEngine::tune() {}

Response TrajectoryPlannerEngine::generate(const Request& request) {
    (void)request;
    throw ConfigurationError("generate() requires task=token_generation; this engine is task=trajectory_planning");
}

TensorMap TrajectoryPlannerEngine::run_stage(
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

TensorMap TrajectoryPlannerEngine::plan(int32_t request_id, const TensorRefMap& inputs) {
    const nlohmann::json planner_config = config_.planner();
    const std::string kind = resolve_planner_kind(planner_config);
    auto start = std::chrono::steady_clock::now();

    TensorMap outputs;
    if (kind == "single_stage") {
        outputs = plan_single_stage(request_id, inputs, planner_config);
    } else if (kind == "candidate_scoring") {
        outputs = plan_candidate_scoring(request_id, inputs, planner_config);
    } else if (kind == "iterative_denoise") {
        outputs = plan_iterative_denoise(request_id, inputs, planner_config);
    } else {
        throw ConfigurationError("Unsupported planner.kind: " + kind);
    }

    auto end = std::chrono::steady_clock::now();
    last_plan_metrics_["plan_ms"] = elapsed_ms(start, end);
    last_plan_metrics_["plan_outputs"] = static_cast<double>(outputs.size());
    return outputs;
}

TensorMap TrajectoryPlannerEngine::plan_single_stage(
    int32_t request_id,
    const TensorRefMap& inputs,
    const nlohmann::json& planner_config)
{
    const std::string stage = planner_config.value("stage", std::string("plan"));
    const std::string output_name = planner_config.value("output_tensor", std::string("trajectory"));
    TensorMap outputs = run_stage(request_id, stage, inputs);
    if (output_name != "trajectory") {
        outputs["trajectory"] = clone_output_as_trajectory(outputs, output_name);
    }
    if (outputs.find("trajectory") == outputs.end()) {
        outputs["trajectory"] = clone_output_as_trajectory(outputs, output_name);
    }
    last_plan_metrics_ = {{"stage_calls", 1.0}, {"plan_steps", 1.0}};
    return outputs;
}

TensorMap TrajectoryPlannerEngine::plan_candidate_scoring(
    int32_t request_id,
    const TensorRefMap& inputs,
    const nlohmann::json& planner_config)
{
    const std::string stage = planner_config.value("stage", std::string("score"));
    const std::string candidate_name = planner_config.value("candidate_tensor", std::string("candidate_trajectories"));
    const std::string score_name = planner_config.value("score_tensor", std::string("candidate_scores"));
    TensorMap outputs = run_stage(request_id, stage, inputs);
    auto candidate_it = outputs.find(candidate_name);
    auto score_it = outputs.find(score_name);
    if (candidate_it == outputs.end() || score_it == outputs.end()) {
        throw InternalError("candidate_scoring planner requires candidate_trajectories and candidate_scores outputs");
    }
    Tensor selected_index;
    Tensor trajectory = select_argmax_trajectory(candidate_it->second, score_it->second, &selected_index);
    outputs["selected_index"] = std::move(selected_index);
    outputs["trajectory"] = std::move(trajectory);
    last_plan_metrics_ = {{"stage_calls", 1.0}, {"plan_steps", 1.0}};
    return outputs;
}

TensorMap TrajectoryPlannerEngine::plan_iterative_denoise(
    int32_t request_id,
    const TensorRefMap& inputs,
    const nlohmann::json& planner_config)
{
    const int32_t num_steps = std::max<int32_t>(1, planner_config.value("num_steps", 1));
    const std::string state_name = planner_config.value("state_tensor", std::string("current_actions"));
    const std::string step_stage = planner_config.value("step_stage", std::string("step"));
    const std::string step_output_name = planner_config.value("step_output_tensor", std::string("velocity"));
    const std::string timestep_name = planner_config.value("timestep_tensor", std::string("timestep"));
    const std::string sampler =
        normalize_planner_string(planner_config.value("sampler", std::string("euler_flow")));

    if (planner_config.contains("context_stage")) {
        const std::string context_stage = planner_config.value("context_stage", std::string("context"));
        (void)run_stage(request_id, context_stage, inputs);
    }

    auto input_it = inputs.find(state_name);
    Tensor current = (input_it != inputs.end() && input_it->second != nullptr)
        ? planner::clone_tensor_to_cpu(*input_it->second)
        : make_initial_state_from_config(planner_config);

    for (int32_t step = 0; step < num_steps; ++step) {
        Tensor timestep = planner::make_cpu_float32_tensor(
            {1},
            {timestep_for_step(planner_config, step, num_steps)});
        TensorRefMap step_inputs = inputs;
        step_inputs[state_name] = &current;
        step_inputs[timestep_name] = &timestep;
        TensorMap step_outputs = run_stage(request_id, step_stage, step_inputs);
        auto step_output_it = step_outputs.find(step_output_name);
        if (step_output_it == step_outputs.end()) {
            throw InternalError("iterative_denoise step did not produce tensor: " + step_output_name);
        }
        if (sampler == "euler_flow") {
            const float dt = planner_config.value("dt", 1.0f / static_cast<float>(num_steps));
            current = add_scaled(current, step_output_it->second, dt);
        } else if (sampler == "ddim" || sampler == "ddim_like") {
            current = planner::clone_tensor_to_cpu(step_output_it->second);
        } else {
            throw ConfigurationError("Unsupported iterative_denoise sampler: " + sampler);
        }
    }

    TensorMap outputs;
    const std::string output_name = planner_config.value("output_tensor", std::string("trajectory"));
    outputs[output_name] = planner::clone_tensor_to_cpu(current);
    if (output_name != "trajectory") {
        outputs["trajectory"] = planner::clone_tensor_to_cpu(current);
    }
    last_plan_metrics_ = {
        {"stage_calls", static_cast<double>(num_steps + (planner_config.contains("context_stage") ? 1 : 0))},
        {"plan_steps", static_cast<double>(num_steps)},
    };
    return outputs;
}

std::unordered_map<std::string, double> TrajectoryPlannerEngine::get_last_generate_metrics() const {
    return {};
}

std::unordered_map<std::string, double> TrajectoryPlannerEngine::get_last_plan_metrics() const {
    return last_plan_metrics_;
}

std::unordered_map<std::string, double> TrajectoryPlannerEngine::get_last_stage_metrics() const {
    return last_stage_metrics_;
}

} // namespace edge_fm
