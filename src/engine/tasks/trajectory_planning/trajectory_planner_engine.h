#pragma once

#include "engine/engine.h"
#include "engine/tasks/stage_execution/mock_stage_runner.h"
#include "engine/tasks/trajectory_planning/planner_state_manager.h"

#include <unordered_map>

namespace edge_fm {

class TrajectoryPlannerEngine : public Engine {
public:
    explicit TrajectoryPlannerEngine(const EngineConfig& config);

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    TensorMap plan(int32_t request_id, const TensorRefMap& inputs) override;
    TensorMap run_stage(int32_t request_id, const std::string& stage_name, const TensorRefMap& inputs) override;
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;
    std::unordered_map<std::string, double> get_last_plan_metrics() const override;
    std::unordered_map<std::string, double> get_last_stage_metrics() const override;

private:
    TensorMap plan_single_stage(int32_t request_id, const TensorRefMap& inputs, const nlohmann::json& planner_config);
    TensorMap plan_candidate_scoring(int32_t request_id, const TensorRefMap& inputs, const nlohmann::json& planner_config);
    TensorMap plan_iterative_denoise(int32_t request_id, const TensorRefMap& inputs, const nlohmann::json& planner_config);

    MockStageRunner mock_stage_runner_;
    PlannerStateManager state_manager_;
    std::unordered_map<std::string, double> last_plan_metrics_{};
    std::unordered_map<std::string, double> last_stage_metrics_{};
};

} // namespace edge_fm
