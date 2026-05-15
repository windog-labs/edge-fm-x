#pragma once

#include "engine/engine.h"
#include "engine/tasks/stage_execution/mock_stage_runner.h"
#include "engine/tasks/trajectory_planning/planner_state_manager.h"

#include <unordered_map>

namespace edge_fm {

class StageExecutionEngine : public Engine {
public:
    explicit StageExecutionEngine(const EngineConfig& config);

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    TensorMap run_stage(int32_t request_id, const std::string& stage_name, const TensorRefMap& inputs) override;
    std::unordered_map<std::string, double> get_last_generate_metrics() const override;
    std::unordered_map<std::string, double> get_last_stage_metrics() const override;

private:
    MockStageRunner mock_stage_runner_;
    PlannerStateManager state_manager_;
    std::unordered_map<std::string, double> last_stage_metrics_{};
};

} // namespace edge_fm
