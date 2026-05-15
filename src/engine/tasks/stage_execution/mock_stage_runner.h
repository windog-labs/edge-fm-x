#pragma once

#include "engine/engine.h"

#include <nlohmann/json.hpp>

#include <string>

namespace edge_fm {

class MockStageRunner {
public:
    explicit MockStageRunner(const EngineConfig& config);

    TensorMap run(
        const std::string& stage_name,
        const TensorRefMap& inputs,
        const TensorRefMap& cached_inputs = TensorRefMap());

private:
    const nlohmann::json& require_stage(const std::string& stage_name) const;
    TensorMap run_mock_stage(
        const std::string& stage_name,
        const nlohmann::json& stage,
        const TensorRefMap& resolved_inputs) const;

    EngineConfig config_;
};

} // namespace edge_fm
