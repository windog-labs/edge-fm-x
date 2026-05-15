#pragma once

#include "engine/engine.h"

#include <edge-fm/core.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

class PlannerStateManager {
public:
    void put(int32_t request_id, const std::string& name, const Tensor& tensor);
    void put_all(int32_t request_id, const TensorMap& tensors);
    const Tensor* get(int32_t request_id, const std::string& name) const;
    TensorRefMap refs(int32_t request_id) const;
    void clear(int32_t request_id);
    void clear_all();

private:
    std::unordered_map<int32_t, TensorMap> states_;
};

class StageRuntime {
public:
    explicit StageRuntime(const EngineConfig& config);

    TensorMap run(
        int32_t request_id,
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

namespace planner {

DType dtype_from_planner_string(const std::string& raw);
int64_t tensor_numel(const std::vector<int64_t>& shape);
Tensor clone_tensor_to_cpu(const Tensor& src);
Tensor make_cpu_float32_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values);
Tensor make_cpu_int32_tensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& values);
Tensor make_cpu_int64_tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values);
Tensor make_cpu_uint8_tensor(const std::vector<int64_t>& shape, const std::vector<uint8_t>& values);
Tensor make_cpu_int8_tensor(const std::vector<int64_t>& shape, const std::vector<int8_t>& values);
const float* require_cpu_float32(const Tensor& tensor, const std::string& name);

} // namespace planner
} // namespace edge_fm
