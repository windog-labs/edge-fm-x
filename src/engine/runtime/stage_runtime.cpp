#include "engine/runtime/stage_runtime.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace edge_fm {
namespace {

std::vector<int64_t> read_shape(const nlohmann::json& spec, const std::string& tensor_name) {
    if (!spec.contains("shape") || !spec["shape"].is_array()) {
        throw ConfigurationError("Mock stage tensor is missing shape: " + tensor_name);
    }
    return spec["shape"].get<std::vector<int64_t>>();
}

template <typename T>
std::vector<T> read_integral_values(const nlohmann::json& spec, const std::string& tensor_name) {
    if (!spec.contains("values") || !spec["values"].is_array()) {
        throw ConfigurationError("Mock integer tensor is missing values: " + tensor_name);
    }
    std::vector<T> values;
    values.reserve(spec["values"].size());
    for (const auto& item : spec["values"]) {
        const int64_t value = item.get<int64_t>();
        if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
            value > static_cast<int64_t>(std::numeric_limits<T>::max()))
        {
            throw ConfigurationError("Mock integer tensor value is out of range: " + tensor_name);
        }
        values.push_back(static_cast<T>(value));
    }
    return values;
}

std::vector<float> read_float_values(const nlohmann::json& spec, const std::string& tensor_name) {
    if (!spec.contains("values") || !spec["values"].is_array()) {
        throw ConfigurationError("Mock float32 tensor is missing values: " + tensor_name);
    }
    std::vector<float> values;
    values.reserve(spec["values"].size());
    for (const auto& item : spec["values"]) {
        values.push_back(item.get<float>());
    }
    return values;
}

Tensor make_mock_tensor_from_spec(const std::string& name, const nlohmann::json& spec) {
    const std::vector<int64_t> shape = read_shape(spec, name);
    const std::string dtype = spec.value("dtype", std::string("float32"));
    if (dtype == "float32" || dtype == "fp32") {
        return planner::make_cpu_float32_tensor(shape, read_float_values(spec, name));
    }
    if (dtype == "int32") {
        return planner::make_cpu_int32_tensor(shape, read_integral_values<int32_t>(spec, name));
    }
    if (dtype == "int64") {
        return planner::make_cpu_int64_tensor(shape, read_integral_values<int64_t>(spec, name));
    }
    if (dtype == "uint8") {
        return planner::make_cpu_uint8_tensor(shape, read_integral_values<uint8_t>(spec, name));
    }
    if (dtype == "int8") {
        return planner::make_cpu_int8_tensor(shape, read_integral_values<int8_t>(spec, name));
    }

    (void)planner::dtype_from_planner_string(dtype);
    throw ConfigurationError(
        "Mock StageRuntime only supports float32, int32, int64, uint8, and int8 tensors for now");
}

template <typename T>
Tensor make_cpu_tensor(const std::vector<int64_t>& shape, const std::vector<T>& values, DType dtype) {
    const int64_t expected = planner::tensor_numel(shape);
    if (expected != static_cast<int64_t>(values.size())) {
        throw ConfigurationError(
            "Planner mock tensor value count does not match shape: expected " +
            std::to_string(expected) + ", got " + std::to_string(values.size()));
    }
    return Tensor::clone_from(
        values.data(),
        shape,
        dtype,
        Device::CPU,
        0,
        Device::CPU,
        0,
        MemoryOwnership::OwnCpuMalloc);
}

} // namespace

void PlannerStateManager::put(int32_t request_id, const std::string& name, const Tensor& tensor) {
    states_[request_id][name] = planner::clone_tensor_to_cpu(tensor);
}

void PlannerStateManager::put_all(int32_t request_id, const TensorMap& tensors) {
    for (const auto& item : tensors) {
        put(request_id, item.first, item.second);
    }
}

const Tensor* PlannerStateManager::get(int32_t request_id, const std::string& name) const {
    auto state_it = states_.find(request_id);
    if (state_it == states_.end()) {
        return nullptr;
    }
    auto tensor_it = state_it->second.find(name);
    if (tensor_it == state_it->second.end()) {
        return nullptr;
    }
    return &tensor_it->second;
}

TensorRefMap PlannerStateManager::refs(int32_t request_id) const {
    TensorRefMap out;
    auto state_it = states_.find(request_id);
    if (state_it == states_.end()) {
        return out;
    }
    for (const auto& item : state_it->second) {
        out.emplace(item.first, &item.second);
    }
    return out;
}

void PlannerStateManager::clear(int32_t request_id) {
    states_.erase(request_id);
}

void PlannerStateManager::clear_all() {
    states_.clear();
}

StageRuntime::StageRuntime(const EngineConfig& config)
    : config_(config)
{}

const nlohmann::json& StageRuntime::require_stage(const std::string& stage_name) const {
    if (!config_.raw().contains("stages")) {
        throw ConfigurationError("StageRuntime requires a top-level stages object or array");
    }
    const nlohmann::json& stages = config_.raw()["stages"];
    if (stages.is_object()) {
        auto it = stages.find(stage_name);
        if (it != stages.end() && it->is_object()) {
            return *it;
        }
    }
    if (stages.is_array()) {
        for (const auto& stage : stages) {
            if (stage.is_object() && stage.value("name", std::string()) == stage_name) {
                return stage;
            }
        }
    }
    throw ConfigurationError("StageRuntime is missing stage: " + stage_name);
}

TensorMap StageRuntime::run(
    int32_t request_id,
    const std::string& stage_name,
    const TensorRefMap& inputs,
    const TensorRefMap& cached_inputs)
{
    (void)request_id;
    const nlohmann::json& stage = require_stage(stage_name);
    const std::string backend = stage.value("backend", stage.value("runtime", std::string("")));
    if (backend != "mock") {
        throw ConfigurationError(
            "StageRuntime currently supports mock stages in the generic engine. "
            "Use HorizonEngine for HBM stages or add a backend adapter for stage: " + stage_name);
    }

    TensorMap default_tensors;
    const nlohmann::json* defaults = nullptr;
    if (stage.contains("defaults") && stage["defaults"].is_object()) {
        defaults = &stage["defaults"];
    } else if (stage.contains("default_inputs") && stage["default_inputs"].is_object()) {
        defaults = &stage["default_inputs"];
    }
    if (defaults != nullptr) {
        for (auto it = defaults->begin(); it != defaults->end(); ++it) {
            if (!it.value().is_object()) {
                throw ConfigurationError("Stage default tensor spec must be an object: " + it.key());
            }
            default_tensors.emplace(it.key(), make_mock_tensor_from_spec(it.key(), it.value()));
        }
    }

    TensorRefMap resolved_inputs;
    for (const auto& item : default_tensors) {
        resolved_inputs[item.first] = &item.second;
    }
    for (const auto& item : cached_inputs) {
        if (item.second != nullptr) {
            resolved_inputs[item.first] = item.second;
        }
    }
    for (const auto& item : inputs) {
        if (item.second != nullptr) {
            resolved_inputs[item.first] = item.second;
        }
    }
    return run_mock_stage(stage_name, stage, resolved_inputs);
}

TensorMap StageRuntime::run_mock_stage(
    const std::string& stage_name,
    const nlohmann::json& stage,
    const TensorRefMap& resolved_inputs) const
{
    (void)stage_name;
    if (!stage.contains("outputs") || !stage["outputs"].is_object()) {
        throw ConfigurationError("Mock stage must define an outputs object");
    }

    TensorMap outputs;
    for (auto it = stage["outputs"].begin(); it != stage["outputs"].end(); ++it) {
        const std::string name = it.key();
        const nlohmann::json& spec = it.value();
        if (!spec.is_object()) {
            throw ConfigurationError("Mock stage output spec must be an object: " + name);
        }
        if (spec.contains("source")) {
            const std::string source_name = spec.value("source", std::string());
            auto input_it = resolved_inputs.find(source_name);
            if (source_name.empty() || input_it == resolved_inputs.end() || input_it->second == nullptr) {
                throw InvalidRequestError("Mock stage output source is missing input tensor: " + source_name);
            }
            outputs.emplace(name, planner::clone_tensor_to_cpu(*input_it->second));
            continue;
        }
        outputs.emplace(name, make_mock_tensor_from_spec(name, spec));
    }
    return outputs;
}

namespace planner {

DType dtype_from_planner_string(const std::string& raw) {
    if (raw == "float32" || raw == "fp32") {
        return DType::Float32;
    }
    if (raw == "float16" || raw == "fp16") {
        return DType::Float16;
    }
    if (raw == "bfloat16" || raw == "bf16") {
        return DType::BFloat16;
    }
    if (raw == "int32") {
        return DType::Int32;
    }
    if (raw == "int64") {
        return DType::Int64;
    }
    if (raw == "uint8") {
        return DType::UInt8;
    }
    if (raw == "int8") {
        return DType::Int8;
    }
    throw ConfigurationError("Unsupported planner tensor dtype: " + raw);
}

int64_t tensor_numel(const std::vector<int64_t>& shape) {
    int64_t out = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw ConfigurationError("Planner tensor shapes must be non-negative");
        }
        out *= dim;
    }
    return out;
}

Tensor clone_tensor_to_cpu(const Tensor& src) {
    auto [src_device, src_device_id] = src.device();
    return Tensor::clone_from(
        src.data_ptr(),
        src.shape(),
        src.dtype(),
        src_device,
        src_device_id,
        Device::CPU,
        0,
        MemoryOwnership::OwnCpuMalloc);
}

Tensor make_cpu_float32_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    return make_cpu_tensor(shape, values, DType::Float32);
}

Tensor make_cpu_int32_tensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& values) {
    return make_cpu_tensor(shape, values, DType::Int32);
}

Tensor make_cpu_int64_tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) {
    return make_cpu_tensor(shape, values, DType::Int64);
}

Tensor make_cpu_uint8_tensor(const std::vector<int64_t>& shape, const std::vector<uint8_t>& values) {
    return make_cpu_tensor(shape, values, DType::UInt8);
}

Tensor make_cpu_int8_tensor(const std::vector<int64_t>& shape, const std::vector<int8_t>& values) {
    return make_cpu_tensor(shape, values, DType::Int8);
}

const float* require_cpu_float32(const Tensor& tensor, const std::string& name) {
    auto [device, device_id] = tensor.device();
    (void)device_id;
    if (device != Device::CPU || tensor.dtype() != DType::Float32) {
        throw InvalidRequestError(name + " must be a CPU float32 tensor for planner v1");
    }
    return static_cast<const float*>(tensor.data_ptr());
}

} // namespace planner
} // namespace edge_fm
