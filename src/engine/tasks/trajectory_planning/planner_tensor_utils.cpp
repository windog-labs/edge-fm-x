#include "engine/tasks/trajectory_planning/planner_tensor_utils.h"

namespace edge_fm::planner {
namespace {

template <typename T>
Tensor make_cpu_tensor(const std::vector<int64_t>& shape, const std::vector<T>& values, DType dtype) {
    const int64_t expected = tensor_numel(shape);
    if (expected != static_cast<int64_t>(values.size())) {
        throw ConfigurationError(
            "Planner tensor value count does not match shape: expected " +
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

} // namespace edge_fm::planner
