#pragma once

#include <edge-fm/core.h>

#include <string>
#include <vector>

namespace edge_fm::planner {

DType dtype_from_planner_string(const std::string& raw);
int64_t tensor_numel(const std::vector<int64_t>& shape);
Tensor clone_tensor_to_cpu(const Tensor& src);
Tensor make_cpu_float32_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values);
Tensor make_cpu_int32_tensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& values);
Tensor make_cpu_int64_tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values);
Tensor make_cpu_uint8_tensor(const std::vector<int64_t>& shape, const std::vector<uint8_t>& values);
Tensor make_cpu_int8_tensor(const std::vector<int64_t>& shape, const std::vector<int8_t>& values);
const float* require_cpu_float32(const Tensor& tensor, const std::string& name);

} // namespace edge_fm::planner
