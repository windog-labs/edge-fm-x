#include "engine/tasks/trajectory_planning/planner_state_manager.h"

#include <edge-fm/core.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

edge_fm::Tensor make_float_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    return edge_fm::Tensor::clone_from(
        values.data(),
        shape,
        edge_fm::DType::Float32,
        edge_fm::Device::CPU,
        0,
        edge_fm::Device::CPU,
        0,
        edge_fm::MemoryOwnership::OwnCpuMalloc);
}

std::vector<float> tensor_values(const edge_fm::Tensor& tensor) {
    auto [device, device_id] = tensor.device();
    (void)device_id;
    require(device == edge_fm::Device::CPU, "expected CPU tensor");
    require(tensor.dtype() == edge_fm::DType::Float32, "expected float32 tensor");
    int64_t count = 1;
    for (int64_t dim : tensor.shape()) {
        count *= dim;
    }
    const float* data = static_cast<const float*>(tensor.data_ptr());
    return std::vector<float>(data, data + count);
}

void test_put_overwrite_refs_and_clear() {
    edge_fm::PlannerStateManager manager;
    edge_fm::Tensor first = make_float_tensor({1, 2}, {1.0f, 2.0f});
    edge_fm::Tensor second = make_float_tensor({1, 2}, {3.0f, 4.0f});

    manager.put(7, "planner_state", first);
    const edge_fm::Tensor* cached = manager.get(7, "planner_state");
    require(cached != nullptr, "put should cache tensor");
    require((tensor_values(*cached) == std::vector<float>{1.0f, 2.0f}), "cached tensor values mismatch");

    manager.put(7, "planner_state", second);
    cached = manager.get(7, "planner_state");
    require(cached != nullptr, "overwrite should keep tensor available");
    require((tensor_values(*cached) == std::vector<float>{3.0f, 4.0f}), "overwrite values mismatch");

    edge_fm::TensorRefMap refs = manager.refs(7);
    require(refs.size() == 1, "refs should expose cached tensor");
    require(refs.at("planner_state") == cached, "refs should point at cached tensor");

    manager.clear(7);
    require(manager.get(7, "planner_state") == nullptr, "clear should remove request-local state");
    require(manager.refs(7).empty(), "refs should be empty after clear");
}

void test_put_all_and_clear_all() {
    edge_fm::PlannerStateManager manager;
    edge_fm::TensorMap tensors;
    tensors.emplace("context", make_float_tensor({1}, {5.0f}));
    tensors.emplace("actions", make_float_tensor({2}, {6.0f, 7.0f}));

    manager.put_all(1, tensors);
    manager.put_all(2, tensors);
    require(manager.refs(1).size() == 2, "put_all should cache all tensors for request 1");
    require(manager.refs(2).size() == 2, "put_all should cache all tensors for request 2");

    manager.clear_all();
    require(manager.refs(1).empty(), "clear_all should remove request 1");
    require(manager.refs(2).empty(), "clear_all should remove request 2");
}

} // namespace

int main() {
    try {
        test_put_overwrite_refs_and_clear();
        test_put_all_and_clear_all();
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
