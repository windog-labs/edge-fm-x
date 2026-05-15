#include <edge-fm/core.h>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_cpu_clone_and_dlpack() {
    std::vector<int32_t> values = {1, 2, 3, 4, 5, 6};
    edge_fm::Tensor tensor = edge_fm::Tensor::clone_from(
        values.data(),
        {2, 3},
        edge_fm::DType::Int32,
        edge_fm::Device::CPU,
        0,
        edge_fm::Device::CPU,
        0,
        edge_fm::MemoryOwnership::OwnCpuMalloc);

    require(!tensor.empty(), "CPU clone should create a non-empty tensor");
    require(tensor.dtype() == edge_fm::DType::Int32, "CPU clone dtype mismatch");
    require(tensor.shape() == std::vector<int64_t>({2, 3}), "CPU clone shape mismatch");
    require(
        std::equal(values.begin(), values.end(), static_cast<int32_t*>(tensor.data_ptr())),
        "CPU clone data mismatch");

    DLManagedTensor* dlpack = tensor.to_dlpack();
    require(dlpack != nullptr, "to_dlpack returned null");
    require(dlpack->dl_tensor.device.device_type == kDLCPU, "DLPack device should be CPU");
    require(dlpack->dl_tensor.ndim == 2, "DLPack ndim mismatch");
    require(dlpack->dl_tensor.shape[0] == 2 && dlpack->dl_tensor.shape[1] == 3, "DLPack shape mismatch");
    require(dlpack->dl_tensor.dtype.code == kDLInt && dlpack->dl_tensor.dtype.bits == 32, "DLPack dtype mismatch");
    dlpack->deleter(dlpack);
}

void test_empty_cpu_clone_preserves_metadata() {
    edge_fm::Tensor tensor = edge_fm::Tensor::clone_from(
        nullptr,
        {1, 0, 2},
        edge_fm::DType::Float32,
        edge_fm::Device::CPU,
        0,
        edge_fm::Device::CPU,
        0,
        edge_fm::MemoryOwnership::OwnCpuMalloc);

    require(tensor.empty(), "Zero-element clone should create an empty tensor");
    require(tensor.dtype() == edge_fm::DType::Float32, "Zero-element clone dtype mismatch");
    require(tensor.shape() == std::vector<int64_t>({1, 0, 2}), "Zero-element clone shape mismatch");
    auto [device, device_id] = tensor.device();
    require(device == edge_fm::Device::CPU, "Zero-element clone device mismatch");
    require(device_id == 0, "Zero-element clone device id mismatch");
}

} // namespace

int main() {
    try {
        test_cpu_clone_and_dlpack();
        test_empty_cpu_clone_preserves_metadata();
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
