#include "engine/runtime_state_arena.h"

#include "utils/check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

namespace edge_fm {

namespace {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Int64:
            return 8;
        case DType::UInt8:
        case DType::Int8:
            return 1;
        default:
            throw InvalidRequestError("Unsupported dtype for RuntimeStateArena");
    }
}

size_t num_elements(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }

    size_t result = 1;
    for (int64_t dim : shape) {
        check<InvalidRequestError>(dim >= 0, "RuntimeStateArena tensor shape must be non-negative");
        const size_t unsigned_dim = static_cast<size_t>(dim);
        if (unsigned_dim == 0) {
            return 0;
        }
        check<OutOfMemoryError>(
            result <= std::numeric_limits<size_t>::max() / unsigned_dim,
            "RuntimeStateArena tensor shape is too large");
        result *= unsigned_dim;
    }
    return result;
}

MemoryOwnership resolve_ownership(Device device, MemoryOwnership ownership, void* stream_handle) {
    (void)stream_handle;
    if (ownership != MemoryOwnership::ViewExternal) {
        return ownership;
    }
    if (device == Device::GPU) {
        return MemoryOwnership::OwnCudaMalloc;
    }
    return MemoryOwnership::OwnCpuMalloc;
}

void check_tensor_metadata(const std::string& name,
                           const Tensor& tensor,
                           const std::vector<int64_t>& shape,
                           DType dtype,
                           Device device,
                           int32_t device_id) {
    auto [existing_device, existing_device_id] = tensor.device();
    check<InvalidRequestError>(
        tensor.shape() == shape && tensor.dtype() == dtype && existing_device == device &&
            existing_device_id == device_id,
        "RuntimeStateArena tensor '" + name + "' already exists with incompatible metadata");
}

void check_tensor_data_ptr(const std::string& name, const Tensor& tensor, const void* data_ptr) {
    check<InvalidRequestError>(
        tensor.data_ptr() == data_ptr,
        "RuntimeStateArena tensor '" + name + "' already exists with a different data pointer");
}

} // namespace

Tensor& RuntimeStateArena::get_or_create(const std::string& name,
                                         const std::vector<int64_t>& shape,
                                         DType dtype,
                                         Device device,
                                         int32_t device_id,
                                         MemoryOwnership ownership,
                                         void* stream_handle) {
    check<InvalidRequestError>(!name.empty(), "RuntimeStateArena tensor name must not be empty");

    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        check_tensor_metadata(name, it->second, shape, dtype, device, device_id);
        return it->second;
    }

    const size_t elements = num_elements(shape);
    const size_t bytes_per_element = dtype_size(dtype);
    check<OutOfMemoryError>(
        elements <= std::numeric_limits<size_t>::max() / bytes_per_element,
        "RuntimeStateArena tensor byte size is too large");

    const size_t byte_size = elements * bytes_per_element;
    Tensor tensor;
    if (byte_size == 0) {
        tensor = Tensor::view(nullptr, shape, dtype, device, device_id);
    } else {
        std::vector<uint8_t> zeros(byte_size, 0);
        tensor = Tensor::clone_from(zeros.data(),
                                    shape,
                                    dtype,
                                    Device::CPU,
                                    0,
                                    device,
                                    device_id,
                                    resolve_ownership(device, ownership, stream_handle),
                                    stream_handle);
    }

    auto inserted = tensors_.emplace(name, std::move(tensor));
    return inserted.first->second;
}

Tensor& RuntimeStateArena::bind_external_view(const std::string& name,
                                              void* data_ptr,
                                              const std::vector<int64_t>& shape,
                                              DType dtype,
                                              Device device,
                                              int32_t device_id) {
    check<InvalidRequestError>(!name.empty(), "RuntimeStateArena tensor name must not be empty");
    if (num_elements(shape) > 0) {
        check<InvalidRequestError>(data_ptr != nullptr, "RuntimeStateArena external view data_ptr must not be null");
    }

    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        check_tensor_metadata(name, it->second, shape, dtype, device, device_id);
        check_tensor_data_ptr(name, it->second, data_ptr);
        return it->second;
    }

    Tensor tensor = Tensor::view(data_ptr, shape, dtype, device, device_id);
    auto inserted = tensors_.emplace(name, std::move(tensor));
    return inserted.first->second;
}

Tensor RuntimeStateArena::view(const std::string& name, const std::vector<int64_t>& shape) const {
    const Tensor* tensor = find(name);
    check<InvalidRequestError>(tensor != nullptr, "RuntimeStateArena tensor '" + name + "' does not exist");
    check<InvalidRequestError>(
        num_elements(shape) == num_elements(tensor->shape()),
        "RuntimeStateArena view for tensor '" + name + "' must preserve element count");

    auto [device, device_id] = tensor->device();
    return Tensor::view(tensor->data_ptr(), shape, tensor->dtype(), device, device_id);
}

Tensor* RuntimeStateArena::find(const std::string& name) {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor* RuntimeStateArena::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool RuntimeStateArena::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> RuntimeStateArena::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& item : tensors_) {
        out.push_back(item.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

void RuntimeStateArena::clear() {
    tensors_.clear();
}

} // namespace edge_fm
