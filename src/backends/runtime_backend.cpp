#include "backends/runtime_backend.h"

#include <edge-fm/core.h>

#include <cstring>
#include <numeric>

namespace edge_fm {

#ifdef EDGE_FM_ENABLE_HORIZON_RUNTIME
std::unique_ptr<IRuntimeBackend> create_horizon_runtime_backend();
#endif

namespace {

bool has_invalid_shape_dim(const RuntimeTensorView& view) {
    for (int64_t dim : view.shape) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

size_t logical_last_dim_bytes(const RuntimeTensorView& view) {
    if (view.shape.empty()) {
        return 0;
    }
    return static_cast<size_t>(view.shape.back()) * runtime_dtype_size(view.dtype);
}

size_t logical_outer_rows(const RuntimeTensorView& view) {
    if (view.shape.empty()) {
        return 0;
    }
    size_t rows = 1;
    for (size_t i = 0; i + 1 < view.shape.size(); ++i) {
        rows *= static_cast<size_t>(view.shape[i]);
    }
    return rows;
}

size_t strided_row_offset(const RuntimeTensorView& view, size_t row) {
    if (view.shape.size() <= 1) {
        return 0;
    }

    size_t offset = 0;
    for (int dim = static_cast<int>(view.shape.size()) - 2; dim >= 0; --dim) {
        const size_t extent = static_cast<size_t>(view.shape[static_cast<size_t>(dim)]);
        const size_t index = extent == 0 ? 0 : row % extent;
        row = extent == 0 ? 0 : row / extent;
        offset += index * static_cast<size_t>(view.stride[static_cast<size_t>(dim)]);
    }
    return offset;
}

void validate_runtime_view(const RuntimeTensorView& view, const char* context) {
    if (view.data == nullptr) {
        throw InvalidRequestError(std::string(context) + " requires non-null tensor data");
    }
    if (has_invalid_shape_dim(view)) {
        throw InvalidRequestError(std::string(context) + " requires non-negative tensor shape");
    }
    if (!view.stride.empty() && view.stride.size() != view.shape.size()) {
        throw InvalidRequestError(std::string(context) + " requires stride to be empty or match tensor rank");
    }
    if (!view.stride.empty()) {
        for (int64_t stride : view.stride) {
            if (stride <= 0) {
                throw InvalidRequestError(std::string(context) + " requires positive byte strides");
            }
        }
        if (!view.shape.empty() &&
            static_cast<size_t>(view.stride.back()) != runtime_dtype_size(view.dtype)) {
            throw InvalidRequestError(
                std::string(context) +
                " requires the innermost dimension to be contiguous for row-wise copy");
        }
    }
}

} // namespace

size_t runtime_dtype_size(RuntimeDType dtype) {
    switch (dtype) {
        case RuntimeDType::Float32:
        case RuntimeDType::Int32:
            return 4;
        case RuntimeDType::Float16:
            return 2;
        case RuntimeDType::Int8:
        case RuntimeDType::UInt8:
            return 1;
    }
    return 0;
}

size_t runtime_tensor_num_elements(const RuntimeTensorView& view) {
    if (view.shape.empty()) {
        return 0;
    }
    if (has_invalid_shape_dim(view)) {
        return 0;
    }
    return std::accumulate(
        view.shape.begin(),
        view.shape.end(),
        static_cast<size_t>(1),
        [](size_t acc, int64_t dim) {
            return acc * static_cast<size_t>(dim);
        });
}

bool runtime_tensor_is_contiguous(const RuntimeTensorView& view) {
    if (view.stride.empty()) {
        return true;
    }
    if (view.stride.size() != view.shape.size()) {
        return false;
    }

    size_t expected = runtime_dtype_size(view.dtype);
    for (int dim = static_cast<int>(view.shape.size()) - 1; dim >= 0; --dim) {
        if (static_cast<size_t>(view.stride[static_cast<size_t>(dim)]) != expected) {
            return false;
        }
        expected *= static_cast<size_t>(view.shape[static_cast<size_t>(dim)] > 0
            ? view.shape[static_cast<size_t>(dim)]
            : 0);
    }
    return true;
}

size_t runtime_tensor_byte_size(const RuntimeTensorView& view) {
    if (view.shape.empty() || has_invalid_shape_dim(view)) {
        return 0;
    }
    if (view.stride.empty()) {
        return runtime_tensor_num_elements(view) * runtime_dtype_size(view.dtype);
    }
    if (view.stride.size() != view.shape.size()) {
        return 0;
    }

    int64_t max_offset = 0;
    for (size_t i = 0; i < view.shape.size(); ++i) {
        if (view.shape[i] <= 0 || view.stride[i] <= 0) {
            return 0;
        }
        max_offset += (view.shape[i] - 1) * view.stride[i];
    }
    return static_cast<size_t>(max_offset) + runtime_dtype_size(view.dtype);
}

void copy_contiguous_to_runtime_buffer(const void* src, const RuntimeTensorView& dst) {
    if (src == nullptr) {
        throw InvalidRequestError("copy_contiguous_to_runtime_buffer requires non-null source data");
    }
    validate_runtime_view(dst, "copy_contiguous_to_runtime_buffer");

    const size_t logical_bytes = runtime_tensor_num_elements(dst) * runtime_dtype_size(dst.dtype);
    if (logical_bytes == 0) {
        return;
    }
    if (runtime_tensor_is_contiguous(dst)) {
        std::memcpy(dst.data, src, logical_bytes);
        return;
    }

    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst.data);
    const size_t row_bytes = logical_last_dim_bytes(dst);
    const size_t rows = logical_outer_rows(dst);
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(dst_bytes + strided_row_offset(dst, row), src_bytes + row * row_bytes, row_bytes);
    }
}

void copy_runtime_buffer_to_contiguous(const RuntimeTensorView& src, void* dst) {
    if (dst == nullptr) {
        throw InvalidRequestError("copy_runtime_buffer_to_contiguous requires non-null destination data");
    }
    validate_runtime_view(src, "copy_runtime_buffer_to_contiguous");

    const size_t logical_bytes = runtime_tensor_num_elements(src) * runtime_dtype_size(src.dtype);
    if (logical_bytes == 0) {
        return;
    }
    if (runtime_tensor_is_contiguous(src)) {
        std::memcpy(dst, src.data, logical_bytes);
        return;
    }

    const auto* src_bytes = static_cast<const uint8_t*>(src.data);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    const size_t row_bytes = logical_last_dim_bytes(src);
    const size_t rows = logical_outer_rows(src);
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(dst_bytes + row * row_bytes, src_bytes + strided_row_offset(src, row), row_bytes);
    }
}

std::unique_ptr<IRuntimeBackend> create_runtime_backend(BackendTarget backend) {
    switch (backend) {
        case BackendTarget::Horizon:
#ifdef EDGE_FM_ENABLE_HORIZON_RUNTIME
            return create_horizon_runtime_backend();
#else
            throw ConfigurationError("Horizon runtime backend is not compiled");
#endif
        case BackendTarget::Cuda:
            throw ConfigurationError(
                "CUDA backend uses StandardEngine and does not use the whole-graph runtime backend interface");
    }
    throw ConfigurationError("Unsupported runtime backend");
}

} // namespace edge_fm
