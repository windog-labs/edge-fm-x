#include <edge-fm/core.h>

#include <dlpack/dlpack.h>

#if defined(EDGE_FM_ENABLE_CUDA)
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace edge_fm {

namespace detail {

namespace {

[[noreturn]] void throw_cuda_tensor_unavailable(const char* message) {
    throw ConfigurationError(message);
}

#if defined(EDGE_FM_ENABLE_CUDA)

DLDevice cuda_tensor_device_to_dlpack(int32_t device_id) {
    DLDevice dl_device{};
    dl_device.device_type = kDLCUDA;
    dl_device.device_id = device_id;
    return dl_device;
}

void cuda_tensor_device_from_dlpack(const DLDevice& dl_device, Device* device, int32_t* device_id) {
    if (device == nullptr || device_id == nullptr) {
        throw InvalidRequestError("tensor_device_from_dlpack requires non-null outputs");
    }
    if (dl_device.device_type == kDLCUDA || dl_device.device_type == kDLCUDAHost) {
        *device = Device::GPU;
        *device_id = dl_device.device_id;
        return;
    }
    throw InvalidRequestError("Unsupported DLPack device type");
}

void check_cuda_tensor_ownership_supported(MemoryOwnership) {}

void check_cuda_tensor_clone_supported(Device, Device, MemoryOwnership) {}

void* allocate_cuda_tensor_data(size_t byte_size,
                                int32_t device_id,
                                MemoryOwnership ownership,
                                void* stream_handle) {
    void* data = nullptr;
    CUDA_CHECK_THROW(cudaSetDevice(device_id), "Failed to set CUDA device for tensor allocation");
    if (ownership == MemoryOwnership::OwnCudaPool) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
        data = MemoryPool::instance().allocate(byte_size, stream, device_id);
    } else if (ownership == MemoryOwnership::OwnCudaMalloc) {
        CUDA_CHECK_THROW_EX(
            cudaMalloc(&data, byte_size),
            "Failed to allocate GPU memory",
            OutOfMemoryError);
    } else {
        throw InvalidRequestError("GPU tensor allocation requires CUDA ownership");
    }
    return data;
}

void free_cuda_tensor_data(void* data,
                           int32_t device_id,
                           MemoryOwnership ownership,
                           void* stream_handle) noexcept {
    cudaSetDevice(device_id);
    if (ownership == MemoryOwnership::OwnCudaPool) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
        cudaFreeAsync(data, stream ? stream : 0);
    } else if (ownership == MemoryOwnership::OwnCudaMalloc) {
        cudaFree(data);
    }
}

void copy_cuda_tensor_data(void* dst,
                           Device dst_device,
                           int32_t dst_device_id,
                           MemoryOwnership dst_ownership,
                           void* dst_stream_handle,
                           const void* src,
                           Device src_device,
                           int32_t src_device_id,
                           size_t byte_size) {
    const bool cuda_async =
        dst_device == Device::GPU && dst_ownership == MemoryOwnership::OwnCudaPool;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(dst_stream_handle);

    if (dst_device == Device::GPU && src_device == Device::GPU) {
        CUDA_CHECK_THROW(cudaSetDevice(dst_device_id), "Failed to set CUDA device for GPU to GPU copy");
        if (dst_device_id != src_device_id) {
            if (cuda_async) {
                CUDA_CHECK_THROW(
                    cudaMemcpyPeerAsync(dst, dst_device_id, src, src_device_id, byte_size, stream),
                    "Failed to copy GPU to GPU (async)");
            } else {
                CUDA_CHECK_THROW(
                    cudaMemcpyPeer(dst, dst_device_id, src, src_device_id, byte_size),
                    "Failed to copy GPU to GPU");
            }
        } else if (cuda_async) {
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDeviceToDevice, stream),
                "Failed to copy GPU to GPU (async)");
        } else {
            CUDA_CHECK_THROW(
                cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToDevice),
                "Failed to copy GPU to GPU");
        }
        return;
    }

    if (dst_device == Device::GPU && src_device == Device::CPU) {
        CUDA_CHECK_THROW(cudaSetDevice(dst_device_id), "Failed to set CUDA device for CPU to GPU copy");
        if (cuda_async) {
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyHostToDevice, stream),
                "Failed to copy CPU to GPU (async)");
        } else {
            CUDA_CHECK_THROW(
                cudaMemcpy(dst, src, byte_size, cudaMemcpyHostToDevice),
                "Failed to copy CPU to GPU");
        }
        return;
    }

    if (dst_device == Device::CPU && src_device == Device::GPU) {
        CUDA_CHECK_THROW(cudaSetDevice(src_device_id), "Failed to set CUDA device for GPU to CPU copy");
        CUDA_CHECK_THROW(
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost),
            "Failed to copy GPU to CPU");
        return;
    }

    std::memcpy(dst, src, byte_size);
}

void copy_cuda_tensor_to_cpu_for_dump(void* dst,
                                      const void* src,
                                      size_t byte_size,
                                      int32_t src_device_id) {
    CUDA_CHECK_THROW(cudaSetDevice(src_device_id), "Failed to set CUDA device for tensor dump");
    CUDA_CHECK_THROW(
        cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost),
        "Failed to copy GPU tensor to CPU for dumping");
}

#else

DLDevice cuda_tensor_device_to_dlpack(int32_t device_id) {
    (void)device_id;
    throw_cuda_tensor_unavailable("Cannot export CUDA DLPack tensor because CUDA backend is not compiled");
}

void cuda_tensor_device_from_dlpack(const DLDevice& dl_device, Device* device, int32_t* device_id) {
    (void)dl_device;
    (void)device;
    (void)device_id;
    throw_cuda_tensor_unavailable("DLPack CUDA tensor requested but CUDA backend is not compiled");
}

void check_cuda_tensor_ownership_supported(MemoryOwnership ownership) {
    (void)ownership;
    throw_cuda_tensor_unavailable("CUDA tensor ownership requested but CUDA backend is not compiled");
}

void check_cuda_tensor_clone_supported(Device src_device, Device dst_device, MemoryOwnership ownership) {
    (void)src_device;
    (void)dst_device;
    (void)ownership;
    throw_cuda_tensor_unavailable("CUDA tensor clone requested but CUDA backend is not compiled");
}

void* allocate_cuda_tensor_data(size_t byte_size,
                                int32_t device_id,
                                MemoryOwnership ownership,
                                void* stream_handle) {
    (void)byte_size;
    (void)device_id;
    (void)ownership;
    (void)stream_handle;
    throw_cuda_tensor_unavailable("CUDA tensor allocation requested but CUDA backend is not compiled");
}

void free_cuda_tensor_data(void* data,
                           int32_t device_id,
                           MemoryOwnership ownership,
                           void* stream_handle) noexcept {
    (void)data;
    (void)device_id;
    (void)ownership;
    (void)stream_handle;
}

void copy_cuda_tensor_data(void* dst,
                           Device dst_device,
                           int32_t dst_device_id,
                           MemoryOwnership dst_ownership,
                           void* dst_stream_handle,
                           const void* src,
                           Device src_device,
                           int32_t src_device_id,
                           size_t byte_size) {
    (void)dst;
    (void)dst_device;
    (void)dst_device_id;
    (void)dst_ownership;
    (void)dst_stream_handle;
    (void)src;
    (void)src_device;
    (void)src_device_id;
    (void)byte_size;
    throw_cuda_tensor_unavailable("CUDA tensor copy requested but CUDA backend is not compiled");
}

void copy_cuda_tensor_to_cpu_for_dump(void* dst,
                                      const void* src,
                                      size_t byte_size,
                                      int32_t src_device_id) {
    (void)dst;
    (void)src;
    (void)byte_size;
    (void)src_device_id;
    throw_cuda_tensor_unavailable("Cannot dump CUDA tensor because CUDA backend is not compiled");
}

#endif

} // namespace

DLDevice tensor_device_to_dlpack(Device device, int32_t device_id) {
    if (device == Device::CPU) {
        DLDevice dl_device{};
        dl_device.device_type = kDLCPU;
        dl_device.device_id = 0;
        return dl_device;
    }
    if (device == Device::GPU) {
        return cuda_tensor_device_to_dlpack(device_id);
    }
    throw InvalidRequestError("Unsupported Device for DLPack conversion");
}

void tensor_device_from_dlpack(const DLDevice& dl_device, Device* device, int32_t* device_id) {
    if (device == nullptr || device_id == nullptr) {
        throw InvalidRequestError("tensor_device_from_dlpack requires non-null outputs");
    }
    if (dl_device.device_type == kDLCPU) {
        *device = Device::CPU;
        *device_id = 0;
        return;
    }
    cuda_tensor_device_from_dlpack(dl_device, device, device_id);
}

void check_tensor_ownership_supported(MemoryOwnership ownership) {
    if (ownership == MemoryOwnership::OwnCudaPool || ownership == MemoryOwnership::OwnCudaMalloc) {
        check_cuda_tensor_ownership_supported(ownership);
    }
}

void check_tensor_clone_supported(Device src_device, Device dst_device, MemoryOwnership ownership) {
    if (ownership == MemoryOwnership::OwnCudaPool || ownership == MemoryOwnership::OwnCudaMalloc ||
        src_device == Device::GPU || dst_device == Device::GPU) {
        check_cuda_tensor_clone_supported(src_device, dst_device, ownership);
    }
}

void* allocate_tensor_data(size_t byte_size,
                           Device device,
                           int32_t device_id,
                           MemoryOwnership ownership,
                           void* stream_handle) {
    if (byte_size == 0) {
        return nullptr;
    }
    if (device == Device::CPU) {
        void* data = std::malloc(byte_size);
        if (data == nullptr) {
            throw OutOfMemoryError("Failed to allocate CPU memory");
        }
        return data;
    }
    return allocate_cuda_tensor_data(byte_size, device_id, ownership, stream_handle);
}

void free_tensor_data(void* data,
                      Device device,
                      int32_t device_id,
                      MemoryOwnership ownership,
                      void* stream_handle) noexcept {
    if (data == nullptr) {
        return;
    }
    if (ownership == MemoryOwnership::OwnCpuMalloc) {
        std::free(data);
        return;
    }
    if (device == Device::GPU) {
        free_cuda_tensor_data(data, device_id, ownership, stream_handle);
    }
}

void copy_tensor_data(void* dst,
                      Device dst_device,
                      int32_t dst_device_id,
                      MemoryOwnership dst_ownership,
                      void* dst_stream_handle,
                      const void* src,
                      Device src_device,
                      int32_t src_device_id,
                      size_t byte_size) {
    if (byte_size == 0) {
        return;
    }
    if (dst_device == Device::CPU && src_device == Device::CPU) {
        std::memcpy(dst, src, byte_size);
        return;
    }
    copy_cuda_tensor_data(
        dst,
        dst_device,
        dst_device_id,
        dst_ownership,
        dst_stream_handle,
        src,
        src_device,
        src_device_id,
        byte_size);
}

void copy_tensor_to_cpu_for_dump(void* dst,
                                 const void* src,
                                 size_t byte_size,
                                 int32_t src_device_id) {
    copy_cuda_tensor_to_cpu_for_dump(dst, src, byte_size, src_device_id);
}

} // namespace detail

namespace {

DType dlpack_to_dtype(DLDataType dl_dtype) {
    if (dl_dtype.code == kDLFloat) {
        if (dl_dtype.bits == 32) return DType::Float32;
        if (dl_dtype.bits == 16) return DType::Float16;
    } else if (dl_dtype.code == kDLInt) {
        if (dl_dtype.bits == 32) return DType::Int32;
        if (dl_dtype.bits == 64) return DType::Int64;
        if (dl_dtype.bits == 8) return DType::Int8;
    } else if (dl_dtype.code == kDLUInt) {
        if (dl_dtype.bits == 8) return DType::UInt8;
    } else if (dl_dtype.code == kDLBfloat) {
        if (dl_dtype.bits == 16) return DType::BFloat16;
    }
    throw InvalidRequestError("Unsupported DLPack data type");
}

DLDataType dtype_to_dlpack(DType dtype) {
    DLDataType dl_dtype{};
    dl_dtype.lanes = 1;

    switch (dtype) {
        case DType::Float32:
            dl_dtype.code = kDLFloat;
            dl_dtype.bits = 32;
            break;
        case DType::Float16:
            dl_dtype.code = kDLFloat;
            dl_dtype.bits = 16;
            break;
        case DType::BFloat16:
            dl_dtype.code = kDLBfloat;
            dl_dtype.bits = 16;
            break;
        case DType::Int32:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 32;
            break;
        case DType::Int64:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 64;
            break;
        case DType::Int8:
            dl_dtype.code = kDLInt;
            dl_dtype.bits = 8;
            break;
        case DType::UInt8:
            dl_dtype.code = kDLUInt;
            dl_dtype.bits = 8;
            break;
        default:
            throw InvalidRequestError("Unsupported DType for DLPack conversion");
    }

    return dl_dtype;
}

void dlpack_deleter(DLManagedTensor* self) {
    if (self == nullptr) {
        return;
    }
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete self;
}

size_t calculate_num_elements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t num_elements = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) return 0;
        num_elements *= static_cast<size_t>(dim);
    }
    return num_elements;
}

bool is_contiguous_row_major(const DLTensor* tensor) {
    if (tensor->ndim == 0) return true;
    if (tensor->strides == nullptr) return true;

    int64_t expected_stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; --i) {
        if (tensor->shape[i] > 1 && tensor->strides[i] != expected_stride) {
            return false;
        }
        if (tensor->shape[i] > 0) {
            expected_stride *= tensor->shape[i];
        }
    }
    return true;
}

template<typename T>
void write_tensor_data(std::ofstream& file, const void* data, size_t num_elements) {
    const T* typed_data = static_cast<const T*>(data);
    for (size_t i = 0; i < num_elements; ++i) {
        if (i > 0) file << " ";
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            file << static_cast<int>(typed_data[i]);
        } else {
            file << typed_data[i];
        }
    }
}

float uint32_bits_to_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float float16_bits_to_float(uint16_t half_bits) {
    const uint32_t sign = (static_cast<uint32_t>(half_bits & 0x8000u)) << 16;
    uint32_t exponent = (half_bits >> 10) & 0x1fu;
    uint32_t mantissa = half_bits & 0x03ffu;

    if (exponent == 0) {
        if (mantissa == 0) {
            return uint32_bits_to_float(sign);
        }
        int32_t normalized_exponent = -14;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            --normalized_exponent;
        }
        mantissa &= 0x03ffu;
        return uint32_bits_to_float(
            sign |
            (static_cast<uint32_t>(normalized_exponent + 127) << 23) |
            (mantissa << 13));
    }
    if (exponent == 31) {
        return uint32_bits_to_float(sign | 0x7f800000u | (mantissa << 13));
    }

    exponent = exponent + (127 - 15);
    return uint32_bits_to_float(sign | (exponent << 23) | (mantissa << 13));
}

float bfloat16_bits_to_float(uint16_t bfloat_bits) {
    return uint32_bits_to_float(static_cast<uint32_t>(bfloat_bits) << 16);
}

void write_float16_data(std::ofstream& file, const void* data, size_t num_elements) {
    const uint16_t* typed_data = static_cast<const uint16_t*>(data);
    for (size_t i = 0; i < num_elements; ++i) {
        if (i > 0) file << " ";
        file << float16_bits_to_float(typed_data[i]);
    }
}

void write_bfloat16_data(std::ofstream& file, const void* data, size_t num_elements) {
    const uint16_t* typed_data = static_cast<const uint16_t*>(data);
    for (size_t i = 0; i < num_elements; ++i) {
        if (i > 0) file << " ";
        file << bfloat16_bits_to_float(typed_data[i]);
    }
}

} // namespace

struct Tensor::Impl {
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;
    int32_t device_id_ = 0;
    size_t num_elements_ = 0;
    size_t data_size_bytes_ = 0;
    MemoryOwnership ownership_ = MemoryOwnership::ViewExternal;
    void* device_stream_ = nullptr;

    Impl() = default;
    ~Impl() noexcept { reset(); }

    void free_data() noexcept {
        if (data_ == nullptr || ownership_ == MemoryOwnership::ViewExternal) {
            return;
        }
        detail::free_tensor_data(data_, device_, device_id_, ownership_, device_stream_);
    }

    void reset() noexcept {
        free_data();
        data_ = nullptr;
        ownership_ = MemoryOwnership::ViewExternal;
        device_stream_ = nullptr;
    }

    void allocate(size_t num_elements, DType dtype, Device device, int32_t device_id) {
        free_data();
        data_ = nullptr;

        num_elements_ = num_elements;
        dtype_ = dtype;
        device_ = device;
        device_id_ = device_id;
        data_size_bytes_ = num_elements * get_dtype_size(dtype);

        if (data_size_bytes_ == 0) {
            data_ = nullptr;
            ownership_ = MemoryOwnership::ViewExternal;
            device_stream_ = nullptr;
            return;
        }

        data_ = detail::allocate_tensor_data(
            data_size_bytes_,
            device_,
            device_id_,
            ownership_,
            device_stream_);
    }

    void copy_from(const void* src, size_t num_elements, DType dtype, Device src_device, int32_t src_device_id) {
        if (num_elements == 0 || data_size_bytes_ == 0) {
            return;
        }

        const size_t src_size_bytes = num_elements * get_dtype_size(dtype);
        detail::copy_tensor_data(
            data_,
            device_,
            device_id_,
            ownership_,
            device_stream_,
            src,
            src_device,
            src_device_id,
            src_size_bytes);
    }
};

Tensor::Tensor() noexcept : impl_(std::make_unique<Impl>()) {}

Tensor::~Tensor() noexcept = default;

Tensor Tensor::view(void* data,
                    const std::vector<int64_t>& shape,
                    DType dtype,
                    Device device,
                    int32_t device_id) {
    if (data == nullptr || calculate_num_elements(shape) == 0) {
        return Tensor();
    }

    Tensor t;
    const size_t num_elements = calculate_num_elements(shape);
    t.impl_->shape_ = shape;
    t.impl_->dtype_ = dtype;
    t.impl_->device_ = device;
    t.impl_->device_id_ = device_id;
    t.impl_->num_elements_ = num_elements;
    t.impl_->data_size_bytes_ = num_elements * get_dtype_size(dtype);
    t.impl_->data_ = data;
    t.impl_->ownership_ = MemoryOwnership::ViewExternal;
    t.impl_->device_stream_ = nullptr;
    return t;
}

Tensor Tensor::adopt(void* data,
                     const std::vector<int64_t>& shape,
                     DType dtype,
                     Device device,
                     int32_t device_id,
                     MemoryOwnership ownership,
                     void* stream_handle) {
    Tensor t;
    const size_t num_elements = calculate_num_elements(shape);
    if (data == nullptr || num_elements == 0) {
        return t;
    }

    detail::check_tensor_ownership_supported(ownership);
    if (ownership == MemoryOwnership::OwnCudaPool) {
        if (device != Device::GPU) {
            throw DeviceError("Tensor::adopt with OwnCudaPool requires device=GPU");
        }
        if (stream_handle == nullptr) {
            throw InvalidRequestError("Tensor::adopt with OwnCudaPool requires non-null stream_handle");
        }
    } else if (ownership == MemoryOwnership::OwnCudaMalloc) {
        if (device != Device::GPU) {
            throw DeviceError("Tensor::adopt with OwnCudaMalloc requires device=GPU");
        }
    } else if (ownership == MemoryOwnership::OwnCpuMalloc) {
        if (device != Device::CPU) {
            throw DeviceError("Tensor::adopt with OwnCpuMalloc requires device=CPU");
        }
    }

    t.impl_->shape_ = shape;
    t.impl_->dtype_ = dtype;
    t.impl_->device_ = device;
    t.impl_->device_id_ = device_id;
    t.impl_->num_elements_ = num_elements;
    t.impl_->data_size_bytes_ = num_elements * get_dtype_size(dtype);
    t.impl_->data_ = data;
    t.impl_->ownership_ = ownership;
    t.impl_->device_stream_ = stream_handle;
    return t;
}

Tensor Tensor::clone_from(const void* src,
                          const std::vector<int64_t>& shape,
                          DType dtype,
                          Device src_device,
                          int32_t src_device_id,
                          Device dst_device,
                          int32_t dst_device_id,
                          MemoryOwnership ownership,
                          void* stream_handle) {
    if (src == nullptr) {
        throw InvalidRequestError("Tensor::clone_from requires non-null src");
    }

    detail::check_tensor_clone_supported(src_device, dst_device, ownership);
    if (ownership == MemoryOwnership::OwnCudaPool) {
        if (dst_device != Device::GPU) {
            throw DeviceError("Tensor::clone_from with OwnCudaPool requires dst_device=GPU");
        }
        if (stream_handle == nullptr) {
            throw InvalidRequestError("Tensor::clone_from with OwnCudaPool requires non-null dst_stream_handle");
        }
    } else if (ownership == MemoryOwnership::OwnCudaMalloc) {
        if (dst_device != Device::GPU) {
            throw DeviceError("Tensor::clone_from with OwnCudaMalloc requires dst_device=GPU");
        }
    } else if (ownership == MemoryOwnership::OwnCpuMalloc) {
        if (dst_device != Device::CPU) {
            throw DeviceError("Tensor::clone_from with OwnCpuMalloc requires dst_device=CPU");
        }
    }

    Tensor dst;
    const size_t num_elements = calculate_num_elements(shape);
    dst.impl_->shape_ = shape;
    dst.impl_->dtype_ = dtype;
    dst.impl_->device_ = dst_device;
    dst.impl_->device_id_ = dst_device_id;
    dst.impl_->num_elements_ = num_elements;
    dst.impl_->data_size_bytes_ = num_elements * get_dtype_size(dtype);
    dst.impl_->ownership_ = ownership;
    dst.impl_->device_stream_ = stream_handle;
    if (num_elements == 0) {
        dst.impl_->ownership_ = MemoryOwnership::ViewExternal;
        dst.impl_->device_stream_ = nullptr;
        return dst;
    }
    dst.impl_->allocate(num_elements, dtype, dst_device, dst_device_id);
    dst.impl_->copy_from(src, num_elements, dtype, src_device, src_device_id);
    return dst;
}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {
    other.impl_ = std::make_unique<Impl>();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        other.impl_ = std::make_unique<Impl>();
    }
    return *this;
}

Tensor Tensor::from_dlpack(DLManagedTensor* managed_tensor) {
    if (managed_tensor == nullptr || managed_tensor->dl_tensor.data == nullptr) {
        throw InvalidRequestError("Invalid DLManagedTensor: null pointer");
    }

    const DLTensor* dl_tensor = &managed_tensor->dl_tensor;
    if (dl_tensor->ndim > 0 && !is_contiguous_row_major(dl_tensor)) {
        throw InvalidRequestError("DLPack tensor must be contiguous and row-major");
    }

    Device device = Device::CPU;
    int32_t device_id = 0;
    detail::tensor_device_from_dlpack(dl_tensor->device, &device, &device_id);

    std::vector<int64_t> shape;
    if (dl_tensor->ndim > 0) {
        shape.assign(dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim);
    }

    const DType dtype = dlpack_to_dtype(dl_tensor->dtype);
    const size_t num_elements = calculate_num_elements(shape);

    Tensor tensor;
    tensor.impl_->shape_ = shape;
    tensor.impl_->ownership_ = device == Device::GPU
        ? MemoryOwnership::OwnCudaMalloc
        : MemoryOwnership::OwnCpuMalloc;
    tensor.impl_->device_stream_ = nullptr;
    tensor.impl_->allocate(num_elements, dtype, device, device_id);

    if (num_elements > 0) {
        void* src_data = static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset;
        tensor.impl_->copy_from(src_data, num_elements, dtype, device, device_id);
    }

    return tensor;
}

DLManagedTensor* Tensor::to_dlpack() const {
    if (empty()) {
        throw InvalidRequestError("Cannot convert empty tensor to DLPack");
    }

    const DType dtype = impl_->dtype_;
    const std::vector<int64_t>& shape = impl_->shape_;
    const Device device = impl_->device_;
    const int32_t device_id = impl_->device_id_;
    void* data_ptr = impl_->data_;

    DLManagedTensor* managed_tensor = new (std::nothrow) DLManagedTensor();
    if (managed_tensor == nullptr) {
        throw OutOfMemoryError("Failed to allocate DLManagedTensor");
    }

    managed_tensor->dl_tensor.data = data_ptr;
    managed_tensor->dl_tensor.device = detail::tensor_device_to_dlpack(device, device_id);
    managed_tensor->dl_tensor.ndim = static_cast<int32_t>(shape.size());
    managed_tensor->dl_tensor.dtype = dtype_to_dlpack(dtype);
    managed_tensor->dl_tensor.byte_offset = 0;

    if (shape.empty()) {
        managed_tensor->dl_tensor.shape = nullptr;
    } else {
        managed_tensor->dl_tensor.shape = new (std::nothrow) int64_t[shape.size()];
        if (managed_tensor->dl_tensor.shape == nullptr) {
            delete managed_tensor;
            throw OutOfMemoryError("Failed to allocate shape array for DLManagedTensor");
        }
        std::copy(shape.begin(), shape.end(), managed_tensor->dl_tensor.shape);
    }

    if (shape.empty()) {
        managed_tensor->dl_tensor.strides = nullptr;
    } else {
        managed_tensor->dl_tensor.strides = new (std::nothrow) int64_t[shape.size()];
        if (managed_tensor->dl_tensor.strides == nullptr) {
            delete[] managed_tensor->dl_tensor.shape;
            delete managed_tensor;
            throw OutOfMemoryError("Failed to allocate strides array for DLManagedTensor");
        }
        int64_t stride = 1;
        for (int32_t i = static_cast<int32_t>(shape.size()) - 1; i >= 0; --i) {
            managed_tensor->dl_tensor.strides[i] = stride;
            stride *= shape[static_cast<size_t>(i)];
        }
    }

    managed_tensor->manager_ctx = nullptr;
    managed_tensor->deleter = dlpack_deleter;
    return managed_tensor;
}

bool Tensor::empty() const noexcept {
    return impl_->num_elements_ == 0 || impl_->data_ == nullptr;
}

DType Tensor::dtype() const noexcept {
    return impl_->dtype_;
}

const std::vector<int64_t>& Tensor::shape() const noexcept {
    return impl_->shape_;
}

std::tuple<Device, int32_t> Tensor::device() const noexcept {
    return std::make_tuple(impl_->device_, impl_->device_id_);
}

void* Tensor::data_ptr() const noexcept {
    return impl_->data_;
}

bool Tensor::operator==(const Tensor& other) const noexcept {
    return impl_->data_ == other.impl_->data_ && impl_->data_ != nullptr;
}

void Tensor::dump(const std::string& file_path) const {
    if (empty()) {
        throw InvalidRequestError("Cannot dump empty tensor");
    }

    const void* data_to_write = nullptr;
    std::unique_ptr<char[]> cpu_buffer;
    if (impl_->device_ == Device::GPU) {
        cpu_buffer = std::make_unique<char[]>(impl_->data_size_bytes_);
        detail::copy_tensor_to_cpu_for_dump(
            cpu_buffer.get(),
            impl_->data_,
            impl_->data_size_bytes_,
            impl_->device_id_);
        data_to_write = cpu_buffer.get();
    } else {
        data_to_write = impl_->data_;
    }

    std::ofstream file(file_path, std::ios::out);
    if (!file.is_open()) {
        throw InternalError("Failed to open file for writing: " + file_path);
    }

    file << "# Tensor shape: [";
    for (size_t i = 0; i < impl_->shape_.size(); ++i) {
        if (i > 0) file << ", ";
        file << impl_->shape_[i];
    }
    file << "]\n";
    file << "# Data type: ";
    switch (impl_->dtype_) {
        case DType::Float32: file << "Float32"; break;
        case DType::Float16: file << "Float16"; break;
        case DType::BFloat16: file << "BFloat16"; break;
        case DType::Int32: file << "Int32"; break;
        case DType::Int64: file << "Int64"; break;
        case DType::UInt8: file << "UInt8"; break;
        case DType::Int8: file << "Int8"; break;
    }
    file << "\n";
    file << "# Number of elements: " << impl_->num_elements_ << "\n";
    file << "# Data (row-major layout):\n";

    const size_t num_elements = impl_->num_elements_;
    switch (impl_->dtype_) {
        case DType::Float32: write_tensor_data<float>(file, data_to_write, num_elements); break;
        case DType::Float16: write_float16_data(file, data_to_write, num_elements); break;
        case DType::BFloat16: write_bfloat16_data(file, data_to_write, num_elements); break;
        case DType::Int32: write_tensor_data<int32_t>(file, data_to_write, num_elements); break;
        case DType::Int64: write_tensor_data<int64_t>(file, data_to_write, num_elements); break;
        case DType::UInt8: write_tensor_data<uint8_t>(file, data_to_write, num_elements); break;
        case DType::Int8: write_tensor_data<int8_t>(file, data_to_write, num_elements); break;
        default:
            throw InternalError("Unsupported dtype for text dump");
    }
    file << "\n";

    if (!file.good()) {
        throw InternalError("Failed to write tensor data to file: " + file_path);
    }
}

} // namespace edge_fm
