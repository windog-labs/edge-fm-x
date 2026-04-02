#include <edge-fm/core.h>
#include <edge-fm/edge-fm.h>
#include <dlpack/dlpack.h>
#include "engine/engine.h"
#include "engine/horizon_engine.h"
#include "engine/stardard_engine.h"
#include "utils/device/weight_loader.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

namespace edge_fm {

namespace {
    // Convert DLPack DLDataType to DType
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

    // Convert DType to DLDataType
    DLDataType dtype_to_dlpack(DType dtype) {
        DLDataType dl_dtype;
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

    // Convert Device to DLDevice
    DLDevice device_to_dlpack(Device device, int32_t device_id) {
        DLDevice dl_device;
        
        switch (device) {
            case Device::CPU:
                dl_device.device_type = kDLCPU;
                dl_device.device_id = 0;
                break;
            case Device::GPU:
                dl_device.device_type = kDLCUDA;
                dl_device.device_id = device_id;
                break;
            default:
                throw InvalidRequestError("Unsupported Device for DLPack conversion");
        }
        
        return dl_device;
    }

    // DLPack deleter function: 释放 DLManagedTensor 及其相关内存（但不释放数据本身）
    void dlpack_deleter(DLManagedTensor* self) {
        if (self != nullptr) {
            // 释放 shape 数组
            if (self->dl_tensor.shape != nullptr) {
                delete[] self->dl_tensor.shape;
            }
            
            // 释放 strides 数组
            if (self->dl_tensor.strides != nullptr) {
                delete[] self->dl_tensor.strides;
            }
            
            // 释放 DLManagedTensor 结构本身
            delete self;
        }
    }

    // Calculate total number of elements
    size_t calculate_num_elements(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        size_t num_elements = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) return 0;
            num_elements *= static_cast<size_t>(dim);
        }
        return num_elements;
    }

    // Check if tensor is contiguous and row-major (required for DLPack)
    bool is_contiguous_row_major(const DLTensor* tensor) {
        if (tensor->ndim == 0) return true;
        
        // According to DLPack specification: if strides is nullptr, it means the tensor
        // is in compact row-major (C-style) contiguous layout. This is commonly used by
        // NumPy when exporting DLPack tensors, while PyTorch typically provides explicit
        // strides even for contiguous tensors.
        if (tensor->strides == nullptr) {
            return true;  // nullptr strides means contiguous row-major by DLPack spec
        }
        
        // Check if strides indicate row-major (C-style) contiguous layout.
        // For dimensions with shape[i]==1 (degenerate), stride is irrelevant since we only
        // have index 0; PyTorch may export stride 1 instead of the full product.
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

    // Helper template function to write tensor data to file (all in one line)
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

    // Helper function to convert Float16 to float and write (all in one line)
    void write_float16_data(std::ofstream& file, const void* data, size_t num_elements) {
        const __half* typed_data = static_cast<const __half*>(data);
        for (size_t i = 0; i < num_elements; ++i) {
            if (i > 0) file << " ";
            float val = __half2float(typed_data[i]);
            file << val;
        }
    }

    // Helper function to convert BFloat16 to float and write (all in one line)
    void write_bfloat16_data(std::ofstream& file, const void* data, size_t num_elements) {
        const __nv_bfloat16* typed_data = static_cast<const __nv_bfloat16*>(data);
        for (size_t i = 0; i < num_elements; ++i) {
            if (i > 0) file << " ";
            float val = __bfloat162float(typed_data[i]);
            file << val;
        }
    }
}

// Tensor::Impl structure
struct Tensor::Impl {
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;
    int32_t device_id_ = 0;
    size_t num_elements_ = 0;
    size_t data_size_bytes_ = 0;
    MemoryOwnership ownership_ = MemoryOwnership::ViewExternal;
    void* device_stream_ = nullptr;  // e.g., cudaStream_t for CUDA

    Impl() = default;
    ~Impl() noexcept { reset(); }

    // 仅释放内存，不重置 ownership_ 和 device_stream_
    void free_data() noexcept {
        if (data_ != nullptr && ownership_ != MemoryOwnership::ViewExternal) {
            if (ownership_ == MemoryOwnership::OwnCudaPool) {
                cudaSetDevice(device_id_);
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(device_stream_);
                cudaFreeAsync(data_, stream ? stream : 0);
            } else if (ownership_ == MemoryOwnership::OwnCudaMalloc) {
                cudaSetDevice(device_id_);
                cudaFree(data_);
            } else if (ownership_ == MemoryOwnership::OwnCpuMalloc) {
                std::free(data_);
            }
        }
    }

    // 释放内存并重置所有状态到初始值
    void reset() noexcept {
        free_data();
        data_ = nullptr;
        ownership_ = MemoryOwnership::ViewExternal;
        device_stream_ = nullptr;
    }

    void allocate(size_t num_elements, DType dtype, Device device, int32_t device_id) {
        free_data();
        
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

        if (device == Device::GPU) {
            cudaSetDevice(device_id);
            if (ownership_ == MemoryOwnership::OwnCudaPool) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(device_stream_);
                data_ = MemoryPool::instance().allocate(data_size_bytes_, stream, device_id);
            } else if (ownership_ == MemoryOwnership::OwnCudaMalloc) {
                CUDA_CHECK_THROW_EX(cudaMalloc(&data_, data_size_bytes_), 
                                    "Failed to allocate GPU memory", OutOfMemoryError);
            }
        } else {  // CPU
            data_ = std::malloc(data_size_bytes_);
            if (data_ == nullptr) {
                throw OutOfMemoryError("Failed to allocate CPU memory");
            }
        }
    }

    void copy_from(const void* src, size_t num_elements, DType dtype, Device src_device, int32_t src_device_id) {
        if (num_elements == 0 || data_size_bytes_ == 0) return;

        size_t src_size_bytes = num_elements * get_dtype_size(dtype);
        
        // Prefer async copy when destination is GPU and memory is from pool (stream-ordered)
        bool cuda_async = (device_ == Device::GPU) && (ownership_ == MemoryOwnership::OwnCudaPool);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(device_stream_);

        if (device_ == Device::GPU && src_device == Device::GPU) {  // GPU -> GPU
            cudaSetDevice(device_id_);
            if (device_id_ != src_device_id) {
                if (cuda_async) {
                    CUDA_CHECK_THROW(cudaMemcpyPeerAsync(data_, device_id_, src, src_device_id, src_size_bytes, stream), "Failed to copy GPU to GPU (async)");
                } else {
                    CUDA_CHECK_THROW(cudaMemcpyPeer(data_, device_id_, src, src_device_id, src_size_bytes), "Failed to copy GPU to GPU");
                }
            } else {
                if (cuda_async) {
                    CUDA_CHECK_THROW(cudaMemcpyAsync(data_, src, src_size_bytes, cudaMemcpyDeviceToDevice, stream), "Failed to copy GPU to GPU (async)");
                } else {
                    CUDA_CHECK_THROW(cudaMemcpy(data_, src, src_size_bytes, cudaMemcpyDeviceToDevice), "Failed to copy GPU to GPU");
                }
            }
        } else if (device_ == Device::GPU && src_device == Device::CPU) {  // CPU -> GPU
            cudaSetDevice(device_id_);
            if (cuda_async) {
                CUDA_CHECK_THROW(cudaMemcpyAsync(data_, src, src_size_bytes, cudaMemcpyHostToDevice, stream), "Failed to copy CPU to GPU (async)");
            } else {
                CUDA_CHECK_THROW(cudaMemcpy(data_, src, src_size_bytes, cudaMemcpyHostToDevice), "Failed to copy CPU to GPU");
            }
        } else if (device_ == Device::CPU && src_device == Device::GPU) {  // GPU -> CPU
            cudaSetDevice(src_device_id);
            CUDA_CHECK_THROW(cudaMemcpy(data_, src, src_size_bytes, cudaMemcpyDeviceToHost), "Failed to copy GPU to CPU");
        } else {  // CPU -> CPU
            std::memcpy(data_, src, src_size_bytes);
        }
    }
};

// Tensor constructors and destructor
Tensor::Tensor() noexcept : impl_(std::make_unique<Impl>()) {}

Tensor::~Tensor() noexcept = default;

// ============================== Tensor factories ==============================
Tensor Tensor::view(void* data,
                    const std::vector<int64_t>& shape,
                    DType dtype,
                    Device device,
                    int32_t device_id) {
    if (data == nullptr || calculate_num_elements(shape) == 0) {
        return Tensor();  // empty view
    }
    Tensor t;
    size_t num_elements = calculate_num_elements(shape);
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
    size_t num_elements = calculate_num_elements(shape);
    if (data == nullptr || num_elements == 0) {
        return t;  // empty tensor
    }
    // Validate ownership/device constraints
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
    // Validate ownership/device constraints
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
    // Allocate destination (inline allocation)
    Tensor dst;
    size_t num_elements = calculate_num_elements(shape);
    dst.impl_->shape_ = shape;
    dst.impl_->dtype_ = dtype;
    dst.impl_->device_ = dst_device;
    dst.impl_->device_id_ = dst_device_id;
    dst.impl_->num_elements_ = num_elements;
    dst.impl_->data_size_bytes_ = num_elements * get_dtype_size(dtype);
    dst.impl_->ownership_ = ownership;
    dst.impl_->device_stream_ = stream_handle;
    if (num_elements == 0) {
        // keep empty
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

// Tensor static method
Tensor Tensor::from_dlpack(DLManagedTensor* managed_tensor) {
    if (managed_tensor == nullptr || managed_tensor->dl_tensor.data == nullptr) {
        throw InvalidRequestError("Invalid DLManagedTensor: null pointer");
    }

    const DLTensor* dl_tensor = &managed_tensor->dl_tensor;
    
    // Check if tensor is contiguous and row-major
    if (dl_tensor->ndim > 0 && !is_contiguous_row_major(dl_tensor)) {
        throw InvalidRequestError("DLPack tensor must be contiguous and row-major");
    }

    // Convert DLPack device to our Device enum
    Device device;
    int32_t device_id = 0;
    if (dl_tensor->device.device_type == kDLCPU) {
        device = Device::CPU;
    } else if (dl_tensor->device.device_type == kDLCUDA || 
               dl_tensor->device.device_type == kDLCUDAHost) {
        device = Device::GPU;
        device_id = dl_tensor->device.device_id;
    } else {
        throw InvalidRequestError("Unsupported DLPack device type");
    }

    // Convert shape
    std::vector<int64_t> shape;
    if (dl_tensor->ndim > 0) {
        shape.assign(dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim);
    }

    // Convert dtype
    DType dtype = dlpack_to_dtype(dl_tensor->dtype);

    // Create new tensor and copy data
    Tensor tensor;
    size_t num_elements = calculate_num_elements(shape);
    tensor.impl_->shape_ = shape;
    tensor.impl_->ownership_ = (device == Device::GPU) 
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

// Tensor methods
DLManagedTensor* Tensor::to_dlpack() const {
    // 如果 tensor 为空，抛出异常
    if (empty()) {
        throw InvalidRequestError("Cannot convert empty tensor to DLPack");
    }
    
    // 获取 tensor 信息
    DType dtype = impl_->dtype_;
    const std::vector<int64_t>& shape = impl_->shape_;
    Device device = impl_->device_;
    int32_t device_id = impl_->device_id_;
    void* data_ptr = impl_->data_;
    
    DLManagedTensor* managed_tensor = new (std::nothrow) DLManagedTensor();
    if (managed_tensor == nullptr) {
        throw OutOfMemoryError("Failed to allocate DLManagedTensor");
    }
    
    managed_tensor->dl_tensor.data = data_ptr;
    managed_tensor->dl_tensor.device = device_to_dlpack(device, device_id);
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
            if (managed_tensor->dl_tensor.shape != nullptr) {
                delete[] managed_tensor->dl_tensor.shape;
            }
            delete managed_tensor;
            throw OutOfMemoryError("Failed to allocate strides array for DLManagedTensor");
        }
        int64_t stride = 1;
        for (int32_t i = static_cast<int32_t>(shape.size()) - 1; i >= 0; --i) {
            managed_tensor->dl_tensor.strides[i] = stride;
            stride *= shape[i];
        }
    }
    
    managed_tensor->manager_ctx = nullptr;
    managed_tensor->deleter = dlpack_deleter;
    
    return managed_tensor;
}

bool                        Tensor::empty() const noexcept { return impl_->num_elements_ == 0 || impl_->data_ == nullptr; }
DType                       Tensor::dtype() const noexcept { return impl_->dtype_; }
const std::vector<int64_t>& Tensor::shape() const noexcept { return impl_->shape_; }
std::tuple<Device, int32_t> Tensor::device() const noexcept { return std::make_tuple(impl_->device_, impl_->device_id_); }
void*                       Tensor::data_ptr() const noexcept { return impl_->data_; }

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
        cudaSetDevice(impl_->device_id_);
        CUDA_CHECK_THROW(cudaMemcpy(cpu_buffer.get(), impl_->data_, impl_->data_size_bytes_, cudaMemcpyDeviceToHost), "Failed to copy GPU tensor to CPU for dumping");
        data_to_write = cpu_buffer.get();
    } else {
        data_to_write = impl_->data_;
    }

    // Write data to file as text
    std::ofstream file(file_path, std::ios::out);
    if (!file.is_open()) {
        throw InternalError("Failed to open file for writing: " + file_path);
    }

    // Write tensor metadata (optional, for readability)
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

    // Write tensor data based on dtype (all in one line, no formatting)
    size_t num_elements = impl_->num_elements_;
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

    file.close();
}

// ============================================================================
// EdgeFM implementation
// ============================================================================

struct EdgeFM::Impl {
    std::unique_ptr<Engine> engine;
};

namespace {
    /// 收集模型目录下待加载的 safetensors 文件：若存在 model.safetensors 则只返回该文件；否则返回所有 model-*-of-*.safetensors 并排序。
    std::vector<std::string> collect_safetensors_files(const std::string& model_dir) {
        std::filesystem::path dir(model_dir);
        std::vector<std::string> out;
        std::string single = model_dir + "/model.safetensors";
        if (std::filesystem::exists(single) && std::filesystem::is_regular_file(single)) {
            out.push_back(single);
            return out;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string name = entry.path().filename().string();
            if (name.size() > 18 && name.compare(0, 6, "model-") == 0 &&
                name.find("-of-") != std::string::npos &&
                name.size() >= 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0) {
                out.push_back(entry.path().string());
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
}  // namespace

EdgeFM::EdgeFM(const std::string& config_path) : impl_(std::make_unique<Impl>()) {
    EngineConfig config(config_path);
    bool speculative_enabled = config.speculative().value("enabled", false);
    if (speculative_enabled) {
        throw std::runtime_error("Speculative decoding (EagleEngine) not yet supported in EdgeFM facade");
    }
    const std::string backend_target = config.backend_target();
    if (backend_target == "horizon") {
        impl_->engine = std::make_unique<HorizonEngine>(config);
        impl_->engine->warmup();
        return;
    }
    if (backend_target != "cuda") {
        throw ConfigurationError("Unsupported backend_target: " + backend_target);
    }

    WeightLoader& loader = WeightLoader::instance();
    // Clear cache before loading: FusedQKVLinearLayer modifies cache in-place (erases q/k/v_proj),
    // so a previously created engine leaves the cache in an invalid state for a new engine.
    loader.clear_stage(ModelStage::Prefill);
    loader.clear_stage(ModelStage::Decode);
    std::string prefill_path = config.prefill_model_path();
    int32_t device_id = config.runtime_device_id();

    // 模型族由 engine.json 中显式的 model_name 决定；装权重时不再从模型文件推断 text / vl。
    const bool is_vlm = (config.resolved_model_name() == "qwen2_5_vl");

    std::vector<std::string> prefill_files = collect_safetensors_files(prefill_path);
    if (prefill_files.empty()) {
        throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in: " + prefill_path);
    }

    if (is_vlm) {
        auto vlm_filter = [](const std::string& name) {
            return name.rfind("model.", 0) == 0;
        };
        auto vlm_key_mapper = [](const std::string& name) {
            if (name.rfind("model.model.", 0) == 0) {
                return name.substr(6);  // "model.model.xxx" -> "model.xxx"
            }
            return name;
        };
        for (const auto& f : prefill_files) {
            loader.load_weights_from_file(ModelStage::Prefill, f, Device::GPU, device_id, true, vlm_filter, vlm_key_mapper);
        }
        std::string decode_path = config.decode_model_path();
        std::vector<std::string> decode_files = (decode_path.empty() || decode_path == prefill_path)
            ? prefill_files
            : collect_safetensors_files(decode_path);
        if (decode_files.empty()) {
            throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in decode path");
        }
        for (const auto& f : decode_files) {
            loader.load_weights_from_file(ModelStage::Decode, f, Device::GPU, device_id, true, vlm_filter, vlm_key_mapper);
        }
    } else {
        for (const auto& f : prefill_files) {
            loader.load_weights_from_file(ModelStage::Prefill, f, Device::GPU, device_id, true);
        }
        std::string decode_path = config.decode_model_path();
        std::vector<std::string> decode_files = (decode_path.empty() || decode_path == prefill_path)
            ? prefill_files
            : collect_safetensors_files(decode_path);
        if (decode_files.empty()) {
            throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in decode path");
        }
        for (const auto& f : decode_files) {
            loader.load_weights_from_file(ModelStage::Decode, f, Device::GPU, device_id, true);
        }
    }
    impl_->engine = std::make_unique<StandardEngine>(config);
    impl_->engine->warmup();
}

EdgeFM::~EdgeFM() noexcept = default;

Response EdgeFM::generate(const Request& request) const {
    return impl_->engine->generate(request);
}

void EdgeFM::tune() {
    impl_->engine->tune();
}

} // namespace edge_fm
