#pragma once
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include <dlpack/dlpack.h>

namespace edge_fm {

enum class Device {
    CPU,
    GPU
};

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt8,
    Int8
};

/**
 * @brief 表示 Tensor 对底层数据缓冲区的所有权/释放策略
 *
 * 注意：MemoryOwnership 仅描述 **Tensor 是否负责释放以及如何释放**，
 * 不描述 "是否分配/是否拷贝"。分配/拷贝语义由具体工厂方法决定：
 * - Tensor::view  : 不分配、不拷贝、不释放（外部负责生命周期）
 * - Tensor::adopt : 不分配、不拷贝，但 Tensor 析构负责释放（接管已有 buffer）
 * - Tensor::clone_from : 分配并拷贝（Tensor 析构负责释放）
 *
 * 取值含义：
 * - ViewExternal: 非拥有视图。Tensor 仅持有指针，不负责释放。
 * - OwnCpuMalloc: Tensor 拥有该 buffer，析构时用 std::free 释放（CPU-only）。
 * - OwnCudaMalloc: Tensor 拥有该 buffer，析构时用 cudaFree 释放（GPU-only）。
 * - OwnCudaPool: Tensor 拥有该 buffer，析构时用 cudaFreeAsync(stream) 归还到 CUDA mempool（GPU-only）。
 *               为保证 stream-ordered 语义，通常要求提供非空 stream_handle。
 */
enum class MemoryOwnership {
    ViewExternal,   // 非拥有：不释放（外部管理）
    OwnCpuMalloc,   // 拥有：析构时 std::free（CPU-only）
    OwnCudaMalloc,  // 拥有：析构时 cudaFree（GPU-only）
    OwnCudaPool,    // 拥有：析构时 cudaFreeAsync(stream)（GPU-only）
};

class Tensor {
public:
    // Constructor and Destructor
    Tensor() noexcept;
    ~Tensor() noexcept;

    /**
     * @brief Create a non-owning view of an external buffer (no allocation, no copy, no free).
     * 
     * This function only wraps an existing contiguous buffer. The Tensor will NOT
     * allocate, copy, or free the memory. The caller must guarantee that:
     * - The lifetime of the buffer outlives the returned Tensor.
     * - The buffer is contiguous and row-major (C-style) for the given shape.
     * - The buffer is actually located on the specified device/device_id.
     * 
     * Typical use cases:
     * - Wrap static or framework-owned buffers (e.g., KV cache slices, pinned host memory).
     * - Zero-copy integration where ownership must remain external.
     */
    static Tensor view(void* data,
                       const std::vector<int64_t>& shape,
                       DType dtype,
                       Device device,
                       int32_t device_id = 0);

    /**
     * @brief Adopt an existing buffer and take ownership (no copy).
     * 
     * Tensor will assume responsibility to release the buffer in its destructor,
     * according to the specified ownership strategy:
     * - OwnCudaMalloc: freed via cudaFree() for GPU or std::free() for CPU.
     * - OwnCudaPool: freed via cudaFreeAsync(stream) (stream-ordered).
     * 
     * Requirements:
     * - The pointer must come from a compatible allocator for the chosen ownership
     *   (e.g., cudaMalloc for OwnCudaMalloc, cudaMallocAsync/mempool for OwnCudaPool).
     * - If ownership is OwnCudaPool, device must be GPU and stream_handle must be non-null.
     * - The buffer must be contiguous and row-major for the given shape.
     * - After adoption, the caller must NOT free the buffer.
     */
    static Tensor adopt(void* data,
                        const std::vector<int64_t>& shape,
                        DType dtype,
                        Device device,
                        int32_t device_id = 0,
                        MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
                        void* stream_handle = nullptr);

    /**
     * @brief Clone from an existing contiguous buffer into a newly allocated buffer (allocate + copy).
     * 
     * The destination buffer is allocated according to 'ownership' on the given dst device/device_id,
     * then contents are copied from 'src' located on src device/device_id.
     * 
     * Assumptions & guidance:
     * - 'src' must point to a contiguous row-major buffer matching 'shape' and 'dtype'.
     * - Cross-device copies are supported by specifying src_* and dst_* explicitly.
     * - If ownership is OwnCudaPool, dst_device must be GPU and dst_stream_handle must be non-null
     *   (copy is performed with cudaMemcpyAsync/cudaMemcpyPeerAsync on the provided stream).
     */
    static Tensor clone_from(const void* src,
                             const std::vector<int64_t>& shape,
                             DType dtype,
                             Device src_device,
                             int32_t src_device_id,
                             Device dst_device,
                             int32_t dst_device_id = 0,
                             MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
                             void* stream_handle = nullptr);

    // disable copy and assignment
    Tensor(const Tensor& other) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other) = delete;
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief Create tensor from DLPack managed tensor
     * 
     * @param managed_tensor Pointer to DLManagedTensor (DLPack standard structure)
     * 
     * @note The data from managed_tensor will be copied to a new Tensor object.
     *       The managed_tensor must be valid and follow DLPack specification.
     *       After this call, the managed_tensor can be safely released.
     * 
     * @note **重要约束**：由于 Tensor 不支持 strides 参数，输入的 DLPack tensor 必须满足以下要求：
     *       - 内存必须是连续的（contiguous）
     *       - 数据布局必须是 row-major（C-style，即最后一个维度连续）
     *       如果输入 tensor 不满足这些条件，将抛出 InvalidRequestError 异常。
     * 
     * @return New Tensor object created from DLPack tensor (data is copied)
     * 
     * @throws InvalidRequestError if the managed_tensor is invalid, unsupported, or not contiguous/row-major
     * @throws DeviceError if device operations fail
     * @throws OutOfMemoryError if memory allocation fails
     */
    static Tensor from_dlpack(DLManagedTensor* managed_tensor);

    /**
     * @brief Convert tensor to DLPack managed tensor
     * 
     * @return Pointer to DLManagedTensor (DLPack standard structure)
     * 
     * @note The returned DLManagedTensor is allocated and must be freed by the caller
     *       using the deleter function in the DLManagedTensor structure.
     * 
     * @throws DeviceError if device operations fail
     * @throws OutOfMemoryError if memory allocation fails
     */
    DLManagedTensor* to_dlpack() const;

    // Get tensor information
    bool empty() const noexcept;
    DType dtype() const noexcept;
    const std::vector<int64_t>& shape() const noexcept;
    std::tuple<Device, int32_t> device() const noexcept;
    
    /**
     * @brief Get the raw data pointer
     * 
     * @return Raw pointer to the tensor data
     * 
     * @note This method is intended for internal use by layers and kernels.
     *       The returned pointer is valid as long as the Tensor object exists.
     */
    void* data_ptr() const noexcept;

    /**
     * @brief Check if two tensors refer to the same underlying data buffer
     * 
     * @param other The other tensor to compare with
     * @return true if both tensors point to the same data buffer, false otherwise
     * 
     * @note This operator compares the data pointers, not the tensor values.
     *       Two tensors are considered equal if they point to the same memory location.
     */
    bool operator==(const Tensor& other) const noexcept;

    /**
     * @brief Dump tensor data to a text file
     * 
     * @param file_path Path to the output file
     * 
     * @note The file format is text (human-readable). For GPU tensors, data will be copied to CPU first.
     *       The file contains:
     *       - Header lines with tensor metadata (shape, dtype, number of elements)
     *       - Data values separated by spaces in row-major (C-style) layout
     *       - Floating point numbers are output in scientific notation with 9-digit precision
     *       - Integer types are output as decimal numbers
     *       - Float16 and BFloat16 are converted to float representation
     * 
     * @throws InvalidRequestError if tensor is empty
     * @throws DeviceError if device operations fail (e.g., GPU to CPU copy fails)
     * @throws InternalError if file operations fail
     */
    void dump(const std::string& file_path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ***************************** exception classes *****************************
/**
 * @brief Base exception class for all EdgeFM errors
 */
class Error : public std::exception {
public:
    explicit Error(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
protected:
    std::string message_;
};

/**
 * @brief Exception thrown when the request is invalid
 * 
 * This includes cases such as:
 * - Empty token_ids
 * - Request sequence length exceeds the maximum allowed
 * - Invalid request format
 * - Embedding and embedding_indices mismatch
 */
struct InvalidRequestError : public Error { using Error::Error; };

struct Request {
    Request(int32_t request_id, const std::vector<int32_t>& token_ids) noexcept
        : request_id_(request_id), token_ids_(token_ids), embed_token_id_(-1) {}
    
    Request(int32_t request_id, 
            const std::vector<int32_t>& token_ids, 
            const Tensor& embedding,
            int32_t embed_token_id=-1,
            Device device=Device::GPU,
            int32_t device_id = 0,
            MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
            void* stream_handle = nullptr
        ) : 
            request_id_(request_id), 
            token_ids_(token_ids), 
            embed_token_id_(embed_token_id)
    {
        embedding_ = embedding.clone_from(embedding.data_ptr(), 
                                          embedding.shape(), 
                                          embedding.dtype(), 
                                          std::get<0>(embedding.device()), 
                                          std::get<1>(embedding.device()),
                                          device,
                                          device_id,
                                          ownership,
                                          stream_handle);
    }

    Request(int32_t request_id,
            const std::vector<int32_t>& token_ids,
            const Tensor& embedding,
            int32_t embed_token_id,
            const Tensor& position_ids,
            const std::vector<int32_t>& mrope_last_pos,
            Device device=Device::GPU,
            int32_t device_id = 0,
            MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
            void* stream_handle = nullptr
        ) :
            request_id_(request_id),
            token_ids_(token_ids),
            embed_token_id_(embed_token_id)
    {
        embedding_ = embedding.clone_from(embedding.data_ptr(),
                                          embedding.shape(),
                                          embedding.dtype(),
                                          std::get<0>(embedding.device()),
                                          std::get<1>(embedding.device()),
                                          device, device_id,
                                          ownership, stream_handle);
        position_ids_ = position_ids.clone_from(position_ids.data_ptr(),
                                                position_ids.shape(),
                                                position_ids.dtype(),
                                                std::get<0>(position_ids.device()),
                                                std::get<1>(position_ids.device()),
                                                device, device_id,
                                                ownership, stream_handle);
        mrope_last_pos_ = mrope_last_pos;
    }

    Request(int32_t request_id,
            const std::vector<int32_t>& token_ids,
            const Tensor& embedding,
            int32_t embed_token_id,
            const Tensor& position_ids,
            Device device=Device::GPU,
            int32_t device_id = 0,
            MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
            void* stream_handle = nullptr
        ) :
            Request(
                request_id,
                token_ids,
                embedding,
                embed_token_id,
                position_ids,
                std::vector<int32_t>{},
                device,
                device_id,
                ownership,
                stream_handle) {}
    
    Request(Request&& other) noexcept = default;
    Request& operator=(Request&& other) noexcept = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    
    int32_t request_id() const noexcept { return request_id_; }
    const std::vector<int32_t>& token_ids() const noexcept { return token_ids_; }
    const Tensor& embedding() const noexcept { return embedding_; }
    int32_t embed_token_id() const noexcept { return embed_token_id_; }
    bool has_embedding() const noexcept { return !embedding_.empty(); }
    const Tensor& position_ids() const noexcept { return position_ids_; }
    bool has_position_ids() const noexcept { return !position_ids_.empty(); }
    const std::vector<int32_t>& mrope_last_pos() const noexcept { return mrope_last_pos_; }
    bool has_mrope_last_pos() const noexcept { return !mrope_last_pos_.empty(); }

    void set_stop_token_ids(const std::vector<int32_t>& ids) { stop_token_ids_ = ids; }
    const std::vector<int32_t>& stop_token_ids() const noexcept { return stop_token_ids_; }

    void set_ignore_stop_tokens(bool v) { ignore_stop_tokens_ = v; }
    bool ignore_stop_tokens() const noexcept { return ignore_stop_tokens_; }

private:
    int32_t request_id_;
    std::vector<int32_t> token_ids_;
    int32_t embed_token_id_;
    Tensor embedding_;
    Tensor position_ids_;
    std::vector<int32_t> mrope_last_pos_;
    std::vector<int32_t> stop_token_ids_;
    bool ignore_stop_tokens_ = false;
};

class Response {
public:
    // Constructor and Destructor
    Response() noexcept = default;
    ~Response() noexcept = default;
    
    // Move constructor and assignment operator
    Response(Response&& other) noexcept = default;
    Response& operator=(Response&& other) noexcept = default;
    Response(const Response&) = delete;
    Response& operator=(const Response&) = delete;
    
    // Get token ids
    std::vector<int32_t>& token_ids() noexcept { return token_ids_; }
    const std::vector<int32_t>& token_ids() const noexcept { return token_ids_; }
    
private:
    std::vector<int32_t> token_ids_;
};

// ***************************** exception classes (continued) *****************************
/**
 @brief Exception thrown when device operations fail (e.g., kernel execution, device errors)
 */
struct DeviceError : public Error { using Error::Error; };

/**
 * @brief Exception thrown when argument is invalid 
 */
struct InvalidArgumentError : public Error { using Error::Error; };

/**
 * @brief Exception thrown when configuration is invalid or cannot be loaded
 */
struct ConfigurationError : public Error { using Error::Error; };

/**
 * @brief Exception thrown when model is not loaded or model loading fails
 */
struct ModelNotLoadedError : public Error { using Error::Error; };

/**
 * @brief Exception thrown when there is insufficient memory (GPU or CPU)
 */
struct OutOfMemoryError : public Error { using Error::Error; };

/**
 * @brief Exception thrown when internal system errors occur
 * 
 * This includes cases such as:
 * - Device unavailable or not accessible
 * - Internal kernel execution failures
 * - System-level errors
 * - Unexpected internal state errors
 */
struct InternalError : public Error { using Error::Error; };

/**
 * @brief Convert string to Device enum
 * 
 * @param device_str Device string ("cpu" or "cuda")
 * @return Device enum value
 * @throws ConfigurationError if device_str is not supported
 */
inline Device device_from_string(const std::string& device_str) {
    if (device_str == "cpu") {
        return Device::CPU;
    } else if (device_str == "cuda") {
        return Device::GPU;
    } else {
        throw ConfigurationError("Unsupported device: " + device_str + ". Expected 'cuda' or 'cpu'");
    }
}

/**
 * @brief Convert string to DType enum
 * 
 * @param dtype_str Data type string ("fp32", "fp16", "bf16", "float32", "float16", "bfloat16")
 * @return DType enum value
 * @throws ConfigurationError if dtype_str is not supported
 */
inline DType dtype_from_string(const std::string& dtype_str) {
    if (dtype_str == "fp32" || dtype_str == "float32") {
        return DType::Float32;
    } else if (dtype_str == "fp16" || dtype_str == "float16") {
        return DType::Float16;
    } else if (dtype_str == "bf16" || dtype_str == "bfloat16") {
        return DType::BFloat16;
    } else if (dtype_str == "int32") {
        return DType::Int32;
    } else if (dtype_str == "int64") {
        return DType::Int64;
    } else if (dtype_str == "int8") {
        return DType::Int8;
    } else if (dtype_str == "uint8") {
        return DType::UInt8;
    } else {
        throw ConfigurationError("Unsupported dtype: " + dtype_str + ". Expected 'fp32', 'fp16', or 'bf16'");
    }
}

/**
 * @brief Get the size in bytes for a given DType
 * 
 * @param dtype The data type
 * @return Size in bytes, or 0 for unsupported types
 */
inline size_t get_dtype_size(DType dtype) {
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
            return 0;
    }
}

} // namespace edge_fm
