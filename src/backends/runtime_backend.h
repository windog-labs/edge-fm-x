#pragma once

#include "backends/backend_target.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

enum class RuntimeDType {
    Float32,
    Float16,
    Int32,
    Int8,
    UInt8,
};

enum class RuntimeDevice {
    CPU,
    GPU,
    BPU,
};

using RuntimeStreamHandle = void*;

struct RuntimeInitParams {
    std::string program_path;
    std::string model_name;
    std::string model_path;
    int max_batch_size = 1;
    std::unordered_map<std::string, std::vector<int64_t>> input_shape_overrides;
    std::vector<int64_t> input_shape_hint;
};

struct RuntimeTensorView {
    void* data = nullptr;
    std::vector<int64_t> shape;
    RuntimeDType dtype = RuntimeDType::Float32;
    RuntimeDevice device = RuntimeDevice::CPU;
    // Byte strides. Empty means compact row-major layout.
    std::vector<int64_t> stride;
};

size_t runtime_dtype_size(RuntimeDType dtype);
size_t runtime_tensor_num_elements(const RuntimeTensorView& view);
bool runtime_tensor_is_contiguous(const RuntimeTensorView& view);
size_t runtime_tensor_byte_size(const RuntimeTensorView& view);

void copy_contiguous_to_runtime_buffer(const void* src, const RuntimeTensorView& dst);
void copy_runtime_buffer_to_contiguous(const RuntimeTensorView& src, void* dst);

class IRuntimeBackend {
public:
    virtual ~IRuntimeBackend() = default;

    virtual bool init(const RuntimeInitParams& params) = 0;
    virtual bool warmup(int batch_size = 1) = 0;
    virtual RuntimeStreamHandle default_stream() = 0;

    virtual int forward_sync() = 0;
    virtual int forward_async(RuntimeStreamHandle stream = nullptr) = 0;
    virtual int wait(RuntimeStreamHandle stream = nullptr) = 0;

    virtual std::vector<std::string> input_names() const = 0;
    virtual std::vector<std::string> output_names() const = 0;
    virtual bool get_input_shape(const std::string& name, std::vector<int64_t>* out_shape) const = 0;
    virtual bool get_output_shape(const std::string& name, std::vector<int64_t>* out_shape) const = 0;
    virtual bool get_input_buffer(const std::string& name, RuntimeTensorView* out_tensor) = 0;
    virtual bool get_output_buffer(const std::string& name, RuntimeTensorView* out_tensor) = 0;
    virtual std::string last_error() const = 0;
};

std::unique_ptr<IRuntimeBackend> create_runtime_backend(BackendTarget backend);

} // namespace edge_fm
