#include "backends/horizon_runtime_backend.h"

#include <edge-fm/core.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

namespace edge_fm {

namespace {

constexpr int kHorizonWaitTimeoutMs = 30000;

int64_t align_value(int64_t value, int64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int64_t bpu_align(int64_t value) {
    return align_value(value, 32);
}

size_t horizon_dtype_size(hbDNNDataType dtype) {
    switch (dtype) {
        case HB_DNN_TENSOR_TYPE_F32:
        case HB_DNN_TENSOR_TYPE_S32:
            return 4;
        case HB_DNN_TENSOR_TYPE_F16:
        case HB_DNN_TENSOR_TYPE_S16:
        case HB_DNN_TENSOR_TYPE_U16:
            return 2;
        case HB_DNN_TENSOR_TYPE_U8:
        case HB_DNN_TENSOR_TYPE_S8:
        case HB_DNN_TENSOR_TYPE_BOOL8:
            return 1;
        default:
            return 4;
    }
}

RuntimeDType to_runtime_dtype(hbDNNDataType dtype) {
    switch (dtype) {
        case HB_DNN_TENSOR_TYPE_F32:
            return RuntimeDType::Float32;
        case HB_DNN_TENSOR_TYPE_F16:
            return RuntimeDType::Float16;
        case HB_DNN_TENSOR_TYPE_S32:
            return RuntimeDType::Int32;
        case HB_DNN_TENSOR_TYPE_S8:
            return RuntimeDType::Int8;
        case HB_DNN_TENSOR_TYPE_U8:
            return RuntimeDType::UInt8;
        default:
            return RuntimeDType::Float32;
    }
}

std::vector<int64_t> valid_shape_to_vector(const hbDNNTensorShape& shape) {
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(std::max(shape.numDimensions, 0)));
    for (int i = 0; i < shape.numDimensions; ++i) {
        out.push_back(static_cast<int64_t>(shape.dimensionSize[i]));
    }
    return out;
}

const std::vector<int64_t>* find_named_input_shape_override(
    const RuntimeInitParams& params,
    const std::string& input_name)
{
    auto it = params.input_shape_overrides.find(input_name);
    return it == params.input_shape_overrides.end() ? nullptr : &it->second;
}

bool apply_shape_override(hbDNNTensorProperties* properties,
                          const std::vector<int64_t>* override_shape) {
    if (properties == nullptr || override_shape == nullptr) {
        return true;
    }
    hbDNNTensorShape& valid_shape = properties->validShape;
    if (valid_shape.numDimensions <= 0 ||
        override_shape->size() != static_cast<size_t>(valid_shape.numDimensions)) {
        return false;
    }
    for (int i = 0; i < valid_shape.numDimensions; ++i) {
        int64_t requested = (*override_shape)[static_cast<size_t>(i)];
        if (requested <= 0) {
            continue;
        }
        if (valid_shape.dimensionSize[i] > 0 && valid_shape.dimensionSize[i] != requested) {
            return false;
        }
        if (valid_shape.dimensionSize[i] <= 0) {
            valid_shape.dimensionSize[i] = requested;
        }
    }
    return true;
}

bool infer_spatial_from_hint(const std::vector<int64_t>* shape_hint,
                             int* out_height,
                             int* out_width) {
    if (shape_hint == nullptr || out_height == nullptr || out_width == nullptr ||
        shape_hint->size() != 4) {
        return false;
    }
    const int64_t d1 = (*shape_hint)[1];
    const int64_t d2 = (*shape_hint)[2];
    const int64_t d3 = (*shape_hint)[3];
    if (d1 > 0 && d1 <= 4 && d2 > 0 && d3 > 0) {
        *out_height = static_cast<int>(d2);
        *out_width = static_cast<int>(d3);
        return true;
    }
    if (d1 > 0 && d2 > 0 && d3 > 0 && d3 <= 4) {
        *out_height = static_cast<int>(d1);
        *out_width = static_cast<int>(d2);
        return true;
    }
    return false;
}

bool resolve_dynamic_shape(hbDNNTensorProperties* properties,
                           const std::vector<int64_t>* named_shape_override,
                           const std::vector<int64_t>* input_shape_hint,
                           int max_batch_size) {
    if (properties == nullptr) {
        return false;
    }
    hbDNNTensorShape& shape = properties->validShape;
    if (shape.numDimensions <= 0) {
        return false;
    }
    if (!apply_shape_override(properties, named_shape_override)) {
        return false;
    }
    if (shape.dimensionSize[0] <= 0) {
        shape.dimensionSize[0] = std::max(1, max_batch_size);
    }

    if (shape.numDimensions == 4) {
        const bool looks_like_nhwc = shape.dimensionSize[3] > 0;
        const bool looks_like_nchw = shape.dimensionSize[1] > 0 && shape.dimensionSize[3] <= 0;
        int hinted_height = 0;
        int hinted_width = 0;
        if ((shape.dimensionSize[1] <= 0 || shape.dimensionSize[2] <= 0 ||
             shape.dimensionSize[3] <= 0) &&
            infer_spatial_from_hint(input_shape_hint, &hinted_height, &hinted_width)) {
            if (looks_like_nhwc) {
                int resolved_height = hinted_height;
                int resolved_width = hinted_width;
                if (properties->tensorType == HB_DNN_TENSOR_TYPE_U8 &&
                    shape.dimensionSize[3] == 2) {
                    resolved_height = hinted_height / 2;
                    resolved_width = hinted_width / 2;
                }
                if (shape.dimensionSize[1] <= 0) {
                    shape.dimensionSize[1] = resolved_height;
                }
                if (shape.dimensionSize[2] <= 0) {
                    shape.dimensionSize[2] = resolved_width;
                }
            } else if (looks_like_nchw) {
                if (shape.dimensionSize[2] <= 0) {
                    shape.dimensionSize[2] = hinted_height;
                }
                if (shape.dimensionSize[3] <= 0) {
                    shape.dimensionSize[3] = hinted_width;
                }
            }
        }
    }

    for (int i = 0; i < shape.numDimensions; ++i) {
        if (shape.dimensionSize[i] <= 0) {
            return false;
        }
    }
    return true;
}

bool resolve_stride_and_byte_size(hbDNNTensorProperties* properties) {
    if (properties == nullptr || properties->validShape.numDimensions <= 0) {
        return false;
    }
    const int dim_count = properties->validShape.numDimensions;
    for (int dim = dim_count - 1; dim >= 0; --dim) {
        if (properties->stride[dim] == -1) {
            const int next = dim + 1;
            const int64_t stride =
                next < dim_count
                    ? properties->stride[next] * properties->validShape.dimensionSize[next]
                    : static_cast<int64_t>(horizon_dtype_size(
                        static_cast<hbDNNDataType>(properties->tensorType)));
            properties->stride[dim] = bpu_align(stride);
        }
    }
    if (properties->stride[0] <= 0 || properties->validShape.dimensionSize[0] <= 0) {
        return false;
    }
    properties->alignedByteSize =
        properties->stride[0] * properties->validShape.dimensionSize[0];
    return properties->alignedByteSize > 0;
}

} // namespace

struct HorizonRuntimeBackend::Impl {
    struct IOTensor {
        std::string name;
        hbDNNTensor tensor{};
    };

    ~Impl() {
        release_last_task();
        for (auto& tensor : input_tensors) {
            if (tensor.tensor.sysMem.virAddr != nullptr) {
                hbUCPFree(&tensor.tensor.sysMem);
                tensor.tensor.sysMem.virAddr = nullptr;
            }
        }
        for (auto& tensor : output_tensors) {
            if (tensor.tensor.sysMem.virAddr != nullptr) {
                hbUCPFree(&tensor.tensor.sysMem);
                tensor.tensor.sysMem.virAddr = nullptr;
            }
        }
        if (packed_handle != nullptr) {
            hbDNNRelease(packed_handle);
            packed_handle = nullptr;
            dnn_handle = nullptr;
        }
    }

    std::string locate_model_file(const std::string& suffix) const {
        const std::vector<std::string> dirs = {"/model/", "/parameters/", "/data/", "/"};
        for (const auto& dir : dirs) {
            std::string path = program_path + dir + suffix;
            std::ifstream file(path);
            if (file.is_open()) {
                return path;
            }
        }
        return {};
    }

    void set_error(const std::string& message) {
        last_error = message;
    }

    bool init(const RuntimeInitParams& params) {
        program_path = params.program_path;
        model_name = params.model_name;
        max_batch_size = params.max_batch_size > 0 ? params.max_batch_size : 1;

        std::string model_file_path = params.model_path;
        if (!model_file_path.empty()) {
            std::ifstream file(model_file_path);
            if (!file.is_open()) {
                set_error("Cannot open Horizon HBM model file: " + model_file_path);
                return false;
            }
        } else {
            model_file_path = locate_model_file(model_name + ".hbm");
            if (model_file_path.empty()) {
                set_error("Cannot locate Horizon HBM model file for model_name=" + model_name);
                return false;
            }
        }

        const char* model_file_cstr = model_file_path.c_str();
        int32_t ret = hbDNNInitializeFromFiles(&packed_handle, &model_file_cstr, 1);
        if (ret != 0) {
            set_error("hbDNNInitializeFromFiles failed, ret=" + std::to_string(ret));
            return false;
        }

        const char** model_name_list = nullptr;
        int model_count = 0;
        ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_handle);
        if (ret != 0 || model_count <= 0 || model_name_list == nullptr) {
            set_error("hbDNNGetModelNameList failed, ret=" + std::to_string(ret));
            return false;
        }

        ret = hbDNNGetModelHandle(&dnn_handle, packed_handle, model_name_list[0]);
        if (ret != 0) {
            set_error("hbDNNGetModelHandle failed, ret=" + std::to_string(ret));
            return false;
        }

        int input_count = 0;
        int output_count = 0;
        ret = hbDNNGetInputCount(&input_count, dnn_handle);
        if (ret != 0) {
            set_error("hbDNNGetInputCount failed, ret=" + std::to_string(ret));
            return false;
        }
        ret = hbDNNGetOutputCount(&output_count, dnn_handle);
        if (ret != 0) {
            set_error("hbDNNGetOutputCount failed, ret=" + std::to_string(ret));
            return false;
        }

        input_tensors.resize(static_cast<size_t>(input_count));
        output_tensors.resize(static_cast<size_t>(output_count));

        const std::vector<int64_t>* shape_hint =
            params.input_shape_hint.empty() ? nullptr : &params.input_shape_hint;
        for (int i = 0; i < input_count; ++i) {
            IOTensor& io = input_tensors[static_cast<size_t>(i)];
            ret = hbDNNGetInputTensorProperties(&io.tensor.properties, dnn_handle, i);
            if (ret != 0) {
                set_error("hbDNNGetInputTensorProperties failed, ret=" + std::to_string(ret));
                return false;
            }
            const char* name = nullptr;
            ret = hbDNNGetInputName(&name, dnn_handle, i);
            if (ret != 0) {
                set_error("hbDNNGetInputName failed, ret=" + std::to_string(ret));
                return false;
            }
            if (name != nullptr) {
                io.name = name;
            }
            if (!resolve_dynamic_shape(
                    &io.tensor.properties,
                    find_named_input_shape_override(params, io.name),
                    shape_hint,
                    max_batch_size)) {
                set_error("Failed to resolve Horizon input shape for " + io.name);
                return false;
            }
            if (!resolve_stride_and_byte_size(&io.tensor.properties)) {
                set_error("Failed to resolve Horizon input stride for " + io.name);
                return false;
            }
            ret = hbUCPMallocCached(
                &io.tensor.sysMem,
                static_cast<int>(io.tensor.properties.alignedByteSize),
                0);
            if (ret != 0) {
                set_error("hbUCPMallocCached input failed, ret=" + std::to_string(ret));
                return false;
            }
        }

        for (int i = 0; i < output_count; ++i) {
            IOTensor& io = output_tensors[static_cast<size_t>(i)];
            ret = hbDNNGetOutputTensorProperties(&io.tensor.properties, dnn_handle, i);
            if (ret != 0) {
                set_error("hbDNNGetOutputTensorProperties failed, ret=" + std::to_string(ret));
                return false;
            }
            const char* name = nullptr;
            ret = hbDNNGetOutputName(&name, dnn_handle, i);
            if (ret != 0) {
                set_error("hbDNNGetOutputName failed, ret=" + std::to_string(ret));
                return false;
            }
            if (name != nullptr) {
                io.name = name;
            }
            ret = hbUCPMallocCached(
                &io.tensor.sysMem,
                static_cast<int>(io.tensor.properties.alignedByteSize),
                0);
            if (ret != 0) {
                set_error("hbUCPMallocCached output failed, ret=" + std::to_string(ret));
                return false;
            }
        }

        input_tensor_views.clear();
        input_tensor_views.reserve(input_tensors.size());
        for (const auto& tensor : input_tensors) {
            input_tensor_views.push_back(tensor.tensor);
        }
        output_tensor_views.clear();
        output_tensor_views.reserve(output_tensors.size());
        for (const auto& tensor : output_tensors) {
            output_tensor_views.push_back(tensor.tensor);
        }

        HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
        sched_param.backend = HB_UCP_BPU_CORE_ANY;

        initialized = true;
        last_error.clear();
        return true;
    }

    bool warmup(int batch_size) {
        if (!initialized || batch_size <= 0) {
            return true;
        }
        for (auto& tensor : input_tensors) {
            if (tensor.tensor.sysMem.virAddr != nullptr) {
                std::memset(tensor.tensor.sysMem.virAddr, 0, tensor.tensor.sysMem.memSize);
            }
        }
        for (auto& tensor : output_tensors) {
            if (tensor.tensor.sysMem.virAddr != nullptr) {
                std::memset(tensor.tensor.sysMem.virAddr, 0, tensor.tensor.sysMem.memSize);
            }
        }
        return forward_sync() == 0;
    }

    int ensure_no_pending_task() {
        if (last_task_handle == nullptr) {
            return 0;
        }
        return wait(nullptr);
    }

    void release_last_task() {
        if (last_task_handle != nullptr) {
            hbUCPReleaseTask(last_task_handle);
            last_task_handle = nullptr;
        }
    }

    int submit_task() {
        int32_t ret = hbDNNInferV2(
            &last_task_handle,
            output_tensor_views.empty() ? nullptr : output_tensor_views.data(),
            input_tensor_views.empty() ? nullptr : input_tensor_views.data(),
            dnn_handle);
        if (ret != 0) {
            set_error("hbDNNInferV2 failed, ret=" + std::to_string(ret));
            return -1;
        }
        ret = hbUCPSubmitTask(last_task_handle, &sched_param);
        if (ret != 0) {
            release_last_task();
            set_error("hbUCPSubmitTask failed, ret=" + std::to_string(ret));
            return -1;
        }
        return 0;
    }

    int forward_sync() {
        if (!initialized) {
            set_error("Horizon runtime backend is not initialized");
            return -1;
        }
        if (ensure_no_pending_task() != 0) {
            return -1;
        }
        for (auto& tensor : input_tensors) {
            hbUCPMemFlush(&tensor.tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
        }
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            input_tensor_views[i] = input_tensors[i].tensor;
        }
        if (submit_task() != 0) {
            return -1;
        }
        return wait(nullptr);
    }

    int forward_async(RuntimeStreamHandle) {
        if (!initialized) {
            set_error("Horizon runtime backend is not initialized");
            return -1;
        }
        if (ensure_no_pending_task() != 0) {
            return -1;
        }
        for (auto& tensor : input_tensors) {
            hbUCPMemFlush(&tensor.tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
        }
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            input_tensor_views[i] = input_tensors[i].tensor;
        }
        return submit_task();
    }

    int wait(RuntimeStreamHandle) {
        if (!initialized) {
            set_error("Horizon runtime backend is not initialized");
            return -1;
        }
        if (last_task_handle == nullptr) {
            return 0;
        }
        int32_t ret = hbUCPWaitTaskDone(last_task_handle, kHorizonWaitTimeoutMs);
        if (ret != 0) {
            release_last_task();
            set_error("hbUCPWaitTaskDone failed, ret=" + std::to_string(ret));
            return -1;
        }
        for (auto& tensor : output_tensors) {
            hbUCPMemFlush(&tensor.tensor.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        }
        release_last_task();
        return 0;
    }

    std::vector<std::string> input_names() const {
        std::vector<std::string> names;
        names.reserve(input_tensors.size());
        for (const auto& tensor : input_tensors) {
            names.push_back(tensor.name);
        }
        return names;
    }

    std::vector<std::string> output_names() const {
        std::vector<std::string> names;
        names.reserve(output_tensors.size());
        for (const auto& tensor : output_tensors) {
            names.push_back(tensor.name);
        }
        return names;
    }

    bool get_input_shape(const std::string& name, std::vector<int64_t>* out_shape) const {
        if (out_shape == nullptr) {
            return false;
        }
        for (const auto& tensor : input_tensors) {
            if (tensor.name == name) {
                *out_shape = valid_shape_to_vector(tensor.tensor.properties.validShape);
                return true;
            }
        }
        return false;
    }

    bool get_output_shape(const std::string& name, std::vector<int64_t>* out_shape) const {
        if (out_shape == nullptr) {
            return false;
        }
        for (const auto& tensor : output_tensors) {
            if (tensor.name == name) {
                *out_shape = valid_shape_to_vector(tensor.tensor.properties.validShape);
                return true;
            }
        }
        return false;
    }

    static RuntimeTensorView make_view(const hbDNNTensor& tensor) {
        RuntimeTensorView view;
        view.data = tensor.sysMem.virAddr;
        view.shape = valid_shape_to_vector(tensor.properties.validShape);
        view.dtype = to_runtime_dtype(static_cast<hbDNNDataType>(tensor.properties.tensorType));
        view.device = RuntimeDevice::BPU;
        view.stride.clear();
        const int dim_count = tensor.properties.validShape.numDimensions;
        if (dim_count > 0) {
            view.stride.reserve(static_cast<size_t>(dim_count));
            for (int i = 0; i < dim_count; ++i) {
                view.stride.push_back(static_cast<int64_t>(tensor.properties.stride[i]));
            }
        }
        return view;
    }

    bool get_input_buffer(const std::string& name, RuntimeTensorView* out_tensor) {
        if (out_tensor == nullptr) {
            return false;
        }
        for (const auto& tensor : input_tensors) {
            if (tensor.name == name) {
                *out_tensor = make_view(tensor.tensor);
                return out_tensor->data != nullptr;
            }
        }
        return false;
    }

    bool get_output_buffer(const std::string& name, RuntimeTensorView* out_tensor) {
        if (out_tensor == nullptr) {
            return false;
        }
        for (const auto& tensor : output_tensors) {
            if (tensor.name == name) {
                *out_tensor = make_view(tensor.tensor);
                return out_tensor->data != nullptr;
            }
        }
        return false;
    }

    std::string program_path;
    std::string model_name;
    int max_batch_size = 1;
    hbUCPSchedParam sched_param{};

    hbDNNPackedHandle_t packed_handle = nullptr;
    hbDNNHandle_t dnn_handle = nullptr;
    hbUCPTaskHandle_t last_task_handle = nullptr;

    std::vector<IOTensor> input_tensors;
    std::vector<IOTensor> output_tensors;
    std::vector<hbDNNTensor> input_tensor_views;
    std::vector<hbDNNTensor> output_tensor_views;

    bool initialized = false;
    std::string last_error;
};

HorizonRuntimeBackend::HorizonRuntimeBackend()
    : impl_(std::make_unique<Impl>())
{}

HorizonRuntimeBackend::~HorizonRuntimeBackend() = default;

bool HorizonRuntimeBackend::init(const RuntimeInitParams& params) {
    return impl_->init(params);
}

bool HorizonRuntimeBackend::warmup(int batch_size) {
    return impl_->warmup(batch_size);
}

RuntimeStreamHandle HorizonRuntimeBackend::default_stream() {
    return nullptr;
}

int HorizonRuntimeBackend::forward_sync() {
    return impl_->forward_sync();
}

int HorizonRuntimeBackend::forward_async(RuntimeStreamHandle stream) {
    return impl_->forward_async(stream);
}

int HorizonRuntimeBackend::wait(RuntimeStreamHandle stream) {
    return impl_->wait(stream);
}

std::vector<std::string> HorizonRuntimeBackend::input_names() const {
    return impl_->input_names();
}

std::vector<std::string> HorizonRuntimeBackend::output_names() const {
    return impl_->output_names();
}

bool HorizonRuntimeBackend::get_input_shape(
    const std::string& name,
    std::vector<int64_t>* out_shape) const
{
    return impl_->get_input_shape(name, out_shape);
}

bool HorizonRuntimeBackend::get_output_shape(
    const std::string& name,
    std::vector<int64_t>* out_shape) const
{
    return impl_->get_output_shape(name, out_shape);
}

bool HorizonRuntimeBackend::get_input_buffer(
    const std::string& name,
    RuntimeTensorView* out_tensor)
{
    return impl_->get_input_buffer(name, out_tensor);
}

bool HorizonRuntimeBackend::get_output_buffer(
    const std::string& name,
    RuntimeTensorView* out_tensor)
{
    return impl_->get_output_buffer(name, out_tensor);
}

std::string HorizonRuntimeBackend::last_error() const {
    return impl_->last_error;
}

std::unique_ptr<IRuntimeBackend> create_horizon_runtime_backend() {
    return std::make_unique<HorizonRuntimeBackend>();
}

} // namespace edge_fm
