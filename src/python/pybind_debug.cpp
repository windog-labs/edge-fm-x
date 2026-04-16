#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <Python.h>  // For PyCapsule API
#include <cstdint>   // For uintptr_t
#include <string>    // For std::string
#include <regex>     // For std::regex
#include <edge-fm/edge-fm.h>
#include <edge-fm/core.h>
#include <dlpack/dlpack.h>
#include "layers/attention.h"
#include "layers/layernorm.h"
#include "layers/activation.h"
#include "layers/sampler.h"
#include "layers/embed_head.h"
#include "layers/linear.h"
#include "engine/engine.h"
#include "models/model.h"
#include "engine/kv_manager.h"
#include "utils/device/weight_loader.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <exception>
#include <filesystem>
#include <sstream>
#include <unistd.h>

namespace py = pybind11;
using namespace edge_fm;

namespace {

std::vector<std::string> collect_safetensors_files(const std::string& model_dir) {
    std::vector<std::string> out;
    std::filesystem::path dir(model_dir);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return out;
    }

    const std::string single = model_dir + "/model.safetensors";
    if (std::filesystem::exists(single) && std::filesystem::is_regular_file(single)) {
        out.push_back(single);
        return out;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() > 18 && name.compare(0, 6, "model-") == 0 &&
            name.find("-of-") != std::string::npos &&
            name.size() >= 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0) {
            out.push_back(entry.path().string());
        }
    }

    std::sort(out.begin(), out.end());
    return out;
}

const std::unordered_map<std::string, Tensor>& load_stage_weights_for_debug_layer(
    WeightLoader& loader,
    const EngineConfig& config,
    ModelStage stage,
    int32_t runtime_device_id,
    const std::unordered_map<std::string, Tensor>* fallback_weights = nullptr) {
    const std::string model_path =
        (stage == ModelStage::Decode && !config.decode_model_path().empty())
            ? config.decode_model_path()
            : config.prefill_model_path();
    const auto safetensors_files = collect_safetensors_files(model_path);
    if (safetensors_files.empty()) {
        if (fallback_weights != nullptr) {
            return *fallback_weights;
        }
        throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in: " + model_path);
    }

    const bool is_vlm = (config.resolved_model_name() == "qwen2_5_vl");
    if (is_vlm) {
        auto vlm_filter = [](const std::string& name) {
            return name.rfind("model.", 0) == 0 ||
                   name.rfind("language_model.", 0) == 0 ||
                   name.rfind("lm_head.", 0) == 0;
        };
        auto vlm_key_mapper = [](const std::string& name) {
            if (name.rfind("model.model.", 0) == 0) {
                return name.substr(6);
            }
            if (name.rfind("language_model.", 0) == 0) {
                return name.substr(std::string("language_model.").size());
            }
            return name;
        };
        for (const auto& file : safetensors_files) {
            loader.load_weights_from_file(stage, file, Device::GPU, runtime_device_id, true, vlm_filter, vlm_key_mapper);
        }
    } else {
        for (const auto& file : safetensors_files) {
            loader.load_weights_from_file(stage, file, Device::GPU, runtime_device_id, true);
        }
    }

    return loader.get(stage);
}

}  // namespace

// 异常转换函数
// 注意：对于已经通过 py::register_exception 注册的异常类型，不应该在这里捕获
// 让它们直接通过，pybind11 会自动处理并转换为对应的 Python 异常类
// 这个转换器只处理未注册的异常类型
void translate_exception(std::exception_ptr p) {
    try {
        if (p) { std::rethrow_exception(p); }
    } catch (const DeviceError&) {
        throw;
    } catch (const ConfigurationError&) {
        throw;
    } catch (const ModelNotLoadedError&) {
        throw;
    } catch (const InvalidRequestError&) {
        throw;
    } catch (const OutOfMemoryError&) {
        throw;
    } catch (const InternalError&) {
        throw;
    } catch (const std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception occurred");
    }
}

// 从 void* 指针创建 Tensor 的辅助函数
Tensor tensor_from_void_ptr(uintptr_t data_ptr,
                            const std::vector<int64_t>& shape,
                            DType dtype,
                            Device device,
                            int32_t device_id = 0,
                            bool copy_data = false) {
    const void* data = reinterpret_cast<const void*>(data_ptr);
    if (copy_data) {
        // Clone into a newly allocated buffer on the specified device
        MemoryOwnership ownership = (device == Device::GPU) 
            ? MemoryOwnership::OwnCudaMalloc 
            : MemoryOwnership::OwnCpuMalloc;
        return Tensor::clone_from(data, shape, dtype,
                                  device, device_id,
                                  device, device_id,
                                  ownership, nullptr);
    } else {
        // Create a non-owning view
        return Tensor::view(const_cast<void*>(data), shape, dtype, device, device_id);
    }
}

// 从 PyTorch tensor 创建 edge_fm.Tensor 视图（零拷贝，要求 tensor 已 contiguous）
Tensor tensor_from_pytorch(py::object torch_tensor) {
    if (py::isinstance<Tensor>(torch_tensor)) {
        Tensor& t = py::cast<Tensor&>(torch_tensor);
        auto [dev, dev_id] = t.device();
        return Tensor::view(t.data_ptr(), t.shape(), t.dtype(), dev, dev_id);
    }
    // PyTorch tensor: 需 contiguous
    py::object t = torch_tensor.attr("contiguous")();
    uintptr_t ptr = py::cast<uintptr_t>(t.attr("data_ptr")());
    py::object shape_obj = t.attr("shape");
    std::vector<int64_t> shape;
    for (auto dim : shape_obj) {
        shape.push_back(py::cast<int64_t>(dim));
    }
    std::string dtype_str = py::str(t.attr("dtype"));
    DType dtype;
    // 必须先检查 bfloat16，否则 "bfloat16" 会匹配 "float16"
    if (dtype_str.find("bfloat16") != std::string::npos) {
        dtype = DType::BFloat16;
    } else if (dtype_str.find("float16") != std::string::npos || dtype_str.find("half") != std::string::npos) {
        dtype = DType::Float16;
    } else if (dtype_str.find("float32") != std::string::npos) {
        dtype = DType::Float32;
    } else if (dtype_str.find("int32") != std::string::npos) {
        dtype = DType::Int32;
    } else {
        throw std::runtime_error("Unsupported PyTorch dtype: " + dtype_str);
    }
    std::string device_str = py::str(t.attr("device"));
    Device device;
    int32_t device_id = 0;
    if (device_str.find("cuda") != std::string::npos) {
        device = Device::GPU;
        size_t colon = device_str.find(':');
        if (colon != std::string::npos && colon + 1 < device_str.size()) {
            device_id = std::stoi(device_str.substr(colon + 1));
        }
    } else {
        device = Device::CPU;
    }
    return Tensor::view(reinterpret_cast<void*>(ptr), shape, dtype, device, device_id);
}

// 从 Python DLPack capsule 创建 Tensor 的辅助函数
Tensor tensor_from_dlpack_capsule(py::object capsule) {
    if (!PyCapsule_CheckExact(capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object for DLPack tensor");
    }
    
    const char* name = PyCapsule_GetName(capsule.ptr());
    if (name == nullptr || std::string(name) != "dltensor") {
        throw std::runtime_error("Invalid DLPack capsule name. Expected 'dltensor'");
    }
    
    DLManagedTensor* managed_tensor = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor")
    );
    
    if (managed_tensor == nullptr) {
        throw std::runtime_error("Failed to get DLManagedTensor from capsule");
    }
    
    Tensor tensor = Tensor::from_dlpack(managed_tensor);
    
    return tensor;
}

// DLPack capsule deleter: 释放 DLManagedTensor 及其相关内存
void dlpack_capsule_deleter(PyObject* capsule) {
    if (capsule == nullptr) {
        return;
    }
    
    // 检查 capsule 的名称是否正确
    const char* name = PyCapsule_GetName(capsule);
    if (name == nullptr || std::string(name) != "dltensor") {
        return;
    }
    
    DLManagedTensor* managed_tensor = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor")
    );
    
    if (managed_tensor != nullptr) {
        // 释放 shape 数组
        if (managed_tensor->dl_tensor.shape != nullptr) {
            delete[] managed_tensor->dl_tensor.shape;
        }
        
        // 释放 strides 数组
        if (managed_tensor->dl_tensor.strides != nullptr) {
            delete[] managed_tensor->dl_tensor.strides;
        }
        
        // 释放 DLManagedTensor 结构本身
        delete managed_tensor;
    }
}

// 将 Tensor 转换为 DLPack capsule 的辅助函数
py::object tensor_to_dlpack_capsule(const Tensor& tensor) {
    try {
        // 使用 Tensor::to_dlpack() 获取 DLManagedTensor*
        // 注意：to_dlpack() 现在会抛出异常，而不是返回 nullptr
        DLManagedTensor* managed_tensor = tensor.to_dlpack();
        
        // 创建 PyCapsule
        // 注意：PyCapsule 的 deleter 会调用 dlpack_capsule_deleter，它会释放 DLManagedTensor 及其相关内存
        py::object capsule = py::reinterpret_steal<py::object>(
            PyCapsule_New(managed_tensor, "dltensor", dlpack_capsule_deleter)
        );
        
        if (!capsule) {
            // 如果创建失败，需要手动释放 managed_tensor
            // 注意：Tensor::to_dlpack() 返回的 managed_tensor 已经设置了 deleter
            // 但我们可以直接调用 dlpack_capsule_deleter 的逻辑来释放
            if (managed_tensor->dl_tensor.shape != nullptr) {
                delete[] managed_tensor->dl_tensor.shape;
            }
            if (managed_tensor->dl_tensor.strides != nullptr) {
                delete[] managed_tensor->dl_tensor.strides;
            }
            delete managed_tensor;
            throw std::runtime_error("Failed to create DLPack capsule");
        }
        
        return capsule;
    } catch (...) {
        // 让 pybind11 处理异常转换
        throw;
    }
}

static WeightLoader& loader = WeightLoader::instance();

PYBIND11_MODULE(edge_fm, m) {
    m.doc() = "EdgeFM: 边缘端基础模型推理引擎";

    // ============================================================================
    // 异常类绑定
    // ============================================================================
    py::register_exception<Error>(m, "Error", PyExc_RuntimeError);
    py::register_exception<DeviceError>(m, "DeviceError", m.attr("Error"));
    py::register_exception<ConfigurationError>(m, "ConfigurationError", m.attr("Error"));
    py::register_exception<ModelNotLoadedError>(m, "ModelNotLoadedError", m.attr("Error"));
    py::register_exception<InvalidRequestError>(m, "InvalidRequestError", m.attr("Error"));
    py::register_exception<OutOfMemoryError>(m, "OutOfMemoryError", m.attr("Error"));
    py::register_exception<InternalError>(m, "InternalError", m.attr("Error"));

    // 注意：异常转换器只处理未注册的异常类型
    // 已注册的异常类型（如 InvalidRequestError）会由 pybind11 自动处理
    py::register_exception_translator(translate_exception);

    // ============================================================================
    // 枚举类型绑定
    // ============================================================================
    py::enum_<Device>(m, "Device", "设备类型枚举")
        .value("CPU", Device::CPU, "CPU 设备")
        .value("GPU", Device::GPU, "GPU 设备")
        .export_values();

    py::enum_<DType>(m, "DType", "数据类型枚举")
        .value("Float32", DType::Float32, "32位浮点数")
        .value("Float16", DType::Float16, "16位浮点数")
        .value("BFloat16", DType::BFloat16, "16位 BFloat")
        .value("Int32", DType::Int32, "32位整数")
        .value("Int64", DType::Int64, "64位整数")
        .value("UInt8", DType::UInt8, "8位无符号整数")
        .value("Int8", DType::Int8, "8位整数")
        .export_values();

    py::enum_<ModelStage>(m, "ModelStage", "模型阶段枚举")
        .value("Prefill", ModelStage::Prefill, "Prefill 阶段")
        .value("Decode", ModelStage::Decode, "Decode 阶段")
        .export_values();

    py::enum_<MemoryOwnership>(m, "MemoryOwnership", "内存所有权/释放策略枚举")
        .value("ViewExternal", MemoryOwnership::ViewExternal, "非拥有视图（不释放）")
        .value("OwnCpuMalloc", MemoryOwnership::OwnCpuMalloc, "CPU 内存（std::free 释放）")
        .value("OwnCudaMalloc", MemoryOwnership::OwnCudaMalloc, "GPU 内存（cudaFree 释放）")
        .value("OwnCudaPool", MemoryOwnership::OwnCudaPool, "GPU 内存池（cudaFreeAsync 释放）")
        .export_values();

    // ============================================================================
    // Tensor 类绑定
    // ============================================================================
    py::class_<Tensor>(m, "Tensor", "张量类，用于存储多维数组数据")
        // 从 void* 指针构造（通过整数地址传递）
        .def(py::init(&tensor_from_void_ptr),
             py::arg("data_ptr"),
             py::arg("shape"),
             py::arg("dtype"),
             py::arg("device"),
             py::arg("device_id") = 0,
             py::arg("copy_data") = false,
             "从内存指针创建张量\n\n"
             "参数:\n"
             "    data_ptr: 内存指针地址（整数，可通过 numpy.ndarray.ctypes.data_as(ctypes.c_void_p).value 获取）\n"
             "    shape: 张量的形状维度\n"
             "    dtype: 张量元素的数据类型\n"
             "    device: 目标设备（CPU 或 GPU）\n"
             "    device_id: 设备 ID（默认: 0）\n"
             "    copy_data: 如果为 True，复制数据到新缓冲区；如果为 False，直接使用缓冲区（默认: False）\n\n"
             "注意:\n"
             "    内存缓冲区必须是连续的内存且为行优先（C 风格）布局。\n"
             "    当 copy_data 为 False 时，调用者必须确保缓冲区在此 Tensor 对象的生命周期内保持有效，\n"
             "    并且数据缓冲区必须已经正确分配到目标设备上。")
        // 静态工厂方法
        .def_static("view", 
                    [](uintptr_t data_ptr, const std::vector<int64_t>& shape, 
                       DType dtype, Device device, int32_t device_id) {
                        return Tensor::view(reinterpret_cast<void*>(data_ptr), shape, dtype, device, device_id);
                    },
                    py::arg("data_ptr"), py::arg("shape"), py::arg("dtype"), 
                    py::arg("device"), py::arg("device_id") = 0,
                    "创建非拥有视图张量（不复制数据，不负责释放）\n\n"
                    "参数:\n"
                    "    data_ptr: 数据指针地址\n"
                    "    shape: 张量形状\n"
                    "    dtype: 数据类型\n"
                    "    device: 设备类型\n"
                    "    device_id: 设备 ID（默认: 0）")
        .def_static("adopt",
                    [](uintptr_t data_ptr, const std::vector<int64_t>& shape,
                       DType dtype, Device device, int32_t device_id,
                       MemoryOwnership ownership, uintptr_t stream_handle) {
                        return Tensor::adopt(reinterpret_cast<void*>(data_ptr), shape, dtype, 
                                           device, device_id, ownership, 
                                           reinterpret_cast<void*>(stream_handle));
                    },
                    py::arg("data_ptr"), py::arg("shape"), py::arg("dtype"),
                    py::arg("device"), py::arg("device_id") = 0,
                    py::arg("ownership") = MemoryOwnership::OwnCudaMalloc,
                    py::arg("stream_handle") = 0,
                    "接管已分配的缓冲区（不复制数据，但负责释放）\n\n"
                    "参数:\n"
                    "    data_ptr: 数据指针地址\n"
                    "    shape: 张量形状\n"
                    "    dtype: 数据类型\n"
                    "    device: 设备类型\n"
                    "    device_id: 设备 ID（默认: 0）\n"
                    "    ownership: 内存所有权类型（默认: OwnCudaMalloc）\n"
                    "    stream_handle: CUDA 流句柄（仅 OwnCudaPool 需要）")
        .def_static("clone_from",
                    [](uintptr_t src_ptr, const std::vector<int64_t>& shape,
                       DType dtype, Device src_device, int32_t src_device_id,
                       Device dst_device, int32_t dst_device_id,
                       MemoryOwnership ownership, uintptr_t stream_handle) {
                        return Tensor::clone_from(reinterpret_cast<const void*>(src_ptr), shape, dtype,
                                                src_device, src_device_id, dst_device, dst_device_id,
                                                ownership, reinterpret_cast<void*>(stream_handle));
                    },
                    py::arg("src_ptr"), py::arg("shape"), py::arg("dtype"),
                    py::arg("src_device"), py::arg("src_device_id"),
                    py::arg("dst_device"), py::arg("dst_device_id") = 0,
                    py::arg("ownership") = MemoryOwnership::OwnCudaMalloc,
                    py::arg("stream_handle") = 0,
                    "从源缓冲区克隆数据到新张量（分配并拷贝）\n\n"
                    "参数:\n"
                    "    src_ptr: 源数据指针地址\n"
                    "    shape: 张量形状\n"
                    "    dtype: 数据类型\n"
                    "    src_device: 源设备类型\n"
                    "    src_device_id: 源设备 ID\n"
                    "    dst_device: 目标设备类型\n"
                    "    dst_device_id: 目标设备 ID（默认: 0）\n"
                    "    ownership: 目标内存所有权类型（默认: OwnCudaMalloc）\n"
                    "    stream_handle: CUDA 流句柄（仅 OwnCudaPool 需要）")
        // 从 DLPack capsule 创建的便捷方法
        .def_static("from_dlpack", &tensor_from_dlpack_capsule,
                    py::arg("capsule"),
                    "从 DLPack capsule 创建张量\n\n"
                    "参数:\n"
                    "    capsule: DLPack 兼容的 PyCapsule 对象（例如来自 PyTorch 的 tensor.__dlpack__()）\n\n"
                    "返回:\n"
                    "    新创建的张量（数据会被复制）\n\n"
                    "注意:\n"
                    "    数据会被复制到新的 Tensor 对象中，原始的 capsule 可以被安全释放。")
        // 转换为 DLPack capsule 的方法
        .def("to_dlpack", &tensor_to_dlpack_capsule,
             "将张量转换为 DLPack capsule\n\n"
             "返回:\n"
             "    DLPack 兼容的 PyCapsule 对象，可以被 PyTorch 等框架使用\n\n"
             "注意:\n"
             "    返回的 capsule 与原始 Tensor 共享数据缓冲区（不复制数据）。\n"
             "    调用者必须确保 Tensor 对象在 capsule 使用期间保持有效。\n"
             "    可以使用 torch.from_dlpack(capsule) 从 capsule 创建 PyTorch tensor。")
        // 属性接口
        .def("empty", &Tensor::empty, "检查张量是否为空")
        .def("dtype", &Tensor::dtype, "获取张量的数据类型")
        .def("shape", &Tensor::shape, "获取张量的形状")
        .def("device", &Tensor::device, "获取张量所在的设备")
        .def("data_ptr", [](const Tensor& self) { 
            return reinterpret_cast<uintptr_t>(self.data_ptr()); 
        }, "获取张量数据指针地址（返回整数）")
        // 数据导出接口
        .def("dump", &Tensor::dump,
             py::arg("file_path"),
             "将张量数据导出到文本文件\n\n"
             "参数:\n"
             "    file_path: 输出文件路径\n\n"
             "注意:\n"
             "    文件格式为文本格式，包含元数据（形状、数据类型等）和数据。\n"
             "    对于 GPU 张量，数据会先复制到 CPU。");

    // ============================================================================
    // AttentionLayer 类绑定
    // ============================================================================
    py::class_<AttentionLayer>(m, "AttentionLayer", "注意力层，实现 FlashInfer attention 计算")
        .def(py::init([](const std::string& config_path) {
                 EngineConfig config(config_path);
                 return std::make_unique<AttentionLayer>(config);
             }),
             py::arg("config_path"),
             "创建注意力层\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "示例:\n"
             "    layer = AttentionLayer(\"/path/to/engine_config.json\")")
        .def("forward_prefill", [](AttentionLayer& self,
                                   const Tensor& q,
                                   const Tensor& k,
                                   const Tensor& v,
                                   Tensor& o,
                                   bool causal = true,
                                   uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                self.forward_prefill(q, k, v, o, causal, stream);
            },
            py::arg("q"),
            py::arg("k"),
            py::arg("v"),
            py::arg("o"),
            py::arg("causal") = true,
            py::arg("stream") = 0,
            "执行 Prefill 模式的前向传播\n\n"
            "参数:\n"
            "    q: 查询张量，形状 [qo_len, num_qo_heads, head_dim]\n"
            "    k: 键张量，形状 [kv_len, num_kv_heads, head_dim]\n"
            "    v: 值张量，形状 [kv_len, num_kv_heads, head_dim]\n"
            "    o: 输出张量，形状 [qo_len, num_qo_heads, head_dim]（用于存储输出）\n"
            "    causal: 是否使用因果 mask（默认: True）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 stream.synchronize() 或 cudaDeviceSynchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream 和 causal mask\n"
            "    layer.forward_prefill(q, k, v, o)\n\n"
            "    # 使用 non-causal mask\n"
            "    layer.forward_prefill(q, k, v, o, causal=False)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_prefill(q, k, v, o, True, torch.cuda.current_stream().cuda_stream)")
        .def("forward_decode", [](AttentionLayer& self,
                                  const Tensor& q,
                                  const Tensor& k,
                                  const Tensor& v,
                                  Tensor& o,
                                  uintptr_t stream_ptr = 0,
                                  uint32_t max_kv_len = 0,
                                  const Tensor* d_kv_len_tensor = nullptr) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                uint32_t* d_kv_len_ptr = nullptr;
                if (d_kv_len_tensor != nullptr) {
                    d_kv_len_ptr = static_cast<uint32_t*>(d_kv_len_tensor->data_ptr());
                }
                self.forward_decode(q, k, v, o, stream, d_kv_len_ptr, max_kv_len);
            },
            py::arg("q"),
            py::arg("k"),
            py::arg("v"),
            py::arg("o"),
            py::arg("stream") = 0,
            py::arg("max_kv_len") = 0,
            py::arg("d_kv_len") = static_cast<const Tensor*>(nullptr),
            R"doc(执行 Decode 模式的前向传播

参数:
    q: 查询张量，形状 [1, num_qo_heads, head_dim]
    k: 键张量（来自 KV cache），形状 [kv_len 或 max_kv_len, num_kv_heads, head_dim]
    v: 值张量（来自 KV cache），形状 [kv_len 或 max_kv_len, num_kv_heads, head_dim]
    o: 输出张量，形状 [1, num_qo_heads, head_dim]（用于存储输出）
    stream: CUDA stream 指针地址（整数），0 表示默认 stream
    max_kv_len: 可选的最大 KV 长度，用于 decode/cuda graph 场景（默认: 0）
    d_kv_len: 可选的 device-side KV 长度张量（shape=[1]，int32/uint32），
              与 max_kv_len 搭配用于 graph-stable decode（默认: None）

注意:
    - 函数本身是异步的，kernel 启动后立即返回
    - 如果需要在函数返回后立即使用结果，需要手动调用 stream.synchronize() 或 cudaDeviceSynchronize()

示例:
    # 使用默认 stream
    layer.forward_decode(q, k, v, o)

    # 使用自定义 stream
    layer.forward_decode(q, k, v, o, stream)

    # 使用 decode/cuda graph 的 max_kv_len
    layer.forward_decode(q, k, v, o, stream, max_kv_len)

    # 使用 device-side d_kv_len + max_kv_len
    layer.forward_decode(q, k_full, v_full, o, stream, max_kv_len, d_kv_len))doc");

    // ============================================================================
    // RMSNormLayer 类绑定
    // ============================================================================
    py::class_<RMSNormLayer>(m, "RMSNormLayer", "RMSNorm 层，实现 FlashInfer RMSNorm 计算")
        .def(py::init([](uint32_t layer_id, const std::string& config_path, const std::string& weight_type) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();
                 
                 // 加载权重
                 WeightLoader& loader = WeightLoader::instance();
                 const std::string safetensors_path = 
                    config.prefill_model_path() + "/model.safetensors";
                 loader.load_weights_from_file(
                     ModelStage::Prefill, safetensors_path, Device::GPU, runtime_device_id);
                 const auto& prefill_weights = loader.get(ModelStage::Prefill);
                 
                 // 尝试加载 decode 权重（如果存在）
                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         const std::string decode_safetensors_path = decode_model_path + "/model.safetensors";
                         try {
                             loader.load_weights_from_file(
                                 ModelStage::Decode,
                                 decode_safetensors_path,
                                 Device::GPU,
                                 runtime_device_id);
                             return loader.get(ModelStage::Decode);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();

                 NormWeightType wt = NormWeightType::Input;
                 if (weight_type == "post_attention") {
                     wt = NormWeightType::PostAttention;
                 } else if (weight_type == "final") {
                     wt = NormWeightType::Final;
                 }

                 auto layer = std::make_unique<RMSNormLayer>(layer_id, wt, config);
                 layer->load_weights(prefill_weights, decode_weights);

                 return layer;
             }),
             py::arg("layer_id"),
             py::arg("config_path"),
             py::arg("weight_type") = "input",
             "创建 RMSNorm 层\n\n"
             "参数:\n"
             "    layer_id: 层 ID（用于确定加载哪个层的权重）\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "示例:\n"
             "    layer = RMSNormLayer(layer_id=0, config_path=\"/path/to/engine_config.json\")")
        .def("forward_rmsnorm", [](RMSNormLayer& self,
                                   const Tensor& input,
                                   Tensor& output,
                                   uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                self.forward_rmsnorm(input, output, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            "执行单纯的 RMSNorm 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, hidden_size]\n"
            "    output: 输出张量，形状 [batch_size, hidden_size]（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    layer.forward_rmsnorm(input, output)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_rmsnorm(input, output, torch.cuda.current_stream().cuda_stream)")
        .def("forward_fused_add_rmsnorm", [](RMSNormLayer& self,
                                             Tensor& inout,
                                             Tensor& residual,
                                             uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                self.forward_fused_add_rmsnorm(inout, residual, stream);
            },
            py::arg("inout"),
            py::arg("residual"),
            py::arg("stream") = 0,
            "执行 Fused Add + RMSNorm 前向传播\n\n"
            "功能:\n"
            "    1. residual = inout + residual (相加结果存到 residual)\n"
            "    2. inout = RMSNorm(residual) (RMSNorm 结果存到 inout)\n\n"
            "参数:\n"
            "    inout: 输入/输出张量，形状 [batch_size, hidden_size]（会被修改为 RMSNorm(inout + residual)）\n"
            "    residual: 残差张量，形状 [batch_size, hidden_size]（会被修改为 inout + residual）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - inout 和 residual 会被原地修改\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    layer.forward_fused_add_rmsnorm(inout, residual)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_fused_add_rmsnorm(inout, residual, torch.cuda.current_stream().cuda_stream)");

    // ============================================================================
    // ActivationLayer 类绑定
    // ============================================================================
    py::class_<ActivationLayer>(m, "ActivationLayer", "激活层，实现 FlashInfer SiLU and Mul 计算")
        .def(py::init([](const std::string& config_path) {
                 EngineConfig config(config_path);
                 return std::make_unique<ActivationLayer>(config);
             }),
             py::arg("config_path"),
             "创建激活层\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "注意:\n"
             "    模型配置中的 hidden_act 必须为 \"silu\"，当前仅支持 SiLU 激活函数。\n\n"
             "示例:\n"
             "    layer = ActivationLayer(\"/path/to/engine_config.json\")")
        .def("forward_silu_and_mul", [](ActivationLayer& self,
                                        const Tensor& input,
                                        Tensor& output,
                                        uintptr_t stream_ptr = 0,
                                        const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_silu_and_mul(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行 SiLU and Mul 前向传播\n\n"
            "功能:\n"
            "    计算: output = silu(input[..., :hidden_size]) * input[..., hidden_size:]\n"
            "    其中 input 的前半部分是 gate projection，后半部分是 up projection\n\n"
            "参数:\n"
            "    input: 输入张量，形状 (..., 2 * hidden_size)\n"
            "           最后一维必须是 2 * hidden_size（前半部分是 gate，后半部分是 up）\n"
            "    output: 输出张量，形状 (..., hidden_size)（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 算子阶段，\"Prefill\" 或 \"Decode\"，默认 \"Prefill\"\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n"
            "    - 输入和输出张量的 dtype 必须相同（支持 Float16 和 BFloat16）\n"
            "    - 输入张量的最后一维必须等于 2 * hidden_size（从模型配置中读取）\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    # input shape: [batch_size, 2 * hidden_size]\n"
            "    # output shape: [batch_size, hidden_size]\n"
            "    layer.forward_silu_and_mul(input, output)\n"
            "    layer.forward_silu_and_mul(input, output, stage=\"Decode\")\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_silu_and_mul(input, output, torch.cuda.current_stream().cuda_stream)")
        .def("forward_silu_and_mul_up_gate", [](ActivationLayer& self,
                                                const Tensor& input,
                                                Tensor& output,
                                                uintptr_t stream_ptr = 0,
                                                const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_silu_and_mul_up_gate(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行 SiLU and Mul 前向传播（输入布局为 [up, gate]）\n\n"
            "功能:\n"
            "    计算: output = silu(input[..., hidden_size:]) * input[..., :hidden_size]\n"
            "    其中 input 的前半部分是 up projection，后半部分是 gate projection\n\n"
            "参数:\n"
            "    input: 输入张量，形状 (..., 2 * hidden_size)\n"
            "    output: 输出张量，形状 (..., hidden_size)\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 算子阶段，\"Prefill\" 或 \"Decode\"，默认 \"Prefill\"");

    // ============================================================================
    // SamplerLayer 类绑定
    // ============================================================================
    py::class_<SamplerLayer>(m, "SamplerLayer", "采样层，实现 FlashInfer 采样计算")
        .def(py::init([](const std::string& config_path) {
                 EngineConfig config(config_path);
                 auto layer = std::make_unique<SamplerLayer>(config);
                 std::unordered_map<std::string, Tensor> empty_weights;
                 layer->load_weights(empty_weights, empty_weights);
                 return layer;
             }),
             py::arg("config_path"),
             "创建采样层\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "示例:\n"
             "    layer = SamplerLayer(\"/path/to/engine_config.json\")")
        .def("forward", [](SamplerLayer& self,
                                    const Tensor& logits,  // 按值传递，允许 move
                                    Tensor& token_ids,
                                    uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                // 直接调用 forward_sampling，避免创建 map 的开销
                self.forward_sampling(logits, token_ids, stream);
            },
            py::arg("logits"),
            py::arg("token_ids"),
            py::arg("stream") = 0,
            "执行采样前向传播\n\n"
            "功能:\n"
            "    从 logits 中采样 token IDs，使用 temperature scaling 和 Gumbel-max 采样\n\n"
            "参数:\n"
            "    logits: 输入 logits 张量，形状 [batch_size, vocab_size]，dtype 必须是 Float32\n"
            "    token_ids: 输出 token IDs 张量，形状 [batch_size]，dtype 必须是 Int32（会被修改）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - token_ids 会被原地修改为采样结果\n"
            "    - logits 的 dtype 必须是 Float32（flashinfer 库的限制）\n"
            "    - token_ids 的 dtype 必须是 Int32\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    # logits shape: [batch_size, vocab_size], dtype: float32\n"
            "    # token_ids shape: [batch_size], dtype: int32\n"
            "    layer.forward(logits, token_ids)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward(logits, token_ids, torch.cuda.current_stream().cuda_stream)");

    // ============================================================================
    // LinearLayer 类绑定
    // ============================================================================
    py::class_<LinearLayer>(m, "LinearLayer", "线性层，支持 FP16/BF16 和 INT4 量化")
        // 构造函数: 接受 layer_prefix, config_path, in_features, out_features
        .def(py::init([](const std::string& layer_prefix, const std::string& config_path, 
                         uint32_t in_features, uint32_t out_features) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();
                 
                 // 加载权重
                 WeightLoader& loader = WeightLoader::instance();
                 const auto& prefill_weights = load_stage_weights_for_debug_layer(
                     loader, config, ModelStage::Prefill, runtime_device_id);
                 
                 // 尝试加载 decode 权重（如果存在）
                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         try {
                             return load_stage_weights_for_debug_layer(
                                 loader, config, ModelStage::Decode, runtime_device_id, &prefill_weights);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();
                 
                 auto layer = std::make_unique<LinearLayer>(layer_prefix, config, in_features, out_features);
                 layer->load_weights(prefill_weights, decode_weights);
                 
                 return layer;
             }),
             py::arg("layer_prefix"),
             py::arg("config_path"),
             py::arg("in_features"),
             py::arg("out_features"),
             "创建线性层\n\n"
             "参数:\n"
             "    layer_prefix: 层名称前缀（例如：\"model.layers.0.mlp.gate_proj\"，不需要转义）\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n"
             "    in_features: 输入特征数\n"
             "    out_features: 输出特征数\n\n"
             "示例:\n"
             "    layer = LinearLayer(\"model.layers.0.mlp.gate_proj\", \"/path/to/engine_config.json\", 4096, 11008)")
        .def("forward_fp16_bf16", [](LinearLayer& self,
                                     const Tensor& input,
                                     Tensor& output,
                                     uintptr_t stream_ptr = 0,
                                     const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_fp16_bf16(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行 FP16/BF16 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, in_features]，dtype 为 Float16 或 BFloat16\n"
            "    output: 输出张量，形状 [batch_size, out_features]，dtype 为 Float16 或 BFloat16（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"（默认: \"Prefill\"）\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream 和 Prefill 阶段\n"
            "    layer.forward_fp16_bf16(input, output)\n\n"
            "    # 使用 Decode 阶段\n"
            "    layer.forward_fp16_bf16(input, output, 0, \"Decode\")")
        .def("forward_int4_groupwise", [](LinearLayer& self,
                                          const Tensor& input,
                                          Tensor& output,
                                          uintptr_t stream_ptr = 0,
                                          const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_int4_groupwise(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行 INT4 Groupwise 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, in_features]，dtype 为 Float16\n"
            "    output: 输出张量，形状 [batch_size, out_features]，dtype 为 Float16（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"（默认: \"Prefill\"）\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n"
            "    - 权重必须是 INT4 groupwise 量化的（包含 .qweight 和 .scaling_factors）\n\n"
            "示例:\n"
            "    # 使用默认 stream 和 Prefill 阶段\n"
            "    layer.forward_int4_groupwise(input, output)\n\n"
            "    # 使用 Decode 阶段\n"
            "    layer.forward_int4_groupwise(input, output, 0, \"Decode\")")
        .def("debug_cached_impl_info", [](LinearLayer& self,
                                          const std::string& stage_str = "Decode",
                                          int32_t m = 1) {
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                return self.debug_cached_impl_info(stage, m).dump();
            },
            py::arg("stage") = "Decode",
            py::arg("m") = 1,
            "返回当前 layer 最近一次命中的缓存 impl / algo 调试信息（JSON 字符串）。\n\n"
            "参数:\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"\n"
            "    m: Prefill 阶段对应 batch/seq 维度；Decode 阶段通常固定为 1\n")
        .def("debug_enumerate_cublaslt_candidates", [](LinearLayer& self,
                                                       const Tensor& input,
                                                       Tensor& output,
                                                       const std::string& stage_str = "Prefill",
                                                       int32_t max_algo_ids = 64,
                                                       int32_t top_k = 128) {
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                return self.debug_enumerate_cublaslt_candidates(
                    input, output, stage, max_algo_ids, top_k).dump();
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stage") = "Prefill",
            py::arg("max_algo_ids") = 64,
            py::arg("top_k") = 128,
            "枚举当前 linear shape 可用的 cublasLt low-level algo config（JSON 字符串）。\n\n"
            "参数:\n"
            "    input: 输入张量\n"
            "    output: 输出张量\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"\n"
            "    max_algo_ids: 最多枚举多少个 algo_id\n"
            "    top_k: 最多返回多少个基于 algo_id 受限 heuristic 的候选\n")
        .def("debug_describe_cublaslt_algo", [](LinearLayer& self,
                                                const Tensor& input,
                                                Tensor& output,
                                                int32_t algo_id,
                                                const std::string& stage_str = "Prefill") {
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                return self.debug_describe_cublaslt_algo(input, output, algo_id, stage).dump();
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("algo_id"),
            py::arg("stage") = "Prefill",
            "返回给定 algo_id 的 cublasLt config 与 capability 信息（JSON 字符串）。\n\n"
            "参数:\n"
            "    input: 输入张量\n"
            "    output: 输出张量\n"
            "    algo_id: 目标 cublasLt algo id\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"\n");

    // ============================================================================
    // FusedQKVLinearLayer 类绑定
    // ============================================================================
    py::class_<FusedQKVLinearLayer, LinearLayer>(m, "FusedQKVLinearLayer", "融合的 QKV 线性层，将 Q、K、V 三个投影合并为一个")
        // 构造函数: 接受 layer_prefix_base, config_path, in_features, q_out_features, k_out_features, v_out_features
        .def(py::init([](const std::string& layer_prefix_base, const std::string& config_path, 
                         uint32_t in_features, uint32_t q_out_features, 
                         uint32_t k_out_features, uint32_t v_out_features) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();
                 
                 // 加载权重
                 WeightLoader& loader = WeightLoader::instance();
                 const auto& prefill_weights = load_stage_weights_for_debug_layer(
                     loader, config, ModelStage::Prefill, runtime_device_id);
                 
                 // 尝试加载 decode 权重（如果存在）
                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         try {
                             return load_stage_weights_for_debug_layer(
                                 loader, config, ModelStage::Decode, runtime_device_id, &prefill_weights);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();
                 
                 auto layer = std::make_unique<FusedQKVLinearLayer>(
                     layer_prefix_base, config, in_features, 
                     q_out_features, k_out_features, v_out_features);
                 layer->load_weights(prefill_weights, decode_weights);
                 
                 return layer;
             }),
             py::arg("layer_prefix_base"),
             py::arg("config_path"),
             py::arg("in_features"),
             py::arg("q_out_features"),
             py::arg("k_out_features"),
             py::arg("v_out_features"),
             "创建融合的 QKV 线性层\n\n"
             "参数:\n"
             "    layer_prefix_base: 层名称基础前缀（例如：\"model.layers.0.attn\"，不需要转义）\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n"
             "    in_features: 输入特征数\n"
             "    q_out_features: Q 投影输出特征数\n"
             "    k_out_features: K 投影输出特征数\n"
             "    v_out_features: V 投影输出特征数\n\n"
             "示例:\n"
             "    layer = FusedQKVLinearLayer(\"model.layers.0.attn\", \"/path/to/engine_config.json\", 4096, 4096, 1024, 1024)")
        .def("forward_fp16_bf16", [](FusedQKVLinearLayer& self,
                                     const Tensor& input,
                                     Tensor& output,
                                     uintptr_t stream_ptr = 0,
                                     const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_fp16_bf16(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行融合 QKV 的 FP16/BF16 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, in_features]，dtype 为 Float16 或 BFloat16\n"
            "    output: 输出张量，形状 [batch_size, q_out + k_out + v_out]，dtype 为 Float16 或 BFloat16（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"（默认: \"Prefill\"）\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n"
            "    - 输出布局: [Q: q_out_features, K: k_out_features, V: v_out_features]\n\n"
            "示例:\n"
            "    # 使用默认 stream 和 Prefill 阶段\n"
            "    layer.forward_fp16_bf16(input, output)\n\n"
            "    # 使用 Decode 阶段\n"
            "    layer.forward_fp16_bf16(input, output, 0, \"Decode\")");

    // ============================================================================
    // FusedGateUpLinearLayer 类绑定
    // ============================================================================
    py::class_<FusedGateUpLinearLayer, LinearLayer>(m, "FusedGateUpLinearLayer", "融合的 Gate+Up 线性层，将 gate_proj 和 up_proj 合并为一个")
        .def(py::init([](const std::string& layer_prefix_base, const std::string& config_path,
                         uint32_t in_features, uint32_t gate_out_features,
                         uint32_t up_out_features) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();

                 WeightLoader& loader = WeightLoader::instance();
                 const auto& prefill_weights = load_stage_weights_for_debug_layer(
                     loader, config, ModelStage::Prefill, runtime_device_id);

                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         try {
                             return load_stage_weights_for_debug_layer(
                                 loader, config, ModelStage::Decode, runtime_device_id, &prefill_weights);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();

                 auto layer = std::make_unique<FusedGateUpLinearLayer>(
                     layer_prefix_base, config, in_features,
                     gate_out_features, up_out_features);
                 layer->load_weights(prefill_weights, decode_weights);

                 return layer;
             }),
             py::arg("layer_prefix_base"),
             py::arg("config_path"),
             py::arg("in_features"),
             py::arg("gate_out_features"),
             py::arg("up_out_features"),
             "创建融合的 Gate+Up 线性层\n\n"
             "参数:\n"
             "    layer_prefix_base: 层名称基础前缀（例如：\"model.layers.0.mlp\"）\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n"
             "    in_features: 输入特征数\n"
             "    gate_out_features: Gate 投影输出特征数\n"
             "    up_out_features: Up 投影输出特征数\n\n"
             "示例:\n"
             "    layer = FusedGateUpLinearLayer(\"model.layers.0.mlp\", \"/path/to/engine_config.json\", 4096, 11008, 11008)")
        .def("forward_fp16_bf16", [](FusedGateUpLinearLayer& self,
                                     const Tensor& input,
                                     Tensor& output,
                                     uintptr_t stream_ptr = 0,
                                     const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_fp16_bf16(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行融合 Gate+Up 的 FP16/BF16 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, in_features]，dtype 为 Float16 或 BFloat16\n"
            "    output: 输出张量，形状 [batch_size, gate_out + up_out]，dtype 为 Float16 或 BFloat16（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"（默认: \"Prefill\"）\n\n"
            "注意:\n"
            "    - 输出布局: [Up: up_out_features, Gate: gate_out_features]\n\n"
            "示例:\n"
            "    layer.forward_fp16_bf16(input, output)\n")
        .def("try_forward_decode_swiglu_fused", [](FusedGateUpLinearLayer& self,
                                                   const Tensor& input,
                                                   Tensor& output,
                                                   uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                return self.try_forward_decode_swiglu_fused(input, output, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            "尝试执行 decode-only 的 fused gate_up + SwiGLU\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [1, in_features]\n"
            "    output: 输出张量，形状 [1, up_out_features]\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "返回:\n"
            "    如果当前设备/shape/dtype 支持 fused fast path，则返回 True 并写入 output；否则返回 False。");

    // ============================================================================
    // LMHeadLinearLayer 类绑定
    // ============================================================================
    py::class_<LMHeadLinearLayer, LinearLayer>(m, "LMHeadLinearLayer", "LM Head 线性层，支持 lm_head.weight 或 model.embed_tokens.weight (tied)")
        .def(py::init([](const std::string& config_path,
                         uint32_t in_features,
                         uint32_t out_features,
                         const std::string& layer_prefix) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();

                 WeightLoader& loader = WeightLoader::instance();
                 const auto& prefill_weights = load_stage_weights_for_debug_layer(
                     loader, config, ModelStage::Prefill, runtime_device_id);

                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         try {
                             return load_stage_weights_for_debug_layer(
                                 loader, config, ModelStage::Decode, runtime_device_id, &prefill_weights);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();

                 auto layer = std::make_unique<LMHeadLinearLayer>(layer_prefix, config, in_features, out_features);
                 layer->load_weights(prefill_weights, decode_weights);

                 return layer;
             }),
             py::arg("config_path"),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("layer_prefix") = "lm_head",
             "创建 LM Head 线性层\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n"
             "    in_features: 输入特征数（hidden_size）\n"
             "    out_features: 输出特征数（vocab_size）\n"
             "    layer_prefix: 权重名称前缀，默认 \"lm_head\"。会依次尝试 layer_prefix.weight、model.layer_prefix.weight、model.embed_tokens.weight\n\n"
             "示例:\n"
             "    layer = LMHeadLinearLayer(\"/path/to/engine_config.json\", 4096, 151936)\n"
             "    layer = LMHeadLinearLayer(config_path, 4096, 151936, layer_prefix=\"lm_head\")")
        .def("forward_fp16_bf16", [](LMHeadLinearLayer& self,
                                     const Tensor& input,
                                     Tensor& output,
                                     uintptr_t stream_ptr = 0,
                                     const std::string& stage_str = "Prefill") {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                ModelStage stage = (stage_str == "Decode") ? ModelStage::Decode : ModelStage::Prefill;
                self.forward_fp16_bf16(input, output, stream, stage);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("stream") = 0,
            py::arg("stage") = "Prefill",
            "执行 LM Head 的 FP16/BF16 前向传播\n\n"
            "参数:\n"
            "    input: 输入张量，形状 [batch_size, hidden_size]，dtype 为 Float16 或 BFloat16\n"
            "    output: 输出张量，形状 [batch_size, vocab_size]，dtype 为 Float16 或 BFloat16（用于存储输出）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n"
            "    stage: 模型阶段，\"Prefill\" 或 \"Decode\"（默认: \"Prefill\"）\n\n"
            "示例:\n"
            "    layer.forward_fp16_bf16(input, output)\n");

    // ============================================================================
    // EmbedHeadLayer 类绑定
    // ============================================================================
    py::class_<EmbedHeadLayer>(m, "EmbedHeadLayer", "Embedding 层，实现 token embedding 查找")
        .def(py::init([](const std::string& config_path) {
                 EngineConfig config(config_path);
                 const int32_t runtime_device_id = config.runtime_device_id();
                 
                 // 加载权重
                 WeightLoader& loader = WeightLoader::instance();
                 const std::string safetensors_path =
                    config.prefill_model_path() + "/model.safetensors";
                 loader.load_weights_from_file(
                     ModelStage::Prefill, safetensors_path, Device::GPU, runtime_device_id, true);
                 const auto& prefill_weights = loader.get(ModelStage::Prefill);
                 
                 // 尝试加载 decode 权重（如果存在）
                 const auto& decode_weights = [&]() -> const std::unordered_map<std::string, Tensor>& {
                     std::string decode_model_path = config.decode_model_path();
                     if (!decode_model_path.empty() && decode_model_path != config.prefill_model_path()) {
                         const std::string decode_safetensors_path = decode_model_path + "/model.safetensors";
                         try {
                             loader.load_weights_from_file(
                                 ModelStage::Decode,
                                 decode_safetensors_path,
                                 Device::GPU,
                                 runtime_device_id,
                                 true);
                             return loader.get(ModelStage::Decode);
                         } catch (...) {
                             return prefill_weights;
                         }
                     } else {
                         return prefill_weights;
                     }
                 }();
                 
                 auto layer = std::make_unique<EmbedHeadLayer>(config);
                 layer->load_weights(prefill_weights, decode_weights);

                 return layer;
             }),
             py::arg("config_path"),
             "创建 Embedding 层\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "示例:\n"
             "    layer = EmbedHeadLayer(\"/path/to/engine_config.json\")")
        .def("forward_for_tokens", [](EmbedHeadLayer& self,
                                      const Tensor& token_ids,
                                      Tensor& output,
                                      uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                self.forward_for_tokens(token_ids, output, stream);
            },
            py::arg("token_ids"),
            py::arg("output"),
            py::arg("stream") = 0,
            "执行 token embedding 查找\n\n"
            "参数:\n"
            "    token_ids: 输入 token IDs 张量，形状 [batch_size, seq_len]，dtype 必须是 Int32\n"
            "    output: 输出 embedding 张量，形状 [batch_size, seq_len, hidden_size]，dtype 必须与 embedding table 一致（会被修改）\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - output 会被原地修改为 embedding 结果\n"
            "    - token_ids 的 dtype 必须是 Int32\n"
            "    - output 的 dtype 必须与 embedding table 的 dtype 一致（Float16 或 BFloat16）\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    # token_ids shape: [batch_size, seq_len], dtype: int32\n"
            "    # output shape: [batch_size, seq_len, hidden_size], dtype: float16 or bfloat16\n"
            "    layer.forward_for_tokens(token_ids, output)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_for_tokens(token_ids, output, torch.cuda.current_stream().cuda_stream)")
        .def("forward_for_embeddings", [](EmbedHeadLayer& self,
                                          const Tensor& token_ids,
                                          const Tensor& embeddings,
                                          Tensor& output,
                                          int32_t embed_token_id = -1,
                                          uintptr_t stream_ptr = 0) {
                cudaStream_t stream = (stream_ptr == 0) ? nullptr : reinterpret_cast<cudaStream_t>(stream_ptr);
                self.forward_for_embeddings(token_ids, embeddings, output, embed_token_id, stream);
            },
            py::arg("token_ids"),
            py::arg("embeddings"),
            py::arg("output"),
            py::arg("embed_token_id") = -1,
            py::arg("stream") = 0,
            "执行 token embedding 查找并插入自定义 embeddings\n\n"
            "参数:\n"
            "    token_ids: 输入 token IDs 张量，形状 [batch_size, seq_len]，dtype 必须是 Int32\n"
            "             如果 embed_token_id >= 0，则 token_ids 中值为 [embed_token_id, embed_token_id + num_custom_embeddings) 的 token\n"
            "             将使用自定义 embeddings，其他 token 使用标准的 embedding_table\n"
            "    embeddings: 自定义 embeddings 张量，形状 [num_custom_embeddings, hidden_size]，dtype 必须与 embedding table 一致\n"
            "    output: 输出 embedding 张量，形状 [batch_size, seq_len, hidden_size]，dtype 必须与 embedding table 一致（会被修改）\n"
            "    embed_token_id: 自定义 embedding 的起始 token ID。如果 >= 0，则 token_ids 中值为 [embed_token_id, embed_token_id + num_custom_embeddings)\n"
            "                    的 token 将使用自定义 embeddings。默认值为 -1，表示不使用自定义 embeddings\n"
            "    stream: CUDA stream 指针地址（整数），0 表示默认 stream\n\n"
            "注意:\n"
            "    - 函数本身是异步的，kernel 启动后立即返回\n"
            "    - output 会被原地修改为 embedding 结果\n"
            "    - 如果 token_id == embed_token_id + i (i < num_custom_embeddings)，则使用 embeddings[i]\n"
            "    - 如果需要在函数返回后立即使用结果，需要手动调用 torch.cuda.synchronize()\n\n"
            "示例:\n"
            "    # 使用默认 stream\n"
            "    # token_ids shape: [batch_size, seq_len], dtype: int32\n"
            "    # embeddings shape: [num_custom_embeddings, hidden_size], dtype: float16 or bfloat16\n"
            "    # output shape: [batch_size, seq_len, hidden_size], dtype: float16 or bfloat16\n"
            "    # 假设 embed_token_id = 32000，num_custom_embeddings = 4\n"
            "    # 则 token_ids 中值为 32000, 32001, 32002, 32003 的 token 将使用自定义 embeddings\n"
            "    layer.forward_for_embeddings(token_ids, embeddings, output, embed_token_id=32000)\n\n"
            "    # 使用自定义 stream\n"
            "    layer.forward_for_embeddings(token_ids, embeddings, output, embed_token_id=32000, stream=torch.cuda.current_stream().cuda_stream)");

    // ============================================================================
    // ModelTensors 常量（调试用）
    // ============================================================================
    py::dict model_tensors;
    model_tensors["NORM_OUTPUT"] = ModelTensors::NORM_OUTPUT;
    model_tensors["QKV_PROJ_OUTPUT"] = ModelTensors::QKV_PROJ_OUTPUT;
    model_tensors["ATTENTION_OUTPUT"] = ModelTensors::ATTENTION_OUTPUT;
    model_tensors["MLP_ACTIVATION_INPUT"] = ModelTensors::MLP_ACTIVATION_INPUT;
    model_tensors["MLP_INTERMEDIATE"] = ModelTensors::MLP_INTERMEDIATE;
    m.attr("ModelTensors") = model_tensors;

    // ============================================================================
    // WeightLoader 类绑定
    // ============================================================================
    // 注意：WeightLoader 是单例类
    py::class_<WeightLoader, std::unique_ptr<WeightLoader, py::nodelete>>(
        m, "WeightLoader", "权重加载器（单例模式）")
        .def_static("instance", &WeightLoader::instance,
                    py::return_value_policy::reference,
                    "获取单例实例\n\n"
                    "返回:\n"
                    "    WeightLoader& 单例引用\n\n"
                    "示例:\n"
                    "    loader = edge_fm.WeightLoader.instance()")
        .def("get", [](WeightLoader& self, ModelStage cache_key) -> py::dict {
                const auto& weights = self.get(cache_key);
                py::dict result;
                for (const auto& [name, tensor] : weights) {
                    // 直接返回 tensor 的引用，不克隆
                    // 使用 reference_internal 策略：Python 对象的生命周期依赖于 WeightLoader（单例）
                    result[py::str(name)] = py::cast(&tensor, py::return_value_policy::reference_internal, py::cast(&self));
                }
                return result;
            },
            py::arg("cache_key"),
            "获取缓存的权重\n\n"
            "参数:\n"
            "    cache_key: 缓存键（ModelStage 类型），可以是 ModelStage.Prefill 或 ModelStage.Decode\n\n"
            "返回:\n"
            "    dict[str, Tensor]: 权重映射表，键为权重名称，值为 Tensor 对象\n\n"
            "抛出:\n"
            "    ConfigurationError: 如果 cache_key 对应的权重不存在\n\n"
            "示例:\n"
            "    loader = edge_fm.WeightLoader.instance()\n"
            "    weights = loader.get(edge_fm.ModelStage.Prefill)")
        .def("clear_stage", &WeightLoader::clear_stage,
            py::arg("cache_key"),
            "清空指定 stage 的权重缓存\n\n"
            "参数:\n"
            "    cache_key: 缓存键（ModelStage 类型）\n\n"
            "示例:\n"
            "    loader = edge_fm.WeightLoader.instance()\n"
            "    loader.clear_stage(edge_fm.ModelStage.Prefill)")
        .def("load_weights_from_file", &WeightLoader::load_weights_from_file,
            py::arg("cache_key"),
            py::arg("safetensors_file"),
            py::arg("device"),
            py::arg("device_id"),
            py::arg("overwrite_if_exists") = false,
            py::arg("weight_filter") = py::none(),
            py::arg("key_mapper") = py::none(),
            "从 safetensors 文件加载权重\n\n"
            "参数:\n"
            "    cache_key: 缓存键（ModelStage 类型）\n"
            "    safetensors_file: safetensors 文件路径\n"
            "    device: 设备类型（Device.CPU 或 Device.GPU）\n"
            "    device_id: 设备 ID（默认: 0）\n"
            "    overwrite_if_exists: 如果为 True，当发现重复的 tensor 时会覆盖；如果为 False，会记录警告但不覆盖（默认: False）\n"
            "    weight_filter: 可选的权重名称过滤器（可调用对象），如果提供，只加载匹配过滤器的权重；如果为 None，加载所有权重（默认: None）\n"
            "    key_mapper: 可选的键映射函数，将原始权重名称映射为缓存键名（默认: None）\n\n"
            "抛出:\n"
            "    ConfigurationError: 如果文件加载失败\n"
            "    DeviceError: 如果设备操作失败\n\n"
            "注意:\n"
            "    加载的权重会存储到全局缓存中，使用 cache_key 作为键\n"
            "    如果同一个文件已经被加载过，会直接返回，不会重复加载\n"
            "    如果同一个 cache_key 对应的缓存已存在，新加载的权重会合并到现有缓存中\n"
            "    当发现重复的 tensor 名称时，如果 overwrite_if_exists 为 False，会记录警告信息但不覆盖；如果为 True，会覆盖现有值\n\n"
            "示例:\n"
            "    loader = edge_fm.WeightLoader.instance()\n"
            "    loader.load_weights_from_file(\n"
            "        edge_fm.ModelStage.Prefill,\n"
            "        \"/path/to/model.safetensors\",\n"
            "        edge_fm.Device.GPU,\n"
            "        device_id=0\n"
            "    )");

    // ============================================================================
    // KVManager 相关结构体和枚举绑定
    // ============================================================================
    py::enum_<AttentionType>(m, "AttentionType", "注意力类型枚举")
        .value("MHA", AttentionType::MHA, "Multi-Head Attention")
        .value("GQA", AttentionType::GQA, "Grouped Query Attention")
        .value("MQA", AttentionType::MQA, "Multi-Query Attention")
        .value("MLA", AttentionType::MLA, "Multi-head Latent Attention")
        .export_values();

    py::class_<KVSlotStatus>(m, "KVSlotStatus", "KV Cache Slot 状态信息")
        .def_readonly("request_id", &KVSlotStatus::request_id, "请求 ID")
        .def_readonly("prefix_token_ids", &KVSlotStatus::prefix_token_ids, "Prefix token IDs")
        .def_readonly("prefix_size", &KVSlotStatus::prefix_size, "Prefix 大小")
        .def_readonly("max_tokens", &KVSlotStatus::max_tokens, "最大 token 数")
        .def_readonly("allocated_size", &KVSlotStatus::allocated_size, "已分配的内存大小（字节）");

    py::class_<KVManagerStatus>(m, "KVManagerStatus", "KV Manager 状态信息")
        .def_readonly("slots", &KVManagerStatus::slots, "所有 slot 的状态列表")
        .def_readonly("device", &KVManagerStatus::device, "设备类型")
        .def_readonly("device_id", &KVManagerStatus::device_id, "设备 ID");

    // ============================================================================
    // KVManager 类绑定
    // ============================================================================
    py::class_<KVManager>(m, "KVManager", "KV Cache 管理器，负责管理多个请求的 KV cache 内存分配")
        .def(py::init([](const std::string& config_path) {
                 EngineConfig config(config_path);
                 return std::make_unique<KVManager>(config);
             }),
             py::arg("config_path"),
             "创建 KV Manager\n\n"
             "参数:\n"
             "    config_path: 引擎配置文件路径（JSON 文件）\n\n"
             "注意:\n"
             "    配置文件必须包含 kvcache 配置，包括 attention_type、dtype 和 requests 数组\n\n"
             "示例:\n"
             "    kv_manager = edge_fm.KVManager(\"/path/to/engine_config.json\")")
        .def("get_read_kvcache", [](const KVManager& self, int32_t request_id) {
                std::vector<void*> ptrs = self.get_read_kvcache(request_id);
                // 将 void* 指针转换为整数地址（Python 可以处理）
                std::vector<uintptr_t> ptr_ints;
                ptr_ints.reserve(ptrs.size());
                for (void* ptr : ptrs) {
                    ptr_ints.push_back(reinterpret_cast<uintptr_t>(ptr));
                }
                return ptr_ints;
            },
            py::arg("request_id"),
            "获取指定请求的所有层的 KV cache 读指针\n\n"
            "参数:\n"
            "    request_id: 请求 ID\n\n"
            "返回:\n"
            "    list[int]: 每层的 KV cache 读指针地址列表（整数地址）\n\n"
            "抛出:\n"
            "    InvalidRequestError: 如果 request_id 无效\n\n"
            "注意:\n"
            "    返回的是指针地址（整数），可以用于后续的内存操作\n\n"
            "示例:\n"
            "    read_ptrs = kv_manager.get_read_kvcache(request_id=0)")
        .def("get_write_kvcache", [](const KVManager& self, int32_t request_id) {
                std::vector<void*> ptrs = self.get_write_kvcache(request_id);
                // 将 void* 指针转换为整数地址（Python 可以处理）
                std::vector<uintptr_t> ptr_ints;
                ptr_ints.reserve(ptrs.size());
                for (void* ptr : ptrs) {
                    ptr_ints.push_back(reinterpret_cast<uintptr_t>(ptr));
                }
                return ptr_ints;
            },
            py::arg("request_id"),
            "获取指定请求的所有层的 KV cache 写指针\n\n"
            "参数:\n"
            "    request_id: 请求 ID\n\n"
            "返回:\n"
            "    list[int]: 每层的 KV cache 写指针地址列表（整数地址）\n\n"
            "抛出:\n"
            "    InvalidRequestError: 如果 request_id 无效\n\n"
            "注意:\n"
            "    写指针指向 prefill 后的第一个 token 位置（如果有 prefix，则指向 prefix 之后）\n"
            "    返回的是指针地址（整数），可以用于后续的内存操作\n\n"
            "示例:\n"
            "    write_ptrs = kv_manager.get_write_kvcache(request_id=0)")
        .def("get_status", &KVManager::get_status,
            "获取 KV Manager 的状态信息\n\n"
            "返回:\n"
            "    KVManagerStatus: 包含所有 slot 状态和设备信息的对象\n\n"
            "示例:\n"
            "    status = kv_manager.get_status()\n"
            "    print(f\"Device: {status.device}, Device ID: {status.device_id}\")\n"
            "    for slot in status.slots:\n"
            "        print(f\"Request {slot.request_id}: max_tokens={slot.max_tokens}\")")
        .def("is_request_valid", &KVManager::is_request_valid,
            py::arg("request_id"),
            "检查请求 ID 是否有效\n\n"
            "参数:\n"
            "    request_id: 请求 ID\n\n"
            "返回:\n"
            "    bool: 如果 request_id 存在且有效返回 True，否则返回 False\n\n"
            "示例:\n"
            "    if kv_manager.is_request_valid(request_id=0):\n"
            "        print(\"Request is valid\")");

    // ============================================================================
    // Request 类绑定
    // ============================================================================
    py::class_<Request>(m, "Request", "推理请求类")
        .def(py::init<int32_t, const std::vector<int32_t>&>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             "创建仅包含 token IDs 的请求")
        .def(py::init<int32_t, const std::vector<int32_t>&, const Tensor&, int32_t>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             py::arg("embedding"),
             py::arg("embed_token_id") = -1,
             "创建包含图像嵌入的请求")
        .def(py::init<int32_t, const std::vector<int32_t>&, const Tensor&, int32_t, const Tensor&>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             py::arg("embedding"),
             py::arg("embed_token_id"),
             py::arg("position_ids"),
             "创建包含图像嵌入和 M-RoPE position_ids 的请求")
        .def(py::init<int32_t, const std::vector<int32_t>&, const Tensor&, int32_t, const Tensor&, const std::vector<int32_t>&>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             py::arg("embedding"),
             py::arg("embed_token_id"),
             py::arg("position_ids"),
             py::arg("mrope_last_pos"),
             "创建包含图像嵌入、M-RoPE position_ids 和预计算 mrope_last_pos 的请求")
        .def("request_id", &Request::request_id,
             "获取请求 ID")
        .def("token_ids", &Request::token_ids,
             "获取 token IDs",
             py::return_value_policy::reference_internal)
        .def("embedding", &Request::embedding,
             "获取 embedding tensor",
             py::return_value_policy::reference_internal)
        .def("embed_token_id", &Request::embed_token_id,
             "获取自定义 embedding 的起始 token ID")
        .def("has_embedding", &Request::has_embedding,
             "检查是否有 embedding")
        .def("position_ids", &Request::position_ids,
             "获取 M-RoPE position_ids tensor",
             py::return_value_policy::reference_internal)
        .def("has_position_ids", &Request::has_position_ids,
             "检查是否有 M-RoPE position_ids")
        .def("mrope_last_pos", &Request::mrope_last_pos,
             "获取预计算的 M-RoPE last_pos",
             py::return_value_policy::reference_internal)
        .def("has_mrope_last_pos", &Request::has_mrope_last_pos,
             "检查是否有预计算的 M-RoPE last_pos")
        .def("set_stop_token_ids", &Request::set_stop_token_ids,
             py::arg("stop_token_ids"),
             "设置自定义停止 token IDs 列表（生成到这些 token 时会提前终止）")
        .def("stop_token_ids", &Request::stop_token_ids,
             "获取自定义停止 token IDs",
             py::return_value_policy::reference_internal)
        .def("set_ignore_stop_tokens", &Request::set_ignore_stop_tokens,
             py::arg("ignore") = true,
             "对齐测试用：忽略 EOS/stop，生成满 max_tokens 步")
        .def("ignore_stop_tokens", &Request::ignore_stop_tokens,
             "是否忽略 stop tokens");

    // ============================================================================
    // Response 类绑定
    // ============================================================================
    py::class_<Response>(m, "Response", "推理响应类")
        .def(py::init<>(), "创建空响应")
        .def("token_ids",
             py::overload_cast<>(&Response::token_ids, py::const_),
             "获取生成的 token IDs",
             py::return_value_policy::reference_internal);

    // ============================================================================
    // EdgeFM 类绑定
    // ============================================================================
    py::class_<EdgeFM>(m, "EdgeFM", "EdgeFM 推理引擎")
        .def(py::init<const std::string&>(),
             py::arg("config_path"),
             "构造 EdgeFM 推理引擎")
        .def_static("from_model",
             [](py::object /*model*/, const std::string& engine_json) {
                 (void)engine_json;
                 throw ConfigurationError(
                     "EdgeFM.from_model(...) has been deprecated. "
                     "Use EdgeFM(config_path) and set model_name explicitly in engine.json.");
             },
             py::arg("model"),
             py::arg("engine_json"),
             "已废弃。请改用 EdgeFM(config_path)，并在 engine.json 中显式设置 model_name。")
        .def("tune", &EdgeFM::tune,
             "对当前 backend 生成或调优执行产物，并将结果持久化缓存。")
        .def("generate", &EdgeFM::generate,
             py::arg("request"),
             "从给定请求生成响应 token")
        .def("last_generate_metrics", &EdgeFM::last_generate_metrics,
             "返回最近一次 generate() 的 stage timing 指标。");

    // ============================================================================
    // M-RoPE standalone function
    // ============================================================================
    m.def("apply_mrope", [](Tensor& q, Tensor& k,
                            const Tensor& position_ids,
                            const Tensor& mrope_section_cumsum,
                            int32_t seq_len, int32_t num_qo_heads, int32_t num_kv_heads,
                            int32_t head_dim, float rope_theta, float rope_scale,
                            const std::string& dtype_str) {
        DType dtype = (dtype_str == "bfloat16") ? DType::BFloat16 : DType::Float16;
        cudaStream_t stream = nullptr;
        AttentionLayer::apply_mrope(q.data_ptr(), k.data_ptr(),
                    static_cast<const int32_t*>(position_ids.data_ptr()),
                    static_cast<const int32_t*>(mrope_section_cumsum.data_ptr()),
                    seq_len, num_qo_heads, num_kv_heads, head_dim,
                    rope_theta, rope_scale, dtype, stream);
        cudaStreamSynchronize(stream);
    },
    py::arg("q"), py::arg("k"),
    py::arg("position_ids"), py::arg("mrope_section_cumsum"),
    py::arg("seq_len"), py::arg("num_qo_heads"), py::arg("num_kv_heads"),
    py::arg("head_dim"), py::arg("rope_theta"), py::arg("rope_scale"),
    py::arg("dtype"));
}
