#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <Python.h>  // For PyCapsule API
#include <cstdint>   // For uintptr_t
#include <edge-fm/edge-fm.h>
#include <edge-fm/core.h>
#include <dlpack/dlpack.h>

namespace py = pybind11;
using namespace edge_fm;

// 异常转换函数
void translate_exception(std::exception_ptr p) {
    try {
        if (p) {
            std::rethrow_exception(p);
        }
    } catch (const DeviceError& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
    } catch (const ConfigurationError& err) {
        PyErr_SetString(PyExc_ValueError, err.what());
    } catch (const ModelNotLoadedError& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
    } catch (const InvalidRequestError& err) {
        PyErr_SetString(PyExc_ValueError, err.what());
    } catch (const OutOfMemoryError& err) {
        PyErr_SetString(PyExc_MemoryError, err.what());
    } catch (const InternalError& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
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

// 从 Python DLPack capsule 创建 Tensor 的辅助函数
Tensor tensor_from_dlpack_capsule(py::object capsule) {
    // 检查是否是 PyCapsule
    if (!PyCapsule_CheckExact(capsule.ptr())) {
        throw std::runtime_error("Expected a PyCapsule object for DLPack tensor");
    }
    
    // 检查 capsule 名称（DLPack 规范要求名称必须是 "dltensor"）
    const char* name = PyCapsule_GetName(capsule.ptr());
    if (name == nullptr || std::string(name) != "dltensor") {
        throw std::runtime_error("Invalid DLPack capsule name. Expected 'dltensor'");
    }
    
    // 获取 DLManagedTensor 指针（不获取所有权）
    DLManagedTensor* managed_tensor = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor")
    );
    
    if (managed_tensor == nullptr) {
        throw std::runtime_error("Failed to get DLManagedTensor from capsule");
    }
    
    // 调用 C++ 接口创建 Tensor（数据会被复制，capsule 可以被安全释放）
    Tensor tensor = Tensor::from_dlpack(managed_tensor);
    
    return tensor;
}

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
        // 属性接口
        .def("empty", &Tensor::empty, "检查张量是否为空")
        .def("dtype", &Tensor::dtype, "获取张量的数据类型")
        .def("shape", &Tensor::shape, "获取张量的形状")
        .def("device", &Tensor::device, "获取张量所在的设备")
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
    // Request 类绑定
    // ============================================================================
    py::class_<Request>(m, "Request", "推理请求类")
        .def(py::init<int, const std::vector<int32_t>&>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             "创建仅包含 token IDs 的请求")
        .def(py::init<int, const std::vector<int32_t>&, const Tensor&, int32_t>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             py::arg("embedding"),
             py::arg("embed_token_id") = -1,
             "创建包含图像嵌入的请求（embed_token_id 为占位 token 的起始 ID）")
        .def(py::init<int, const std::vector<int32_t>&, const Tensor&, int32_t, const Tensor&>(),
             py::arg("request_id"),
             py::arg("token_ids"),
             py::arg("embedding"),
             py::arg("embed_token_id"),
             py::arg("position_ids"),
             "创建包含图像嵌入和 M-RoPE position_ids 的请求")
        .def(py::init<int, const std::vector<int32_t>&, const Tensor&, int32_t, const Tensor&, const std::vector<int32_t>&>(),
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
        .def("generate", &EdgeFM::generate,
             py::arg("request"),
             "从给定请求生成响应 token");
}
