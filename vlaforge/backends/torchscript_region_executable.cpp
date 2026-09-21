#include "vlaforge/backends/torchscript_region_executable.h"

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#ifdef VLAFORGE_TORCHSCRIPT_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kMaximumBindings = 128u;
constexpr std::size_t kErrorCapacity = 256u;

struct Binding {
  VLAForgeTensorView view{};
  bool bound = false;
};

c10::ScalarType ToScalarType(VLAForgeDType dtype) {
  switch (dtype) {
    case VLAFORGE_DTYPE_BOOL:
      return c10::ScalarType::Bool;
    case VLAFORGE_DTYPE_U8:
      return c10::ScalarType::Byte;
    case VLAFORGE_DTYPE_I32:
      return c10::ScalarType::Int;
    case VLAFORGE_DTYPE_I64:
      return c10::ScalarType::Long;
    case VLAFORGE_DTYPE_F16:
      return c10::ScalarType::Half;
    case VLAFORGE_DTYPE_BF16:
      return c10::ScalarType::BFloat16;
    case VLAFORGE_DTYPE_F32:
      return c10::ScalarType::Float;
    case VLAFORGE_DTYPE_F64:
      return c10::ScalarType::Double;
    case VLAFORGE_DTYPE_INVALID:
    case VLAFORGE_DTYPE_U64:
      break;
  }
  throw std::invalid_argument("unsupported TorchScript tensor dtype");
}

std::size_t ElementSize(VLAForgeDType dtype) {
  switch (dtype) {
    case VLAFORGE_DTYPE_BOOL:
    case VLAFORGE_DTYPE_U8:
      return 1u;
    case VLAFORGE_DTYPE_F16:
    case VLAFORGE_DTYPE_BF16:
      return 2u;
    case VLAFORGE_DTYPE_I32:
    case VLAFORGE_DTYPE_F32:
      return 4u;
    case VLAFORGE_DTYPE_I64:
    case VLAFORGE_DTYPE_F64:
      return 8u;
    case VLAFORGE_DTYPE_INVALID:
    case VLAFORGE_DTYPE_U64:
      return 0u;
  }
  return 0u;
}

bool ValidDevice(VLAForgeDevice device) {
  if (device.kind == VLAFORGE_DEVICE_CPU) {
    return device.ordinal == 0;
  }
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
  int count = 0;
  return device.kind == VLAFORGE_DEVICE_CUDA && device.ordinal >= 0 &&
         cudaGetDeviceCount(&count) == cudaSuccess && device.ordinal < count;
#else
  return false;
#endif
}

c10::Device ToDevice(VLAForgeDevice device) {
  return device.kind == VLAFORGE_DEVICE_CPU
      ? c10::Device(c10::DeviceType::CPU)
      : c10::Device(c10::DeviceType::CUDA,
                    static_cast<c10::DeviceIndex>(device.ordinal));
}

bool ValidTensorView(const VLAForgeTensorView& view) {
  if (view.data == nullptr || !ValidDevice(view.device) ||
      (view.rank != 0u && view.dimensions == nullptr) ||
      ElementSize(view.dtype) == 0u) {
    return false;
  }
  std::uint64_t elements = 1u;
  for (std::uint32_t index = 0; index < view.rank; ++index) {
    if (view.dimensions[index] < 0) {
      return false;
    }
    const auto dimension =
        static_cast<std::uint64_t>(view.dimensions[index]);
    if (dimension != 0u &&
        elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
      return false;
    }
    elements *= dimension;
  }
  return elements <= std::numeric_limits<std::uint64_t>::max() /
                         ElementSize(view.dtype) &&
         view.size_bytes == elements * ElementSize(view.dtype);
}

at::Tensor TensorFromView(const VLAForgeTensorView& view) {
  return at::from_blob(
      view.data, c10::IntArrayRef(view.dimensions, view.rank),
      at::TensorOptions().dtype(ToScalarType(view.dtype)).device(ToDevice(view.device)));
}

bool SameMetadata(const at::Tensor& tensor,
                  const VLAForgeTensorView& view) {
  if (tensor.device() != ToDevice(view.device) ||
      tensor.scalar_type() != ToScalarType(view.dtype) ||
      tensor.dim() != static_cast<std::int64_t>(view.rank)) {
    return false;
  }
  for (std::uint32_t index = 0; index < view.rank; ++index) {
    if (tensor.size(index) != view.dimensions[index]) {
      return false;
    }
  }
  return true;
}

bool FlattenOutputs(const torch::jit::IValue& value,
                    std::vector<at::Tensor>* outputs) {
  if (value.isTensor()) {
    outputs->push_back(value.toTensor());
    return true;
  }
  if (!value.isTuple()) {
    return false;
  }
  for (const auto& item : value.toTupleRef().elements()) {
    if (!item.isTensor()) {
      return false;
    }
    outputs->push_back(item.toTensor());
  }
  return true;
}

bool ValidArchivedValue(const torch::jit::IValue& value,
                        VLAForgeDevice device) {
  if (value.isTensor()) {
    return value.toTensor().device().is_cpu() ||
           value.toTensor().device() == ToDevice(device);
  }
  if (value.isDevice()) {
    return value.toDevice().is_cpu() || value.toDevice() == ToDevice(device);
  }
  if (value.isTuple()) {
    for (const auto& item : value.toTupleRef().elements()) {
      if (!ValidArchivedValue(item, device)) return false;
    }
  } else if (value.isList()) {
    for (const auto& item : value.toListRef()) {
      if (!ValidArchivedValue(item, device)) return false;
    }
  } else if (value.isGenericDict()) {
    for (const auto& item : value.toGenericDict()) {
      if (!ValidArchivedValue(item.key(), device) ||
          !ValidArchivedValue(item.value(), device)) return false;
    }
  } else if (value.isObject()) {
    return false;
  }
  return true;
}

void ValidateGraphDevices(const torch::jit::Block* block,
                          VLAForgeDevice device) {
  for (const auto* node : block->nodes()) {
    const std::string kind(node->kind().toQualString());
    if (kind == "prim::PythonOp" || kind == "prim::fork" ||
        kind == "aten::cuda" || kind == "aten::record_stream" ||
        kind.rfind("cuda::", 0u) == 0u) {
      throw std::invalid_argument("TorchScript profile rejects Python or unmanaged CUDA work");
    }
    if (kind == "prim::Constant") {
      for (const auto* output : node->outputs()) {
        const auto value = torch::jit::toIValue(output);
        if (value.has_value() && !ValidArchivedValue(*value, device)) {
          throw std::invalid_argument("TorchScript archived constant device mismatch");
        }
      }
    }
    for (const auto* nested : node->blocks()) {
      ValidateGraphDevices(nested, device);
    }
  }
}

std::shared_ptr<torch::jit::Module> LoadSharedModule(
    const std::string& archive_path, bool native_aten,
    VLAForgeDevice device) {
  static std::mutex mutex;
  static std::unordered_map<
      std::string, std::weak_ptr<torch::jit::Module>>
      modules;
  // Legacy CPU archives retain their old cache. Native Sessions own modules:
  // even identical bytes can contain mutable buffers or JIT executor state.
  std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
  if (!native_aten) lock.lock();
  const std::string key = archive_path + (native_aten ? "|aten|" : "|legacy|") +
      std::to_string(device.kind) + ":" + std::to_string(device.ordinal);
  if (!native_aten) {
    const auto found = modules.find(key);
    if (found != modules.end()) {
      if (auto existing = found->second.lock()) {
        return existing;
      }
    }
  }
  auto module = std::make_shared<torch::jit::Module>(
      torch::jit::load(
          archive_path, native_aten ? std::optional<c10::Device>()
                                   : std::optional<c10::Device>(c10::Device(c10::DeviceType::CPU))));
  if (native_aten) {
    const auto valid = [&](const at::Tensor& value) {
      return value.device().is_cpu() || value.device() == ToDevice(device);
    };
    for (const auto& value : module->parameters()) {
      if (!valid(value)) {
        throw std::invalid_argument("TorchScript archived parameter device mismatch");
      }
    }
    for (const auto& value : module->buffers()) {
      if (!valid(value)) {
        throw std::invalid_argument("TorchScript archived buffer device mismatch");
      }
    }
    for (const auto& child : module->modules()) {
      for (const auto& method : child.get_methods()) {
        ValidateGraphDevices(method.graph()->block(), device);
      }
    }
  }
  module->eval();
  if (!native_aten) modules[key] = module;
  return module;
}

}  // namespace

struct VLAForgeRegionExecutable {
  std::uint32_t region_id = 0u;
  std::shared_ptr<torch::jit::Module> module;
  std::optional<torch::jit::Method> method;
  std::array<Binding, kMaximumBindings> inputs{};
  std::array<Binding, kMaximumBindings> outputs{};
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  VLAForgeDevice device{VLAFORGE_DEVICE_CPU, 0};
  bool native_aten = false;
  bool context_capable = false;
  bool pending_work = false;
  std::optional<VLAForgeExecutionContextView> execution_context;
  bool poisoned = false;
  std::array<char, kErrorCapacity> error{};

  VLAForgeStatus RecordError(const char* message) noexcept {
    const char* text =
        message == nullptr ? "TorchScript backend error" : message;
    std::snprintf(error.data(), error.size(), "%s", text);
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }
};

namespace {

#ifdef VLAFORGE_TORCHSCRIPT_CUDA
c10::cuda::CUDAStream ActiveStream(const VLAForgeRegionExecutable& executable) {
  return executable.execution_context.has_value()
      ? c10::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(executable.execution_context->native_stream),
            executable.device.ordinal)
      : c10::cuda::getDefaultCUDAStream(executable.device.ordinal);
}
#endif

template <std::size_t Size>
bool HasVariant(const VLAForgeArtifactDescriptor& artifact, const char (&name)[Size]) {
  return artifact.backend_variant != nullptr && artifact.backend_variant_size == Size - 1u &&
         std::memcmp(artifact.backend_variant, name, Size - 1u) == 0;
}

bool DrainFailedLoad(VLAForgeRegionExecutable& executable) noexcept {
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
  if (executable.pending_work) {
    try {
      const c10::cuda::CUDAGuard guard(executable.device.ordinal);
      ActiveStream(executable).synchronize();
      executable.pending_work = false;
    } catch (...) {
      return false;
    }
  }
#else
  (void)executable;
#endif
  return true;
}

VLAForgeStatus CreateImpl(const VLAForgeRegionCreateOptions* options,
                         VLAForgeRegionExecutable** output, bool native_aten) {
  if (output != nullptr) {
    *output = nullptr;
  }
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != (native_aten ? VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION
                                          : VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) ||
      !ValidDevice(options->device) ||
      (!native_aten && options->device.kind != VLAFORGE_DEVICE_CPU)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_OUT_OF_MEMORY,
        "TorchScript executable allocation failed");
  }
  executable->region_id = options->region_id;
  executable->device = options->device;
  executable->native_aten = native_aten;
  *output = executable;
  return vlaforge_status_ok();
}

VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
                      VLAForgeRegionExecutable** output) {
  return CreateImpl(options, output, false);
}

VLAForgeStatus CreateValue(const VLAForgeRegionCreateOptions* options,
                           VLAForgeRegionExecutable** output) {
  return CreateImpl(options, output, true);
}

VLAForgeStatus Load(VLAForgeRegionExecutable* executable,
                    const VLAForgeArtifactDescriptor* artifact) {
  if (executable != nullptr && executable->pending_work) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "synchronize TorchScript before reload");
  }
  if (executable != nullptr && executable->native_aten) {
    executable->poisoned = true;
    executable->method.reset();
    executable->module.reset();
    executable->inputs.fill(Binding{});
    executable->outputs.fill(Binding{});
    executable->input_count = executable->output_count = 0u;
  }
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version != (executable->native_aten
          ? VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION
          : VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) ||
      artifact->path == nullptr || artifact->path_size == 0u) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript artifact descriptor");
  }
  if (executable->native_aten) {
    const bool context = HasVariant(*artifact, "torchscript-aten-context/1");
    if ((!context && !HasVariant(*artifact, "torchscript-aten/1")) ||
        (context && executable->device.kind != VLAFORGE_DEVICE_CUDA) ||
        (!context && executable->execution_context.has_value())) {
      return executable->RecordError("unsupported TorchScript variant or execution context");
    }
    executable->context_capable = context;
  }
  try {
    const c10::OptionalDeviceGuard device_guard(ToDevice(executable->device));
    const torch::jit::GraphOptimizerEnabledGuard optimize(
        executable->native_aten ? false : torch::jit::getGraphExecutorOptimize());
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (executable->device.kind == VLAFORGE_DEVICE_CUDA) {
      stream_guard.reset_stream(ActiveStream(*executable));
    }
#endif
    const std::string artifact_spec(
        artifact->path, artifact->path_size);
    const auto fragment = artifact_spec.rfind('#');
    const std::string archive_path =
        fragment == std::string::npos
        ? artifact_spec
        : artifact_spec.substr(0u, fragment);
    const std::string method_name =
        fragment == std::string::npos
        ? "forward"
        : artifact_spec.substr(fragment + 1u);
    if (archive_path.empty() || method_name.empty()) {
      return executable->RecordError(
          "invalid TorchScript archive entrypoint");
    }
    executable->pending_work = executable->device.kind == VLAFORGE_DEVICE_CUDA;
    auto module = LoadSharedModule(
        archive_path, executable->native_aten, executable->device);
    if (executable->native_aten) executable->module = module;
    auto method = module->get_method(method_name);
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
    if (executable->device.kind == VLAFORGE_DEVICE_CUDA) {
      ActiveStream(*executable).synchronize();
    }
#endif
    executable->module = std::move(module);
    executable->method.emplace(std::move(method));
    executable->pending_work = false;
    executable->poisoned = false;
  } catch (const std::exception& error) {
    if (!DrainFailedLoad(*executable)) return executable->RecordError("TorchScript load completion failed");
    if (executable->native_aten) {
      executable->method.reset();
      executable->module.reset();
    }
    return executable->RecordError(error.what());
  } catch (...) {
    if (!DrainFailedLoad(*executable)) return executable->RecordError("TorchScript load completion failed");
    if (executable->native_aten) {
      executable->method.reset();
      executable->module.reset();
    }
    return executable->RecordError("unknown TorchScript load exception");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus QueryWorkspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == nullptr || requirement == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript workspace query");
  }
  requirement->size_bytes = 0u;
  requirement->alignment = 1u;
  requirement->device = executable->device;
  return vlaforge_status_ok();
}

VLAForgeStatus Bind(VLAForgeRegionExecutable* executable,
                    std::uint32_t index,
                    const VLAForgeTensorView* tensor,
                    bool input) {
  if (executable != nullptr && executable->native_aten && executable->pending_work) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "synchronize TorchScript before tensor rebind");
  }
  if (executable == nullptr || tensor == nullptr ||
      index >= kMaximumBindings || !ValidTensorView(*tensor) ||
      !((tensor->device.kind == executable->device.kind &&
         tensor->device.ordinal == executable->device.ordinal) ||
        (executable->device.kind == VLAFORGE_DEVICE_CUDA &&
         tensor->device.kind == VLAFORGE_DEVICE_CPU))) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript tensor binding");
  }
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
  if (tensor->device.kind == VLAFORGE_DEVICE_CUDA) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, tensor->data) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice ||
        attributes.device != tensor->device.ordinal) {
      return executable->RecordError("TorchScript CUDA pointer device mismatch");
    }
  }
#endif
  auto& bindings = input ? executable->inputs : executable->outputs;
  auto& count = input ? executable->input_count : executable->output_count;
  bindings[index] = Binding{*tensor, true};
  count = std::max(count, static_cast<std::size_t>(index) + 1u);
  return vlaforge_status_ok();
}

VLAForgeStatus BindInput(VLAForgeRegionExecutable* executable,
                         std::uint32_t index,
                         const VLAForgeTensorView* tensor) {
  return Bind(executable, index, tensor, true);
}

VLAForgeStatus BindOutput(VLAForgeRegionExecutable* executable,
                          std::uint32_t index,
                          const VLAForgeTensorView* tensor) {
  return Bind(executable, index, tensor, false);
}

VLAForgeStatus BindValue(VLAForgeRegionExecutable* executable,
                         std::uint32_t index, const VLAForgeValueView* value,
                         bool input) {
  if (value == nullptr || value->struct_size < sizeof(*value) ||
      value->kind != VLAFORGE_VALUE_TENSOR) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "TorchScript Value ABI accepts tensors only");
  }
  const auto& tensor = value->value.tensor;
  if (tensor.struct_size < sizeof(tensor) ||
      tensor.layout != VLAFORGE_LAYOUT_CONTIGUOUS ||
      tensor.alignment == 0u ||
      (tensor.alignment & (tensor.alignment - 1u)) != 0u ||
      reinterpret_cast<std::uintptr_t>(tensor.tensor.data) % tensor.alignment != 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid TorchScript bound tensor");
  }
  return Bind(executable, index, &tensor.tensor, input);
}

VLAForgeStatus BindInputValue(VLAForgeRegionExecutable* executable,
                              std::uint32_t index, const VLAForgeValueView* value) {
  return BindValue(executable, index, value, true);
}

VLAForgeStatus BindOutputValue(VLAForgeRegionExecutable* executable,
                               std::uint32_t index, const VLAForgeValueView* value) {
  return BindValue(executable, index, value, false);
}

VLAForgeStatus BindWorkspace(VLAForgeRegionExecutable* executable,
                             void* workspace,
                             std::uint64_t workspace_size) {
  if (executable == nullptr ||
      (workspace_size != 0u && workspace == nullptr)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript workspace binding");
  }
  if (workspace_size != 0u) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TorchScript archive owns its workspace");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable* executable);

VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || executable->module == nullptr ||
      !executable->method.has_value() || executable->poisoned) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TorchScript executable is not loaded");
  }
  for (std::size_t index = 0; index < executable->input_count; ++index) {
    if (!executable->inputs[index].bound) {
      return vlaforge_status_error(
          VLAFORGE_STATUS_FAILED_PRECONDITION,
          "TorchScript input binding has a gap");
    }
  }
  for (std::size_t index = 0; index < executable->output_count; ++index) {
    if (!executable->outputs[index].bound) {
      return vlaforge_status_error(
          VLAFORGE_STATUS_FAILED_PRECONDITION,
          "TorchScript output binding has a gap");
    }
  }
  try {
    const c10::OptionalDeviceGuard device_guard(ToDevice(executable->device));
    const torch::jit::GraphOptimizerEnabledGuard optimize(
        executable->native_aten ? false : torch::jit::getGraphExecutorOptimize());
    const c10::InferenceMode inference(
        executable->native_aten || c10::InferenceMode::is_enabled());
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (executable->device.kind == VLAFORGE_DEVICE_CUDA) {
      const auto stream = ActiveStream(*executable);
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream.stream(), &capture) != cudaSuccess ||
          (capture != cudaStreamCaptureStatusNone &&
           !(executable->context_capable && executable->execution_context.has_value()))) {
        return executable->RecordError("synchronous TorchScript cannot run during capture");
      }
      stream_guard.reset_stream(stream);
      executable->pending_work = true;
    }
#endif
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(executable->input_count);
    for (std::size_t index = 0; index < executable->input_count; ++index) {
      inputs.emplace_back(
          TensorFromView(executable->inputs[index].view));
    }
    std::vector<at::Tensor> outputs;
    if (!FlattenOutputs((*executable->method)(inputs), &outputs)) {
      throw std::invalid_argument("TorchScript output is not a tensor or flat tensor tuple");
    }
    if (outputs.size() != executable->output_count) {
      throw std::invalid_argument("TorchScript output count mismatch");
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& view = executable->outputs[index].view;
      if (!SameMetadata(outputs[index], view)) {
        throw std::invalid_argument("TorchScript output metadata mismatch");
      }
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& view = executable->outputs[index].view;
      TensorFromView(view).copy_(outputs[index]);
    }
    return executable->execution_context.has_value()
        ? vlaforge_status_ok() : Synchronize(executable);
  } catch (const std::exception& error) {
    if (executable->execution_context.has_value()) return executable->RecordError(error.what());
    const auto status = Synchronize(executable);
    if (status.code != VLAFORGE_STATUS_OK) {
      return status;
    }
    return executable->RecordError(error.what());
  } catch (...) {
    if (!executable->execution_context.has_value()) {
      const auto status = Synchronize(executable);
      if (status.code != VLAFORGE_STATUS_OK) return status;
    }
    return executable->RecordError("unknown TorchScript execution exception");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TorchScript executable is null");
  }
  if (executable->poisoned) {
    return executable->RecordError("TorchScript execution is poisoned");
  }
#ifdef VLAFORGE_TORCHSCRIPT_CUDA
  if (executable->device.kind == VLAFORGE_DEVICE_CUDA) {
    try {
      const c10::cuda::CUDAGuard guard(executable->device.ordinal);
      const auto stream = ActiveStream(*executable);
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream.stream(), &capture) != cudaSuccess) {
        executable->poisoned = true;
        return executable->RecordError("TorchScript capture status query failed");
      }
      if (capture != cudaStreamCaptureStatusNone) {
        return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                     "TorchScript synchronize is not permitted during capture");
      }
      stream.synchronize();
      executable->pending_work = false;
    } catch (const std::exception& error) {
      executable->poisoned = true;
      return executable->RecordError(error.what());
    } catch (...) {
      executable->poisoned = true;
      return executable->RecordError("unknown TorchScript completion exception");
    }
  }
#endif
  return vlaforge_status_ok();
}

void Destroy(VLAForgeRegionExecutable* executable) {
  if (executable != nullptr && executable->pending_work &&
      Synchronize(executable).code != VLAFORGE_STATUS_OK) {
    // Retain module, storages and borrowed stream view when completion is unknown.
    return;
  }
  delete executable;
}

VLAForgeStatus BindExecutionContext(VLAForgeRegionExecutable* executable,
                                    const VLAForgeExecutionContextView* view) {
  if (executable == nullptr || !executable->native_aten ||
      executable->device.kind != VLAFORGE_DEVICE_CUDA) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "TorchScript shared context requires native CUDA");
  }
  if (executable->pending_work || executable->poisoned ||
      (executable->module != nullptr && !executable->context_capable)) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "TorchScript context unavailable or requires synchronize");
  }
  if (view == nullptr) {
    executable->execution_context.reset();
    return vlaforge_status_ok();
  }
  auto status = vlaforge_execution_context_view_validate(view);
  if (status.code != VLAFORGE_STATUS_OK) return status;
  if (view->device.kind != executable->device.kind || view->device.ordinal != executable->device.ordinal) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "TorchScript execution context device mismatch");
  }
  executable->execution_context = *view;
  return vlaforge_status_ok();
}

const VLAForgeRegionExecutionExtensionApi kExecutionExtension = {
    sizeof(VLAForgeRegionExecutionExtensionApi), VLAFORGE_REGION_EXECUTION_EXTENSION_ABI_VERSION,
    VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT, &BindExecutionContext,
};

const VLAForgeRegionExecutableApi kTorchScriptApi = {
    sizeof(VLAForgeRegionExecutableApi),
    VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
    &Create,
    &Load,
    &QueryWorkspace,
    &BindInput,
    &BindOutput,
    &BindWorkspace,
    &Run,
    &Synchronize,
    &Destroy,
};

const VLAForgeRegionExecutableValueApi kTorchScriptValueApi = {
    sizeof(VLAForgeRegionExecutableValueApi),
    VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    &CreateValue, &Load, &QueryWorkspace, &BindInputValue, &BindOutputValue,
    &BindWorkspace, &Run, &Synchronize, &Destroy,
};

}  // namespace

extern "C" const VLAForgeRegionExecutableApi*
vlaforge_torchscript_region_executable_api(void) {
  return &kTorchScriptApi;
}

extern "C" const VLAForgeRegionExecutableValueApi*
vlaforge_torchscript_region_executable_value_api(void) {
  return &kTorchScriptValueApi;
}

extern "C" const VLAForgeRegionExecutionExtensionApi*
vlaforge_torchscript_region_execution_extension_api(void) {
  return &kExecutionExtension;
}
