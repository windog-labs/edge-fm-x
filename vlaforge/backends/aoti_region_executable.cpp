#include "vlaforge/backends/aoti_region_executable.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
    case VLAFORGE_DTYPE_U64:
      return c10::ScalarType::UInt64;
    case VLAFORGE_DTYPE_U8:
      return c10::ScalarType::Byte;
    case VLAFORGE_DTYPE_INVALID:
      break;
  }
  throw std::invalid_argument("unsupported AOTI tensor dtype");
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
    case VLAFORGE_DTYPE_U64:
    case VLAFORGE_DTYPE_F64:
      return 8u;
    case VLAFORGE_DTYPE_INVALID:
      return 0u;
  }
  return 0u;
}

bool ValidTensorView(const VLAForgeTensorView& view,
                     VLAForgeDeviceKind device_kind,
                     int device_ordinal) {
  if (view.data == nullptr || view.device.kind != device_kind ||
      view.device.ordinal != device_ordinal ||
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
  return view.size_bytes == elements * ElementSize(view.dtype);
}

at::Tensor TensorFromView(const VLAForgeTensorView& view) {
  const c10::IntArrayRef shape(view.dimensions, view.rank);
  const c10::Device device =
      view.device.kind == VLAFORGE_DEVICE_CUDA
      ? c10::Device(c10::DeviceType::CUDA, view.device.ordinal)
      : c10::Device(c10::DeviceType::CPU);
  const auto options =
      at::TensorOptions()
          .dtype(ToScalarType(view.dtype))
          .device(device);
  return at::from_blob(view.data, shape, options);
}

bool SameShapeAndDType(const at::Tensor& tensor,
                       const VLAForgeTensorView& view) {
  if (tensor.scalar_type() != ToScalarType(view.dtype) ||
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

bool ValidBoundTensor(const VLAForgeBoundTensor& value) {
  if (value.struct_size < sizeof(VLAForgeBoundTensor) ||
      value.layout != VLAFORGE_LAYOUT_CONTIGUOUS ||
      value.alignment == 0u ||
      (value.alignment & (value.alignment - 1u)) != 0u) {
    return false;
  }
  const auto address =
      reinterpret_cast<std::uintptr_t>(value.tensor.data);
  return address % value.alignment == 0u;
}

}  // namespace

struct VLAForgeRegionExecutable {
  std::uint32_t region_id = 0u;
  std::uint32_t abi_version = 0u;
  VLAForgeDeviceKind device_kind = VLAFORGE_DEVICE_CPU;
  int device_ordinal = 0;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::array<Binding, kMaximumBindings> inputs{};
  std::array<Binding, kMaximumBindings> outputs{};
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  std::array<char, kErrorCapacity> error{};

  VLAForgeStatus RecordError(const char* message) noexcept {
    const char* text = message == nullptr ? "AOTI backend error" : message;
    std::snprintf(error.data(), error.size(), "%s", text);
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }
};

namespace {

bool OnExecutableDevice(
    const at::Tensor& tensor,
    const VLAForgeRegionExecutable& executable) {
  return executable.device_kind == VLAFORGE_DEVICE_CUDA
      ? tensor.is_cuda() &&
            tensor.get_device() == executable.device_ordinal
      : tensor.device().is_cpu();
}

bool TargetMatches(const VLAForgeRegionExecutable& executable,
                   const VLAForgeArtifactDescriptor& artifact) {
  if (artifact.target == nullptr || artifact.target_size == 0u) {
    return true;
  }
  const std::string_view target(artifact.target, artifact.target_size);
  if (executable.device_kind == VLAFORGE_DEVICE_CPU) {
    return target == "cpu";
  }
  if (target.size() != 5u || target.substr(0u, 3u) != "sm_" ||
      target[3] < '0' || target[3] > '9' ||
      target[4] < '0' || target[4] > '9') {
    return false;
  }
  const auto* properties =
      at::cuda::getDeviceProperties(executable.device_ordinal);
  return properties != nullptr &&
         properties->major == target[3] - '0' &&
         properties->minor == target[4] - '0';
}

VLAForgeStatus AotiCreate(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** output) {
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(*options) ||
      (options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION &&
       options->abi_version !=
           VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION) ||
      (options->device.kind != VLAFORGE_DEVICE_CPU &&
       options->device.kind != VLAFORGE_DEVICE_CUDA) ||
      options->device.ordinal < 0 ||
      (options->device.kind == VLAFORGE_DEVICE_CPU &&
       options->device.ordinal != 0)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "AOTI executable allocation failed");
  }
  executable->region_id = options->region_id;
  executable->abi_version = options->abi_version;
  executable->device_kind = options->device.kind;
  executable->device_ordinal = options->device.ordinal;
  *output = executable;
  return vlaforge_status_ok();
}

VLAForgeStatus AotiLoad(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version != executable->abi_version ||
      artifact->path == nullptr || artifact->path_size == 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI artifact descriptor");
  }
  if (!TargetMatches(*executable, *artifact)) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "AOTI artifact target mismatch");
  }
  try {
    std::optional<c10::cuda::CUDAGuard> guard;
    if (executable->device_kind == VLAFORGE_DEVICE_CUDA) {
      guard.emplace(executable->device_ordinal);
    }
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
    executable->loader =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            std::string(artifact->path, artifact->path_size), "model",
            true, 1u,
            executable->device_kind == VLAFORGE_DEVICE_CUDA
                ? executable->device_ordinal
                : -1);
#else
    executable->loader =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            std::string(artifact->path, artifact->path_size), "model");
#endif
  } catch (const std::exception& error) {
    return executable->RecordError(error.what());
  }
  return vlaforge_status_ok();
}

VLAForgeStatus AotiQueryWorkspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == nullptr || requirement == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI workspace query");
  }
  requirement->size_bytes = 0u;
  requirement->alignment = 1u;
  requirement->device = {executable->device_kind,
                         executable->device_ordinal};
  return vlaforge_status_ok();
}

VLAForgeStatus Bind(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor, bool input) {
  if (executable == nullptr || tensor == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI tensor binding");
  }
  const bool valid_device =
      ValidTensorView(*tensor, executable->device_kind,
                      executable->device_ordinal) ||
      (executable->device_kind == VLAFORGE_DEVICE_CUDA &&
       ValidTensorView(*tensor, VLAFORGE_DEVICE_CPU, 0));
  if (index >= kMaximumBindings || !valid_device) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI tensor binding");
  }
  auto& bindings = input ? executable->inputs : executable->outputs;
  auto& count = input ? executable->input_count : executable->output_count;
  bindings[index] = Binding{*tensor, true};
  count = std::max(count, static_cast<std::size_t>(index) + 1u);
  return vlaforge_status_ok();
}

VLAForgeStatus AotiBindInput(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor) {
  return Bind(executable, index, tensor, true);
}

VLAForgeStatus AotiBindOutput(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor) {
  return Bind(executable, index, tensor, false);
}

VLAForgeStatus BindValue(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeValueView* value, bool input) {
  if (value == nullptr || value->struct_size < sizeof(*value) ||
      value->kind != VLAFORGE_VALUE_TENSOR ||
      !ValidBoundTensor(value->value.tensor)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI value binding");
  }
  return Bind(executable, index, &value->value.tensor.tensor, input);
}

VLAForgeStatus AotiBindInputValue(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeValueView* value) {
  return BindValue(executable, index, value, true);
}

VLAForgeStatus AotiBindOutputValue(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeValueView* value) {
  return BindValue(executable, index, value, false);
}

VLAForgeStatus AotiBindWorkspace(
    VLAForgeRegionExecutable* executable, void* workspace,
    std::uint64_t workspace_size) {
  if (executable == nullptr ||
      (workspace_size != 0u && workspace == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI workspace binding");
  }
  if (workspace_size != 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "AOTI package owns its workspace");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus AotiRun(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || executable->loader == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "AOTI executable is not loaded");
  }
  for (std::size_t index = 0; index < executable->input_count; ++index) {
    if (!executable->inputs[index].bound) {
      return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                   "AOTI input binding has a gap");
    }
  }
  for (std::size_t index = 0; index < executable->output_count; ++index) {
    if (!executable->outputs[index].bound) {
      return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                   "AOTI output binding has a gap");
    }
  }

  try {
    std::optional<c10::cuda::CUDAGuard> guard;
    if (executable->device_kind == VLAFORGE_DEVICE_CUDA) {
      guard.emplace(executable->device_ordinal);
    }
    std::vector<at::Tensor> inputs;
    inputs.reserve(executable->input_count);
    for (std::size_t index = 0; index < executable->input_count; ++index) {
      inputs.push_back(TensorFromView(executable->inputs[index].view));
    }
    auto outputs = executable->loader->run(inputs);
    if (outputs.size() != executable->output_count) {
      return executable->RecordError("AOTI output count mismatch");
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& view = executable->outputs[index].view;
      if (!OnExecutableDevice(outputs[index], *executable) ||
          !SameShapeAndDType(outputs[index], view)) {
        return executable->RecordError("AOTI output metadata mismatch");
      }
      TensorFromView(view).copy_(outputs[index]);
    }
  } catch (const std::exception& error) {
    return executable->RecordError(error.what());
  }
  return vlaforge_status_ok();
}

VLAForgeStatus AotiSynchronize(
    VLAForgeRegionExecutable* executable) {
  if (executable == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "AOTI executable is null");
  }
  if (executable->device_kind == VLAFORGE_DEVICE_CPU) {
    return vlaforge_status_ok();
  }
  try {
    const c10::cuda::CUDAGuard guard(executable->device_ordinal);
    c10::cuda::device_synchronize();
  } catch (const std::exception& error) {
    return executable->RecordError(error.what());
  }
  return vlaforge_status_ok();
}

void AotiDestroy(VLAForgeRegionExecutable* executable) {
  delete executable;
}

const VLAForgeRegionExecutableApi kAotiApi = {
    sizeof(VLAForgeRegionExecutableApi),
    VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
    &AotiCreate,
    &AotiLoad,
    &AotiQueryWorkspace,
    &AotiBindInput,
    &AotiBindOutput,
    &AotiBindWorkspace,
    &AotiRun,
    &AotiSynchronize,
    &AotiDestroy,
};

const VLAForgeRegionExecutableValueApi kAotiValueApi = {
    sizeof(VLAForgeRegionExecutableValueApi),
    VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    &AotiCreate,
    &AotiLoad,
    &AotiQueryWorkspace,
    &AotiBindInputValue,
    &AotiBindOutputValue,
    &AotiBindWorkspace,
    &AotiRun,
    &AotiSynchronize,
    &AotiDestroy,
};

}  // namespace

extern "C" const VLAForgeRegionExecutableApi*
vlaforge_aoti_region_executable_api(void) {
  return &kAotiApi;
}

extern "C" const VLAForgeRegionExecutableValueApi*
vlaforge_aoti_region_executable_value_api(void) {
  return &kAotiValueApi;
}
