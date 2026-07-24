#include "vlaforge/backends/aoti_region_executable.h"

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaximumBindings = 16u;
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
    case VLAFORGE_DTYPE_INVALID:
      break;
  }
  throw std::invalid_argument("unsupported AOTI tensor dtype");
}

std::size_t ElementSize(VLAForgeDType dtype) {
  switch (dtype) {
    case VLAFORGE_DTYPE_BOOL:
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
      return 0u;
  }
  return 0u;
}

bool ValidTensorView(const VLAForgeTensorView& view, int device_ordinal) {
  if (view.data == nullptr || view.device.kind != VLAFORGE_DEVICE_CUDA ||
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
  const auto options =
      at::TensorOptions()
          .dtype(ToScalarType(view.dtype))
          .device(c10::Device(c10::DeviceType::CUDA, view.device.ordinal));
  return at::from_blob(view.data, shape, options);
}

bool SameMetadata(const at::Tensor& tensor,
                  const VLAForgeTensorView& view) {
  if (!tensor.is_cuda() || tensor.get_device() != view.device.ordinal ||
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

}  // namespace

struct VLAForgeRegionExecutable {
  std::uint32_t region_id = 0u;
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

VLAForgeStatus AotiCreate(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** output) {
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION ||
      options->device.kind != VLAFORGE_DEVICE_CUDA ||
      options->device.ordinal < 0) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid CUDA AOTI create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "AOTI executable allocation failed");
  }
  executable->region_id = options->region_id;
  executable->device_ordinal = options->device.ordinal;
  *output = executable;
  return vlaforge_status_ok();
}

VLAForgeStatus AotiLoad(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_ABI_VERSION ||
      artifact->path == nullptr || artifact->path_size == 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid AOTI artifact descriptor");
  }
  try {
    const c10::cuda::CUDAGuard guard(executable->device_ordinal);
    executable->loader =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            std::string(artifact->path, artifact->path_size), "model",
            true, 1u, executable->device_ordinal);
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
  requirement->device = {VLAFORGE_DEVICE_CUDA,
                         executable->device_ordinal};
  return vlaforge_status_ok();
}

VLAForgeStatus Bind(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor, bool input) {
  if (executable == nullptr || tensor == nullptr ||
      index >= kMaximumBindings ||
      !ValidTensorView(*tensor, executable->device_ordinal)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid CUDA AOTI tensor binding");
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

VLAForgeStatus AotiBindWorkspace(
    VLAForgeRegionExecutable* executable, void* workspace,
    std::uint64_t workspace_size) {
  if (executable == nullptr ||
      (workspace_size != 0u && workspace == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid CUDA AOTI workspace binding");
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
    const c10::cuda::CUDAGuard guard(executable->device_ordinal);
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
      if (!SameMetadata(outputs[index], view)) {
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

}  // namespace

extern "C" const VLAForgeRegionExecutableApi*
vlaforge_aoti_region_executable_api(void) {
  return &kAotiApi;
}
