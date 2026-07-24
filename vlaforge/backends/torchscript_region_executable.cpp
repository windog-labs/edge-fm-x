#include "vlaforge/backends/torchscript_region_executable.h"

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
  throw std::invalid_argument("unsupported TorchScript tensor dtype");
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

bool ValidTensorView(const VLAForgeTensorView& view) {
  if (view.data == nullptr || view.device.kind != VLAFORGE_DEVICE_CPU ||
      view.device.ordinal != 0 ||
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
  return at::from_blob(
      view.data, c10::IntArrayRef(view.dimensions, view.rank),
      at::TensorOptions().dtype(ToScalarType(view.dtype)).device(at::kCPU));
}

bool SameMetadata(const at::Tensor& tensor,
                  const VLAForgeTensorView& view) {
  if (!tensor.device().is_cpu() ||
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

std::shared_ptr<torch::jit::Module> LoadSharedModule(
    const std::string& archive_path) {
  static std::mutex mutex;
  static std::unordered_map<
      std::string, std::weak_ptr<torch::jit::Module>>
      modules;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto found = modules.find(archive_path);
  if (found != modules.end()) {
    if (auto existing = found->second.lock()) {
      return existing;
    }
  }
  auto module = std::make_shared<torch::jit::Module>(
      torch::jit::load(
          archive_path, c10::Device(c10::DeviceType::CPU)));
  module->eval();
  modules[archive_path] = module;
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

VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
                      VLAForgeRegionExecutable** output) {
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION ||
      options->device.kind != VLAFORGE_DEVICE_CPU ||
      options->device.ordinal != 0) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid CPU TorchScript create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_OUT_OF_MEMORY,
        "TorchScript executable allocation failed");
  }
  executable->region_id = options->region_id;
  *output = executable;
  return vlaforge_status_ok();
}

VLAForgeStatus Load(VLAForgeRegionExecutable* executable,
                    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_ABI_VERSION ||
      artifact->path == nullptr || artifact->path_size == 0u) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TorchScript artifact descriptor");
  }
  try {
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
    executable->module = LoadSharedModule(archive_path);
    executable->method.emplace(
        executable->module->get_method(method_name));
  } catch (const std::exception& error) {
    return executable->RecordError(error.what());
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
  requirement->device = {VLAFORGE_DEVICE_CPU, 0};
  return vlaforge_status_ok();
}

VLAForgeStatus Bind(VLAForgeRegionExecutable* executable,
                    std::uint32_t index,
                    const VLAForgeTensorView* tensor,
                    bool input) {
  if (executable == nullptr || tensor == nullptr ||
      index >= kMaximumBindings || !ValidTensorView(*tensor)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid CPU TorchScript tensor binding");
  }
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

VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || executable->module == nullptr ||
      !executable->method.has_value()) {
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
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(executable->input_count);
    for (std::size_t index = 0; index < executable->input_count; ++index) {
      inputs.emplace_back(
          TensorFromView(executable->inputs[index].view));
    }
    std::vector<at::Tensor> outputs;
    if (!FlattenOutputs((*executable->method)(inputs), &outputs)) {
      return executable->RecordError(
          "TorchScript output is not a tensor or flat tensor tuple");
    }
    if (outputs.size() != executable->output_count) {
      return executable->RecordError(
          "TorchScript output count mismatch");
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& view = executable->outputs[index].view;
      if (!SameMetadata(outputs[index], view)) {
        return executable->RecordError(
            "TorchScript output metadata mismatch");
      }
      TensorFromView(view).copy_(outputs[index]);
    }
  } catch (const std::exception& error) {
    return executable->RecordError(error.what());
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TorchScript executable is null");
  }
  return vlaforge_status_ok();
}

void Destroy(VLAForgeRegionExecutable* executable) {
  delete executable;
}

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

}  // namespace

extern "C" const VLAForgeRegionExecutableApi*
vlaforge_torchscript_region_executable_api(void) {
  return &kTorchScriptApi;
}
