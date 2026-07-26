#include "vlaforge/backends/tensorrt_region_executable.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaximumBindings = 128u;
constexpr std::size_t kErrorCapacity = 512u;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR && message != nullptr) {
      std::fprintf(stderr, "VLAForge TensorRT: %s\n", message);
    }
  }
};

Logger& GlobalLogger() {
  static Logger logger;
  return logger;
}

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

struct Binding {
  VLAForgeBoundTensor value{};
  bool bound = false;
};

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
    case VLAFORGE_DTYPE_U64:
      return 8u;
    case VLAFORGE_DTYPE_INVALID:
      return 0u;
  }
  return 0u;
}

bool TensorBytes(const VLAForgeTensorView& view, std::uint64_t* bytes) {
  if (bytes == nullptr || ElementSize(view.dtype) == 0u ||
      (view.rank != 0u && view.dimensions == nullptr)) {
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
  const auto element_size =
      static_cast<std::uint64_t>(ElementSize(view.dtype));
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_size) {
    return false;
  }
  *bytes = elements * element_size;
  return true;
}

bool ValidTensor(const VLAForgeBoundTensor& value, int device_ordinal) {
  const auto& view = value.tensor;
  std::uint64_t expected_bytes = 0u;
  if (value.struct_size < sizeof(value) ||
      value.layout != VLAFORGE_LAYOUT_CONTIGUOUS ||
      value.alignment == 0u ||
      (value.alignment & (value.alignment - 1u)) != 0u ||
      view.data == nullptr ||
      (view.device.kind != VLAFORGE_DEVICE_CUDA &&
       view.device.kind != VLAFORGE_DEVICE_CPU) ||
      (view.device.kind == VLAFORGE_DEVICE_CUDA &&
       view.device.ordinal != device_ordinal) ||
      (view.device.kind == VLAFORGE_DEVICE_CPU &&
       view.device.ordinal != 0) ||
      view.rank > static_cast<std::uint32_t>(nvinfer1::Dims::MAX_DIMS) ||
      !TensorBytes(view, &expected_bytes) ||
      view.size_bytes != expected_bytes) {
    return false;
  }
  const auto address =
      reinterpret_cast<std::uintptr_t>(view.data);
  return address % value.alignment == 0u;
}

bool MatchesDataType(nvinfer1::DataType type, VLAForgeDType dtype) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return dtype == VLAFORGE_DTYPE_F32;
    case nvinfer1::DataType::kHALF:
      return dtype == VLAFORGE_DTYPE_F16;
    case nvinfer1::DataType::kINT32:
      return dtype == VLAFORGE_DTYPE_I32;
    case nvinfer1::DataType::kBOOL:
      return dtype == VLAFORGE_DTYPE_BOOL;
    case nvinfer1::DataType::kUINT8:
      return dtype == VLAFORGE_DTYPE_U8;
    case nvinfer1::DataType::kBF16:
      return dtype == VLAFORGE_DTYPE_BF16;
    case nvinfer1::DataType::kINT64:
      return dtype == VLAFORGE_DTYPE_I64;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kINT4:
      return false;
  }
  return false;
}

bool StaticDimensionsMatch(const nvinfer1::Dims& expected,
                           const VLAForgeTensorView& actual) {
  if (expected.nbDims < 0 ||
      static_cast<std::uint32_t>(expected.nbDims) != actual.rank) {
    return false;
  }
  for (std::int32_t index = 0; index < expected.nbDims; ++index) {
    if (expected.d[index] >= 0 &&
        expected.d[index] != actual.dimensions[index]) {
      return false;
    }
  }
  return true;
}

bool ResolvedDimensionsMatch(const nvinfer1::Dims& expected,
                             const VLAForgeTensorView& actual) {
  if (expected.nbDims < 0 ||
      static_cast<std::uint32_t>(expected.nbDims) != actual.rank) {
    return false;
  }
  for (std::int32_t index = 0; index < expected.nbDims; ++index) {
    if (expected.d[index] < 0 ||
        expected.d[index] != actual.dimensions[index]) {
      return false;
    }
  }
  return true;
}

nvinfer1::Dims ToDims(const VLAForgeTensorView& view) {
  nvinfer1::Dims result{};
  result.nbDims = static_cast<std::int32_t>(view.rank);
  for (std::uint32_t index = 0; index < view.rank; ++index) {
    result.d[index] = view.dimensions[index];
  }
  return result;
}

bool TargetSyntax(std::string_view target) {
  return target.size() == 5u && target.substr(0u, 3u) == "sm_" &&
         target[3] >= '0' && target[3] <= '9' &&
         target[4] >= '0' && target[4] <= '9';
}

bool BackendVariantCompatible(std::string_view variant) {
  constexpr std::string_view kPrefix = "tensorrt-";
  if (variant.size() <= kPrefix.size() ||
      variant.substr(0u, kPrefix.size()) != kPrefix) {
    return false;
  }
  std::uint32_t major = 0u;
  std::size_t cursor = kPrefix.size();
  bool has_digit = false;
  while (cursor < variant.size() &&
         variant[cursor] >= '0' && variant[cursor] <= '9') {
    has_digit = true;
    major = major * 10u +
            static_cast<std::uint32_t>(variant[cursor] - '0');
    ++cursor;
  }
  return has_digit && cursor < variant.size() &&
         variant[cursor] == '.' &&
         major == static_cast<std::uint32_t>(NV_TENSORRT_MAJOR);
}

bool ReadFile(std::string_view path, std::vector<char>* data) {
  if (data == nullptr || path.empty()) {
    return false;
  }
  std::ifstream stream(std::string(path), std::ios::binary | std::ios::ate);
  if (!stream) {
    return false;
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    return false;
  }
  data->resize(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  return static_cast<bool>(
      stream.read(data->data(), static_cast<std::streamsize>(data->size())));
}

}  // namespace

struct VLAForgeRegionExecutable {
  std::uint32_t region_id = 0u;
  std::uint32_t abi_version = 0u;
  int device_ordinal = 0;
  cudaStream_t stream = nullptr;
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
  TrtUniquePtr<nvinfer1::IExecutionContext> context;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::array<Binding, kMaximumBindings> inputs{};
  std::array<Binding, kMaximumBindings> outputs{};
  std::array<char, kErrorCapacity> error{};

  VLAForgeStatus RecordError(const char* message) noexcept {
    const char* text =
        message == nullptr ? "TensorRT backend error" : message;
    std::snprintf(error.data(), error.size(), "%s", text);
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }

  VLAForgeStatus RecordCuda(cudaError_t status,
                            const char* operation) noexcept {
    std::snprintf(error.data(), error.size(), "%s: %s", operation,
                  cudaGetErrorString(status));
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }
};

namespace {

VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
                      VLAForgeRegionExecutable** output) {
  if (output == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT output executable pointer is null");
  }
  *output = nullptr;
  if (options == nullptr ||
      options->struct_size < sizeof(*options) ||
      (options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION &&
       options->abi_version !=
           VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION) ||
      options->device.kind != VLAFORGE_DEVICE_CUDA ||
      options->device.ordinal < 0) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TensorRT create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_OUT_OF_MEMORY,
        "TensorRT executable allocation failed");
  }
  executable->region_id = options->region_id;
  executable->abi_version = options->abi_version;
  executable->device_ordinal = options->device.ordinal;
  auto status = cudaSetDevice(executable->device_ordinal);
  if (status == cudaSuccess) {
    status = cudaStreamCreateWithFlags(
        &executable->stream, cudaStreamNonBlocking);
  }
  if (status != cudaSuccess) {
    delete executable;
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR,
        "TensorRT stream creation failed");
  }
  *output = executable;
  return vlaforge_status_ok();
}

bool TargetMatches(VLAForgeRegionExecutable& executable,
                   const VLAForgeArtifactDescriptor& artifact) {
  if (artifact.target == nullptr || artifact.target_size == 0u) {
    return false;
  }
  const std::string_view target(artifact.target, artifact.target_size);
  if (!TargetSyntax(target)) {
    return false;
  }
  cudaDeviceProp properties{};
  const auto status =
      cudaGetDeviceProperties(&properties, executable.device_ordinal);
  if (status != cudaSuccess) {
    executable.RecordCuda(status, "CUDA device query failed");
    return false;
  }
  return properties.major == target[3] - '0' &&
         properties.minor == target[4] - '0';
}

VLAForgeStatus LoadImpl(VLAForgeRegionExecutable* executable,
                        const VLAForgeArtifactDescriptor* artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version != executable->abi_version ||
      artifact->path == nullptr || artifact->path_size == 0u) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TensorRT artifact descriptor");
  }
  if (!TargetMatches(*executable, *artifact)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT artifact target mismatch");
  }
  if (artifact->backend_variant == nullptr ||
      artifact->backend_variant_size == 0u ||
      !BackendVariantCompatible(std::string_view(
          artifact->backend_variant, artifact->backend_variant_size))) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT backend variant mismatch");
  }
  const auto select_status = cudaSetDevice(executable->device_ordinal);
  if (select_status != cudaSuccess) {
    return executable->RecordCuda(
        select_status, "CUDA device selection failed");
  }
  std::vector<char> engine_data;
  if (!ReadFile(
          std::string_view(artifact->path, artifact->path_size),
          &engine_data)) {
    return executable->RecordError("TensorRT engine read failed");
  }
  executable->runtime.reset(
      nvinfer1::createInferRuntime(GlobalLogger()));
  if (executable->runtime == nullptr) {
    return executable->RecordError("TensorRT runtime creation failed");
  }
  executable->engine.reset(
      executable->runtime->deserializeCudaEngine(
          engine_data.data(), engine_data.size()));
  if (executable->engine == nullptr) {
    return executable->RecordError(
        "TensorRT engine deserialization failed");
  }
  executable->context.reset(
      executable->engine->createExecutionContext());
  if (executable->context == nullptr) {
    return executable->RecordError(
        "TensorRT execution context creation failed");
  }
  executable->input_names.clear();
  executable->output_names.clear();
  executable->inputs.fill(Binding{});
  executable->outputs.fill(Binding{});
  const auto io_count = executable->engine->getNbIOTensors();
  if (io_count < 0 ||
      static_cast<std::size_t>(io_count) > 2u * kMaximumBindings) {
    return executable->RecordError("TensorRT I/O count is unsupported");
  }
  for (std::int32_t index = 0; index < io_count; ++index) {
    const char* name = executable->engine->getIOTensorName(index);
    if (name == nullptr || name[0] == '\0') {
      return executable->RecordError(
          "TensorRT engine contains an unnamed I/O tensor");
    }
    const auto mode = executable->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      executable->input_names.emplace_back(name);
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      executable->output_names.emplace_back(name);
    } else {
      return executable->RecordError(
          "TensorRT engine contains an invalid I/O tensor");
    }
  }
  if (executable->input_names.size() > kMaximumBindings ||
      executable->output_names.size() > kMaximumBindings) {
    return executable->RecordError("TensorRT binding capacity exceeded");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Load(VLAForgeRegionExecutable* executable,
                    const VLAForgeArtifactDescriptor* artifact) {
  try {
    return LoadImpl(executable, artifact);
  } catch (const std::bad_alloc&) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_OUT_OF_MEMORY,
        "TensorRT artifact loading ran out of memory");
  } catch (const std::exception& error) {
    if (executable != nullptr) {
      return executable->RecordError(error.what());
    }
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR,
        "TensorRT artifact loading failed");
  } catch (...) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_BACKEND_ERROR,
        "TensorRT artifact loading failed");
  }
}

VLAForgeStatus QueryWorkspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == nullptr || requirement == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TensorRT workspace query");
  }
  requirement->size_bytes = 0u;
  requirement->alignment = 1u;
  requirement->device = {
      VLAFORGE_DEVICE_CUDA, executable->device_ordinal};
  return vlaforge_status_ok();
}

VLAForgeStatus BindTensor(VLAForgeRegionExecutable* executable,
                          std::uint32_t index,
                          const VLAForgeBoundTensor& value,
                          bool input) {
  if (executable == nullptr ||
      index >= kMaximumBindings ||
      !ValidTensor(value, executable->device_ordinal)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TensorRT tensor binding");
  }
  const auto& names =
      input ? executable->input_names : executable->output_names;
  if (executable->engine != nullptr && index >= names.size()) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT tensor binding index is out of range");
  }
  auto& bindings = input ? executable->inputs : executable->outputs;
  bindings[index] = Binding{value, true};
  return vlaforge_status_ok();
}

VLAForgeStatus BindInput(VLAForgeRegionExecutable* executable,
                         std::uint32_t index,
                         const VLAForgeTensorView* tensor) {
  if (tensor == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT input tensor is null");
  }
  const VLAForgeBoundTensor value{
      sizeof(VLAForgeBoundTensor), *tensor,
      VLAFORGE_LAYOUT_CONTIGUOUS, 1u};
  return BindTensor(executable, index, value, true);
}

VLAForgeStatus BindOutput(VLAForgeRegionExecutable* executable,
                          std::uint32_t index,
                          const VLAForgeTensorView* tensor) {
  if (tensor == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT output tensor is null");
  }
  const VLAForgeBoundTensor value{
      sizeof(VLAForgeBoundTensor), *tensor,
      VLAFORGE_LAYOUT_CONTIGUOUS, 1u};
  return BindTensor(executable, index, value, false);
}

VLAForgeStatus BindValue(VLAForgeRegionExecutable* executable,
                         std::uint32_t index,
                         const VLAForgeValueView* value,
                         bool input) {
  if (value == nullptr || value->struct_size < sizeof(*value) ||
      value->kind != VLAFORGE_VALUE_TENSOR) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT accepts tensor values only");
  }
  return BindTensor(executable, index, value->value.tensor, input);
}

VLAForgeStatus BindInputValue(VLAForgeRegionExecutable* executable,
                              std::uint32_t index,
                              const VLAForgeValueView* value) {
  return BindValue(executable, index, value, true);
}

VLAForgeStatus BindOutputValue(VLAForgeRegionExecutable* executable,
                               std::uint32_t index,
                               const VLAForgeValueView* value) {
  return BindValue(executable, index, value, false);
}

VLAForgeStatus BindWorkspace(VLAForgeRegionExecutable* executable,
                             void* workspace,
                             std::uint64_t workspace_size) {
  if (executable == nullptr ||
      (workspace_size != 0u && workspace == nullptr)) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "invalid TensorRT workspace binding");
  }
  if (workspace_size != 0u) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT execution context owns its workspace");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus BindOne(VLAForgeRegionExecutable& executable,
                       const std::string& name,
                       const Binding& binding,
                       bool input) {
  if (!binding.bound) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT binding is missing");
  }
  const auto& view = binding.value.tensor;
  const auto type = executable.engine->getTensorDataType(name.c_str());
  const auto format = executable.engine->getTensorFormat(name.c_str());
  const auto location =
      executable.engine->getTensorLocation(name.c_str());
  const auto expected_device =
      location == nvinfer1::TensorLocation::kHOST
      ? VLAFORGE_DEVICE_CPU
      : VLAFORGE_DEVICE_CUDA;
  if (format != nvinfer1::TensorFormat::kLINEAR ||
      !MatchesDataType(type, view.dtype) ||
      view.device.kind != expected_device) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT tensor metadata mismatch");
  }
  if (input) {
    const auto engine_shape =
        executable.engine->getTensorShape(name.c_str());
    if (!StaticDimensionsMatch(engine_shape, view)) {
      return vlaforge_status_error(
          VLAFORGE_STATUS_FAILED_PRECONDITION,
          "TensorRT input shape is outside the engine profile");
    }
    if (!executable.context->setInputShape(
            name.c_str(), ToDims(view))) {
      return executable.RecordError(
          "TensorRT dynamic input shape rejected");
    }
  } else {
    const auto resolved_shape =
        executable.context->getTensorShape(name.c_str());
    if (!ResolvedDimensionsMatch(resolved_shape, view)) {
      return vlaforge_status_error(
          VLAFORGE_STATUS_FAILED_PRECONDITION,
          "TensorRT output shape mismatch");
    }
  }
  if (!executable.context->setTensorAddress(
          name.c_str(), view.data)) {
    return executable.RecordError(
        "TensorRT tensor address rejected");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || executable->engine == nullptr ||
      executable->context == nullptr || executable->stream == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_FAILED_PRECONDITION,
        "TensorRT executable is not loaded");
  }
  auto status = cudaSetDevice(executable->device_ordinal);
  if (status != cudaSuccess) {
    return executable->RecordCuda(status, "CUDA device selection failed");
  }
  for (std::size_t index = 0;
       index < executable->input_names.size(); ++index) {
    const auto result = BindOne(
        *executable, executable->input_names[index],
        executable->inputs[index], true);
    if (result.code != VLAFORGE_STATUS_OK) {
      return result;
    }
  }
  for (std::size_t index = 0;
       index < executable->output_names.size(); ++index) {
    const auto result = BindOne(
        *executable, executable->output_names[index],
        executable->outputs[index], false);
    if (result.code != VLAFORGE_STATUS_OK) {
      return result;
    }
  }
  if (!executable->context->allInputDimensionsSpecified()) {
    return executable->RecordError(
        "TensorRT input dimensions are incomplete");
  }
  if (!executable->context->enqueueV3(executable->stream)) {
    return executable->RecordError("TensorRT enqueueV3 failed");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || executable->stream == nullptr) {
    return vlaforge_status_error(
        VLAFORGE_STATUS_INVALID_ARGUMENT,
        "TensorRT executable is null");
  }
  const auto status = cudaStreamSynchronize(executable->stream);
  if (status != cudaSuccess) {
    return executable->RecordCuda(
        status, "TensorRT stream synchronization failed");
  }
  return vlaforge_status_ok();
}

void Destroy(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr) {
    return;
  }
  cudaSetDevice(executable->device_ordinal);
  executable->context.reset();
  executable->engine.reset();
  executable->runtime.reset();
  if (executable->stream != nullptr) {
    cudaStreamDestroy(executable->stream);
    executable->stream = nullptr;
  }
  delete executable;
}

const VLAForgeRegionExecutableApi kApi{
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

const VLAForgeRegionExecutableValueApi kValueApi{
    sizeof(VLAForgeRegionExecutableValueApi),
    VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    &Create,
    &Load,
    &QueryWorkspace,
    &BindInputValue,
    &BindOutputValue,
    &BindWorkspace,
    &Run,
    &Synchronize,
    &Destroy,
};

}  // namespace

extern "C" const VLAForgeRegionExecutableApi*
vlaforge_tensorrt_region_executable_api(void) {
  return &kApi;
}

extern "C" const VLAForgeRegionExecutableValueApi*
vlaforge_tensorrt_region_executable_value_api(void) {
  return &kValueApi;
}
