#include "vlaforge/backends/tensorrt_region_executable.h"
#include "vlaforge/runtime/cuda_graph.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#if defined(VLAFORGE_TENSORRT_STANDARD_PLUGINS)
#include <NvInferPlugin.h>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR && message != nullptr) {
      std::fprintf(stderr, "TensorRT smoke: %s\n", message);
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

bool Check(VLAForgeStatus status, const char* operation) {
  if (status.code == VLAFORGE_STATUS_OK) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %.*s\n", operation,
               static_cast<int>(status.message_size), status.message);
  return false;
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool BuildIdentityEngine(const std::string& path, bool dynamic,
                         bool plugin = false) {
  Logger logger;
#if defined(VLAFORGE_TENSORRT_STANDARD_PLUGINS)
  if (plugin && !initLibNvInferPlugins(&logger, "")) return false;
  std::shared_ptr<void> standard_plugin;
#endif
  TrtUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  if (builder == nullptr) {
    return false;
  }
  std::uint32_t network_flags = 0u;
#if NV_TENSORRT_MAJOR < 10
  network_flags = 1u << static_cast<std::uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  TrtUniquePtr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(network_flags));
  TrtUniquePtr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  if (network == nullptr || config == nullptr) {
    return false;
  }
  nvinfer1::Dims shape{};
  shape.nbDims = plugin ? 3 : 1;
  shape.d[0] = plugin ? 1 : (dynamic ? -1 : 4);
  if (plugin) {
    shape.d[1] = 1;
    shape.d[2] = 768;
  }
  auto* input =
      network->addInput("input", nvinfer1::DataType::kFLOAT, shape);
  if (input == nullptr) {
    return false;
  }
  nvinfer1::ITensor* result = nullptr;
#if defined(VLAFORGE_TENSORRT_STANDARD_PLUGINS)
  std::array<float, 768> gamma{};
  std::array<float, 768> beta{};
  gamma.fill(1.0F);
  if (plugin) {
    auto* creator = getPluginRegistry()->getPluginCreator("LayerNorm", "1", "");
    if (creator == nullptr) return false;
    const float epsilon = 1.0e-6F;
    const nvinfer1::PluginField parameter{
        "epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1};
    const nvinfer1::PluginFieldCollection fields{1, &parameter};
    auto* instance = creator->createPlugin("normalization", &fields);
    if (instance == nullptr) return false;
    standard_plugin = std::shared_ptr<void>(instance, [](auto* value) { value->destroy(); });
    nvinfer1::Dims weights_shape{};
    weights_shape.nbDims = 1;
    weights_shape.d[0] = 768;
    auto* scale = network->addConstant(weights_shape,
        {nvinfer1::DataType::kFLOAT, gamma.data(), 768});
    auto* bias = network->addConstant(weights_shape,
        {nvinfer1::DataType::kFLOAT, beta.data(), 768});
    if (scale == nullptr || bias == nullptr) return false;
    std::array<nvinfer1::ITensor*, 3> operands = {input, scale->getOutput(0), bias->getOutput(0)};
    auto* layer = network->addPluginV2(operands.data(), 3, *instance);
    if (layer == nullptr) return false;
    layer->setPrecision(nvinfer1::DataType::kFLOAT);
    layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    result = layer->getOutput(0);
  }
#endif
  if (!plugin) {
    auto* identity = network->addIdentity(*input);
    if (identity == nullptr) return false;
    result = identity->getOutput(0);
  }
  if (result == nullptr) return false;
  result->setName("output");
  network->markOutput(*result);
  if (dynamic) {
    auto* profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
      return false;
    }
    nvinfer1::Dims bound{};
    bound.nbDims = 1;
    bound.d[0] = 1;
    if (!profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, bound)) {
      return false;
    }
    bound.d[0] = 4;
    if (!profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, bound)) {
      return false;
    }
    bound.d[0] = 8;
    if (!profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, bound) ||
        config->addOptimizationProfile(profile) < 0) {
      return false;
    }
  }
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, 1u << 20u);
  TrtUniquePtr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  if (serialized == nullptr) {
    return false;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(static_cast<const char*>(serialized->data()),
               static_cast<std::streamsize>(serialized->size()));
  return static_cast<bool>(output);
}

bool ExerciseSharedContext(VLAForgeRegionExecutable* executable,
                           const VLAForgeArtifactDescriptor& artifact,
                           void* device_input, void* device_output) {
  const auto* api = vlaforge_tensorrt_region_executable_value_api();
  const auto* extension = vlaforge_tensorrt_region_execution_extension_api();
  const auto* graph_api = vlaforge_cuda_graph_backend_api();
  VLAForgeExecutionContext* owner = nullptr;
  VLAForgeGraphExecutable* graph = nullptr;
  bool capturing = false;
  const VLAForgeExecutionContextOptions options{
      sizeof(VLAForgeExecutionContextOptions), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {VLAFORGE_DEVICE_CUDA, 0}};
  VLAForgeExecutionContextView view{};
  view.struct_size = sizeof(view);
  bool passed = Check(vlaforge_region_execution_extension_api_validate(extension), "extension ABI") &&
      graph_api != nullptr && Check(vlaforge_execution_context_create(&options, &owner), "context create") &&
      Check(vlaforge_execution_context_get_view(owner, &view), "context view");
  if (passed) {
    auto wrong = view;
    wrong.device.ordinal = 1;
    passed = extension->bind_context(executable, &wrong).code == VLAFORGE_STATUS_INVALID_ARGUMENT;
    wrong = view;
    wrong.abi_version = 0u;
    passed = passed && extension->bind_context(executable, &wrong).code == VLAFORGE_STATUS_UNSUPPORTED_ABI;
    passed = passed && Check(extension->bind_context(executable, &view), "bind shared context");
  }
  const std::array<float, 4> changed = {3.0F, 7.0F, -2.0F, 8.0F};
  if (passed) {
    passed = CheckCuda(cudaMemcpyAsync(device_input, changed.data(), sizeof(changed), cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(view.native_stream)), "shared stream input") &&
        Check(api->run(executable), "shared run") &&
        extension->bind_context(executable, nullptr).code == VLAFORGE_STATUS_FAILED_PRECONDITION &&
        api->load(executable, &artifact).code == VLAFORGE_STATUS_FAILED_PRECONDITION &&
        Check(api->synchronize(executable), "shared synchronize") &&
        Check(graph_api->create(&view, &graph), "graph create") &&
        graph_api->launch(graph).code == VLAFORGE_STATUS_FAILED_PRECONDITION;
  }
  if (passed) {
    passed = Check(graph_api->begin(graph), "graph begin");
    capturing = passed;
  }
  if (passed) {
    passed = Check(api->run(executable), "captured run") &&
        api->synchronize(executable).code == VLAFORGE_STATUS_FAILED_PRECONDITION &&
        Check(graph_api->end(graph), "graph end");
    if (passed) { capturing = false; }
  }
  if (passed) {
    const std::array<float, 4> zero{};
    std::array<float, 4> output{};
    passed = CheckCuda(cudaMemset(device_output, 0, sizeof(changed)), "clear output") &&
        Check(graph_api->launch(graph), "graph launch") &&
        Check(api->synchronize(executable), "replay synchronize") &&
        CheckCuda(cudaMemcpy(output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost), "replay output") &&
        output != zero && output == changed;
  }
  if (graph != nullptr && !capturing) {
    graph_api->destroy(graph, 0u);
    graph = nullptr;
  }
  if (passed) {
    passed = Check(graph_api->create(&view, &graph), "abort graph create") &&
        Check(graph_api->begin(graph), "abort graph begin");
    capturing = passed;
    if (passed) {
      passed = Check(api->run(executable), "abort captured run") &&
          Check(graph_api->abort(graph), "graph abort");
      if (passed) { capturing = false; }
    }
  }
  if (capturing) { (void)graph_api->abort(graph); }
  if (graph != nullptr) { graph_api->destroy(graph, 0u); }
  passed = Check(api->synchronize(executable), "context final drain") && passed;
  passed = Check(extension->bind_context(executable, nullptr), "detach context") && passed;
  vlaforge_execution_context_destroy(owner);
  return passed && Check(api->run(executable), "private stream after detach") &&
      Check(api->synchronize(executable), "private stream drain");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--plugin-build") {
    return BuildIdentityEngine(argv[2], false, true) ? 0 : 1;
  }
  const bool dynamic = argc == 2 && std::string(argv[1]) == "--dynamic";
  const bool shared_context = argc == 2 && std::string(argv[1]) == "--context";
  const bool plugin = argc == 2 && std::string(argv[1]) == "--plugin";
  const std::string engine_path =
      "/tmp/vlaforge-tensorrt-smoke-" +
      std::to_string(static_cast<long long>(getpid())) + ".engine";
  bool built = false;
  if (plugin) {
    // Build in a separate image so the loader cannot inherit registration.
    const auto child = fork();
    if (child == 0) {
      execl("/proc/self/exe", argv[0], "--plugin-build", engine_path.c_str(), static_cast<char*>(nullptr));
      _exit(127);
    }
    int status = 0;
    built = child > 0 && waitpid(child, &status, 0) == child &&
        WIFEXITED(status) && WEXITSTATUS(status) == 0;
  } else {
    built = BuildIdentityEngine(engine_path, dynamic);
  }
  if (!built) {
    std::fprintf(stderr, "failed to build TensorRT identity engine\n");
    return 1;
  }

  constexpr std::array<float, 4> kInput = {
      0.25F, -0.5F, 1.5F, 2.0F};
  std::array<float, 768> plugin_input{};
  for (std::size_t index = 0; index < plugin_input.size(); ++index) {
    plugin_input[index] = index % 2 == 0 ? -1.0F : 1.0F;
  }
  const auto* host_input = plugin ? plugin_input.data() : kInput.data();
  const auto input_count = plugin ? plugin_input.size() : kInput.size();
  const auto input_bytes = input_count * sizeof(float);
  std::vector<float> output(input_count);
  void* device_input = nullptr;
  void* device_output = nullptr;
  if (!CheckCuda(
          cudaMalloc(&device_input, input_bytes), "cudaMalloc input") ||
      !CheckCuda(
          cudaMalloc(&device_output, input_bytes), "cudaMalloc output") ||
      !CheckCuda(
          cudaMemcpy(device_input, host_input, input_bytes,
                     cudaMemcpyHostToDevice),
          "copy input")) {
    cudaFree(device_output);
    cudaFree(device_input);
    std::remove(engine_path.c_str());
    return 2;
  }

  const auto* api =
      vlaforge_tensorrt_region_executable_value_api();
  if (!Check(
          vlaforge_region_executable_value_api_validate(api),
          "validate value API")) {
    cudaFree(device_output);
    cudaFree(device_input);
    std::remove(engine_path.c_str());
    return 3;
  }
  VLAForgeRegionExecutable* executable = nullptr;
  const VLAForgeRegionCreateOptions options{
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      0u,
      {VLAFORGE_DEVICE_CUDA, 0},
  };
  const std::string target = "sm_87";
  const std::string variant = "tensorrt-" +
      std::to_string(NV_TENSORRT_MAJOR) + "." +
      std::to_string(NV_TENSORRT_MINOR) + "." +
      std::to_string(NV_TENSORRT_PATCH) + "-cu" +
      std::to_string(CUDART_VERSION / 1000) +
      std::to_string((CUDART_VERSION % 1000) / 10);
  const VLAForgeArtifactDescriptor artifact{
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      engine_path.data(),
      engine_path.size(),
      nullptr,
      0u,
      nullptr,
      0u,
      target.data(),
      target.size(),
      variant.data(),
      variant.size(),
  };
  constexpr std::array<std::int64_t, 1> kShape = {4};
  constexpr std::array<std::int64_t, 3> kPluginShape = {1, 1, 768};
  const auto* dimensions = plugin ? kPluginShape.data() : kShape.data();
  const std::uint32_t rank = plugin ? 3u : 1u;
  std::vector<float> expected(host_input, host_input + input_count);
  if (plugin) {
    for (auto& value : expected) value /= std::sqrt(1.0F + 1.0e-6F);
  }
  VLAForgeValueView input_view{};
  input_view.struct_size = sizeof(VLAForgeValueView);
  input_view.kind = VLAFORGE_VALUE_TENSOR;
  input_view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {device_input, input_bytes, dimensions, rank,
       VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      256u,
  };
  VLAForgeValueView output_view{};
  output_view.struct_size = sizeof(VLAForgeValueView);
  output_view.kind = VLAFORGE_VALUE_TENSOR;
  output_view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {device_output, input_bytes, dimensions, rank,
       VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      256u,
  };

  bool passed =
      Check(api->create(&options, &executable), "create") &&
      Check(api->load(executable, &artifact), "load") &&
      Check(api->bind_output(executable, 0u, &output_view), "bind output");
  if (passed) {
    auto wrong_dtype = input_view;
    wrong_dtype.value.tensor.tensor.dtype = VLAFORGE_DTYPE_F16;
    wrong_dtype.value.tensor.tensor.size_bytes = input_bytes / 2;
    passed = Check(api->bind_input(executable, 0u, &wrong_dtype), "bind wrong dtype") &&
        api->run(executable).code == VLAFORGE_STATUS_FAILED_PRECONDITION;
  }
#if NV_TENSORRT_MAJOR < 10
  if (passed) {
    const std::array<std::int64_t, 1> overflow_shape = {
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1};
    auto overflow = input_view;
    overflow.value.tensor.tensor.dimensions = overflow_shape.data();
    overflow.value.tensor.tensor.rank = 1u;
    overflow.value.tensor.tensor.size_bytes =
        static_cast<std::uint64_t>(overflow_shape[0]) * sizeof(float);
    passed = api->bind_input(executable, 0u, &overflow).code ==
        VLAFORGE_STATUS_INVALID_ARGUMENT;
  }
#endif
  passed = passed &&
      Check(api->bind_input(executable, 0u, &input_view), "bind input") &&
      Check(api->bind_output(executable, 0u, &output_view), "bind output") &&
      Check(api->run(executable), "run") &&
      Check(api->synchronize(executable), "synchronize") &&
      CheckCuda(
          cudaMemcpy(output.data(), device_output, input_bytes,
                     cudaMemcpyDeviceToHost),
          "copy output");
  for (std::size_t index = 0; passed && index < output.size(); ++index) {
    passed = std::fabs(output[index] - expected[index]) <= 1.0e-6F;
    if (!passed) {
      std::fprintf(stderr, "TensorRT output mismatch at %zu: actual %.9g, expected %.9g\n",
                   index, static_cast<double>(output[index]), static_cast<double>(expected[index]));
    }
  }
  if (passed && shared_context) {
    passed = ExerciseSharedContext(executable, artifact, device_input, device_output);
  }
  if (executable != nullptr) {
    api->destroy(executable);
  }
  cudaFree(device_output);
  cudaFree(device_input);
  std::remove(engine_path.c_str());
  if (!passed) {
    return 4;
  }
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (std::fabs(output[index] - expected[index]) > 1.0e-6F) {
      std::fprintf(stderr, "output mismatch at %zu\n", index);
      return 5;
    }
  }
  std::printf("TensorRT Region on-device smoke passed\n");
  return 0;
}
