#include "vlaforge/backends/tensorrt_region_executable.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include <unistd.h>

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

bool BuildIdentityEngine(const std::string& path) {
  Logger logger;
  TrtUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  if (builder == nullptr) {
    return false;
  }
  TrtUniquePtr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0u));
  TrtUniquePtr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  if (network == nullptr || config == nullptr) {
    return false;
  }
  nvinfer1::Dims shape{};
  shape.nbDims = 1;
  shape.d[0] = 4;
  auto* input =
      network->addInput("input", nvinfer1::DataType::kFLOAT, shape);
  if (input == nullptr) {
    return false;
  }
  auto* identity = network->addIdentity(*input);
  if (identity == nullptr || identity->getOutput(0) == nullptr) {
    return false;
  }
  identity->getOutput(0)->setName("output");
  network->markOutput(*identity->getOutput(0));
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

}  // namespace

int main() {
  const std::string engine_path =
      "/tmp/vlaforge-tensorrt-smoke-" +
      std::to_string(static_cast<long long>(getpid())) + ".engine";
  if (!BuildIdentityEngine(engine_path)) {
    std::fprintf(stderr, "failed to build TensorRT identity engine\n");
    return 1;
  }

  constexpr std::array<float, 4> kInput = {
      0.25F, -0.5F, 1.5F, 2.0F};
  std::array<float, 4> output{};
  void* device_input = nullptr;
  void* device_output = nullptr;
  if (!CheckCuda(
          cudaMalloc(&device_input, sizeof(kInput)), "cudaMalloc input") ||
      !CheckCuda(
          cudaMalloc(&device_output, sizeof(output)), "cudaMalloc output") ||
      !CheckCuda(
          cudaMemcpy(device_input, kInput.data(), sizeof(kInput),
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
  const std::string variant = "tensorrt-10.3-cu126";
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
  VLAForgeValueView input_view{};
  input_view.struct_size = sizeof(VLAForgeValueView);
  input_view.kind = VLAFORGE_VALUE_TENSOR;
  input_view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {device_input, sizeof(kInput), kShape.data(), 1u,
       VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      256u,
  };
  VLAForgeValueView output_view{};
  output_view.struct_size = sizeof(VLAForgeValueView);
  output_view.kind = VLAFORGE_VALUE_TENSOR;
  output_view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {device_output, sizeof(output), kShape.data(), 1u,
       VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      256u,
  };

  const bool passed =
      Check(api->create(&options, &executable), "create") &&
      Check(api->load(executable, &artifact), "load") &&
      Check(api->bind_input(executable, 0u, &input_view), "bind input") &&
      Check(api->bind_output(executable, 0u, &output_view), "bind output") &&
      Check(api->run(executable), "run") &&
      Check(api->synchronize(executable), "synchronize") &&
      CheckCuda(
          cudaMemcpy(output.data(), device_output, sizeof(output),
                     cudaMemcpyDeviceToHost),
          "copy output");
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
    if (std::fabs(output[index] - kInput[index]) > 1.0e-6F) {
      std::fprintf(stderr, "output mismatch at %zu\n", index);
      return 5;
    }
  }
  std::printf("TensorRT Region on-device smoke passed\n");
  return 0;
}
