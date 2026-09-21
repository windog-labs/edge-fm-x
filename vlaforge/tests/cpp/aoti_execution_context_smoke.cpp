#include "vlaforge/backends/aoti_region_executable.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void Check(VLAForgeStatus status) {
  if (status.code != VLAFORGE_STATUS_OK) {
    throw std::runtime_error(std::string(status.message, status.message_size));
  }
}

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

VLAForgeValueView View(at::Tensor& tensor) {
  VLAForgeValueView view{};
  view.struct_size = sizeof(view);
  view.kind = VLAFORGE_VALUE_TENSOR;
  view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {tensor.data_ptr(), static_cast<std::uint64_t>(tensor.nbytes()),
       tensor.sizes().data(), static_cast<std::uint32_t>(tensor.dim()),
       VLAFORGE_DTYPE_F32,
       {tensor.is_cuda() ? VLAFORGE_DEVICE_CUDA : VLAFORGE_DEVICE_CPU,
        tensor.is_cuda() ? tensor.get_device() : 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 4u};
  return view;
}

int Run(const std::string& artifact_path, bool cuda, const std::string& target) {
  const auto kind = cuda ? VLAFORGE_DEVICE_CUDA : VLAFORGE_DEVICE_CPU;
  std::optional<c10::cuda::CUDAGuard> guard;
  if (cuda) {
    guard.emplace(0);
  }
  VLAForgeExecutionContext* raw_context = nullptr;
  const VLAForgeExecutionContextOptions context_options{
      sizeof(context_options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {kind, 0}};
  Check(vlaforge_execution_context_create(&context_options, &raw_context));
  std::unique_ptr<VLAForgeExecutionContext,
                  decltype(&vlaforge_execution_context_destroy)>
      context(raw_context, &vlaforge_execution_context_destroy);
  VLAForgeExecutionContextView context_view{};
  context_view.struct_size = sizeof(context_view);
  Check(vlaforge_execution_context_get_view(context.get(), &context_view));
  if (cuda) {
    Require(context_view.native_stream != nullptr &&
                context_view.native_stream !=
                    c10::cuda::getDefaultCUDAStream(0).stream(),
            "runtime context must own a non-default stream");
  }
  const auto* api = vlaforge_aoti_region_executable_value_api();
  const auto* extension = vlaforge_aoti_region_execution_extension_api();
  Check(vlaforge_region_executable_value_api_validate(api));
  Check(vlaforge_region_execution_extension_api_validate(extension));
  using Region = std::unique_ptr<VLAForgeRegionExecutable,
                                  VLAForgeRegionDestroyFn>;
  std::array<Region, 2> regions = {
      Region(nullptr, api->destroy), Region(nullptr, api->destroy)};
  for (std::uint32_t index = 0; index < regions.size(); ++index) {
    VLAForgeRegionExecutable* executable = nullptr;
    const VLAForgeRegionCreateOptions options{
        sizeof(options), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
        index, {kind, 0}};
    Check(api->create(&options, &executable));
    regions[index].reset(executable);
    auto wrong = context_view;
    wrong.device.ordinal = 1;
    Require(extension->bind_context(executable, &wrong).code ==
                VLAFORGE_STATUS_INVALID_ARGUMENT,
            "mismatched context device was accepted");
    Check(extension->bind_context(executable, &context_view));
    const VLAForgeArtifactDescriptor artifact{
        sizeof(artifact), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
        artifact_path.data(), artifact_path.size(), nullptr, 0u,
        nullptr, 0u, target.data(), target.size(), nullptr, 0u};
    Check(api->load(executable, &artifact));
  }

  const auto device = cuda ? at::Device(at::kCUDA) : at::Device(at::kCPU);
  const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
  auto input = at::arange(16, options).reshape({4, 4}).div(8.0);
  auto gain = at::full({}, 0.75, options);
  auto intermediate = at::empty_like(input);
  auto output = at::empty_like(input);
  auto input_view = View(input);
  auto gain_view = View(gain);
  auto intermediate_view = View(intermediate);
  auto output_view = View(output);
  Check(api->bind_input(regions[0].get(), 0u, &input_view));
  Check(api->bind_input(regions[0].get(), 1u, &gain_view));
  Check(api->bind_output(regions[0].get(), 0u, &intermediate_view));
  Check(api->bind_input(regions[1].get(), 0u, &intermediate_view));
  Check(api->bind_input(regions[1].get(), 1u, &gain_view));
  Check(api->bind_output(regions[1].get(), 0u, &output_view));
  if (cuda) {
    c10::cuda::device_synchronize();
  }
  // Mutating the caller's descriptor must not mutate a bound context.
  context_view.native_stream = nullptr;
  for (int run = 0; run < 3; ++run) {
    std::optional<c10::cuda::CUDAStream> previous;
    if (cuda) {
      previous = c10::cuda::getCurrentCUDAStream(0);
    }
    Check(api->run(regions[0].get()));
    Check(api->run(regions[1].get()));
    if (cuda) {
      Require(c10::cuda::getCurrentCUDAStream(0) == *previous,
              "AOTI leaked the borrowed stream into the caller");
      Require(extension->bind_context(regions[0].get(), nullptr).code ==
                  VLAFORGE_STATUS_FAILED_PRECONDITION,
              "pending AOTI execution allowed context rebinding");
    }
    Check(api->synchronize(regions[0].get()));
    Check(api->synchronize(regions[1].get()));
    Check(vlaforge_execution_context_synchronize(context.get()));
    const auto first = (input.sin() + input.square()) * gain;
    const auto expected = (first.sin() + first.square()) * gain;
    Require(at::allclose(output, expected, 1e-5, 1e-6),
            "shared-context Region chain output mismatch");
  }
  // Detaching restores the old ABI's execution path.
  Check(extension->bind_context(regions[0].get(), nullptr));
  Check(api->run(regions[0].get()));
  Check(api->synchronize(regions[0].get()));
  Require(at::allclose(intermediate, (input.sin() + input.square()) * gain,
                       1e-5, 1e-6),
          "legacy path changed after detaching execution context");
  std::printf("EXECUTION_CONTEXT passed device=%s regions=2 runs=3\n",
              cuda ? "cuda" : "cpu");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 || (std::string(argv[2]) != "cuda" &&
                    std::string(argv[2]) != "cpu")) {
    std::fprintf(stderr, "usage: %s AUDIT_ARTIFACT cuda|cpu TARGET\n", argv[0]);
    return 2;
  }
  try {
    return Run(argv[1], std::string(argv[2]) == "cuda", argv[3]);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "execution context smoke: %s\n", error.what());
    return 1;
  }
}
