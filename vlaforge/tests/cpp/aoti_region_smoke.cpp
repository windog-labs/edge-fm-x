#include "vlaforge/backends/aoti_region_executable.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace {

bool Check(VLAForgeStatus status, const char* operation) {
  if (status.code == VLAFORGE_STATUS_OK) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %.*s\n", operation,
               static_cast<int>(status.message_size), status.message);
  return false;
}

VLAForgeTensorView View(at::Tensor& tensor,
                        const std::int64_t* dimensions,
                        std::uint32_t rank,
                        VLAForgeDeviceKind device_kind) {
  return VLAForgeTensorView{
      tensor.data_ptr(),
      static_cast<std::uint64_t>(tensor.nbytes()),
      dimensions,
      rank,
      VLAFORGE_DTYPE_F32,
      {device_kind, tensor.is_cuda() ? tensor.get_device() : 0},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 6) {
    std::fprintf(
        stderr,
        "usage: %s ARTIFACT [cuda|cpu] [target|-] [negative-mode] "
        "[backend-variant|-]\n",
        argv[0]);
    return 2;
  }

  const std::string requested_device = argc >= 3 ? argv[2] : "cuda";
  const bool use_cuda = requested_device == "cuda";
  const std::string target = argc >= 4 ? argv[3] : "";
  const std::string mode = argc >= 5 ? argv[4] : "normal";
  const std::string backend_variant = argc >= 6 ? argv[5] : "";
  if (!use_cuda && requested_device != "cpu") {
    std::fprintf(stderr, "device must be cuda or cpu\n");
    return 2;
  }
  std::optional<c10::cuda::CUDAGuard> guard;
  if (use_cuda) {
    guard.emplace(0);
  }
  const auto device =
      use_cuda ? at::Device(at::kCUDA) : at::Device(at::kCPU);
  const auto device_kind =
      use_cuda ? VLAFORGE_DEVICE_CUDA : VLAFORGE_DEVICE_CPU;
  at::Tensor input =
      at::arange(16, at::TensorOptions().dtype(at::kFloat).device(device))
          .reshape({4, 4})
          .div(8.0);
  at::Tensor gain =
      at::full({}, 0.75,
               at::TensorOptions().dtype(at::kFloat).device(device));
  at::Tensor output = at::empty_like(input);
  constexpr std::array<std::int64_t, 2> kMatrixShape = {4, 4};

  const auto* api = vlaforge_aoti_region_executable_api();
  if (!Check(vlaforge_region_executable_api_validate(api),
             "validate API")) {
    return 3;
  }

  VLAForgeRegionExecutable* executable = nullptr;
  const VLAForgeRegionCreateOptions options{
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      0u,
      {device_kind, 0},
  };
  if (!Check(api->create(&options, &executable), "create")) {
    return 4;
  }

  const std::string package_path(argv[1]);
  const VLAForgeArtifactDescriptor artifact{
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      package_path.data(),
      package_path.size(),
      nullptr,
      0u,
      nullptr,
      0u,
      target.empty() || target == "-" ? nullptr : target.data(),
      target.empty() || target == "-" ? 0u : target.size(),
      backend_variant.empty() || backend_variant == "-"
          ? nullptr
          : backend_variant.data(),
      backend_variant.empty() || backend_variant == "-"
          ? 0u
          : backend_variant.size(),
  };
  auto input_view = View(input, kMatrixShape.data(), 2u, device_kind);
  auto gain_view = View(gain, nullptr, 0u, device_kind);
  auto output_view = View(output, kMatrixShape.data(), 2u, device_kind);
  constexpr std::array<std::int64_t, 2> kWrongShape = {2, 8};
  if (mode == "wrong-output-shape") {
    output_view.dimensions = kWrongShape.data();
  } else if (mode == "missing-input-binding") {
    // Deliberately leave gain unbound below.
  } else if (mode != "normal") {
    std::fprintf(stderr, "unknown negative mode\n");
    api->destroy(executable);
    return 2;
  }

  const bool passed =
      Check(api->load(executable, &artifact), "load") &&
      Check(api->bind_input(executable, 0u, &input_view), "bind input") &&
      (mode == "missing-input-binding" ||
       Check(api->bind_input(executable, 1u, &gain_view), "bind gain")) &&
      Check(api->bind_output(executable, 0u, &output_view), "bind output") &&
      Check(api->run(executable), "run") &&
      Check(api->synchronize(executable), "synchronize");
  api->destroy(executable);
  if (!passed) {
    return 5;
  }

  const at::Tensor cpu = output.cpu().contiguous().view({-1});
  const auto* values = cpu.const_data_ptr<float>();
  std::printf("OUTPUT");
  for (std::int64_t index = 0; index < cpu.numel(); ++index) {
    std::printf(",%.9g", static_cast<double>(values[index]));
  }
  std::printf("\n");
  return 0;
}
