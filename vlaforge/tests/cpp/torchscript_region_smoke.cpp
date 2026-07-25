#include "vlaforge/backends/torchscript_region_executable.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
                        std::uint32_t rank) {
  return VLAForgeTensorView{
      tensor.data_ptr(),
      static_cast<std::uint64_t>(tensor.nbytes()),
      dimensions,
      rank,
      VLAFORGE_DTYPE_F32,
      {VLAFORGE_DEVICE_CPU, 0},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s REGION.pt\n", argv[0]);
    return 2;
  }
  at::Tensor input =
      at::arange(16, at::TensorOptions().dtype(at::kFloat))
          .reshape({4, 4})
          .div(8.0);
  at::Tensor gain =
      at::full({}, 0.75, at::TensorOptions().dtype(at::kFloat));
  at::Tensor output = at::empty_like(input);
  constexpr std::array<std::int64_t, 2> kMatrixShape = {4, 4};
  const auto* api = vlaforge_torchscript_region_executable_api();
  if (!Check(vlaforge_region_executable_api_validate(api),
             "validate API")) {
    return 3;
  }
  VLAForgeRegionExecutable* executable = nullptr;
  const VLAForgeRegionCreateOptions options{
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      0u,
      {VLAFORGE_DEVICE_CPU, 0},
  };
  if (!Check(api->create(&options, &executable), "create")) {
    return 4;
  }
  const std::string archive_path(argv[1]);
  const VLAForgeArtifactDescriptor artifact{
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      archive_path.data(),
      archive_path.size(),
      nullptr,
      0u,
      nullptr,
      0u,
      nullptr,
      0u,
      nullptr,
      0u,
  };
  auto input_view = View(input, kMatrixShape.data(), 2u);
  auto gain_view = View(gain, nullptr, 0u);
  auto output_view = View(output, kMatrixShape.data(), 2u);
  const bool passed =
      Check(api->load(executable, &artifact), "load") &&
      Check(api->bind_input(executable, 0u, &input_view), "bind input") &&
      Check(api->bind_input(executable, 1u, &gain_view), "bind gain") &&
      Check(api->bind_output(executable, 0u, &output_view), "bind output") &&
      Check(api->run(executable), "run") &&
      Check(api->synchronize(executable), "synchronize");
  api->destroy(executable);
  if (!passed) {
    return 5;
  }
  const auto* values = output.contiguous().const_data_ptr<float>();
  std::printf("OUTPUT");
  for (std::int64_t index = 0; index < output.numel(); ++index) {
    std::printf(",%.9g", static_cast<double>(values[index]));
  }
  std::printf("\n");
  return 0;
}
