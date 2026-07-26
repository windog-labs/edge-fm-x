#include "vlaforge/backends/tensorrt_region_executable.h"

#include <cstddef>
#include <cstdint>

int main() {
  const auto* tensor_api = vlaforge_tensorrt_region_executable_api();
  const auto* value_api =
      vlaforge_tensorrt_region_executable_value_api();
  if (vlaforge_region_executable_api_validate(tensor_api).code !=
          VLAFORGE_STATUS_OK ||
      vlaforge_region_executable_value_api_validate(value_api).code !=
          VLAFORGE_STATUS_OK) {
    return 1;
  }

  auto* executable =
      reinterpret_cast<VLAForgeRegionExecutable*>(static_cast<uintptr_t>(1u));
  const VLAForgeRegionCreateOptions invalid_cpu{
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      7u,
      {VLAFORGE_DEVICE_CPU, 0},
  };
  const auto status = value_api->create(&invalid_cpu, &executable);
  if (status.code != VLAFORGE_STATUS_INVALID_ARGUMENT ||
      executable != nullptr) {
    return 2;
  }

  VLAForgeWorkspaceRequirement requirement{};
  if (value_api->query_workspace(nullptr, &requirement).code !=
      VLAFORGE_STATUS_INVALID_ARGUMENT) {
    return 3;
  }
  return 0;
}
