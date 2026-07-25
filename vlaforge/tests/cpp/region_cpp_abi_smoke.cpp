#include "vlaforge/runtime/region_executable.h"

#include <cstddef>
#include <type_traits>

int main() {
  static_assert(std::is_standard_layout_v<VLAForgeStatus>);
  static_assert(std::is_standard_layout_v<VLAForgeTensorView>);
  static_assert(std::is_standard_layout_v<VLAForgeValueView>);
  static_assert(std::is_standard_layout_v<VLAForgeRegionExecutableApi>);
  static_assert(
      std::is_standard_layout_v<VLAForgeRegionExecutableValueApi>);
  static_assert(VLAFORGE_REGION_EXECUTABLE_ABI_VERSION == 1u);
  static_assert(VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION == 2u);

  const VLAForgeStatus status =
      vlaforge_region_executable_api_validate(nullptr);
  if (status.code != VLAFORGE_STATUS_INVALID_ARGUMENT ||
      status.message == nullptr || status.message_size == 0u) {
    return 1;
  }

  VLAForgeRegionExecutableApi wrong_abi{};
  wrong_abi.struct_size = sizeof(wrong_abi);
  wrong_abi.abi_version = VLAFORGE_REGION_EXECUTABLE_ABI_VERSION + 1u;
  const VLAForgeStatus abi_status =
      vlaforge_region_executable_api_validate(&wrong_abi);
  if (abi_status.code != VLAFORGE_STATUS_UNSUPPORTED_ABI) {
    return 2;
  }
  const VLAForgeStatus value_status =
      vlaforge_region_executable_value_api_validate(nullptr);
  return value_status.code == VLAFORGE_STATUS_INVALID_ARGUMENT ? 0 : 3;
}
