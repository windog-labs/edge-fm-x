#include "vlaforge/backends/tensorrt_region_executable.h"

int main() {
  const auto* api =
      vlaforge_tensorrt_region_executable_value_api();
  return api != nullptr &&
                 api->abi_version ==
                     VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION
             ? 0
             : 1;
}
