#include "vlaforge/runtime/region_executable.h"

#include <string.h>

VLAForgeStatus vlaforge_status_ok(void) {
  VLAForgeStatus status = {VLAFORGE_STATUS_OK, NULL, 0u};
  return status;
}

VLAForgeStatus vlaforge_status_error(VLAForgeStatusCode code,
                                     const char* message) {
  VLAForgeStatus status;
  status.code = code;
  status.message = message;
  status.message_size = message == NULL ? 0u : strlen(message);
  return status;
}

VLAForgeStatus vlaforge_region_executable_api_validate(
    const VLAForgeRegionExecutableApi* api) {
  if (api == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "region executable API is null");
  }
  if (api->struct_size < sizeof(VLAForgeRegionExecutableApi)) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "region executable API struct is too small");
  }
  if (api->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported region executable ABI");
  }
  if (api->create == NULL || api->load == NULL ||
      api->query_workspace == NULL || api->bind_input == NULL ||
      api->bind_output == NULL || api->bind_workspace == NULL ||
      api->run == NULL || api->synchronize == NULL ||
      api->destroy == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "region executable API is incomplete");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus vlaforge_region_executable_value_api_validate(
    const VLAForgeRegionExecutableValueApi* api) {
  if (api == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "region value API is null");
  }
  if (api->struct_size < sizeof(VLAForgeRegionExecutableValueApi) ||
      api->abi_version != VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported region value ABI");
  }
  if (api->create == NULL || api->load == NULL ||
      api->query_workspace == NULL || api->bind_input == NULL ||
      api->bind_output == NULL || api->bind_workspace == NULL ||
      api->run == NULL || api->synchronize == NULL ||
      api->destroy == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "region value API is incomplete");
  }
  return vlaforge_status_ok();
}
