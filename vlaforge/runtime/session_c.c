#include "vlaforge/runtime/session_c.h"

#include <string.h>

VLAForgeStatus vlaforge_session_api_validate(
    const VLAForgeSessionApi* api,
    const char* expected_schema_digest,
    size_t expected_schema_digest_size) {
  if (api == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "session API is null");
  }
  if (api->struct_size < sizeof(VLAForgeSessionApi) ||
      api->abi_version != VLAFORGE_SESSION_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported session ABI");
  }
  if (api->schema_digest == NULL ||
      api->schema_digest_size != VLAFORGE_SCHEMA_DIGEST_HEX_SIZE) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid session schema digest");
  }
  if (api->bind_tensor == NULL || api->bind_scalar == NULL ||
      api->run == NULL || api->read_output_tensor == NULL ||
      api->read_output_scalar == NULL || api->reset_episode == NULL ||
      api->destroy == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "session API is incomplete");
  }
  if (expected_schema_digest != NULL &&
      (expected_schema_digest_size != api->schema_digest_size ||
       memcmp(expected_schema_digest, api->schema_digest,
              api->schema_digest_size) != 0)) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "session schema digest mismatch");
  }
  return vlaforge_status_ok();
}
