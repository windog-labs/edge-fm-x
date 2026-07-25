#include <stdint.h>
#include <string.h>

#include "vlaforge/runtime/session_c.h"

struct VLAForgeSession {
  int value;
};

static VLAForgeStatus BindTensor(VLAForgeSession* session, uint32_t input_id,
                                 const VLAForgeBoundTensor* tensor,
                                 const VLAForgeInputStamp* stamp) {
  (void)input_id;
  (void)stamp;
  if (session == NULL || tensor == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "bind");
  }
  return vlaforge_status_ok();
}

static VLAForgeStatus BindScalar(VLAForgeSession* session, uint32_t input_id,
                                 const VLAForgeScalarValue* scalar,
                                 const VLAForgeInputStamp* stamp) {
  (void)input_id;
  (void)stamp;
  if (session == NULL || scalar == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "bind");
  }
  return vlaforge_status_ok();
}

static VLAForgeStatus Run(VLAForgeSession* session) {
  return session == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "run")
             : vlaforge_status_ok();
}

static VLAForgeStatus ReadTensor(const VLAForgeSession* session,
                                 uint32_t output_id,
                                 VLAForgeBoundTensor* output) {
  (void)output_id;
  return session == NULL || output == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "read")
             : vlaforge_status_ok();
}

static VLAForgeStatus ReadScalar(const VLAForgeSession* session,
                                 uint32_t output_id,
                                 VLAForgeScalarValue* output) {
  (void)output_id;
  return session == NULL || output == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "read")
             : vlaforge_status_ok();
}

static VLAForgeStatus Reset(VLAForgeSession* session, uint64_t episode) {
  (void)episode;
  return session == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "reset")
             : vlaforge_status_ok();
}

static void Destroy(VLAForgeSession* session) { (void)session; }

int main(void) {
  const char digest[] =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  VLAForgeSessionApi api = {
      sizeof(VLAForgeSessionApi),
      VLAFORGE_SESSION_ABI_VERSION,
      digest,
      VLAFORGE_SCHEMA_DIGEST_HEX_SIZE,
      BindTensor,
      BindScalar,
      Run,
      ReadTensor,
      ReadScalar,
      Reset,
      Destroy,
  };
  VLAForgeStatus status =
      vlaforge_session_api_validate(&api, digest, strlen(digest));
  if (status.code != VLAFORGE_STATUS_OK) {
    return 1;
  }
  const char wrong[] =
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  status = vlaforge_session_api_validate(&api, wrong, strlen(wrong));
  if (status.code != VLAFORGE_STATUS_FAILED_PRECONDITION) {
    return 2;
  }
  return 0;
}
