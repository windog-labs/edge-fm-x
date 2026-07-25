#ifndef VLAFORGE_RUNTIME_SESSION_C_H_
#define VLAFORGE_RUNTIME_SESSION_C_H_

#include <stddef.h>
#include <stdint.h>

#include "vlaforge/runtime/region_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VLAFORGE_SESSION_ABI_VERSION 2u
#define VLAFORGE_SCHEMA_DIGEST_HEX_SIZE 64u

typedef struct VLAForgeInputStamp {
  uint32_t struct_size;
  uint8_t has_revision;
  uint8_t has_timestamp;
  uint8_t reserved[6];
  uint64_t revision;
  uint64_t timestamp_ns;
} VLAForgeInputStamp;

typedef struct VLAForgeSession VLAForgeSession;

typedef VLAForgeStatus (*VLAForgeSessionBindTensorFn)(
    VLAForgeSession* session, uint32_t input_id,
    const VLAForgeBoundTensor* tensor, const VLAForgeInputStamp* stamp);
typedef VLAForgeStatus (*VLAForgeSessionBindScalarFn)(
    VLAForgeSession* session, uint32_t input_id,
    const VLAForgeScalarValue* scalar, const VLAForgeInputStamp* stamp);
typedef VLAForgeStatus (*VLAForgeSessionRunFn)(VLAForgeSession* session);
typedef VLAForgeStatus (*VLAForgeSessionReadOutputTensorFn)(
    const VLAForgeSession* session, uint32_t output_id,
    VLAForgeBoundTensor* output);
typedef VLAForgeStatus (*VLAForgeSessionReadOutputScalarFn)(
    const VLAForgeSession* session, uint32_t output_id,
    VLAForgeScalarValue* output);
typedef VLAForgeStatus (*VLAForgeSessionResetEpisodeFn)(
    VLAForgeSession* session, uint64_t new_episode);
typedef void (*VLAForgeSessionDestroyFn)(VLAForgeSession* session);

typedef struct VLAForgeSessionApi {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* schema_digest;
  size_t schema_digest_size;
  VLAForgeSessionBindTensorFn bind_tensor;
  VLAForgeSessionBindScalarFn bind_scalar;
  VLAForgeSessionRunFn run;
  VLAForgeSessionReadOutputTensorFn read_output_tensor;
  VLAForgeSessionReadOutputScalarFn read_output_scalar;
  VLAForgeSessionResetEpisodeFn reset_episode;
  VLAForgeSessionDestroyFn destroy;
} VLAForgeSessionApi;

VLAForgeStatus vlaforge_session_api_validate(
    const VLAForgeSessionApi* api,
    const char* expected_schema_digest,
    size_t expected_schema_digest_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_RUNTIME_SESSION_C_H_
