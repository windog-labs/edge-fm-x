#ifndef VLAFORGE_RUNTIME_REGION_EXECUTABLE_H_
#define VLAFORGE_RUNTIME_REGION_EXECUTABLE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VLAFORGE_REGION_EXECUTABLE_ABI_VERSION 1u
#define VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION 2u

typedef enum VLAForgeStatusCode {
  VLAFORGE_STATUS_OK = 0,
  VLAFORGE_STATUS_INVALID_ARGUMENT = 1,
  VLAFORGE_STATUS_UNSUPPORTED_ABI = 2,
  VLAFORGE_STATUS_NOT_FOUND = 3,
  VLAFORGE_STATUS_IO_ERROR = 4,
  VLAFORGE_STATUS_BACKEND_ERROR = 5,
  VLAFORGE_STATUS_OUT_OF_MEMORY = 6,
  VLAFORGE_STATUS_FAILED_PRECONDITION = 7,
  VLAFORGE_STATUS_INTERNAL = 8
} VLAForgeStatusCode;

typedef struct VLAForgeStatus {
  VLAForgeStatusCode code;
  const char* message;
  size_t message_size;
} VLAForgeStatus;

typedef enum VLAForgeDType {
  VLAFORGE_DTYPE_INVALID = 0,
  VLAFORGE_DTYPE_BOOL = 1,
  VLAFORGE_DTYPE_I32 = 2,
  VLAFORGE_DTYPE_I64 = 3,
  VLAFORGE_DTYPE_F16 = 4,
  VLAFORGE_DTYPE_BF16 = 5,
  VLAFORGE_DTYPE_F32 = 6,
  VLAFORGE_DTYPE_F64 = 7,
  VLAFORGE_DTYPE_U64 = 8,
  VLAFORGE_DTYPE_U8 = 9
} VLAForgeDType;

typedef enum VLAForgeDeviceKind {
  VLAFORGE_DEVICE_CPU = 0,
  VLAFORGE_DEVICE_CUDA = 1,
  VLAFORGE_DEVICE_EXTERNAL = 2
} VLAForgeDeviceKind;

typedef struct VLAForgeDevice {
  VLAForgeDeviceKind kind;
  int32_t ordinal;
} VLAForgeDevice;

typedef struct VLAForgeTensorView {
  void* data;
  uint64_t size_bytes;
  const int64_t* dimensions;
  uint32_t rank;
  VLAForgeDType dtype;
  VLAForgeDevice device;
} VLAForgeTensorView;

typedef enum VLAForgeLayout {
  VLAFORGE_LAYOUT_CONTIGUOUS = 0,
  VLAFORGE_LAYOUT_NCHW = 1,
  VLAFORGE_LAYOUT_NHWC = 2,
  VLAFORGE_LAYOUT_CUSTOM = 3
} VLAForgeLayout;

typedef struct VLAForgeBoundTensor {
  uint32_t struct_size;
  VLAForgeTensorView tensor;
  VLAForgeLayout layout;
  uint64_t alignment;
} VLAForgeBoundTensor;

typedef union VLAForgeScalarPayload {
  uint8_t boolean;
  int32_t i32;
  int64_t i64;
  uint64_t u64;
  float f32;
  double f64;
} VLAForgeScalarPayload;

typedef struct VLAForgeScalarValue {
  uint32_t struct_size;
  VLAForgeDType dtype;
  VLAForgeScalarPayload value;
} VLAForgeScalarValue;

typedef enum VLAForgeValueKind {
  VLAFORGE_VALUE_TENSOR = 0,
  VLAFORGE_VALUE_SCALAR = 1
} VLAForgeValueKind;

typedef struct VLAForgeValueView {
  uint32_t struct_size;
  VLAForgeValueKind kind;
  union {
    VLAForgeBoundTensor tensor;
    VLAForgeScalarValue scalar;
  } value;
} VLAForgeValueView;

typedef struct VLAForgeWorkspaceRequirement {
  uint64_t size_bytes;
  uint64_t alignment;
  VLAForgeDevice device;
} VLAForgeWorkspaceRequirement;

typedef struct VLAForgeRegionCreateOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t region_id;
  VLAForgeDevice device;
} VLAForgeRegionCreateOptions;

typedef struct VLAForgeArtifactDescriptor {
  uint32_t struct_size;
  uint32_t callable_abi_version;
  const char* path;
  size_t path_size;
  const uint8_t* sha256;
  uint64_t size_bytes;
} VLAForgeArtifactDescriptor;

typedef struct VLAForgeRegionExecutable VLAForgeRegionExecutable;

typedef VLAForgeStatus (*VLAForgeRegionCreateFn)(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** executable);
typedef VLAForgeStatus (*VLAForgeRegionLoadFn)(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact);
typedef VLAForgeStatus (*VLAForgeRegionQueryWorkspaceFn)(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement);
typedef VLAForgeStatus (*VLAForgeRegionBindTensorFn)(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeTensorView* tensor);
typedef VLAForgeStatus (*VLAForgeRegionBindWorkspaceFn)(
    VLAForgeRegionExecutable* executable, void* workspace,
    uint64_t workspace_size);
typedef VLAForgeStatus (*VLAForgeRegionRunFn)(
    VLAForgeRegionExecutable* executable);
typedef VLAForgeStatus (*VLAForgeRegionSynchronizeFn)(
    VLAForgeRegionExecutable* executable);
typedef void (*VLAForgeRegionDestroyFn)(
    VLAForgeRegionExecutable* executable);

typedef struct VLAForgeRegionExecutableApi {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeRegionCreateFn create;
  VLAForgeRegionLoadFn load;
  VLAForgeRegionQueryWorkspaceFn query_workspace;
  VLAForgeRegionBindTensorFn bind_input;
  VLAForgeRegionBindTensorFn bind_output;
  VLAForgeRegionBindWorkspaceFn bind_workspace;
  VLAForgeRegionRunFn run;
  VLAForgeRegionSynchronizeFn synchronize;
  VLAForgeRegionDestroyFn destroy;
} VLAForgeRegionExecutableApi;

typedef VLAForgeStatus (*VLAForgeRegionBindValueFn)(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeValueView* value);

typedef struct VLAForgeRegionExecutableValueApi {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeRegionCreateFn create;
  VLAForgeRegionLoadFn load;
  VLAForgeRegionQueryWorkspaceFn query_workspace;
  VLAForgeRegionBindValueFn bind_input;
  VLAForgeRegionBindValueFn bind_output;
  VLAForgeRegionBindWorkspaceFn bind_workspace;
  VLAForgeRegionRunFn run;
  VLAForgeRegionSynchronizeFn synchronize;
  VLAForgeRegionDestroyFn destroy;
} VLAForgeRegionExecutableValueApi;

VLAForgeStatus vlaforge_status_ok(void);
VLAForgeStatus vlaforge_status_error(VLAForgeStatusCode code,
                                     const char* message);
VLAForgeStatus vlaforge_region_executable_api_validate(
    const VLAForgeRegionExecutableApi* api);
VLAForgeStatus vlaforge_region_executable_value_api_validate(
    const VLAForgeRegionExecutableValueApi* api);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_RUNTIME_REGION_EXECUTABLE_H_
