#ifndef VLAFORGE_RUNTIME_EXECUTION_CONTEXT_H_
#define VLAFORGE_RUNTIME_EXECUTION_CONTEXT_H_

#include "vlaforge/runtime/region_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION 1u
#define VLAFORGE_REGION_EXECUTION_EXTENSION_ABI_VERSION 1u
#define VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT 1u

typedef struct VLAForgeExecutionContext VLAForgeExecutionContext;

typedef struct VLAForgeExecutionContextOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeDevice device;
} VLAForgeExecutionContextOptions;

/* The runtime owns the native stream. CPU contexts have a null stream.
 * A copied view borrows that stream until all bound Regions are destroyed or
 * detached after synchronization. Context and Region operations are serial;
 * neither object is safe for concurrent Run/rebind/destroy calls. Poisoning
 * invalidates every borrowed view: callers must stop direct Region operations
 * too. A view does not hold an owner reference or independently check health. */
typedef struct VLAForgeExecutionContextView {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeDevice device;
  void* native_stream;
} VLAForgeExecutionContextView;

VLAForgeStatus vlaforge_execution_context_create(
    const VLAForgeExecutionContextOptions* options,
    VLAForgeExecutionContext** context);
VLAForgeStatus vlaforge_execution_context_get_view(
    const VLAForgeExecutionContext* context,
    VLAForgeExecutionContextView* view);
VLAForgeStatus vlaforge_execution_context_view_validate(
    const VLAForgeExecutionContextView* view);
VLAForgeStatus vlaforge_execution_context_synchronize(
    VLAForgeExecutionContext* context);
VLAForgeStatus vlaforge_execution_context_status(
    const VLAForgeExecutionContext* context);
void vlaforge_execution_context_poison(VLAForgeExecutionContext* context);
/* Enqueue an equal-device byte copy. Both storages live through synchronize;
 * overlapping distinct ranges and host/device transfers are rejected. */
VLAForgeStatus vlaforge_execution_context_copy(
    VLAForgeExecutionContext* context, const VLAForgeTensorView* destination,
    const VLAForgeTensorView* source, uint64_t size_bytes);
/* A poisoned CUDA context intentionally retains its native stream until
 * process exit: completion and allocator cleanup are no longer guaranteed. */
void vlaforge_execution_context_destroy(VLAForgeExecutionContext* context);

/* Optional sidecar: the v1/v2 Region executable tables remain unchanged.
 * A null view detaches a previously borrowed context. Rebinding while work is
 * pending must fail. SHARED_CONTEXT alone makes no graph-capture guarantee. */
typedef VLAForgeStatus (*VLAForgeRegionBindExecutionContextFn)(
    VLAForgeRegionExecutable* executable,
    const VLAForgeExecutionContextView* view);

typedef struct VLAForgeRegionExecutionExtensionApi {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t capabilities;
  VLAForgeRegionBindExecutionContextFn bind_context;
} VLAForgeRegionExecutionExtensionApi;

#define VLAFORGE_REGION_EXECUTION_EXTENSION_API_SYMBOL \
  "vlaforge_region_execution_extension_api"
typedef const VLAForgeRegionExecutionExtensionApi*
    (*VLAForgeRegionExecutionExtensionApiProviderFn)(void);

VLAForgeStatus vlaforge_region_execution_extension_api_validate(
    const VLAForgeRegionExecutionExtensionApi* api);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_RUNTIME_EXECUTION_CONTEXT_H_
