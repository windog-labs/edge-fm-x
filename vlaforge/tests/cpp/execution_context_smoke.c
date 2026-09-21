#include "vlaforge/runtime/execution_context.h"

#include <stdio.h>
#include <string.h>

#define REQUIRE(condition)                                                \
  do {                                                                    \
    if (!(condition)) {                                                   \
      fprintf(stderr, "execution context check failed at %d\n", __LINE__); \
      return 1;                                                           \
    }                                                                     \
  } while (0)

static VLAForgeExecutionContextView copied_view;

static VLAForgeStatus BindContext(VLAForgeRegionExecutable* executable,
                                  const VLAForgeExecutionContextView* view) {
  (void)executable;
  if (view == NULL) {
    memset(&copied_view, 0, sizeof(copied_view));
    return vlaforge_status_ok();
  }
  VLAForgeStatus status = vlaforge_execution_context_view_validate(view);
  if (status.code == VLAFORGE_STATUS_OK) {
    copied_view = *view;
  }
  return status;
}

int main(void) {
  VLAForgeExecutionContextOptions options = {
      sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {VLAFORGE_DEVICE_CPU, 0}};
  VLAForgeExecutionContext* context = NULL;
  REQUIRE(vlaforge_execution_context_create(&options, &context).code ==
          VLAFORGE_STATUS_OK);
  REQUIRE(context != NULL);
  VLAForgeExecutionContextView view = {0};
  REQUIRE(vlaforge_execution_context_get_view(context, &view).code ==
          VLAFORGE_STATUS_INVALID_ARGUMENT);
  view.struct_size = sizeof(view);
  REQUIRE(vlaforge_execution_context_get_view(context, &view).code ==
          VLAFORGE_STATUS_OK);
  REQUIRE(view.device.kind == VLAFORGE_DEVICE_CPU &&
          view.device.ordinal == 0 && view.native_stream == NULL);
  REQUIRE(vlaforge_execution_context_view_validate(&view).code ==
          VLAFORGE_STATUS_OK);
  REQUIRE(vlaforge_execution_context_synchronize(context).code ==
          VLAFORGE_STATUS_OK);

  VLAForgeRegionExecutionExtensionApi api = {
      sizeof(api), VLAFORGE_REGION_EXECUTION_EXTENSION_ABI_VERSION,
      VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT, &BindContext};
  REQUIRE(vlaforge_region_execution_extension_api_validate(&api).code ==
          VLAFORGE_STATUS_OK);
  REQUIRE(api.bind_context(NULL, &view).code == VLAFORGE_STATUS_OK);
  view.device.ordinal = 1;
  REQUIRE(copied_view.device.ordinal == 0);
  REQUIRE(vlaforge_execution_context_view_validate(&view).code ==
          VLAFORGE_STATUS_INVALID_ARGUMENT);
  view.device.ordinal = 0;
  view.abi_version += 1;
  REQUIRE(vlaforge_execution_context_view_validate(&view).code ==
          VLAFORGE_STATUS_UNSUPPORTED_ABI);
  REQUIRE(api.bind_context(NULL, NULL).code == VLAFORGE_STATUS_OK);
  REQUIRE(copied_view.struct_size == 0);
  api.struct_size -= 1;
  REQUIRE(vlaforge_region_execution_extension_api_validate(&api).code ==
          VLAFORGE_STATUS_UNSUPPORTED_ABI);
  api.struct_size = sizeof(api);
  api.bind_context = NULL;
  REQUIRE(vlaforge_region_execution_extension_api_validate(&api).code ==
          VLAFORGE_STATUS_INVALID_ARGUMENT);
  vlaforge_execution_context_destroy(context);
  context = NULL;
  vlaforge_execution_context_destroy(NULL);
  REQUIRE(vlaforge_execution_context_synchronize(NULL).code ==
          VLAFORGE_STATUS_INVALID_ARGUMENT);
  options.abi_version += 1;
  REQUIRE(vlaforge_execution_context_create(&options, &context).code ==
          VLAFORGE_STATUS_UNSUPPORTED_ABI);
  REQUIRE(context == NULL);
  options.abi_version = VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION;
  options.device.kind = VLAFORGE_DEVICE_EXTERNAL;
  REQUIRE(vlaforge_execution_context_create(&options, &context).code ==
          VLAFORGE_STATUS_INVALID_ARGUMENT);
#if !defined(VLAFORGE_ENABLE_CUDA_ARENA)
  options.device.kind = VLAFORGE_DEVICE_CUDA;
  REQUIRE(vlaforge_execution_context_create(&options, &context).code ==
          VLAFORGE_STATUS_FAILED_PRECONDITION);
  REQUIRE(context == NULL);
#endif
  return 0;
}
