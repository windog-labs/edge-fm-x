#include "vlaforge/runtime/region_executable.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct VLAForgeRegionExecutable {
  uint32_t region_id;
  int loaded;
  VLAForgeTensorView input;
  VLAForgeTensorView output;
  void* workspace;
  uint64_t workspace_size;
};

static VLAForgeStatus fixture_create(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** executable) {
  if (options == NULL || executable == NULL ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture create arguments");
  }
  *executable = (VLAForgeRegionExecutable*)calloc(1u, sizeof(**executable));
  if (*executable == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "fixture allocation failed");
  }
  (*executable)->region_id = options->region_id;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_load(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == NULL || artifact == NULL ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_ABI_VERSION ||
      artifact->path == NULL || artifact->path_size == 0u ||
      artifact->sha256 == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture artifact");
  }
  executable->loaded = 1;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_query_workspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == NULL || requirement == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid workspace query");
  }
  requirement->size_bytes = 64u;
  requirement->alignment = 64u;
  requirement->device.kind = VLAFORGE_DEVICE_CPU;
  requirement->device.ordinal = 0;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_input(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeTensorView* tensor) {
  if (executable == NULL || tensor == NULL || index != 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture input");
  }
  executable->input = *tensor;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_output(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeTensorView* tensor) {
  if (executable == NULL || tensor == NULL || index != 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture output");
  }
  executable->output = *tensor;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_workspace(
    VLAForgeRegionExecutable* executable, void* workspace,
    uint64_t workspace_size) {
  if (executable == NULL || workspace == NULL || workspace_size < 64u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture workspace");
  }
  executable->workspace = workspace;
  executable->workspace_size = workspace_size;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_run(VLAForgeRegionExecutable* executable) {
  uint64_t index;
  const float* input;
  float* output;
  if (executable == NULL || !executable->loaded ||
      executable->input.data == NULL || executable->output.data == NULL ||
      executable->workspace == NULL ||
      executable->input.dtype != VLAFORGE_DTYPE_F32 ||
      executable->output.dtype != VLAFORGE_DTYPE_F32 ||
      executable->input.size_bytes != executable->output.size_bytes) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "fixture is not fully bound");
  }
  input = (const float*)executable->input.data;
  output = (float*)executable->output.data;
  for (index = 0u; index < executable->input.size_bytes / sizeof(float);
       ++index) {
    output[index] = input[index] * 2.0f;
  }
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_synchronize(
    VLAForgeRegionExecutable* executable) {
  if (executable == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "fixture executable is null");
  }
  return vlaforge_status_ok();
}

static void fixture_destroy(VLAForgeRegionExecutable* executable) {
  free(executable);
}

static int check_ok(VLAForgeStatus status) {
  return status.code == VLAFORGE_STATUS_OK;
}

int main(void) {
  const VLAForgeRegionExecutableApi api = {
      sizeof(VLAForgeRegionExecutableApi),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      fixture_create,
      fixture_load,
      fixture_query_workspace,
      fixture_bind_input,
      fixture_bind_output,
      fixture_bind_workspace,
      fixture_run,
      fixture_synchronize,
      fixture_destroy};
  const uint8_t digest[32] = {0u};
  const char artifact_path[] = "artifacts/fixture.bin";
  const VLAForgeArtifactDescriptor artifact = {
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      artifact_path,
      sizeof(artifact_path) - 1u,
      digest,
      0u};
  const VLAForgeRegionCreateOptions options = {
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
      7u,
      {VLAFORGE_DEVICE_CPU, 0}};
  int64_t dimensions[] = {4};
  float input[] = {1.0f, -2.0f, 3.5f, 0.25f};
  float output[] = {0.0f, 0.0f, 0.0f, 0.0f};
  unsigned char workspace_storage[64];
  VLAForgeTensorView input_view = {
      input, sizeof(input), dimensions, 1u, VLAFORGE_DTYPE_F32,
      {VLAFORGE_DEVICE_CPU, 0}};
  VLAForgeTensorView output_view = {
      output, sizeof(output), dimensions, 1u, VLAFORGE_DTYPE_F32,
      {VLAFORGE_DEVICE_CPU, 0}};
  VLAForgeWorkspaceRequirement requirement;
  VLAForgeRegionExecutable* executable = NULL;

  if (!check_ok(vlaforge_region_executable_api_validate(&api)) ||
      !check_ok(api.create(&options, &executable)) ||
      !check_ok(api.load(executable, &artifact)) ||
      !check_ok(api.query_workspace(executable, &requirement)) ||
      requirement.size_bytes != 64u || requirement.alignment != 64u ||
      !check_ok(api.bind_input(executable, 0u, &input_view)) ||
      !check_ok(api.bind_output(executable, 0u, &output_view)) ||
      !check_ok(api.bind_workspace(executable, workspace_storage,
                                   sizeof(workspace_storage))) ||
      !check_ok(api.run(executable)) ||
      !check_ok(api.synchronize(executable))) {
    api.destroy(executable);
    return 1;
  }
  api.destroy(executable);
  return memcmp(output, (float[]){2.0f, -4.0f, 7.0f, 0.5f},
                sizeof(output)) == 0
             ? 0
             : 2;
}
