#include "vlaforge/runtime/region_executable.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct VLAForgeRegionExecutable {
  int loaded;
  VLAForgeValueView inputs[2];
  VLAForgeValueView outputs[1];
  void* workspace;
  uint64_t workspace_size;
};

static VLAForgeStatus fixture_create(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** executable) {
  if (options == NULL || executable == NULL ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value Region create arguments");
  }
  *executable = (VLAForgeRegionExecutable*)calloc(1u, sizeof(**executable));
  return *executable == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                     "value Region allocation failed")
             : vlaforge_status_ok();
}

static VLAForgeStatus fixture_load(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == NULL || artifact == NULL ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION ||
      artifact->path == NULL || artifact->path_size == 0u ||
      artifact->sha256 == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value Region artifact");
  }
  executable->loaded = 1;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_query_workspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == NULL || requirement == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value workspace query");
  }
  requirement->size_bytes = 32u;
  requirement->alignment = 16u;
  requirement->device.kind = VLAFORGE_DEVICE_CPU;
  requirement->device.ordinal = 0;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_input(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeValueView* value) {
  if (executable == NULL || value == NULL || index >= 2u ||
      value->struct_size < sizeof(*value)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value Region input");
  }
  if ((index == 0u && value->kind != VLAFORGE_VALUE_TENSOR) ||
      (index == 1u && (value->kind != VLAFORGE_VALUE_SCALAR ||
                       value->value.scalar.dtype != VLAFORGE_DTYPE_F32))) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "value Region input kind mismatch");
  }
  executable->inputs[index] = *value;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_output(
    VLAForgeRegionExecutable* executable, uint32_t index,
    const VLAForgeValueView* value) {
  if (executable == NULL || value == NULL || index != 0u ||
      value->struct_size < sizeof(*value) ||
      value->kind != VLAFORGE_VALUE_TENSOR) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value Region output");
  }
  executable->outputs[index] = *value;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_bind_workspace(
    VLAForgeRegionExecutable* executable, void* workspace,
    uint64_t workspace_size) {
  if (executable == NULL || workspace == NULL || workspace_size < 32u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid value Region workspace");
  }
  executable->workspace = workspace;
  executable->workspace_size = workspace_size;
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_run(VLAForgeRegionExecutable* executable) {
  uint64_t index;
  const VLAForgeBoundTensor* input;
  const VLAForgeScalarValue* scale;
  VLAForgeBoundTensor* output;
  if (executable == NULL || !executable->loaded ||
      executable->workspace == NULL) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "value Region is not ready");
  }
  input = &executable->inputs[0].value.tensor;
  scale = &executable->inputs[1].value.scalar;
  output = &executable->outputs[0].value.tensor;
  if (input->tensor.data == NULL || output->tensor.data == NULL ||
      input->tensor.dtype != VLAFORGE_DTYPE_F32 ||
      output->tensor.dtype != VLAFORGE_DTYPE_F32 ||
      input->tensor.size_bytes != output->tensor.size_bytes) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "value Region tensor contract mismatch");
  }
  for (index = 0u; index < input->tensor.size_bytes / sizeof(float); ++index) {
    ((float*)output->tensor.data)[index] =
        ((const float*)input->tensor.data)[index] * scale->value.f32;
  }
  return vlaforge_status_ok();
}

static VLAForgeStatus fixture_synchronize(
    VLAForgeRegionExecutable* executable) {
  return executable == NULL
             ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                     "value Region is null")
             : vlaforge_status_ok();
}

static void fixture_destroy(VLAForgeRegionExecutable* executable) {
  free(executable);
}

static int check_ok(VLAForgeStatus status) {
  return status.code == VLAFORGE_STATUS_OK;
}

int main(void) {
  const VLAForgeRegionExecutableValueApi api = {
      sizeof(VLAForgeRegionExecutableValueApi),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
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
  const char artifact_path[] = "artifacts/value_fixture.bin";
  const VLAForgeArtifactDescriptor artifact = {
      sizeof(VLAForgeArtifactDescriptor),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      artifact_path,
      sizeof(artifact_path) - 1u,
      digest,
      0u,
      NULL,
      0u,
      NULL,
      0u,
      NULL,
      0u};
  const VLAForgeRegionCreateOptions options = {
      sizeof(VLAForgeRegionCreateOptions),
      VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
      9u,
      {VLAFORGE_DEVICE_CPU, 0}};
  int64_t dimensions[] = {4};
  float input[] = {1.0f, -2.0f, 3.5f, 0.25f};
  float output[] = {0.0f, 0.0f, 0.0f, 0.0f};
  unsigned char workspace_storage[32];
  VLAForgeValueView input_value = {
      sizeof(VLAForgeValueView),
      VLAFORGE_VALUE_TENSOR,
      {.tensor = {sizeof(VLAForgeBoundTensor),
                  {input, sizeof(input), dimensions, 1u, VLAFORGE_DTYPE_F32,
                   {VLAFORGE_DEVICE_CPU, 0}},
                  VLAFORGE_LAYOUT_CONTIGUOUS,
                  1u}}};
  VLAForgeValueView scale_value = {
      sizeof(VLAForgeValueView),
      VLAFORGE_VALUE_SCALAR,
      {.scalar = {sizeof(VLAForgeScalarValue),
                  VLAFORGE_DTYPE_F32,
                  {.f32 = 3.0f}}}};
  VLAForgeValueView output_value = {
      sizeof(VLAForgeValueView),
      VLAFORGE_VALUE_TENSOR,
      {.tensor = {sizeof(VLAForgeBoundTensor),
                  {output, sizeof(output), dimensions, 1u, VLAFORGE_DTYPE_F32,
                   {VLAFORGE_DEVICE_CPU, 0}},
                  VLAFORGE_LAYOUT_CONTIGUOUS,
                  1u}}};
  VLAForgeWorkspaceRequirement requirement;
  VLAForgeRegionExecutable* executable = NULL;

  if (!check_ok(vlaforge_region_executable_value_api_validate(&api)) ||
      !check_ok(api.create(&options, &executable)) ||
      !check_ok(api.load(executable, &artifact)) ||
      !check_ok(api.query_workspace(executable, &requirement)) ||
      requirement.size_bytes != 32u || requirement.alignment != 16u ||
      !check_ok(api.bind_input(executable, 0u, &input_value)) ||
      !check_ok(api.bind_input(executable, 1u, &scale_value)) ||
      !check_ok(api.bind_output(executable, 0u, &output_value)) ||
      !check_ok(api.bind_workspace(executable, workspace_storage,
                                   sizeof(workspace_storage))) ||
      !check_ok(api.run(executable)) ||
      !check_ok(api.synchronize(executable))) {
    api.destroy(executable);
    return 1;
  }
  api.destroy(executable);
  return memcmp(output, (float[]){3.0f, -6.0f, 10.5f, 0.75f},
                sizeof(output)) == 0
             ? 0
             : 2;
}
