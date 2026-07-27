#include "vlaforge/runtime/region_executable.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string_view>

#ifndef VLAFORGE_EXPECTED_SCHEMA_DIGEST
#error "VLAFORGE_EXPECTED_SCHEMA_DIGEST must be defined"
#endif

namespace {

constexpr std::size_t kMaximumBindings = 4u;
constexpr std::string_view kExpectedSchema = VLAFORGE_EXPECTED_SCHEMA_DIGEST;
constexpr std::string_view kExpectedTarget = "cpu";
constexpr std::string_view kExpectedVariant = "shared-plugin/1";

VLAForgeStatus Ok() { return VLAForgeStatus{VLAFORGE_STATUS_OK, nullptr, 0u}; }

VLAForgeStatus Error(VLAForgeStatusCode code, const char *message) {
  return VLAForgeStatus{code, message, std::strlen(message)};
}

bool TextEquals(const char *data, std::size_t size, std::string_view expected) {
  return data != nullptr && size == expected.size() &&
         std::string_view(data, size) == expected;
}

bool TensorMatches(const VLAForgeValueView *value, VLAForgeDType dtype,
                   const std::int64_t *shape, std::size_t rank,
                   std::size_t size_bytes) {
  if (value == nullptr || value->struct_size < sizeof(VLAForgeValueView) ||
      value->kind != VLAFORGE_VALUE_TENSOR ||
      value->value.tensor.struct_size < sizeof(VLAForgeBoundTensor)) {
    return false;
  }
  const auto &tensor = value->value.tensor.tensor;
  if (tensor.data == nullptr || tensor.dtype != dtype ||
      tensor.device.kind != VLAFORGE_DEVICE_CPU || tensor.device.ordinal != 0 ||
      tensor.rank != rank || tensor.size_bytes != size_bytes ||
      (rank != 0u && tensor.dimensions == nullptr)) {
    return false;
  }
  for (std::size_t index = 0; index < rank; ++index) {
    if (tensor.dimensions[index] != shape[index]) {
      return false;
    }
  }
  return true;
}

bool ScalarMatches(const VLAForgeValueView *value, VLAForgeDType dtype) {
  return value != nullptr && value->struct_size >= sizeof(VLAForgeValueView) &&
         value->kind == VLAFORGE_VALUE_SCALAR &&
         value->value.scalar.struct_size >= sizeof(VLAForgeScalarValue) &&
         value->value.scalar.dtype == dtype;
}

} // namespace

struct VLAForgeRegionExecutable {
  std::uint32_t region_id = 0u;
  bool loaded = false;
  std::array<const VLAForgeValueView *, kMaximumBindings> inputs{};
  std::array<const VLAForgeValueView *, kMaximumBindings> outputs{};
};

namespace {

VLAForgeStatus Create(const VLAForgeRegionCreateOptions *options,
                      VLAForgeRegionExecutable **output) {
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(VLAForgeRegionCreateOptions) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION ||
      options->region_id > 1u || options->device.kind != VLAFORGE_DEVICE_CPU ||
      options->device.ordinal != 0) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                 "external BEV plugin create contract mismatch");
  }
  auto *executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return Error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                 "external BEV plugin allocation failed");
  }
  executable->region_id = options->region_id;
  *output = executable;
  return Ok();
}

VLAForgeStatus Load(VLAForgeRegionExecutable *executable,
                    const VLAForgeArtifactDescriptor *artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(VLAForgeArtifactDescriptor) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION ||
      artifact->sha256 == nullptr ||
      !TextEquals(artifact->io_schema_digest, artifact->io_schema_digest_size,
                  kExpectedSchema) ||
      !TextEquals(artifact->target, artifact->target_size, kExpectedTarget) ||
      !TextEquals(artifact->backend_variant, artifact->backend_variant_size,
                  kExpectedVariant)) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external BEV plugin artifact contract mismatch");
  }
  executable->loaded = true;
  return Ok();
}

VLAForgeStatus QueryWorkspace(const VLAForgeRegionExecutable *executable,
                              VLAForgeWorkspaceRequirement *requirement) {
  if (executable == nullptr || !executable->loaded || requirement == nullptr) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                 "external BEV plugin workspace query is invalid");
  }
  *requirement = VLAForgeWorkspaceRequirement{0u, 1u, {VLAFORGE_DEVICE_CPU, 0}};
  return Ok();
}

VLAForgeStatus BindValue(VLAForgeRegionExecutable *executable,
                         std::uint32_t index, const VLAForgeValueView *value,
                         bool input) {
  if (executable == nullptr || !executable->loaded || value == nullptr ||
      index >= kMaximumBindings) {
    return Error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                 "external BEV plugin value binding is invalid");
  }
  (input ? executable->inputs : executable->outputs)[index] = value;
  return Ok();
}

VLAForgeStatus BindInput(VLAForgeRegionExecutable *executable,
                         std::uint32_t index, const VLAForgeValueView *value) {
  return BindValue(executable, index, value, true);
}

VLAForgeStatus BindOutput(VLAForgeRegionExecutable *executable,
                          std::uint32_t index, const VLAForgeValueView *value) {
  return BindValue(executable, index, value, false);
}

VLAForgeStatus BindWorkspace(VLAForgeRegionExecutable *executable,
                             void *workspace, std::uint64_t workspace_size) {
  if (executable == nullptr || !executable->loaded || workspace != nullptr ||
      workspace_size != 0u) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external BEV plugin requires zero workspace");
  }
  return Ok();
}

VLAForgeStatus RunPreprocess(VLAForgeRegionExecutable *executable) {
  constexpr std::int64_t kInputShape[] = {4, 4};
  constexpr std::int64_t kOutputShape[] = {4};
  if (!TensorMatches(executable->inputs[0], VLAFORGE_DTYPE_F32, kInputShape, 2u,
                     16u * sizeof(float)) ||
      !TensorMatches(executable->outputs[0], VLAFORGE_DTYPE_F32, kOutputShape,
                     1u, 4u * sizeof(float))) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external BEV preprocess binding mismatch");
  }
  const auto *input = static_cast<const float *>(
      executable->inputs[0]->value.tensor.tensor.data);
  if (input[0] < -900.0f) {
    return Error(VLAFORGE_STATUS_BACKEND_ERROR,
                 "external BEV preprocess injected failure");
  }
  auto *output =
      static_cast<float *>(executable->outputs[0]->value.tensor.tensor.data);
  for (std::size_t column = 0; column < 4u; ++column) {
    output[column] = 0.0f;
    for (std::size_t row = 0; row < 4u; ++row) {
      output[column] += input[row * 4u + column];
    }
  }
  return Ok();
}

VLAForgeStatus RunPlanner(VLAForgeRegionExecutable *executable) {
  constexpr std::int64_t kBevShape[] = {4};
  constexpr std::int64_t kAgentShape[] = {6, 3};
  constexpr std::int64_t kRouteShape[] = {3};
  constexpr std::int64_t kPlanningShape[] = {6, 2};
  if (!TensorMatches(executable->inputs[0], VLAFORGE_DTYPE_F32, kBevShape, 1u,
                     4u * sizeof(float)) ||
      !TensorMatches(executable->inputs[1], VLAFORGE_DTYPE_F32, kAgentShape, 2u,
                     18u * sizeof(float)) ||
      !ScalarMatches(executable->inputs[2], VLAFORGE_DTYPE_I32) ||
      !TensorMatches(executable->inputs[3], VLAFORGE_DTYPE_F32, kRouteShape, 1u,
                     3u * sizeof(float)) ||
      !TensorMatches(executable->outputs[0], VLAFORGE_DTYPE_F32, kPlanningShape,
                     2u, 12u * sizeof(float)) ||
      !TensorMatches(executable->outputs[1], VLAFORGE_DTYPE_F32, kPlanningShape,
                     2u, 12u * sizeof(float)) ||
      !ScalarMatches(executable->outputs[2], VLAFORGE_DTYPE_I64)) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external hybrid planner binding mismatch");
  }
  const auto *bev = static_cast<const float *>(
      executable->inputs[0]->value.tensor.tensor.data);
  const auto *agents = static_cast<const float *>(
      executable->inputs[1]->value.tensor.tensor.data);
  const auto count = executable->inputs[2]->value.scalar.value.i32;
  const auto *route = static_cast<const float *>(
      executable->inputs[3]->value.tensor.tensor.data);
  auto *trajectory =
      static_cast<float *>(executable->outputs[0]->value.tensor.tensor.data);
  auto *prediction =
      static_cast<float *>(executable->outputs[1]->value.tensor.tensor.data);
  for (std::size_t step = 0; step < 6u; ++step) {
    trajectory[step * 2u] = static_cast<float>(step) * 0.4f + bev[0] * 0.01f;
    trajectory[step * 2u + 1u] = route[1] + bev[1] * 0.01f;
    if (count > 0) {
      const auto agent = step % static_cast<std::size_t>(count);
      prediction[step * 2u] =
          agents[agent * 3u] + static_cast<float>(step) * 0.1f;
      prediction[step * 2u + 1u] = agents[agent * 3u + 1u];
    } else {
      prediction[step * 2u] = static_cast<float>(step) * 0.1f;
      prediction[step * 2u + 1u] = 0.0f;
    }
  }
  const auto token =
      static_cast<std::int64_t>(
          std::fabs(bev[0] + bev[1] + bev[2] + bev[3] + route[0]) * 10.0f) %
      1024;
  auto *scalar =
      const_cast<VLAForgeScalarValue *>(&executable->outputs[2]->value.scalar);
  scalar->value.i64 = token;
  return Ok();
}

VLAForgeStatus Run(VLAForgeRegionExecutable *executable) {
  if (executable == nullptr || !executable->loaded) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external BEV plugin is not loaded");
  }
  return executable->region_id == 0u ? RunPreprocess(executable)
                                     : RunPlanner(executable);
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable *executable) {
  if (executable == nullptr || !executable->loaded) {
    return Error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                 "external BEV plugin is not loaded");
  }
  return Ok();
}

void Destroy(VLAForgeRegionExecutable *executable) { delete executable; }

const VLAForgeRegionExecutableValueApi kApi{
    sizeof(VLAForgeRegionExecutableValueApi),
#ifdef VLAFORGE_PLUGIN_BAD_ABI
    999u,
#else
    VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
#endif
    &Create,
    &Load,
    &QueryWorkspace,
    &BindInput,
    &BindOutput,
    &BindWorkspace,
    &Run,
    &Synchronize,
    &Destroy,
};

} // namespace

#ifndef VLAFORGE_PLUGIN_OMIT_ENTRYPOINT
extern "C" const VLAForgeRegionExecutableValueApi *
vlaforge_region_executable_value_api(void) {
  return &kApi;
}
#endif
