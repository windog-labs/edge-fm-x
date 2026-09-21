#include "vlaforge/runtime/execution_context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace {

void Log(const char* event, unsigned id, const void* pointer = nullptr) {
  const auto* path = std::getenv("VLAFORGE_CONTEXT_TEST_LOG");
  if (path != nullptr) {
    if (auto* stream = std::fopen(path, "a")) {
      std::fprintf(stream, "%s,%u,%p\n", event, id, pointer);
      std::fclose(stream);
    }
  }
}

bool Mode(const char* expected) {
  const auto* mode = std::getenv("VLAFORGE_CONTEXT_TEST_MODE");
  return mode != nullptr && std::strcmp(mode, expected) == 0;
}

VLAForgeStatus Ok() { return {VLAFORGE_STATUS_OK, nullptr, 0u}; }
VLAForgeStatus Error() {
  constexpr char message[] = "injected context fixture failure";
  return {VLAFORGE_STATUS_FAILED_PRECONDITION,
          "injected context fixture failure", sizeof(message) - 1};
}

}  // namespace

struct VLAForgeRegionExecutable {
  unsigned id = 0;
  bool context_bound = false;
  const VLAForgeValueView* input = nullptr;
  const VLAForgeValueView* output = nullptr;
};

namespace {

VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
                      VLAForgeRegionExecutable** result) {
  if (options == nullptr || result == nullptr) { return Error(); }
  *result = nullptr;
  Log("create", options->region_id);
  if (Mode("createfail") && options->region_id == 1u) { return Error(); }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) { return Error(); }
  executable->id = options->region_id;
  *result = executable;
  return Ok();
}

VLAForgeStatus Load(VLAForgeRegionExecutable* executable,
                    const VLAForgeArtifactDescriptor*) {
  Log("load", executable->id);
  return Mode("loadfail") && executable->id == 1u ? Error() : Ok();
}

VLAForgeStatus Query(const VLAForgeRegionExecutable*,
                     VLAForgeWorkspaceRequirement* workspace) {
  *workspace = {0u, 1u, {VLAFORGE_DEVICE_CPU, 0}};
  return Ok();
}

VLAForgeStatus BindInput(VLAForgeRegionExecutable* executable, std::uint32_t,
                         const VLAForgeValueView* value) {
  executable->input = value;
  return Ok();
}

VLAForgeStatus BindOutput(VLAForgeRegionExecutable* executable, std::uint32_t,
                          const VLAForgeValueView* value) {
  executable->output = value;
  return Ok();
}

VLAForgeStatus Workspace(VLAForgeRegionExecutable*, void*, std::uint64_t size) {
  return size == 0u ? Ok() : Error();
}

VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
#if !defined(VLAFORGE_CONTEXT_TEST_LEGACY)
  if (!executable->context_bound) { return Error(); }
#endif
  if (executable->input == nullptr || executable->output == nullptr) {
    return Error();
  }
  const auto* input = static_cast<const float*>(
      executable->input->value.tensor.tensor.data);
  auto* output = static_cast<float*>(
      executable->output->value.tensor.tensor.data);
  for (unsigned index = 0; index < 2u; ++index) {
    output[index] = input[index] + static_cast<float>(executable->id + 1u);
  }
  return Ok();
}

VLAForgeStatus Synchronize(VLAForgeRegionExecutable*) { return Ok(); }
void Destroy(VLAForgeRegionExecutable* executable) {
  Log("destroy", executable->id);
  delete executable;
}

#if !defined(VLAFORGE_CONTEXT_TEST_LEGACY)
VLAForgeStatus BindContext(VLAForgeRegionExecutable* executable,
                           const VLAForgeExecutionContextView* view) {
  Log("bind", executable->id, view);
  if (Mode("bindfail") && executable->id == 1u) { return Error(); }
  if (view == nullptr || view->struct_size < sizeof(*view) ||
      view->device.kind != VLAFORGE_DEVICE_CPU || view->native_stream != nullptr) {
    return Error();
  }
  executable->context_bound = true;
  return Ok();
}
#endif

const VLAForgeRegionExecutableValueApi kApi = {
    sizeof(kApi), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    &Create, &Load, &Query, &BindInput, &BindOutput, &Workspace, &Run,
    &Synchronize, &Destroy};

}  // namespace

extern "C" const VLAForgeRegionExecutableValueApi*
vlaforge_region_executable_value_api(void) { return &kApi; }

#if !defined(VLAFORGE_CONTEXT_TEST_LEGACY)
extern "C" const VLAForgeRegionExecutionExtensionApi*
vlaforge_region_execution_extension_api(void) {
  if (Mode("nullprovider")) { return nullptr; }
  if (Mode("throwprovider")) { throw std::runtime_error("injected provider failure"); }
  static const VLAForgeRegionExecutionExtensionApi valid = {
      sizeof(valid), VLAFORGE_REGION_EXECUTION_EXTENSION_ABI_VERSION,
      VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT, &BindContext};
  static const VLAForgeRegionExecutionExtensionApi invalid = {
      sizeof(invalid), 999u, VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT,
      &BindContext};
  return Mode("badabi") ? &invalid : &valid;
}
#endif
