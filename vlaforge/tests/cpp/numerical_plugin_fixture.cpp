#include "vlaforge/runtime/numerical.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <stdexcept>

namespace {
int current_precision = 1;
unsigned queries = 0, acquired = 0, released = 0, live_regions = 0, next_id = 0;
thread_local bool allowed_thread = true;

struct Lease { int precision; unsigned id; };

void Log(const char* event, unsigned id = 0) {
  if (auto* path = std::getenv("VLAFORGE_NUMERICAL_TEST_LOG")) {
    if (auto* stream = std::fopen(path, "a")) {
      std::fprintf(stream, "%s,%u\n", event, id);
      std::fclose(stream);
    }
  }
}
bool Mode(const char* value) {
  const auto* mode = std::getenv("VLAFORGE_NUMERICAL_TEST_MODE");
  return mode != nullptr && std::strcmp(mode, value) == 0;
}
VLAForgeStatus Ok() { return {VLAFORGE_STATUS_OK, nullptr, 0}; }
VLAForgeStatus Error() {
  return {VLAFORGE_STATUS_FAILED_PRECONDITION, "numerical test failure", 22};
}
bool Equal(const char* data, std::size_t size, const char* expected) {
  return size == std::strlen(expected) && std::memcmp(data, expected, size) == 0;
}
}  // namespace

struct VLAForgeRegionExecutable {
  unsigned id = 0, runs = 0, syncs_since_run = 0;
  Lease* lease = nullptr;
  const VLAForgeValueView* input = nullptr;
  const VLAForgeValueView* output = nullptr;
};

namespace {
VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
                      VLAForgeRegionExecutable** output) {
  Log("create", options->region_id);
  *output = nullptr;
  if (Mode("createfail") && options->region_id == 1) { return Error(); }
  auto* region = new (std::nothrow) VLAForgeRegionExecutable;
  if (!region) { return Error(); }
  region->id = options->region_id;
  ++live_regions;
  *output = region;
  return Ok();
}
VLAForgeStatus Load(VLAForgeRegionExecutable* region, const VLAForgeArtifactDescriptor*) {
  Log("load", region->id);
#if !defined(VLAFORGE_NUMERICAL_TEST_LEGACY)
  const unsigned required = Mode("bundle") ? 1u : 2u;
  if (region->lease == nullptr || acquired - released < required || queries < required) { return Error(); }
#endif
  if (Mode("driftload") && region->id == 1) { current_precision = 2; }
  return Mode("loadfail") && region->id == 1 ? Error() : Ok();
}
VLAForgeStatus Query(const VLAForgeRegionExecutable*, VLAForgeWorkspaceRequirement* output) {
  *output = {0u, 1u, {VLAFORGE_DEVICE_CPU, 0}};
  return Ok();
}
VLAForgeStatus Input(VLAForgeRegionExecutable* region, std::uint32_t, const VLAForgeValueView* value) {
  region->input = value;
  return Ok();
}
VLAForgeStatus Output(VLAForgeRegionExecutable* region, std::uint32_t, const VLAForgeValueView* value) {
  region->output = value;
  return Ok();
}
VLAForgeStatus Workspace(VLAForgeRegionExecutable*, void*, std::uint64_t size) {
  return size == 0 ? Ok() : Error();
}
VLAForgeStatus Run(VLAForgeRegionExecutable* region) {
  Log("run", region->id);
  if (!region->input || !region->output) { return Error(); }
  ++region->runs;
  region->syncs_since_run = 0;
  auto* in = static_cast<const float*>(region->input->value.tensor.tensor.data);
  auto* out = static_cast<float*>(region->output->value.tensor.tensor.data);
  out[0] = in[0] + region->id + 1;
  out[1] = in[1] + region->id + 1;
  if (Mode("exitdrift") && region->id == 1 && region->runs == 2) { current_precision = 2; }
  return Ok();
}
VLAForgeStatus Sync(VLAForgeRegionExecutable* region) {
  Log("drain", region->id);
  ++region->syncs_since_run;
  if (Mode("draindrift") && region->id == 1 && region->runs == 2 && region->syncs_since_run == 2) {
    current_precision = 2;
  }
  if (Mode("drainfail") && region->id == 1 && region->runs == 2 && region->syncs_since_run >= 2) {
    return Error();
  }
  return Ok();
}
void Destroy(VLAForgeRegionExecutable* region) {
  Log("destroy", region->id);
  --live_regions;
  delete region;
}

[[maybe_unused]] VLAForgeStatus Support(const VLAForgeNumericalRequirementView* requirement) {
  Log("query", ++queries);
  if (Mode("queryfail") && queries == 2) { return Error(); }
  if (!Equal(requirement->policy_namespace, requirement->namespace_size, "test.numerics/1") ||
      requirement->entry_count != 4 || requirement->execution_lane != 1) { return Error(); }
  const auto* entries = requirement->entries;
  if (!Equal(entries[0].name, entries[0].name_size, "epsilon") ||
      entries[0].kind != VLAFORGE_NUMERICAL_F64 || entries[0].real != 0.125 ||
      !Equal(entries[1].name, entries[1].name_size, "mode") ||
      entries[1].kind != VLAFORGE_NUMERICAL_STRING ||
      !Equal(entries[1].text, entries[1].text_size, "strict") ||
      !Equal(entries[2].name, entries[2].name_size, "precision") ||
      entries[2].kind != VLAFORGE_NUMERICAL_I64 ||
      !Equal(entries[3].name, entries[3].name_size, "strict") ||
      entries[3].kind != VLAFORGE_NUMERICAL_BOOL || entries[3].integer != 1) { return Error(); }
  return Ok();
}
[[maybe_unused]] VLAForgeStatus Acquire(const VLAForgeNumericalRequirementView* requirement, void** output) {
  Log("acquire", next_id);
  *output = nullptr;
  if ((Mode("acquirefail") && next_id == 1) || !allowed_thread ||
      requirement->entries[2].integer != current_precision) { return Error(); }
  auto* lease = new (std::nothrow) Lease{current_precision, next_id++};
  if (!lease) { return Error(); }
  ++acquired;
  *output = lease;
  return Ok();
}
[[maybe_unused]] VLAForgeStatus Validate(void* opaque, VLAForgeNumericalBoundary boundary) {
  auto* lease = static_cast<Lease*>(opaque);
  Log(boundary == VLAFORGE_NUMERICAL_BEFORE_COMMIT ? "validate_commit" :
      boundary == VLAFORGE_NUMERICAL_RUN_ENTRY ? "validate_entry" :
      boundary == VLAFORGE_NUMERICAL_BEFORE_LOAD ? "validate_preload" : "validate_load", lease->id);
  return allowed_thread && current_precision == lease->precision ? Ok() : Error();
}
[[maybe_unused]] VLAForgeStatus Bind(VLAForgeRegionExecutable* region, void* opaque) {
  Log("bind_policy", region->id);
  if (Mode("bindfail") && region->id == 1) { return Error(); }
  region->lease = static_cast<Lease*>(opaque);
  return Ok();
}
[[maybe_unused]] void Release(void* opaque) {
  auto* lease = static_cast<Lease*>(opaque);
  Log("release", lease->id);
  if (live_regions != 0 && !Mode("multisession")) { Log("release_before_destroy", live_regions); }
  ++released;
  delete lease;
}

const VLAForgeRegionExecutableValueApi api{
    sizeof(api), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    Create, Load, Query, Input, Output, Workspace, Run, Sync, Destroy};
}  // namespace

extern "C" const VLAForgeRegionExecutableValueApi* vlaforge_region_executable_value_api() {
  return &api;
}

#if !defined(VLAFORGE_NUMERICAL_TEST_LEGACY)
extern "C" const VLAForgeNumericalProviderApi* vlaforge_numerical_provider_api() {
  static const VLAForgeNumericalProviderApi api{
      sizeof(api), VLAFORGE_NUMERICAL_ABI_VERSION, "test.process", 12,
      Support, Acquire, Validate, Bind, Release};
  static const VLAForgeNumericalProviderApi bad{
      sizeof(bad), 99, "test.process", 12, Support, Acquire, Validate, Bind, Release};
  if (Mode("badabi")) { return &bad; }
  if (Mode("nullprovider")) { return nullptr; }
  if (Mode("throwprovider")) { throw std::runtime_error("test provider"); }
  return &api;
}
#endif

// Test-driver mutations only. The actual provider has no setting operation.
extern "C" void numerical_test_current(int precision) { current_precision = precision; }
extern "C" void numerical_test_thread(int allowed) { allowed_thread = allowed != 0; }
