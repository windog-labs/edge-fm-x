#include "vlaforge/runtime/numerical.h"

#include <array>
#include <cstring>
#include <limits>

namespace {
constexpr char sha[] = "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char other_sha[] = "2222222222222222222222222222222222222222222222222222222222222222";
VLAForgeNumericalEntry field{"precise", 7, VLAFORGE_NUMERICAL_BOOL, 1, 0.0, nullptr, 0};
VLAForgeNumericalRequirementView requirement{
    sizeof(requirement), VLAFORGE_NUMERICAL_ABI_VERSION, "test/1", 6,
    sha, 64, sha, 64, 1, &field, 1};
int current = 1, active = 0, acquired = 0, released = 0, binds = 0;
bool fail_second = false;
std::array<int, 2> leases{1, 1};
VLAForgeStatus Ok() { return vlaforge_status_ok(); }
VLAForgeStatus Fail() { return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "fake"); }
VLAForgeStatus Query(const VLAForgeNumericalRequirementView* value) {
  return value->namespace_size == 6 && std::memcmp(value->policy_namespace, "test/1", 6) == 0 ? Ok() : Fail();
}
VLAForgeStatus Acquire(const VLAForgeNumericalRequirementView*, void** output) {
  *output = nullptr;
  if ((fail_second && acquired == 1) || current != 1) { return Fail(); }
  *output = &leases[acquired++]; ++active; return Ok();
}
VLAForgeStatus Validate(void* lease, VLAForgeNumericalBoundary) {
  return *static_cast<int*>(lease) == current ? Ok() : Fail();
}
VLAForgeStatus Bind(VLAForgeRegionExecutable*, void*) { ++binds; return Ok(); }
void Release(void*) { --active; ++released; }
const VLAForgeNumericalProviderApi api{
    sizeof(api), VLAFORGE_NUMERICAL_ABI_VERSION, "test", 4,
    Query, Acquire, Validate, Bind, Release};
}  // namespace

int main() {
  if (vlaforge_numerical_provider_api_validate(nullptr).code == VLAFORGE_STATUS_OK ||
      vlaforge_numerical_requirement_validate(nullptr).code == VLAFORGE_STATUS_OK) { return 1; }
  auto bad = requirement;
  bad.abi_version = 99;
  if (vlaforge_numerical_requirement_validate(&bad).code == VLAFORGE_STATUS_OK) { return 2; }
  bad = requirement; bad.policy_sha256_size = 63;
  if (vlaforge_numerical_requirement_validate(&bad).code == VLAFORGE_STATUS_OK) { return 3; }
  auto wrong_field = field; wrong_field.integer = 2;
  bad = requirement; bad.entries = &wrong_field;
  if (vlaforge_numerical_requirement_validate(&bad).code == VLAFORGE_STATUS_OK) { return 4; }
  wrong_field.kind = VLAFORGE_NUMERICAL_F64;
  wrong_field.integer = 0; wrong_field.real = std::numeric_limits<double>::infinity();
  if (vlaforge_numerical_requirement_validate(&bad).code == VLAFORGE_STATUS_OK) { return 5; }
  std::array<VLAForgeNumericalEntry, 2> duplicate{field, field};
  bad = requirement; bad.entries = duplicate.data(); bad.entry_count = 2;
  if (vlaforge_numerical_requirement_validate(&bad).code == VLAFORGE_STATUS_OK) { return 6; }
  vlaforge::runtime::NumericalLeaseSet set;
  if (set.Validate(VLAFORGE_NUMERICAL_RUN_ENTRY).ok() ||
      set.Add(0, nullptr, &requirement).ok() || !set.Add(0, &api, &requirement).ok() ||
      set.Add(0, &api, &requirement).ok()) { return 7; }
  bad = requirement; bad.policy_sha256 = other_sha;
  if (set.Add(1, &api, &bad).ok()) { return 8; }
  wrong_field = field; wrong_field.integer = 0;
  bad = requirement; bad.entries = &wrong_field;
  // Matching claimed digests cannot conceal contradictory typed entries.
  if (set.Add(1, &api, &bad).ok()) { return 9; }
  if (!set.Add(1, &api, &requirement).ok() || !set.AcquireAll().ok() || active != 2 ||
      set.AcquireAll().ok() || set.Add(2, &api, &requirement).ok()) { return 10; }
  if (!set.Validate(VLAFORGE_NUMERICAL_RUN_ENTRY).ok()) { return 11; }
  current = 2;
  if (set.Validate(VLAFORGE_NUMERICAL_BEFORE_COMMIT).ok()) { return 12; }
  set.Clear();
  if (active != 0 || released != 2) { return 13; }
  current = 1; acquired = 0; released = 0; fail_second = true;
  if (!set.Add(0, &api, &requirement).ok() || !set.Add(1, &api, &requirement).ok() ||
      set.AcquireAll().ok() || active != 0 || released != 1) { return 14; }
  return 0;
}
