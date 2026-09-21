#ifndef VLAFORGE_RUNTIME_NUMERICAL_H_
#define VLAFORGE_RUNTIME_NUMERICAL_H_

#include <stddef.h>
#include <stdint.h>

#include "vlaforge/runtime/region_executable.h"

#define VLAFORGE_NUMERICAL_ABI_VERSION 1u
#define VLAFORGE_NUMERICAL_PROVIDER_API_SYMBOL "vlaforge_numerical_provider_api"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VLAForgeNumericalKind {
  VLAFORGE_NUMERICAL_BOOL = 1,
  VLAFORGE_NUMERICAL_I64 = 2,
  VLAFORGE_NUMERICAL_F64 = 3,
  VLAFORGE_NUMERICAL_STRING = 4
} VLAForgeNumericalKind;

/* Unused value members must be zero/null. Strings are sized, not JSON. */
typedef struct VLAForgeNumericalEntry {
  const char* name;
  size_t name_size;
  VLAForgeNumericalKind kind;
  int64_t integer;
  double real;
  const char* text;
  size_t text_size;
} VLAForgeNumericalEntry;

typedef struct VLAForgeNumericalRequirementView {
  size_t struct_size;
  uint32_t abi_version;
  const char* policy_namespace;
  size_t namespace_size;
  const char* policy_sha256;
  size_t policy_sha256_size;
  const char* requirement_sha256;
  size_t requirement_sha256_size;
  /* 1 = same-precision; 2 = quantized. Neither means output parity passed. */
  uint32_t execution_lane;
  const VLAForgeNumericalEntry* entries;
  size_t entry_count;
} VLAForgeNumericalRequirementView;

typedef enum VLAForgeNumericalBoundary {
  VLAFORGE_NUMERICAL_BEFORE_LOAD = 1,
  VLAFORGE_NUMERICAL_AFTER_LOAD = 2,
  VLAFORGE_NUMERICAL_RUN_ENTRY = 3,
  VLAFORGE_NUMERICAL_BEFORE_COMMIT = 4
} VLAForgeNumericalBoundary;

/* Optional, independent of Region/Session ABI. All callbacks are host-only,
 * nonthrowing, validation-only: no setters, restore guards or captured work.
 * query_support validates every namespace/key/type/value without acquiring.
 * acquire_current checks current process AND calling-thread state and returns
 * a stable immutable lease. On failure it must return a null lease.
 * validate_current checks the actual calling thread at each boundary.
 * bind_region is called after create and before load; the lease outlives the
 * Region. release is called after drain and Region destruction, before dlclose.
 * Providers own cross-Session conflict arbitration in their actual process
 * domain. A lease does not prevent unrelated external global-state setters.
 */
typedef struct VLAForgeNumericalProviderApi {
  size_t struct_size;
  uint32_t abi_version;
  const char* process_domain;
  size_t process_domain_size;
  VLAForgeStatus (*query_support)(const VLAForgeNumericalRequirementView*);
  VLAForgeStatus (*acquire_current)(const VLAForgeNumericalRequirementView*, void**);
  VLAForgeStatus (*validate_current)(void*, VLAForgeNumericalBoundary);
  VLAForgeStatus (*bind_region)(VLAForgeRegionExecutable*, void*);
  void (*release)(void*);
} VLAForgeNumericalProviderApi;

typedef const VLAForgeNumericalProviderApi* (*VLAForgeNumericalProviderApiFn)(void);

VLAForgeStatus vlaforge_numerical_provider_api_validate(
    const VLAForgeNumericalProviderApi* api);
VLAForgeStatus vlaforge_numerical_requirement_validate(
    const VLAForgeNumericalRequirementView* requirement);

#ifdef __cplusplus
}  // extern "C"

#include <vector>

#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

// Session-owned staging and lifetime of provider leases; not a process broker.
// Add borrows immutable provider/requirement tables through Clear; generated
// Sessions use static constants. Standalone callers must honor that lifetime.
class NumericalLeaseSet final {
 public:
  NumericalLeaseSet() = default;
  ~NumericalLeaseSet() { Clear(); }
  NumericalLeaseSet(const NumericalLeaseSet&) = delete;
  NumericalLeaseSet& operator=(const NumericalLeaseSet&) = delete;

  Status Add(std::uint32_t region_id, const VLAForgeNumericalProviderApi* provider,
             const VLAForgeNumericalRequirementView* requirement) noexcept;
  Status AcquireAll() noexcept;
  Status Bind(std::uint32_t region_id, VLAForgeRegionExecutable* region) noexcept;
  Status Validate(VLAForgeNumericalBoundary boundary) noexcept;
  void Clear() noexcept;
  // Fatal device/capture failures retain provider/lease ownership until exit.
  void Abandon() noexcept;

 private:
  struct Entry {
    std::uint32_t region_id;
    const VLAForgeNumericalProviderApi* provider;
    const VLAForgeNumericalRequirementView* requirement;
    void* lease = nullptr;
  };
  std::vector<Entry> entries_;
  bool acquired_ = false;
};

}  // namespace vlaforge::runtime
#endif
#endif  // VLAFORGE_RUNTIME_NUMERICAL_H_
