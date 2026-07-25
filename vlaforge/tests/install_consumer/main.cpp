#include "vlaforge/runtime/artifact_verifier.h"
#include "vlaforge/runtime/static_arena.h"

#if defined(VLAFORGE_CONSUMER_HAS_AOTI)
#include "vlaforge/backends/aoti_region_executable.h"
#endif

#include <string>

int main() {
#if defined(VLAFORGE_CONSUMER_HAS_AOTI)
  if (vlaforge_aoti_region_executable_value_api() == nullptr) {
    return 3;
  }
#endif
  vlaforge::runtime::StaticArena arena(64u, 64u);
  if (arena.Resolve(0u, 64u, 64u) == nullptr) {
    return 1;
  }
  std::string resolved;
  const auto status = vlaforge::runtime::VerifyArtifactFile(
      "", "artifact.bin",
      "0000000000000000000000000000000000000000000000000000000000000000",
      0u, &resolved);
  return status.ok() ? 2 : 0;
}
