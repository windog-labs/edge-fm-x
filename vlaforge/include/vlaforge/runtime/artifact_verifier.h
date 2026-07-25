#ifndef VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_
#define VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

// Resolves and authenticates one bundle-relative artifact. The resolved path
// is returned only after containment, regular-file, size, and SHA-256 checks
// all pass.
[[nodiscard]] Status VerifyArtifactFile(
    std::string_view bundle_root, std::string_view relative_path,
    std::string_view expected_sha256, std::uint64_t expected_size,
    std::string* resolved_path) noexcept;

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_
