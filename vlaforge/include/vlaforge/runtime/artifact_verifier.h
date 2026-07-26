#ifndef VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_
#define VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_

#include <cstddef>
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

// ABI-stable bridge for backends built against a LibTorch distribution whose
// libstdc++ ABI differs from the core runtime. The returned strings are owned
// by the runtime and remain valid until the next call on the same thread.
extern "C" std::uint32_t vlaforge_verify_artifact_file_abi(
    const char* bundle_root, std::size_t bundle_root_size,
    const char* relative_path, std::size_t relative_path_size,
    const char* expected_sha256, std::size_t expected_sha256_size,
    std::uint64_t expected_size, const char** resolved_path,
    const char** error_message) noexcept;

#endif  // VLAFORGE_RUNTIME_ARTIFACT_VERIFIER_H_
