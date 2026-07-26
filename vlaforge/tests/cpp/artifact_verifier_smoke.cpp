#include "vlaforge/runtime/artifact_verifier.h"

#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

int ProcessId() {
#if defined(_WIN32)
  return _getpid();
#else
  return getpid();
#endif
}

}  // namespace

int main() {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("vlaforge_artifact_verifier_smoke_" +
       std::to_string(ProcessId()));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directories(root / "artifacts");
  {
    std::ofstream output(root / "artifacts" / "region.bin",
                         std::ios::binary);
    output << "abc";
  }
  std::string resolved;
  const auto passed = vlaforge::runtime::VerifyArtifactFile(
      root.string(), "artifacts/region.bin",
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      3u, &resolved);
  if (!passed.ok() || resolved.empty()) {
    return 1;
  }
  const auto root_text = root.string();
  constexpr char kPath[] = "artifacts/region.bin";
  constexpr char kSha256[] =
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
      "410ff61f20015ad";
  const char* abi_resolved = nullptr;
  const char* abi_error = nullptr;
  const auto abi_status = vlaforge_verify_artifact_file_abi(
      root_text.data(), root_text.size(), kPath, sizeof(kPath) - 1u,
      kSha256, sizeof(kSha256) - 1u, 3u, &abi_resolved, &abi_error);
  if (abi_status != 0u || abi_resolved == nullptr ||
      std::string(abi_resolved).empty() || abi_error == nullptr) {
    return 2;
  }
  const auto bad_hash = vlaforge::runtime::VerifyArtifactFile(
      root.string(), "artifacts/region.bin",
      "0000000000000000000000000000000000000000000000000000000000000000",
      3u, &resolved);
  if (bad_hash.ok()) {
    return 3;
  }
  const auto escape = vlaforge::runtime::VerifyArtifactFile(
      root.string(), "../region.bin",
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      3u, &resolved);
  std::filesystem::remove_all(root, error);
  return escape.ok() ? 4 : 0;
}
