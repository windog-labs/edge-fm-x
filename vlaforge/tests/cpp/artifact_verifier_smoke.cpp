#include "vlaforge/runtime/artifact_verifier.h"

#include <filesystem>
#include <fstream>
#include <string>

int main() {
  const auto root =
      std::filesystem::temp_directory_path() /
      "vlaforge_artifact_verifier_smoke";
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
  const auto bad_hash = vlaforge::runtime::VerifyArtifactFile(
      root.string(), "artifacts/region.bin",
      "0000000000000000000000000000000000000000000000000000000000000000",
      3u, &resolved);
  if (bad_hash.ok()) {
    return 2;
  }
  const auto escape = vlaforge::runtime::VerifyArtifactFile(
      root.string(), "../region.bin",
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      3u, &resolved);
  std::filesystem::remove_all(root, error);
  return escape.ok() ? 3 : 0;
}
