#include "aoti_materialized_package.h"

#include "vlaforge/runtime/artifact_verifier.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

namespace vlaforge::backends {
namespace {
namespace fs = std::filesystem;

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

std::string Token(std::istream& input) {
  std::string value;
  Require(static_cast<bool>(input >> value), "truncated materialized AOTI manifest");
  Require(value.size() <= 4096u, "materialized AOTI token exceeds limit");
  return value;
}

void Expect(std::istream& input, const char* value) {
  Require(Token(input) == value, "unknown materialized AOTI manifest token");
}

std::uint64_t Number(std::istream& input) {
  const auto token = Token(input);
  std::uint64_t value = 0;
  const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
  Require(parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size() &&
              std::to_string(value) == token && value <= std::numeric_limits<std::int64_t>::max(),
          "invalid materialized AOTI number");
  return value;
}

void Digest(const std::string& value) {
  Require(value.size() == 64u, "invalid materialized AOTI SHA-256");
  for (const auto ch : value) {
    Require((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'),
            "invalid materialized AOTI SHA-256");
  }
}

void Name(const std::string& value) {
  const fs::path path(value);
  Require(!value.empty() && value != "-" && !path.is_absolute() &&
              path.generic_string() == value && path.lexically_normal() == path,
          "unsafe materialized AOTI path");
  for (const auto& part : path) Require(part != "." && part != "..", "unsafe materialized AOTI path");
  for (const unsigned char ch : value) {
    Require((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-' || ch == '/',
            "unsafe materialized AOTI path");
  }
}

std::string Verified(const std::string& root, const std::string& name,
                     const std::string& digest, std::uint64_t size) {
  Name(name);
  Digest(digest);
  const char* result = nullptr;
  const char* error = nullptr;
  const auto code = vlaforge_verify_artifact_file_abi(root.data(), root.size(), name.data(), name.size(),
      digest.data(), digest.size(), size, &result, &error);
  if (code != static_cast<std::uint32_t>(runtime::StatusCode::kOk)) {
    throw std::runtime_error(error ? error : "materialized AOTI file verification failed");
  }
  return result;
}
}  // namespace

AotiMaterializedPackage::AotiMaterializedPackage(
    const std::string& manifest, const std::string& sha256,
    std::uint64_t size_bytes, const std::string& device_type) {
  Require(size_bytes > 0u && size_bytes <= 16u * 1024u * 1024u,
          "materialized AOTI manifest size exceeds limit");
  const auto root = fs::path(manifest).parent_path().string();
  const auto verified = Verified(root, fs::path(manifest).filename().string(), sha256, size_bytes);
  std::ifstream input(verified);
  Expect(input, "VLAFORGE_AOTI_MATERIALIZED");
  Require(Number(input) == 1u, "unsupported materialized AOTI version");
  Expect(input, "package");
  Digest(Token(input));
  Require(Number(input) > 0u, "invalid source AOTI package size");
  Expect(input, "model");
  const auto declared_device = Token(input);
  Require((declared_device == "cpu" || declared_device == "cuda") && declared_device == device_type,
          "materialized AOTI device mismatch");
  const auto library = Token(input);
  const auto blob = Token(input);
  Name(library);
  if (blob != "-") Name(blob);
  Require(fs::path(library).extension() == ".so", "materialized AOTI library must be a shared object");
  Expect(input, "files");
  const auto count = Number(input);
  Require(count > 0u && count <= 100000u, "materialized AOTI file count exceeds limit");
  std::map<std::string, std::string> files;
  std::string previous;
  std::size_t libraries = 0, blobs = 0;
  for (std::uint64_t i = 0; i < count; ++i) {
    Expect(input, "file");
    const auto name = Token(input);
    const auto digest = Token(input);
    const auto size = Number(input);
    Require(previous.empty() || previous < name, "materialized AOTI files must be sorted and unique");
    previous = name;
    const auto extension = fs::path(name).extension();
    if (extension == ".so") {
      Require(name == library && size > 0u, "undeclared materialized AOTI shared library");
      ++libraries;
    }
    if (extension == ".blob") {
      Require(name == blob, "undeclared materialized AOTI weight blob");
      ++blobs;
    }
    files.emplace(name, Verified(root, name, digest, size));
  }
  Expect(input, "end");
  std::string trailing;
  Require(!(input >> trailing), "trailing materialized AOTI manifest data");
  Require(libraries == 1u && blobs == (blob == "-" ? 0u : 1u), "materialized AOTI library/blob set incomplete");
  auto metadata = fs::path(library);
  metadata.replace_extension();
  Require(files.count(metadata.string() + "_metadata.json") == 1u, "materialized AOTI library metadata missing");
  library_ = files.at(library);
  if (blob != "-") weight_blob_ = files.at(blob);
}

}  // namespace vlaforge::backends
