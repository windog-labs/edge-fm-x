#include "vlaforge/runtime/artifact_verifier.h"

#include <openssl/evp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>

namespace vlaforge::runtime {
namespace {

constexpr std::size_t kSha256HexSize = 64u;
constexpr std::size_t kReadBufferSize = 1024u * 1024u;

bool IsSafeRelativePath(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute() ||
      path.lexically_normal() != path || path == ".") {
    return false;
  }
  for (const auto& component : path) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

bool IsLowerHex(std::string_view value) {
  if (value.size() != kSha256HexSize) {
    return false;
  }
  for (const char item : value) {
    if (!((item >= '0' && item <= '9') ||
          (item >= 'a' && item <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool IsWithin(const std::filesystem::path& root,
              const std::filesystem::path& candidate) {
  auto root_item = root.begin();
  auto candidate_item = candidate.begin();
  for (; root_item != root.end(); ++root_item, ++candidate_item) {
    if (candidate_item == candidate.end() ||
        *root_item != *candidate_item) {
      return false;
    }
  }
  return true;
}

bool Sha256(const std::filesystem::path& path, std::string* digest) {
  using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  Context context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    return false;
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  std::array<char, kReadBufferSize> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = stream.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      return false;
    }
  }
  if (!stream.eof()) {
    return false;
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int byte_count = 0u;
  if (EVP_DigestFinal_ex(context.get(), bytes.data(), &byte_count) != 1 ||
      byte_count != 32u) {
    return false;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < byte_count; ++index) {
    text << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  *digest = text.str();
  return true;
}

}  // namespace

Status VerifyArtifactFile(
    std::string_view bundle_root, std::string_view relative_path,
    std::string_view expected_sha256, std::uint64_t expected_size,
    std::string* resolved_path) noexcept {
  if (bundle_root.empty() || resolved_path == nullptr ||
      !IsLowerHex(expected_sha256)) {
    return Status::Error(StatusCode::kInvalidArgument, 0u,
                         "invalid artifact verification request");
  }
  try {
    const std::filesystem::path relative{relative_path};
    if (!IsSafeRelativePath(relative)) {
      return Status::Error(StatusCode::kInvalidArgument, 0u,
                           "artifact path must be normalized and relative");
    }
    std::error_code error;
    const auto root = std::filesystem::canonical(
        std::filesystem::path{bundle_root}, error);
    if (error) {
      return Status::Error(StatusCode::kNotFound, 0u,
                           "bundle root does not exist");
    }
    const auto candidate =
        std::filesystem::weakly_canonical(root / relative, error);
    if (error || !IsWithin(root, candidate)) {
      return Status::Error(StatusCode::kFailedPrecondition, 0u,
                           "artifact path escapes bundle root");
    }
    if (!std::filesystem::is_regular_file(candidate, error) || error) {
      return Status::Error(StatusCode::kNotFound, 0u,
                           "artifact file does not exist");
    }
    const auto actual_size = std::filesystem::file_size(candidate, error);
    if (error || actual_size != expected_size) {
      return Status::Error(StatusCode::kFailedPrecondition, 0u,
                           "artifact size mismatch");
    }
    std::string actual_sha256;
    if (!Sha256(candidate, &actual_sha256)) {
      return Status::Error(StatusCode::kInternal, 0u,
                           "artifact SHA-256 computation failed");
    }
    if (actual_sha256 != expected_sha256) {
      return Status::Error(StatusCode::kFailedPrecondition, 0u,
                           "artifact SHA-256 mismatch");
    }
    *resolved_path = candidate.string();
    return Status::Ok();
  } catch (...) {
    return Status::Error(StatusCode::kInternal, 0u,
                         "artifact verification threw an exception");
  }
}

}  // namespace vlaforge::runtime

extern "C" std::uint32_t vlaforge_verify_artifact_file_abi(
    const char* bundle_root, std::size_t bundle_root_size,
    const char* relative_path, std::size_t relative_path_size,
    const char* expected_sha256, std::size_t expected_sha256_size,
    std::uint64_t expected_size, const char** resolved_path,
    const char** error_message) noexcept {
  thread_local std::string resolved_storage;
  if (resolved_path == nullptr || error_message == nullptr ||
      bundle_root == nullptr || relative_path == nullptr ||
      expected_sha256 == nullptr) {
    if (resolved_path != nullptr) {
      *resolved_path = nullptr;
    }
    if (error_message != nullptr) {
      *error_message = "invalid artifact verification ABI request";
    }
    return static_cast<std::uint32_t>(
        vlaforge::runtime::StatusCode::kInvalidArgument);
  }
  resolved_storage.clear();
  const auto status = vlaforge::runtime::VerifyArtifactFile(
      std::string_view(bundle_root, bundle_root_size),
      std::string_view(relative_path, relative_path_size),
      std::string_view(expected_sha256, expected_sha256_size),
      expected_size, &resolved_storage);
  if (!status.ok()) {
    *resolved_path = nullptr;
    *error_message = status.message;
    return static_cast<std::uint32_t>(status.code);
  }
  *resolved_path = resolved_storage.c_str();
  *error_message = "ok";
  return static_cast<std::uint32_t>(
      vlaforge::runtime::StatusCode::kOk);
}
