#include "aoti_extracted_package.h"
#include "aoti_package_config.h"

#include <torch/version.h>

#include <filesystem>
#include <cstdio>
#include <stdexcept>

#if defined(__linux__) && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
#include <caffe2/serialize/inline_container.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>
#endif

namespace vlaforge::backends {
namespace fs = std::filesystem;

std::string DefaultAotiPackageExtractionRoot() {
  return VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT;
}

#if defined(__linux__) && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
namespace {
constexpr std::size_t kChunkBytes = 1024u * 1024u;
constexpr std::size_t kMaximumRecords = 100000u;
constexpr std::string_view kModel = "data/aotinductor/model/";
constexpr std::string_view kConstants = "data/constants/";

bool StartsWith(const std::string& value, std::string_view prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void ValidateName(const std::string& name) {
  const fs::path path(name);
  Require(!name.empty() && name.size() <= 4096u && !path.is_absolute() &&
              path.generic_string() == name && path.lexically_normal() == path &&
              name.back() != '/', "unsafe AOTI package record path");
  for (const auto& component : path) {
    Require(component != "." && component != "..",
            "unsafe AOTI package record component");
  }
  for (const unsigned char ch : name) {
    Require(ch >= 32u && ch != 127u && ch != '\\' && ch != ':',
            "unsafe AOTI package record character");
  }
}

std::string HashStream(std::ifstream& stream, std::uint64_t expected_size) {
  using Digest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  Digest digest(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  Require(digest && EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) == 1,
          "AOTI SHA initialization failed");
  stream.clear();
  stream.seekg(0);
  Require(static_cast<bool>(stream), "AOTI package cannot seek");
  std::vector<char> buffer(kChunkBytes);
  std::uint64_t total = 0;
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = static_cast<std::size_t>(stream.gcount());
    Require(count <= expected_size - total, "AOTI package size mismatch");
    total += count;
    Require(EVP_DigestUpdate(digest.get(), buffer.data(), count) == 1,
            "AOTI SHA update failed");
  }
  Require(stream.eof() && total == expected_size, "AOTI package size mismatch");
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int length = 0;
  Require(EVP_DigestFinal_ex(digest.get(), bytes.data(), &length) == 1 && length == 32u,
          "AOTI SHA finalization failed");
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < length; ++i) result << std::setw(2) << unsigned(bytes[i]);
  stream.clear();
  stream.seekg(0);
  return result.str();
}

std::string SmallRecord(caffe2::serialize::PyTorchStreamReader& reader,
                        const std::string& name) {
  Require(reader.hasRecord(name) && reader.getRecordSize(name) <= 128u,
          "unsupported AOTI package format marker");
  auto record = reader.getRecord(name);
  return {static_cast<const char*>(std::get<0>(record).get()), std::get<1>(record)};
}

bool ClearDirectory(int descriptor) noexcept {
  const int duplicate = dup(descriptor);
  if (duplicate < 0) return false;
  DIR* entries = fdopendir(duplicate);
  if (!entries) { close(duplicate); return false; }
  bool success = true;
  while (const auto* entry = readdir(entries)) {
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..") continue;
    struct stat info {};
    if (fstatat(descriptor, entry->d_name, &info, AT_SYMLINK_NOFOLLOW) != 0) {
      success = false; continue;
    }
    if (S_ISDIR(info.st_mode)) {
      const int child = openat(descriptor, entry->d_name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (child < 0) { success = false; continue; }
      if (!ClearDirectory(child)) success = false;
      close(child);
      if (unlinkat(descriptor, entry->d_name, AT_REMOVEDIR) != 0) success = false;
    } else if (unlinkat(descriptor, entry->d_name, 0) != 0) success = false;
  }
  closedir(entries);
  return success;
}

void RemoveOwnedDirectory(const std::string& path) noexcept {
  if (path.empty()) return;
  // Use fd-relative POSIX operations: never follow a link while cleaning an
  // owned child, and avoid SDK interposition of std::filesystem::remove_all.
  const int descriptor = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  bool success = descriptor >= 0;
  if (success) {
    success = ClearDirectory(descriptor);
    close(descriptor);
    if (rmdir(path.c_str()) != 0) success = false;
  }
  if (!success) std::fprintf(stderr, "AOTI owned extraction cleanup failed: %s\n", path.c_str());
}

struct DirectoryGuard {
  std::string path;
  ~DirectoryGuard() {
    RemoveOwnedDirectory(path);
  }
};
}  // namespace
#endif

AotiExtractedPackage::AotiExtractedPackage(
    const std::string& package, const std::string& root,
    const std::string& sha256, std::uint64_t size_bytes) {
#if defined(__linux__) && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
  Require(sha256.size() == 64u && size_bytes > 0u, "AOTI extraction requires SHA-256 and size");
  for (const char ch : sha256) {
    Require((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'),
            "AOTI extraction requires lowercase SHA-256");
  }
  const fs::path parent(root);
  struct stat info {};
  Require(!root.empty() && parent.is_absolute() && parent.lexically_normal() == parent &&
              parent.string() == root && fs::canonical(parent) == parent &&
              lstat(root.c_str(), &info) == 0 && S_ISDIR(info.st_mode) &&
              info.st_uid == geteuid() && (info.st_mode & 0777) == 0700,
          "AOTI extraction root must be an existing canonical private owned directory (0700)");
  std::ifstream stream(package, std::ios::binary);
  Require(static_cast<bool>(stream), "AOTI package cannot be opened");
  Require(HashStream(stream, size_bytes) == sha256, "AOTI package SHA-256 mismatch");
  caffe2::serialize::PyTorchStreamReader reader(&stream);
  Require(SmallRecord(reader, "archive_format") == "pt2" &&
              SmallRecord(reader, "archive_version") == "0",
          "unsupported AOTI package format/version");
  const auto records = reader.getAllRecords();
  Require(!records.empty() && records.size() <= kMaximumRecords,
          "AOTI package record count exceeds limit");
  std::set<std::string> names;
  std::map<std::string, std::string> selected;
  std::uint64_t extracted_bytes = 0;
  std::string library_name;
  std::string blob_name;
  for (const auto& name : records) {
    ValidateName(name);
    Require(names.insert(name).second, "duplicate AOTI package record");
    std::string output;
    if (StartsWith(name, kModel)) output = name.substr(kModel.size());
    else if (StartsWith(name, kConstants)) output = fs::path(name).filename().string();
    else continue;
    ValidateName(output);
    Require(selected.emplace(output, name).second, "AOTI package flattened record collision");
    const auto bytes = reader.getRecordSize(name);
    Require(bytes <= std::numeric_limits<std::uint64_t>::max() - extracted_bytes,
            "AOTI extraction size overflow");
    extracted_bytes += bytes;
    if (fs::path(output).extension() == ".so") {
      Require(library_name.empty() && fs::path(output).parent_path().empty(),
              "AOTI package requires exactly one top-level compiled shared library");
      library_name = output;
    }
    if (fs::path(output).extension() == ".blob") {
      Require(blob_name.empty(), "AOTI package contains multiple weight blobs");
      blob_name = output;
    }
  }
  Require(!library_name.empty(), "AOTI explicit extraction does not compile source-only packages");
  const auto stem = fs::path(library_name).stem().string();
  Require(selected.count(stem + "_metadata.json") == 1u,
          "AOTI package is missing shared-library metadata");
  Require(fs::space(parent).available >= extracted_bytes, "insufficient AOTI extraction space");
  // All members are materialized as regular bytes. Archive permission/link
  // metadata is never interpreted, so a ZIP link cannot redirect extraction.
  std::string unique_path = (parent / "vlaforge-aoti-XXXXXX").string();
  Require(mkdtemp(unique_path.data()) != nullptr, "AOTI private extraction directory creation failed");
  DirectoryGuard guard{std::move(unique_path)};
  std::vector<char> buffer(kChunkBytes);
  for (const auto& [relative, record] : selected) {
    const auto destination = fs::path(guard.path) / relative;
    fs::create_directories(destination.parent_path());
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    Require(static_cast<bool>(output), "AOTI extracted file cannot be opened");
    const auto size = reader.getRecordSize(record);
    auto chunks = reader.createChunkReaderIter(record, size, buffer.size());
    std::uint64_t total = 0;
    while (const auto count = chunks.next(buffer.data())) {
      output.write(buffer.data(), static_cast<std::streamsize>(count));
      Require(static_cast<bool>(output), "AOTI extracted file write failed");
      total += count;
    }
    output.close();
    Require(output && total == size, "AOTI extracted file size/write mismatch");
  }
  // Re-read the same open package inode before any dlopen. No multi-GB package
  // copy or in-memory weights buffer is introduced by extraction.
  Require(HashStream(stream, size_bytes) == sha256,
          "AOTI package changed during extraction");
  directory_ = guard.path;
  library_ = (fs::path(directory_) / library_name).string();
  if (!blob_name.empty()) weight_blob_ = (fs::path(directory_) / blob_name).string();
  guard.path.clear();
#else
  (void)package; (void)root; (void)sha256; (void)size_bytes;
  throw std::runtime_error("explicit AOTI extraction requires audited Linux LibTorch 2.10");
#endif
}

AotiExtractedPackage::~AotiExtractedPackage() {
#if defined(__linux__) && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
  RemoveOwnedDirectory(directory_);
#endif
}
}  // namespace vlaforge::backends
