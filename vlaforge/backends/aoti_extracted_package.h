#ifndef VLAFORGE_BACKENDS_AOTI_EXTRACTED_PACKAGE_H_
#define VLAFORGE_BACKENDS_AOTI_EXTRACTED_PACKAGE_H_

#include <cstdint>
#include <string>

namespace vlaforge::backends {

std::string DefaultAotiPackageExtractionRoot();

// Declare this owner before runners: their proxy/constants may refer to files
// here until runner destruction. No shared cache and no process-global policy.
class AotiExtractedPackage final {
 public:
  AotiExtractedPackage(const std::string& package, const std::string& root,
                       const std::string& sha256, std::uint64_t size_bytes);
  ~AotiExtractedPackage();
  AotiExtractedPackage(const AotiExtractedPackage&) = delete;
  AotiExtractedPackage& operator=(const AotiExtractedPackage&) = delete;
  const std::string& directory() const noexcept { return directory_; }
  const std::string& library() const noexcept { return library_; }
  const std::string& weight_blob() const noexcept { return weight_blob_; }

 private:
  std::string directory_;
  std::string library_;
  std::string weight_blob_;
};

}  // namespace vlaforge::backends
#endif
