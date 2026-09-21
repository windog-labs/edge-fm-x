#ifndef VLAFORGE_BACKENDS_AOTI_MATERIALIZED_PACKAGE_H_
#define VLAFORGE_BACKENDS_AOTI_MATERIALIZED_PACKAGE_H_

#include <cstdint>
#include <string>

namespace vlaforge::backends {

// Files belong to the deployment bundle, never to this runner's destructor.
class AotiMaterializedPackage final {
 public:
  AotiMaterializedPackage(const std::string& manifest, const std::string& sha256,
                          std::uint64_t size_bytes, const std::string& device_type);
  const std::string& library() const noexcept { return library_; }
  const std::string& weight_blob() const noexcept { return weight_blob_; }

 private:
  std::string library_;
  std::string weight_blob_;
};

}  // namespace vlaforge::backends
#endif
