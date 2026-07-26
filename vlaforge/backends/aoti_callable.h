#ifndef VLAFORGE_BACKENDS_AOTI_CALLABLE_H_
#define VLAFORGE_BACKENDS_AOTI_CALLABLE_H_

#include <ATen/Tensor.h>

#include <memory>
#include <string>
#include <vector>

#include "vlaforge/runtime/region_executable.h"

namespace vlaforge::backends {

// Version-adaptive owner for one LibTorch AOTInductor package or raw shared
// library. This class deliberately hides unstable LibTorch runner types from
// the stable RegionExecutable ABI.
class AotiCallable final {
 public:
  AotiCallable(VLAForgeDeviceKind device_kind, int device_ordinal);
  ~AotiCallable();

  AotiCallable(const AotiCallable&) = delete;
  AotiCallable& operator=(const AotiCallable&) = delete;
  AotiCallable(AotiCallable&&) = delete;
  AotiCallable& operator=(AotiCallable&&) = delete;

  void Load(const std::string& path);
  [[nodiscard]] bool loaded() const noexcept;
  std::vector<at::Tensor> Run(std::vector<at::Tensor>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vlaforge::backends

#endif  // VLAFORGE_BACKENDS_AOTI_CALLABLE_H_
