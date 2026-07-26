#ifndef VLAFORGE_BACKENDS_AOTI_SEQUENCE_RUNNER_H_
#define VLAFORGE_BACKENDS_AOTI_SEQUENCE_RUNNER_H_

#include <ATen/Tensor.h>

#include <memory>
#include <string>
#include <vector>

#include "vlaforge/runtime/region_executable.h"

namespace vlaforge::backends {

// Session-resident executor for the bounded vlaforge.aoti_sequence/1
// backend-artifact format.
class AotiSequenceRunner final {
 public:
  AotiSequenceRunner(VLAForgeDeviceKind device_kind, int device_ordinal);
  ~AotiSequenceRunner();

  AotiSequenceRunner(const AotiSequenceRunner&) = delete;
  AotiSequenceRunner& operator=(const AotiSequenceRunner&) = delete;
  AotiSequenceRunner(AotiSequenceRunner&&) = delete;
  AotiSequenceRunner& operator=(AotiSequenceRunner&&) = delete;

  void Load(const std::string& manifest_path, const std::string& target);
  [[nodiscard]] bool loaded() const noexcept;
  std::vector<at::Tensor> Run(std::vector<at::Tensor>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vlaforge::backends

#endif  // VLAFORGE_BACKENDS_AOTI_SEQUENCE_RUNNER_H_
