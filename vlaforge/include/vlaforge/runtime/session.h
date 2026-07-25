#ifndef VLAFORGE_RUNTIME_SESSION_H_
#define VLAFORGE_RUNTIME_SESSION_H_

#include <cstdint>

#include "vlaforge/runtime/session_c.h"
#include "vlaforge/runtime/status.h"
#include "vlaforge/runtime/trace.h"

namespace vlaforge::runtime {

class Session {
 public:
  virtual ~Session() = default;

  virtual Status ResetEpisode(std::uint64_t new_episode) noexcept = 0;
  virtual Status BindTensor(
      std::uint32_t input_id, const VLAForgeBoundTensor& input,
      const VLAForgeInputStamp* stamp) noexcept = 0;
  virtual Status BindScalar(
      std::uint32_t input_id, const VLAForgeScalarValue& input,
      const VLAForgeInputStamp* stamp) noexcept = 0;
  virtual Status Run() noexcept = 0;
  virtual Status ReadOutputTensor(
      std::uint32_t output_id, VLAForgeBoundTensor* output) const noexcept = 0;
  virtual Status ReadOutputScalar(
      std::uint32_t output_id, VLAForgeScalarValue* output) const noexcept = 0;
  virtual const char* SchemaDigest() const noexcept = 0;
  virtual void SetTraceSink(TraceSink trace) noexcept = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_SESSION_H_
