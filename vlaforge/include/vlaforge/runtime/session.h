#ifndef VLAFORGE_RUNTIME_SESSION_H_
#define VLAFORGE_RUNTIME_SESSION_H_

#include <cstdint>

#include "vlaforge/runtime/epoch.h"
#include "vlaforge/runtime/status.h"
#include "vlaforge/runtime/tensor_view.h"
#include "vlaforge/runtime/trace.h"
#include "vlaforge/runtime/transaction.h"

namespace vlaforge::runtime {

class Session {
 public:
  virtual ~Session() = default;

  virtual Status ResetEpisode(std::uint64_t new_episode) noexcept = 0;
  virtual Status BindInput(std::uint32_t input_id,
                           const TensorView& input) noexcept = 0;
  virtual Status RunTick(const Epoch& tick) noexcept = 0;
  virtual Status ReadCommittedAction(
      CommittedAction* action) const noexcept = 0;
  virtual void SetTraceSink(TraceSink trace) noexcept = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_SESSION_H_
