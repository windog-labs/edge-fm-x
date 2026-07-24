#ifndef VLAFORGE_RUNTIME_ACTION_QUEUE_H_
#define VLAFORGE_RUNTIME_ACTION_QUEUE_H_

#include <cstdint>

#include "vlaforge/runtime/status.h"
#include "vlaforge/runtime/trace.h"
#include "vlaforge/runtime/transaction.h"

namespace vlaforge::runtime {

using ActionPublishFn = Status (*)(void* context,
                                   const CommittedAction* action);

class ActionQueue final {
 public:
  ActionQueue(ActionPublishFn publish = nullptr, void* context = nullptr,
              TraceSink trace = {}) noexcept
      : publish_(publish), context_(context), trace_(trace) {}

  void SetTraceSink(TraceSink trace) noexcept { trace_ = trace; }

  Status Publish(const CommittedAction& action,
                 std::uint32_t task_id) noexcept;
  void Reset() noexcept {
    latest_ = CommittedAction{};
    publish_count_ = 0;
  }

  [[nodiscard]] const CommittedAction& latest() const noexcept {
    return latest_;
  }
  [[nodiscard]] std::uint64_t publish_count() const noexcept {
    return publish_count_;
  }

 private:
  ActionPublishFn publish_ = nullptr;
  void* context_ = nullptr;
  TraceSink trace_{};
  CommittedAction latest_{};
  std::uint64_t publish_count_ = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_ACTION_QUEUE_H_
