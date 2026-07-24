#include "vlaforge/runtime/action_queue.h"

namespace vlaforge::runtime {

Status ActionQueue::Publish(const CommittedAction& action,
                            std::uint32_t task_id) noexcept {
  if (!action.valid ||
      (action.size_bytes != 0 && action.data == nullptr)) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "only a committed action may be published");
  }
  if (publish_ != nullptr) {
    const Status status = publish_(context_, &action);
    if (!status.ok()) {
      return status;
    }
  }
  latest_ = action;
  ++publish_count_;
  EmitTrace(trace_, TraceEvent{TraceKind::kActionPublish, task_id, 0, 0,
                               action.transaction_id, action.epoch});
  return Status::Ok();
}

}  // namespace vlaforge::runtime
