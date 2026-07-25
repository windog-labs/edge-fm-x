#include "vlaforge/runtime/transaction.h"

namespace vlaforge::runtime {

Transaction::Transaction(std::size_t maximum_staged_states)
    : pending_(maximum_staged_states) {}

Status Transaction::Begin(std::uint64_t id,
                          std::uint64_t episode) noexcept {
  if (active()) {
    return Status::Error(StatusCode::kFailedPrecondition, 0,
                         "transaction is already active");
  }
  id_ = id;
  episode_ = episode;
  state_ = TransactionState::kActive;
  pending_count_ = 0;
  return Status::Ok();
}

Status Transaction::Add(const PendingWrite& write) noexcept {
  if (!active()) {
    return Status::Error(StatusCode::kFailedPrecondition, write.state_id,
                         "transaction is not active");
  }
  for (std::size_t index = 0; index < pending_count_; ++index) {
    if (pending_[index].state_id == write.state_id) {
      return Status::Error(StatusCode::kAlreadyExists, write.state_id,
                           "state is already staged");
    }
  }
  if (pending_count_ == pending_.size()) {
    return Status::Error(StatusCode::kResourceExhausted, write.state_id,
                         "transaction staging capacity exhausted");
  }
  pending_[pending_count_++] = write;
  return Status::Ok();
}

void Transaction::Close(TransactionState state) noexcept {
  state_ = state;
  pending_count_ = 0;
}

}  // namespace vlaforge::runtime
