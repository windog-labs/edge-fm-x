#include "vlaforge/runtime/state_store.h"

#include <cstring>
#include <limits>

namespace vlaforge::runtime {
namespace {

constexpr std::uint64_t kNoTransaction =
    std::numeric_limits<std::uint64_t>::max();

bool IsPowerOfTwo(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

StateStore::StateStore(StaticArena& arena,
                       const StateSlotDescriptor* descriptors,
                       std::size_t state_count, TraceSink trace)
    : arena_(arena),
      descriptors_(state_count),
      metadata_offsets_(state_count),
      staging_offsets_(state_count),
      next_versions_(state_count),
      trace_(trace),
      initialization_status_(Status::Ok()) {
  if (state_count != 0 && descriptors == nullptr) {
    initialization_status_ =
        Status::Error(StatusCode::kInvalidArgument, 0,
                      "state descriptor pointer is null");
    return;
  }
  for (std::size_t index = 0; index < state_count; ++index) {
    descriptors_[index] = descriptors[index];
  }
  std::size_t metadata_count = 0;
  std::size_t staging_size = 0;
  for (std::size_t index = 0; index < state_count; ++index) {
    const StateSlotDescriptor& descriptor = descriptors_[index];
    if (descriptor.state_id != index || descriptor.capacity == 0 ||
        descriptor.slot_size == 0 ||
        !IsPowerOfTwo(descriptor.alignment) ||
        descriptor.offset % descriptor.alignment != 0) {
      initialization_status_ =
          Status::Error(StatusCode::kInvalidArgument,
                        descriptor.state_id,
                        "invalid state slot descriptor");
      return;
    }
    const std::size_t bytes =
        descriptor.slot_size * descriptor.capacity;
    if (bytes / descriptor.capacity != descriptor.slot_size ||
        arena_.Resolve(descriptor.offset, bytes, descriptor.alignment) ==
            nullptr) {
      initialization_status_ =
          Status::Error(StatusCode::kOutOfRange, descriptor.state_id,
                        "state slot descriptor exceeds arena");
      return;
    }
    metadata_offsets_[index] = metadata_count;
    metadata_count += descriptor.capacity;
    staging_offsets_[index] = staging_size;
    staging_size += descriptor.slot_size;
  }
  metadata_.resize(metadata_count);
  staging_.resize(staging_size);
}

Status StateStore::Initialize(std::uint32_t state_id, const Epoch& epoch,
                              const void* data,
                              std::size_t size_bytes) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  if (active_transaction_id_ != kNoTransaction) {
    return Status::Error(StatusCode::kFailedPrecondition, state_id,
                         "cannot initialize during a transaction");
  }
  const StateSlotDescriptor* descriptor = Descriptor(state_id);
  if (descriptor == nullptr || (size_bytes != 0 && data == nullptr) ||
      size_bytes > descriptor->slot_size) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid state initializer");
  }
  void* destination = SlotData(*descriptor, 0);
  if (destination == nullptr) {
    return Status::Error(StatusCode::kInternal, state_id,
                         "state slot resolution failed");
  }
  std::memset(destination, 0, descriptor->slot_size);
  if (size_bytes != 0) {
    std::memcpy(destination, data, size_bytes);
  }
  SlotMetadata& metadata = metadata_[MetadataIndex(state_id, 0)];
  metadata.valid = true;
  metadata.logical_version = 0;
  metadata.epoch = epoch;
  next_versions_[state_id] = 1;
  episode_ = epoch.episode;
  return Status::Ok();
}

Status StateStore::Begin(Transaction* transaction, const Epoch& tick,
                         std::uint32_t task_id) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  if (transaction == nullptr ||
      transaction->capacity() < descriptors_.size()) {
    return Status::Error(StatusCode::kInvalidArgument, task_id,
                         "invalid transaction storage");
  }
  if (active_transaction_id_ != kNoTransaction) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "another transaction is active");
  }
  if (tick.episode != episode_) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "tick episode does not match state store");
  }
  const Status status = transaction->Begin(next_transaction_id_++, tick);
  if (!status.ok()) {
    return status;
  }
  active_transaction_id_ = transaction->id();
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionBegin, task_id, 0, 0,
                               transaction->id(), tick});
  return Status::Ok();
}

Status StateStore::ReadLatest(std::uint32_t state_id,
                              std::uint64_t episode,
                              std::uint64_t maximum_sequence,
                              bool exact_sequence,
                              std::uint32_t task_id,
                              StateSnapshot* output) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  const StateSlotDescriptor* descriptor = Descriptor(state_id);
  if (descriptor == nullptr || output == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid state read");
  }
  const SlotMetadata* selected = nullptr;
  std::uint32_t selected_slot = 0;
  for (std::uint32_t slot = 0; slot < descriptor->capacity; ++slot) {
    const SlotMetadata& candidate =
        metadata_[MetadataIndex(state_id, slot)];
    if (!candidate.valid || candidate.epoch.episode != episode ||
        (exact_sequence
             ? candidate.epoch.sequence != maximum_sequence
             : candidate.epoch.sequence > maximum_sequence)) {
      continue;
    }
    if (selected == nullptr ||
        candidate.epoch.sequence > selected->epoch.sequence ||
        (candidate.epoch.sequence == selected->epoch.sequence &&
         candidate.logical_version > selected->logical_version)) {
      selected = &candidate;
      selected_slot = slot;
    }
  }
  if (selected == nullptr) {
    return Status::Error(StatusCode::kNotFound, state_id,
                         "no matching committed state version");
  }
  output->state_id = state_id;
  output->physical_slot = selected_slot;
  output->logical_version = selected->logical_version;
  output->epoch = selected->epoch;
  output->data = SlotData(*descriptor, selected_slot);
  output->size_bytes = descriptor->slot_size;
  EmitTrace(trace_, TraceEvent{TraceKind::kStateRead, task_id, state_id,
                               selected->logical_version, 0,
                               selected->epoch});
  return Status::Ok();
}

Status StateStore::Stage(Transaction* transaction, std::uint32_t state_id,
                         const Epoch& epoch, const void* data,
                         std::size_t size_bytes,
                         std::uint32_t task_id) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  const StateSlotDescriptor* descriptor = Descriptor(state_id);
  if (transaction == nullptr || !IsActive(*transaction) ||
      descriptor == nullptr || (size_bytes != 0 && data == nullptr) ||
      size_bytes > descriptor->slot_size ||
      epoch.episode != episode_) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid staged state write");
  }
  std::byte* staging = StagingData(state_id);
  std::memset(staging, 0, descriptor->slot_size);
  if (size_bytes != 0) {
    std::memcpy(staging, data, size_bytes);
  }
  const Status status = transaction->Add(
      PendingWrite{state_id, epoch, staging, descriptor->slot_size});
  if (!status.ok()) {
    return status;
  }
  EmitTrace(trace_, TraceEvent{TraceKind::kStateStage, task_id, state_id, 0,
                               transaction->id(), epoch});
  return Status::Ok();
}

Status StateStore::Commit(Transaction* transaction,
                          const PendingAction& action,
                          bool validation_passed,
                          std::uint32_t task_id,
                          CommittedAction* output) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  if (transaction == nullptr || output == nullptr ||
      !IsActive(*transaction)) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "transaction cannot be committed");
  }
  if (!validation_passed) {
    const std::uint64_t transaction_id = transaction->id();
    const Epoch tick = transaction->tick();
    transaction->Close(TransactionState::kAborted);
    active_transaction_id_ = kNoTransaction;
    EmitTrace(trace_, TraceEvent{TraceKind::kTransactionAbort, task_id, 0, 0,
                                 transaction_id, tick});
    return Status::Error(StatusCode::kValidationFailed, task_id,
                         "action validation failed");
  }
  if ((action.size_bytes != 0 && action.data == nullptr) ||
      action.epoch.episode != transaction->tick().episode) {
    return Status::Error(StatusCode::kInvalidArgument, task_id,
                         "invalid pending action");
  }
  for (std::size_t index = 0; index < transaction->pending_count();
       ++index) {
    const PendingWrite& pending = transaction->pending(index);
    const StateSlotDescriptor& descriptor =
        descriptors_[pending.state_id];
    const std::uint64_t logical_version =
        next_versions_[pending.state_id]++;
    const std::uint32_t slot = static_cast<std::uint32_t>(
        logical_version % descriptor.capacity);
    void* destination = SlotData(descriptor, slot);
    if (destination == nullptr) {
      return Status::Error(StatusCode::kInternal, pending.state_id,
                           "state slot resolution failed");
    }
    std::memcpy(destination, pending.data, descriptor.slot_size);
    SlotMetadata& metadata =
        metadata_[MetadataIndex(pending.state_id, slot)];
    metadata.valid = true;
    metadata.logical_version = logical_version;
    metadata.epoch = pending.epoch;
    EmitTrace(trace_, TraceEvent{TraceKind::kStateCommit, task_id,
                                 pending.state_id, logical_version,
                                 transaction->id(), pending.epoch});
  }
  const std::uint64_t transaction_id = transaction->id();
  const Epoch tick = transaction->tick();
  transaction->Close(TransactionState::kCommitted);
  active_transaction_id_ = kNoTransaction;
  *output = CommittedAction{action.epoch, action.data, action.size_bytes,
                            transaction_id, true};
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionCommit, task_id, 0, 0,
                               transaction_id, tick});
  EmitTrace(trace_, TraceEvent{TraceKind::kActionCommit, task_id, 0, 0,
                               transaction_id, action.epoch});
  return Status::Ok();
}

Status StateStore::Abort(Transaction* transaction,
                         std::uint32_t task_id) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  if (transaction == nullptr || !IsActive(*transaction)) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "transaction cannot be aborted");
  }
  const std::uint64_t transaction_id = transaction->id();
  const Epoch tick = transaction->tick();
  transaction->Close(TransactionState::kAborted);
  active_transaction_id_ = kNoTransaction;
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionAbort, task_id, 0, 0,
                               transaction_id, tick});
  return Status::Ok();
}

Status StateStore::ResetEpisode(std::uint64_t new_episode,
                                std::uint32_t task_id) noexcept {
  if (!Ready()) {
    return initialization_status_;
  }
  if (active_transaction_id_ != kNoTransaction ||
      new_episode <= episode_) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "invalid episode reset");
  }
  for (std::size_t state = 0; state < descriptors_.size(); ++state) {
    if (!descriptors_[state].reset_on_episode) {
      continue;
    }
    for (std::uint32_t slot = 0;
         slot < descriptors_[state].capacity; ++slot) {
      metadata_[MetadataIndex(static_cast<std::uint32_t>(state), slot)] =
          SlotMetadata{};
    }
    next_versions_[state] = 0;
  }
  episode_ = new_episode;
  EmitTrace(trace_, TraceEvent{TraceKind::kReset, task_id, 0, 0, 0,
                               Epoch{0, 0, 0, new_episode}});
  return Status::Ok();
}

const StateSlotDescriptor* StateStore::Descriptor(
    std::uint32_t state_id) const noexcept {
  return state_id < descriptors_.size() ? &descriptors_[state_id] : nullptr;
}

std::size_t StateStore::MetadataIndex(std::uint32_t state_id,
                                      std::uint32_t slot) const noexcept {
  return metadata_offsets_[state_id] + slot;
}

std::byte* StateStore::StagingData(std::uint32_t state_id) noexcept {
  return staging_.data() + staging_offsets_[state_id];
}

void* StateStore::SlotData(const StateSlotDescriptor& descriptor,
                           std::uint32_t slot) noexcept {
  return arena_.Resolve(
      descriptor.offset + descriptor.slot_size * slot,
      descriptor.slot_size, descriptor.alignment);
}

bool StateStore::IsActive(const Transaction& transaction) const noexcept {
  return transaction.active() &&
         transaction.id() == active_transaction_id_;
}

}  // namespace vlaforge::runtime
