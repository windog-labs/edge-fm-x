#include "vlaforge/runtime/state_store.h"

#include <algorithm>
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
    : arena_(arena), trace_(trace) {
  if (state_count != 0 && descriptors == nullptr) {
    initialization_status_ =
        Status::Error(StatusCode::kInvalidArgument, 0,
                      "state descriptors are null");
    return;
  }
  if (state_count != 0) {
    descriptors_.assign(descriptors, descriptors + state_count);
  }
  metadata_offsets_.resize(state_count);
  staging_offsets_.resize(state_count);
  initialized_.assign(state_count, false);
  next_versions_.assign(state_count, 0);
  std::size_t metadata_count = 0;
  std::size_t staging_size = 0;
  for (std::size_t index = 0; index < state_count; ++index) {
    const auto& descriptor = descriptors_[index];
    if (descriptor.state_id != index || descriptor.capacity == 0 ||
        descriptor.value_size == 0 ||
        descriptor.slot_stride < descriptor.value_size ||
        !IsPowerOfTwo(descriptor.alignment)) {
      initialization_status_ =
          Status::Error(StatusCode::kInvalidArgument,
                        static_cast<std::uint32_t>(index),
                        "invalid state descriptor");
      return;
    }
    const std::size_t total =
        descriptor.slot_stride * descriptor.capacity;
    if (arena_.Resolve(descriptor.offset, total, descriptor.alignment) ==
        nullptr) {
      initialization_status_ =
          Status::Error(StatusCode::kOutOfRange, descriptor.state_id,
                        "state ring is outside the static arena");
      return;
    }
    metadata_offsets_[index] = metadata_count;
    metadata_count += descriptor.capacity;
    staging_offsets_[index] = staging_size;
    staging_size += descriptor.value_size;
  }
  metadata_.resize(metadata_count);
  staging_.resize(staging_size);
  initial_.resize(staging_size);
  initialization_status_ = Status::Ok();
}

Status StateStore::Initialize(std::uint32_t state_id, const void* data,
                              std::size_t size_bytes) noexcept {
  const auto* descriptor = Descriptor(state_id);
  if (descriptor == nullptr || data == nullptr ||
      size_bytes != descriptor->value_size) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid state initializer");
  }
  if (initialized_[state_id]) {
    return Status::Error(StatusCode::kAlreadyExists, state_id,
                         "state is already initialized");
  }
  std::memcpy(InitialData(state_id), data, size_bytes);
  std::memcpy(SlotData(*descriptor, 0), data, size_bytes);
  auto& metadata = metadata_[MetadataIndex(state_id, 0)];
  metadata = SlotMetadata{true, 0, episode_};
  initialized_[state_id] = true;
  next_versions_[state_id] = 1;
  return Status::Ok();
}

Status StateStore::Begin(Transaction* transaction,
                         std::uint32_t task_id) noexcept {
  if (transaction == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument, task_id,
                         "transaction is null");
  }
  if (active_transaction_id_ != kNoTransaction) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "another transaction is active");
  }
  const Status status =
      transaction->Begin(next_transaction_id_++, episode_);
  if (!status.ok()) {
    return status;
  }
  active_transaction_id_ = transaction->id();
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionBegin, task_id, 0, 0,
                               transaction->id(), episode_, run_, 0});
  return Status::Ok();
}

Status StateStore::ReadLatest(std::uint32_t state_id,
                              std::uint32_t task_id,
                              StateSnapshot* output) noexcept {
  const auto* descriptor = Descriptor(state_id);
  if (descriptor == nullptr || output == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid state read");
  }
  const SlotMetadata* selected = nullptr;
  std::uint32_t selected_slot = 0;
  for (std::uint32_t slot = 0; slot < descriptor->capacity; ++slot) {
    const auto& metadata = metadata_[MetadataIndex(state_id, slot)];
    if (metadata.valid && metadata.episode == episode_ &&
        (selected == nullptr ||
         metadata.logical_version > selected->logical_version)) {
      selected = &metadata;
      selected_slot = slot;
    }
  }
  if (selected == nullptr) {
    return Status::Error(StatusCode::kNotFound, state_id,
                         "state has no version in current episode");
  }
  *output = StateSnapshot{state_id,
                          selected_slot,
                          selected->logical_version,
                          selected->episode,
                          SlotData(*descriptor, selected_slot),
                          descriptor->value_size};
  EmitTrace(trace_, TraceEvent{TraceKind::kStateRead, task_id, state_id,
                               selected->logical_version,
                               active_transaction_id_ == kNoTransaction
                                   ? 0
                                   : active_transaction_id_,
                               episode_, run_, 0});
  return Status::Ok();
}

Status StateStore::Stage(Transaction* transaction, std::uint32_t state_id,
                         const void* data, std::size_t size_bytes,
                         std::uint32_t task_id) noexcept {
  const auto* descriptor = Descriptor(state_id);
  if (transaction == nullptr || !IsActive(*transaction) ||
      descriptor == nullptr || data == nullptr ||
      size_bytes != descriptor->value_size) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid staged state write");
  }
  std::byte* staging = StagingData(state_id);
  std::memcpy(staging, data, size_bytes);
  const Status status =
      transaction->Add(PendingWrite{state_id, staging, size_bytes});
  if (status.ok()) {
    EmitTrace(trace_, TraceEvent{TraceKind::kStateStage, task_id, state_id, 0,
                                 transaction->id(), episode_, run_, 0});
  }
  return status;
}

Status StateStore::Commit(Transaction* transaction,
                          std::uint32_t task_id) noexcept {
  if (transaction == nullptr || !IsActive(*transaction)) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "transaction is not active");
  }
  for (std::size_t index = 0; index < transaction->pending_count(); ++index) {
    const auto& pending = transaction->pending(index);
    const auto* descriptor = Descriptor(pending.state_id);
    if (descriptor == nullptr) {
      return Status::Error(StatusCode::kInternal, pending.state_id,
                           "staged state descriptor disappeared");
    }
    const std::uint64_t version = next_versions_[pending.state_id]++;
    const auto slot =
        static_cast<std::uint32_t>(version % descriptor->capacity);
    std::memcpy(SlotData(*descriptor, slot), pending.data,
                pending.size_bytes);
    metadata_[MetadataIndex(pending.state_id, slot)] =
        SlotMetadata{true, version, episode_};
    EmitTrace(trace_, TraceEvent{TraceKind::kStateCommit, task_id,
                                 pending.state_id, version,
                                 transaction->id(), episode_, run_, 0});
  }
  const std::uint64_t transaction_id = transaction->id();
  transaction->Close(TransactionState::kCommitted);
  active_transaction_id_ = kNoTransaction;
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionCommit, task_id, 0, 0,
                               transaction_id, episode_, run_, 0});
  return Status::Ok();
}

Status StateStore::Abort(Transaction* transaction,
                         std::uint32_t task_id) noexcept {
  if (transaction == nullptr || !IsActive(*transaction)) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "transaction is not active");
  }
  const std::uint64_t transaction_id = transaction->id();
  transaction->Close(TransactionState::kAborted);
  active_transaction_id_ = kNoTransaction;
  EmitTrace(trace_, TraceEvent{TraceKind::kTransactionAbort, task_id, 0, 0,
                               transaction_id, episode_, run_, 0});
  return Status::Ok();
}

Status StateStore::ResetEpisode(std::uint64_t new_episode,
                                std::uint32_t task_id) noexcept {
  if (active_transaction_id_ != kNoTransaction ||
      new_episode <= episode_) {
    return Status::Error(StatusCode::kFailedPrecondition, task_id,
                         "invalid episode reset");
  }
  for (const auto& descriptor : descriptors_) {
    StateSnapshot latest{};
    const Status latest_status =
        ReadLatest(descriptor.state_id, task_id, &latest);
    std::vector<std::byte> carried;
    std::uint64_t carried_version = 0;
    if (!descriptor.reset_on_episode && latest_status.ok()) {
      const auto* begin = static_cast<const std::byte*>(latest.data);
      carried.assign(begin, begin + latest.size_bytes);
      carried_version = latest.logical_version;
    }
    for (std::uint32_t slot = 0; slot < descriptor.capacity; ++slot) {
      metadata_[MetadataIndex(descriptor.state_id, slot)].valid = false;
    }
    if (descriptor.reset_on_episode && initialized_[descriptor.state_id]) {
      std::memcpy(SlotData(descriptor, 0),
                  InitialData(descriptor.state_id), descriptor.value_size);
      metadata_[MetadataIndex(descriptor.state_id, 0)] =
          SlotMetadata{true, 0, new_episode};
      next_versions_[descriptor.state_id] = 1;
    } else if (!carried.empty()) {
      const auto slot = static_cast<std::uint32_t>(
          carried_version % descriptor.capacity);
      std::memcpy(SlotData(descriptor, slot), carried.data(),
                  descriptor.value_size);
      metadata_[MetadataIndex(descriptor.state_id, slot)] =
          SlotMetadata{true, carried_version, new_episode};
      next_versions_[descriptor.state_id] = carried_version + 1;
    }
  }
  episode_ = new_episode;
  EmitTrace(trace_, TraceEvent{TraceKind::kReset, task_id, 0, 0, 0,
                               episode_, run_, 0});
  return Status::Ok();
}

const StateSlotDescriptor* StateStore::Descriptor(
    std::uint32_t state_id) const noexcept {
  if (!initialization_status_.ok() || state_id >= descriptors_.size()) {
    return nullptr;
  }
  return &descriptors_[state_id];
}

std::size_t StateStore::MetadataIndex(std::uint32_t state_id,
                                      std::uint32_t slot) const noexcept {
  return metadata_offsets_[state_id] + slot;
}

std::byte* StateStore::StagingData(std::uint32_t state_id) noexcept {
  return staging_.data() + staging_offsets_[state_id];
}

std::byte* StateStore::InitialData(std::uint32_t state_id) noexcept {
  return initial_.data() + staging_offsets_[state_id];
}

void* StateStore::SlotData(const StateSlotDescriptor& descriptor,
                           std::uint32_t slot) noexcept {
  return arena_.Resolve(
      descriptor.offset + slot * descriptor.slot_stride,
      descriptor.value_size, descriptor.alignment);
}

bool StateStore::IsActive(const Transaction& transaction) const noexcept {
  return transaction.active() &&
         transaction.id() == active_transaction_id_ &&
         transaction.episode() == episode_;
}

}  // namespace vlaforge::runtime
