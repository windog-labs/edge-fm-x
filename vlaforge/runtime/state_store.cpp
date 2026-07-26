#include "vlaforge/runtime/state_store.h"

#include <algorithm>
#include <limits>

#include "vlaforge/runtime/device_copy.h"

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
  commit_backup_.resize(staging_size);
  initialization_status_ = Status::Ok();
}

Status StateStore::Initialize(std::uint32_t state_id, const void* data,
                              std::size_t size_bytes,
                              VLAForgeDevice source_device) noexcept {
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
  auto status = CopyBytes(
      InitialData(state_id), {VLAFORGE_DEVICE_CPU, 0},
      data, source_device, size_bytes, state_id);
  if (!status.ok()) {
    return status;
  }
  status = CopyBytes(
      SlotData(*descriptor, 0), arena_.device(),
      InitialData(state_id), {VLAFORGE_DEVICE_CPU, 0},
      size_bytes, state_id);
  if (!status.ok()) {
    return status;
  }
  auto& metadata = metadata_[MetadataIndex(state_id, 0)];
  metadata = SlotMetadata{true, 0, episode_};
  initialized_[state_id] = true;
  next_versions_[state_id] = 1;
  return Status::Ok();
}

Status StateStore::InitializeZero(std::uint32_t state_id) noexcept {
  const auto* descriptor = Descriptor(state_id);
  if (descriptor == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid zero state initializer");
  }
  if (initialized_[state_id]) {
    return Status::Error(StatusCode::kAlreadyExists, state_id,
                         "state is already initialized");
  }
  std::fill_n(
      InitialData(state_id), descriptor->value_size, std::byte{0});
  const auto status = CopyBytes(
      SlotData(*descriptor, 0), arena_.device(),
      InitialData(state_id), {VLAFORGE_DEVICE_CPU, 0},
      descriptor->value_size, state_id);
  if (!status.ok()) {
    return status;
  }
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
                          descriptor->value_size,
                          arena_.device()};
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
                         std::uint32_t task_id,
                         VLAForgeDevice source_device) noexcept {
  const auto* descriptor = Descriptor(state_id);
  if (transaction == nullptr || !IsActive(*transaction) ||
      descriptor == nullptr || data == nullptr ||
      size_bytes != descriptor->value_size) {
    return Status::Error(StatusCode::kInvalidArgument, state_id,
                         "invalid staged state write");
  }
  std::byte* staging = StagingData(state_id);
  auto status = CopyBytes(
      staging, {VLAFORGE_DEVICE_CPU, 0},
      data, source_device, size_bytes, state_id);
  if (!status.ok()) {
    return status;
  }
  status =
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
  // Preserve every destination slot before performing any write. Metadata and
  // logical versions remain untouched until all device copies succeed.
  for (std::size_t index = 0; index < transaction->pending_count(); ++index) {
    const auto& pending = transaction->pending(index);
    const auto* descriptor = Descriptor(pending.state_id);
    if (descriptor == nullptr) {
      return Status::Error(StatusCode::kInternal, pending.state_id,
                           "staged state descriptor disappeared");
    }
    const std::uint64_t version = next_versions_[pending.state_id];
    const auto slot =
        static_cast<std::uint32_t>(version % descriptor->capacity);
    const auto& previous =
        metadata_[MetadataIndex(pending.state_id, slot)];
    if (previous.valid) {
      const auto backup_status = CopyBytes(
          BackupData(pending.state_id), {VLAFORGE_DEVICE_CPU, 0},
          SlotData(*descriptor, slot), arena_.device(),
          pending.size_bytes, pending.state_id);
      if (!backup_status.ok()) {
        return backup_status;
      }
    }
  }
  for (std::size_t index = 0; index < transaction->pending_count(); ++index) {
    const auto& pending = transaction->pending(index);
    const auto* descriptor = Descriptor(pending.state_id);
    if (descriptor == nullptr) {
      return Status::Error(StatusCode::kInternal, pending.state_id,
                           "staged state descriptor disappeared");
    }
    const std::uint64_t version = next_versions_[pending.state_id];
    const auto slot =
        static_cast<std::uint32_t>(version % descriptor->capacity);
    const auto copy_status = CopyBytes(
        SlotData(*descriptor, slot), arena_.device(),
        pending.data, {VLAFORGE_DEVICE_CPU, 0},
        pending.size_bytes, pending.state_id);
    if (!copy_status.ok()) {
      bool rollback_ok = true;
      for (std::size_t rollback = 0; rollback <= index; ++rollback) {
        const auto& prior = transaction->pending(rollback);
        const auto* prior_descriptor = Descriptor(prior.state_id);
        if (prior_descriptor == nullptr) {
          rollback_ok = false;
          continue;
        }
        const auto prior_version = next_versions_[prior.state_id];
        const auto prior_slot = static_cast<std::uint32_t>(
            prior_version % prior_descriptor->capacity);
        const auto& previous =
            metadata_[MetadataIndex(prior.state_id, prior_slot)];
        if (!previous.valid) {
          continue;
        }
        rollback_ok =
            CopyBytes(
                SlotData(*prior_descriptor, prior_slot), arena_.device(),
                BackupData(prior.state_id), {VLAFORGE_DEVICE_CPU, 0},
                prior.size_bytes, prior.state_id)
                .ok() &&
            rollback_ok;
      }
      return rollback_ok
                 ? copy_status
                 : Status::Error(StatusCode::kInternal, pending.state_id,
                                 "state commit rollback failed");
    }
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
    if (!descriptor.reset_on_episode ||
        !initialized_[descriptor.state_id]) {
      continue;
    }
    const auto backup_status = CopyBytes(
        BackupData(descriptor.state_id), {VLAFORGE_DEVICE_CPU, 0},
        SlotData(descriptor, 0), arena_.device(),
        descriptor.value_size, descriptor.state_id);
    if (!backup_status.ok()) {
      return backup_status;
    }
  }
  std::size_t reset_count = 0;
  for (const auto& descriptor : descriptors_) {
    if (!descriptor.reset_on_episode ||
        !initialized_[descriptor.state_id]) {
      continue;
    }
    const auto copy_status = CopyBytes(
        SlotData(descriptor, 0), arena_.device(),
        InitialData(descriptor.state_id), {VLAFORGE_DEVICE_CPU, 0},
        descriptor.value_size, descriptor.state_id);
    if (!copy_status.ok()) {
      bool rollback_ok = true;
      std::size_t rollback_count = 0;
      for (const auto& prior : descriptors_) {
        if (!prior.reset_on_episode ||
            !initialized_[prior.state_id]) {
          continue;
        }
        if (rollback_count++ > reset_count) {
          break;
        }
        rollback_ok =
            CopyBytes(
                SlotData(prior, 0), arena_.device(),
                BackupData(prior.state_id), {VLAFORGE_DEVICE_CPU, 0},
                prior.value_size, prior.state_id)
                .ok() &&
            rollback_ok;
      }
      return rollback_ok
                 ? copy_status
                 : Status::Error(StatusCode::kInternal,
                                 descriptor.state_id,
                                 "episode reset rollback failed");
    }
    ++reset_count;
  }
  for (const auto& descriptor : descriptors_) {
    const SlotMetadata* latest = nullptr;
    std::uint32_t latest_slot = 0;
    for (std::uint32_t slot = 0; slot < descriptor.capacity; ++slot) {
      const auto& candidate =
          metadata_[MetadataIndex(descriptor.state_id, slot)];
      if (candidate.valid && candidate.episode == episode_ &&
          (latest == nullptr ||
           candidate.logical_version > latest->logical_version)) {
        latest = &candidate;
        latest_slot = slot;
      }
    }
    const bool carry = !descriptor.reset_on_episode && latest != nullptr;
    const auto carried_version =
        carry ? latest->logical_version : std::uint64_t{0};
    for (std::uint32_t slot = 0; slot < descriptor.capacity; ++slot) {
      metadata_[MetadataIndex(descriptor.state_id, slot)].valid = false;
    }
    if (descriptor.reset_on_episode && initialized_[descriptor.state_id]) {
      metadata_[MetadataIndex(descriptor.state_id, 0)] =
          SlotMetadata{true, 0, new_episode};
      next_versions_[descriptor.state_id] = 1;
    } else if (carry) {
      metadata_[MetadataIndex(descriptor.state_id, latest_slot)] =
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

std::byte* StateStore::BackupData(std::uint32_t state_id) noexcept {
  return commit_backup_.data() + staging_offsets_[state_id];
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
