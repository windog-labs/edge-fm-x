#ifndef VLAFORGE_RUNTIME_STATE_STORE_H_
#define VLAFORGE_RUNTIME_STATE_STORE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vlaforge/runtime/static_arena.h"
#include "vlaforge/runtime/status.h"
#include "vlaforge/runtime/trace.h"
#include "vlaforge/runtime/transaction.h"

namespace vlaforge::runtime {

struct StateSlotDescriptor final {
  std::uint32_t state_id = 0;
  std::uint32_t capacity = 0;
  std::size_t value_size = 0;
  std::size_t slot_stride = 0;
  std::size_t alignment = 1;
  std::size_t offset = 0;
  bool reset_on_episode = true;
};

struct StateSnapshot final {
  std::uint32_t state_id = 0;
  std::uint32_t physical_slot = 0;
  std::uint64_t logical_version = 0;
  std::uint64_t episode = 0;
  const void* data = nullptr;
  std::size_t size_bytes = 0;
  VLAForgeDevice device{VLAFORGE_DEVICE_CPU, 0};
};

class StateStore final {
 public:
  StateStore(StaticArena& arena, const StateSlotDescriptor* descriptors,
             std::size_t state_count, TraceSink trace = {});

  StateStore(const StateStore&) = delete;
  StateStore& operator=(const StateStore&) = delete;

  [[nodiscard]] const Status& initialization_status() const noexcept {
    return initialization_status_;
  }
  [[nodiscard]] std::uint64_t episode() const noexcept { return episode_; }

  void SetTraceSink(TraceSink trace) noexcept { trace_ = trace; }
  void SetRunIndex(std::uint64_t run) noexcept { run_ = run; }

  Status Initialize(std::uint32_t state_id, const void* data,
                    std::size_t size_bytes,
                    VLAForgeDevice source_device = {
                        VLAFORGE_DEVICE_CPU, 0}) noexcept;
  Status InitializeZero(std::uint32_t state_id) noexcept;
  Status Begin(Transaction* transaction,
               std::uint32_t task_id = 0) noexcept;
  Status ReadLatest(std::uint32_t state_id, std::uint32_t task_id,
                    StateSnapshot* output) noexcept;
  Status Stage(Transaction* transaction, std::uint32_t state_id,
               const void* data, std::size_t size_bytes,
               std::uint32_t task_id,
               VLAForgeDevice source_device = {
                   VLAFORGE_DEVICE_CPU, 0}) noexcept;
  Status Commit(Transaction* transaction,
                std::uint32_t task_id) noexcept;
  Status Abort(Transaction* transaction,
               std::uint32_t task_id) noexcept;
  Status ResetEpisode(std::uint64_t new_episode,
                      std::uint32_t task_id) noexcept;

 private:
  struct SlotMetadata final {
    bool valid = false;
    std::uint64_t logical_version = 0;
    std::uint64_t episode = 0;
  };

  [[nodiscard]] const StateSlotDescriptor* Descriptor(
      std::uint32_t state_id) const noexcept;
  [[nodiscard]] std::size_t MetadataIndex(std::uint32_t state_id,
                                          std::uint32_t slot) const noexcept;
  [[nodiscard]] std::byte* StagingData(
      std::uint32_t state_id) noexcept;
  [[nodiscard]] std::byte* InitialData(
      std::uint32_t state_id) noexcept;
  [[nodiscard]] std::byte* BackupData(
      std::uint32_t state_id) noexcept;
  [[nodiscard]] void* SlotData(const StateSlotDescriptor& descriptor,
                               std::uint32_t slot) noexcept;
  [[nodiscard]] bool IsActive(
      const Transaction& transaction) const noexcept;

  StaticArena& arena_;
  std::vector<StateSlotDescriptor> descriptors_;
  std::vector<std::size_t> metadata_offsets_;
  std::vector<std::size_t> staging_offsets_;
  std::vector<SlotMetadata> metadata_;
  std::vector<std::byte> staging_;
  std::vector<std::byte> initial_;
  std::vector<std::byte> commit_backup_;
  std::vector<bool> initialized_;
  std::vector<std::uint64_t> next_versions_;
  TraceSink trace_{};
  Status initialization_status_{};
  std::uint64_t next_transaction_id_ = 0;
  std::uint64_t active_transaction_id_ = static_cast<std::uint64_t>(-1);
  std::uint64_t episode_ = 0;
  std::uint64_t run_ = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_STATE_STORE_H_
