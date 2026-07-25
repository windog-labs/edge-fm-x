#include <cstdint>
#include <iostream>

#include "vlaforge/runtime/device_copy.h"
#include "vlaforge/runtime/state_store.h"

namespace {

constexpr VLAForgeDevice kCpu{VLAFORGE_DEVICE_CPU, 0};
constexpr VLAForgeDevice kCuda{VLAFORGE_DEVICE_CUDA, 0};

bool Check(const vlaforge::runtime::Status& status, const char* label) {
  if (!status.ok()) {
    std::cerr << label << ": " << status.message << '\n';
    return false;
  }
  return true;
}

bool ReadValue(const vlaforge::runtime::StateSnapshot& snapshot,
               std::int32_t* value) {
  return Check(
      vlaforge::runtime::CopyBytes(
          value, kCpu, snapshot.data, snapshot.device,
          sizeof(*value), snapshot.state_id),
      "read CUDA state");
}

}  // namespace

int main() {
  using vlaforge::runtime::StateSlotDescriptor;
  using vlaforge::runtime::StateSnapshot;
  using vlaforge::runtime::StateStore;
  using vlaforge::runtime::StaticArena;
  using vlaforge::runtime::Transaction;

  StaticArena state_arena(64, 64, kCuda);
  StaticArena source_arena(64, 64, kCuda);
  const StateSlotDescriptor descriptor{
      0u, 2u, sizeof(std::int32_t), sizeof(std::int32_t),
      alignof(std::int32_t), 0u, true};
  StateStore store(state_arena, &descriptor, 1u);
  if (!Check(store.initialization_status(), "construct")) {
    return 1;
  }
  const std::int32_t initial = 7;
  if (!Check(store.Initialize(0u, &initial, sizeof(initial), kCpu),
             "initialize")) {
    return 2;
  }
  StateSnapshot snapshot{};
  std::int32_t observed = 0;
  if (!Check(store.ReadLatest(0u, 1u, &snapshot), "read initial") ||
      snapshot.device.kind != VLAFORGE_DEVICE_CUDA ||
      !ReadValue(snapshot, &observed) || observed != initial ||
      snapshot.logical_version != 0u) {
    return 3;
  }

  const std::int32_t next = 8;
  if (!Check(vlaforge::runtime::CopyBytes(
                 source_arena.data(), kCuda, &next, kCpu, sizeof(next)),
             "upload staged state")) {
    return 4;
  }
  Transaction committed(1u);
  if (!Check(store.Begin(&committed, 2u), "begin commit") ||
      !Check(store.Stage(&committed, 0u, source_arena.data(),
                         sizeof(next), 3u, kCuda),
             "stage CUDA state") ||
      !Check(store.Commit(&committed, 4u), "commit CUDA state") ||
      !Check(store.ReadLatest(0u, 5u, &snapshot), "read committed") ||
      !ReadValue(snapshot, &observed) || observed != next ||
      snapshot.logical_version != 1u) {
    return 5;
  }

  const std::int32_t rejected = 9;
  if (!Check(vlaforge::runtime::CopyBytes(
                 source_arena.data(), kCuda, &rejected, kCpu,
                 sizeof(rejected)),
             "upload rejected state")) {
    return 6;
  }
  Transaction aborted(1u);
  if (!Check(store.Begin(&aborted, 6u), "begin abort") ||
      !Check(store.Stage(&aborted, 0u, source_arena.data(),
                         sizeof(rejected), 7u, kCuda),
             "stage rejected state") ||
      !Check(store.Abort(&aborted, 8u), "abort") ||
      !Check(store.ReadLatest(0u, 9u, &snapshot), "read after abort") ||
      !ReadValue(snapshot, &observed) || observed != next ||
      snapshot.logical_version != 1u) {
    return 7;
  }

  if (!Check(store.ResetEpisode(1u, 10u), "reset") ||
      !Check(store.ReadLatest(0u, 11u, &snapshot), "read reset") ||
      !ReadValue(snapshot, &observed) || observed != initial ||
      snapshot.logical_version != 0u || snapshot.episode != 1u) {
    return 8;
  }
  return 0;
}
