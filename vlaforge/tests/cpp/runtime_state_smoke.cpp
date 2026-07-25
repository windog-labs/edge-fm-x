#include <cstdint>
#include <cstring>
#include <iostream>

#include "vlaforge/runtime/state_store.h"

namespace {

bool Check(const vlaforge::runtime::Status& status, const char* label) {
  if (!status.ok()) {
    std::cerr << label << ": " << status.message << '\n';
    return false;
  }
  return true;
}

std::int32_t Value(const vlaforge::runtime::StateSnapshot& snapshot) {
  std::int32_t value = 0;
  std::memcpy(&value, snapshot.data, sizeof(value));
  return value;
}

}  // namespace

int main() {
  using vlaforge::runtime::StateSlotDescriptor;
  using vlaforge::runtime::StateSnapshot;
  using vlaforge::runtime::StateStore;
  using vlaforge::runtime::StaticArena;
  using vlaforge::runtime::Transaction;

  StaticArena arena(64, 64);
  const StateSlotDescriptor descriptor{
      0u, 2u, sizeof(std::int32_t), sizeof(std::int32_t),
      alignof(std::int32_t), 0u, true};
  StateStore store(arena, &descriptor, 1u);
  if (!Check(store.initialization_status(), "construct")) {
    return 1;
  }
  const std::int32_t initial = 7;
  if (!Check(store.Initialize(0u, &initial, sizeof(initial)), "initialize")) {
    return 2;
  }
  StateSnapshot snapshot{};
  if (!Check(store.ReadLatest(0u, 1u, &snapshot), "read initial") ||
      snapshot.logical_version != 0u || Value(snapshot) != 7) {
    return 3;
  }

  Transaction committed(1u);
  const std::int32_t next = 8;
  if (!Check(store.Begin(&committed, 2u), "begin commit") ||
      !Check(store.Stage(&committed, 0u, &next, sizeof(next), 3u),
             "stage commit") ||
      !Check(store.Commit(&committed, 4u), "commit") ||
      !Check(store.ReadLatest(0u, 5u, &snapshot), "read committed") ||
      snapshot.logical_version != 1u || Value(snapshot) != 8) {
    return 4;
  }

  Transaction aborted(1u);
  const std::int32_t rejected = 9;
  if (!Check(store.Begin(&aborted, 6u), "begin abort") ||
      !Check(store.Stage(&aborted, 0u, &rejected, sizeof(rejected), 7u),
             "stage abort") ||
      !Check(store.Abort(&aborted, 8u), "abort") ||
      !Check(store.ReadLatest(0u, 9u, &snapshot), "read after abort") ||
      snapshot.logical_version != 1u || Value(snapshot) != 8) {
    return 5;
  }

  if (!Check(store.ResetEpisode(1u, 10u), "reset") ||
      !Check(store.ReadLatest(0u, 11u, &snapshot), "read reset") ||
      snapshot.logical_version != 0u || snapshot.episode != 1u ||
      Value(snapshot) != 7) {
    return 6;
  }
  return 0;
}
