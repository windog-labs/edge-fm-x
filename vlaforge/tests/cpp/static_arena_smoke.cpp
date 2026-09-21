#include "vlaforge/runtime/static_arena.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

int main() {
  vlaforge::runtime::StaticArena arena(4096, 256);
  if (arena.data() == nullptr || arena.size_bytes() != 4096 ||
      arena.alignment() != 256) {
    return 1;
  }
  void* first = arena.Resolve(0, 512, 256);
  void* second = arena.Resolve(512, 1024, 128);
  if (first == nullptr || second == nullptr || first == second) {
    return 2;
  }
  if (reinterpret_cast<std::uintptr_t>(first) % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(second) % 128 != 0) {
    return 3;
  }
  if (arena.Resolve(4090, 8) != nullptr ||
      arena.Resolve(1, 1, 256) != nullptr ||
      arena.Resolve(0, 1, 3) != nullptr) {
    return 4;
  }

  vlaforge::runtime::StaticArena moved(std::move(arena));
  if (moved.Resolve(0, 4096, 256) == nullptr || arena.data() != nullptr) {
    return 5;
  }
  void* quarantined = nullptr;
  {
    vlaforge::runtime::StaticArena abandoned(64, 64);
    quarantined = abandoned.data();
    abandoned.Abandon();
    abandoned.Abandon();
    if (abandoned.data() != nullptr || abandoned.size_bytes() != 0 ||
        abandoned.Resolve(0, 0) != nullptr) return 6;
    vlaforge::runtime::StaticArena empty(std::move(abandoned));
    if (empty.data() != nullptr) return 7;
  }
  // CPU-only test owns the raw allocation after proving wrapper destruction
  // did not free it. Actual poisoned CUDA storage stays until process exit.
  static_cast<std::byte*>(quarantined)[0] = std::byte{1};
  std::free(quarantined);
  return 0;
}
