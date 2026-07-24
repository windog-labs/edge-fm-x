#include "vlaforge/runtime/static_arena.h"

#include <cstddef>
#include <cstdint>
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
  return 0;
}
