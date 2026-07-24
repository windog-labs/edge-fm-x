#ifndef VLAFORGE_RUNTIME_EPOCH_H_
#define VLAFORGE_RUNTIME_EPOCH_H_

#include <cstdint>

namespace vlaforge::runtime {

struct Epoch final {
  std::uint32_t clock_id = 0;
  std::uint64_t sequence = 0;
  std::uint64_t timestamp_ns = 0;
  std::uint64_t episode = 0;
};

[[nodiscard]] constexpr bool operator==(const Epoch& left,
                                        const Epoch& right) noexcept {
  return left.clock_id == right.clock_id &&
         left.sequence == right.sequence &&
         left.timestamp_ns == right.timestamp_ns &&
         left.episode == right.episode;
}

[[nodiscard]] constexpr bool operator!=(const Epoch& left,
                                        const Epoch& right) noexcept {
  return !(left == right);
}

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_EPOCH_H_
