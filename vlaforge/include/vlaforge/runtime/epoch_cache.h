#ifndef VLAFORGE_RUNTIME_EPOCH_CACHE_H_
#define VLAFORGE_RUNTIME_EPOCH_CACHE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "vlaforge/runtime/epoch.h"
#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

enum class TemporalDependencyKind : std::uint32_t {
  kInputEpoch = 0,
  kStateVersion = 1,
};

struct TemporalDependency final {
  static constexpr std::uint64_t kUnboundedAge =
      std::numeric_limits<std::uint64_t>::max();

  TemporalDependencyKind kind = TemporalDependencyKind::kInputEpoch;
  std::uint32_t subject_id = 0;
  std::uint64_t logical_version = 0;
  Epoch epoch{};
  std::uint64_t max_age_ns = kUnboundedAge;
  std::uint64_t max_versions = kUnboundedAge;
};

class EpochVersionCacheGuard final {
 public:
  static constexpr std::size_t kMaximumDependencies = 8u;

  [[nodiscard]] bool Lookup(const TemporalDependency* dependencies,
                            std::size_t dependency_count,
                            const Epoch& now) noexcept;

  [[nodiscard]] Status Update(const TemporalDependency* dependencies,
                              std::size_t dependency_count) noexcept;

  void Invalidate() noexcept;

  [[nodiscard]] constexpr std::uint64_t hits() const noexcept {
    return hits_;
  }
  [[nodiscard]] constexpr std::uint64_t misses() const noexcept {
    return misses_;
  }

 private:
  std::array<TemporalDependency, kMaximumDependencies> dependencies_{};
  std::size_t dependency_count_ = 0u;
  std::uint64_t hits_ = 0u;
  std::uint64_t misses_ = 0u;
  bool valid_ = false;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_EPOCH_CACHE_H_
