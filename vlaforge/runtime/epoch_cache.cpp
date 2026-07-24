#include "vlaforge/runtime/epoch_cache.h"

#include <algorithm>

namespace vlaforge::runtime {
namespace {

bool SameDependency(const TemporalDependency& left,
                    const TemporalDependency& right) noexcept {
  return left.kind == right.kind &&
         left.subject_id == right.subject_id &&
         left.logical_version == right.logical_version &&
         left.epoch == right.epoch &&
         left.max_age_ns == right.max_age_ns;
}

bool Fresh(const TemporalDependency& dependency,
           const Epoch& now) noexcept {
  if (dependency.epoch.episode != now.episode ||
      dependency.epoch.timestamp_ns > now.timestamp_ns) {
    return false;
  }
  return dependency.max_age_ns == TemporalDependency::kUnboundedAge ||
         now.timestamp_ns - dependency.epoch.timestamp_ns <=
             dependency.max_age_ns;
}

}  // namespace

bool EpochVersionCacheGuard::Lookup(
    const TemporalDependency* dependencies,
    std::size_t dependency_count,
    const Epoch& now) noexcept {
  if (!valid_ || dependencies == nullptr ||
      dependency_count != dependency_count_ ||
      !std::all_of(
          dependencies,
          dependencies + dependency_count,
          [&now](const TemporalDependency& item) {
            return Fresh(item, now);
          }) ||
      !std::equal(
          dependencies,
          dependencies + dependency_count,
          dependencies_.begin(),
          SameDependency)) {
    ++misses_;
    return false;
  }
  ++hits_;
  return true;
}

Status EpochVersionCacheGuard::Update(
    const TemporalDependency* dependencies,
    std::size_t dependency_count) noexcept {
  if (dependencies == nullptr || dependency_count == 0u ||
      dependency_count > kMaximumDependencies) {
    return Status::Error(
        StatusCode::kInvalidArgument,
        static_cast<std::uint32_t>(dependency_count),
        "invalid temporal cache dependency signature");
  }
  std::copy(
      dependencies,
      dependencies + dependency_count,
      dependencies_.begin());
  dependency_count_ = dependency_count;
  valid_ = true;
  return Status::Ok();
}

void EpochVersionCacheGuard::Invalidate() noexcept {
  valid_ = false;
  dependency_count_ = 0u;
}

}  // namespace vlaforge::runtime
