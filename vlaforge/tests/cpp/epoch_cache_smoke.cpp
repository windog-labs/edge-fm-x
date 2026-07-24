#include "vlaforge/runtime/epoch_cache.h"

#include <cassert>
#include <cstddef>

int main() {
  using vlaforge::runtime::Epoch;
  using vlaforge::runtime::EpochVersionCacheGuard;
  using vlaforge::runtime::TemporalDependency;
  using vlaforge::runtime::TemporalDependencyKind;

  EpochVersionCacheGuard cache;
  const TemporalDependency image{
      TemporalDependencyKind::kInputEpoch,
      3u,
      0u,
      Epoch{0u, 4u, 100u, 2u},
      50u,
  };
  assert(!cache.Lookup(&image, 1u, Epoch{1u, 8u, 110u, 2u}));
  assert(cache.Update(&image, 1u).ok());
  assert(cache.Lookup(&image, 1u, Epoch{1u, 8u, 150u, 2u}));

  // A stale observation, a new input epoch, or an episode reset must miss.
  assert(!cache.Lookup(&image, 1u, Epoch{1u, 9u, 151u, 2u}));
  TemporalDependency newer = image;
  newer.epoch.sequence = 5u;
  assert(!cache.Lookup(&newer, 1u, Epoch{1u, 9u, 140u, 2u}));
  assert(!cache.Lookup(&image, 1u, Epoch{1u, 9u, 140u, 3u}));

  TemporalDependency zero_age = image;
  zero_age.max_age_ns = 0u;
  assert(cache.Update(&zero_age, 1u).ok());
  assert(cache.Lookup(&zero_age, 1u, Epoch{1u, 9u, 100u, 2u}));
  assert(!cache.Lookup(&zero_age, 1u, Epoch{1u, 9u, 101u, 2u}));

  TemporalDependency state_version{
      TemporalDependencyKind::kStateVersion,
      7u,
      12u,
      Epoch{1u, 12u, 100u, 2u},
      TemporalDependency::kUnboundedAge,
  };
  assert(cache.Update(&state_version, 1u).ok());
  assert(cache.Lookup(
      &state_version, 1u, Epoch{1u, 12u, 1000u, 2u}));
  state_version.logical_version = 13u;
  assert(!cache.Lookup(
      &state_version, 1u, Epoch{1u, 13u, 1000u, 2u}));

  cache.Invalidate();
  assert(!cache.Lookup(&image, 1u, Epoch{1u, 9u, 140u, 2u}));
  assert(cache.hits() == 3u);
  assert(cache.misses() == 7u);

  const TemporalDependency invalid[
      EpochVersionCacheGuard::kMaximumDependencies + 1u]{};
  assert(!cache.Update(
                   invalid,
                   EpochVersionCacheGuard::kMaximumDependencies + 1u)
              .ok());
  return 0;
}
