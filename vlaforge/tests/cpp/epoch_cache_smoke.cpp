#include "vlaforge/runtime/epoch_cache.h"
#include "vlaforge/runtime/state_store.h"

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
      TemporalDependency::kUnboundedAge,
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
      0u,
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

  // Missing/unversioned signatures are never cacheable.
  assert(!cache.Update(nullptr, 0u).ok());
  assert(!cache.Lookup(nullptr, 0u, Epoch{1u, 1u, 0u, 0u}));

  // A transaction abort leaves the authoritative StateVersion unchanged, so
  // an exact-version pure-region cache entry remains legal. A later commit
  // advances the version and must miss.
  using vlaforge::runtime::StateSlotDescriptor;
  using vlaforge::runtime::StateSnapshot;
  using vlaforge::runtime::StateStore;
  using vlaforge::runtime::StaticArena;
  using vlaforge::runtime::Transaction;
  StaticArena arena(64u, 64u);
  const StateSlotDescriptor descriptor{0u, 2u, 8u, 8u, 0u, true};
  StateStore store(arena, &descriptor, 1u);
  Transaction transaction(1u);
  const std::uint64_t initial = 7u;
  const std::uint64_t staged = 8u;
  assert(store.Initialize(
                    0u, Epoch{1u, 0u, 0u, 0u}, &initial, sizeof(initial))
             .ok());
  StateSnapshot snapshot;
  assert(store
             .ReadLatest(0u, 0u, TemporalDependency::kUnboundedAge,
                         false, 0u, &snapshot)
             .ok());
  TemporalDependency committed_version{
      TemporalDependencyKind::kStateVersion,
      0u,
      snapshot.logical_version,
      snapshot.epoch,
      TemporalDependency::kUnboundedAge,
      0u,
  };
  assert(cache.Update(&committed_version, 1u).ok());
  assert(store.Begin(&transaction, Epoch{1u, 1u, 10u, 0u}, 0u).ok());
  assert(store
             .Stage(&transaction, 0u, Epoch{1u, 1u, 10u, 0u},
                    &staged, sizeof(staged), 0u)
             .ok());
  assert(store.Abort(&transaction, 0u).ok());
  assert(store
             .ReadLatest(0u, 0u, TemporalDependency::kUnboundedAge,
                         false, 0u, &snapshot)
             .ok());
  assert(snapshot.logical_version == committed_version.logical_version);
  assert(cache.Lookup(
      &committed_version, 1u, Epoch{1u, 1u, 10u, 0u}));

  assert(store.Begin(&transaction, Epoch{1u, 2u, 20u, 0u}, 0u).ok());
  assert(store
             .Stage(&transaction, 0u, Epoch{1u, 2u, 20u, 0u},
                    &staged, sizeof(staged), 0u)
             .ok());
  vlaforge::runtime::CommittedAction action;
  assert(store
             .Commit(
                 &transaction,
                 vlaforge::runtime::PendingAction{
                     Epoch{1u, 2u, 20u, 0u}, &staged, sizeof(staged)},
                 true, 0u, &action)
             .ok());
  assert(store
             .ReadLatest(0u, 0u, TemporalDependency::kUnboundedAge,
                         false, 0u, &snapshot)
             .ok());
  TemporalDependency newer_version = committed_version;
  newer_version.logical_version = snapshot.logical_version;
  newer_version.epoch = snapshot.epoch;
  assert(!cache.Lookup(
      &newer_version, 1u, Epoch{1u, 2u, 20u, 0u}));

  assert(store.ResetEpisode(1u, 0u).ok());
  cache.Invalidate();
  assert(!cache.Lookup(
      &newer_version, 1u, Epoch{1u, 0u, 0u, 1u}));
  return 0;
}
