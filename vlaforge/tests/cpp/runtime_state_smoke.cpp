#include "vlaforge/runtime/action_queue.h"
#include "vlaforge/runtime/state_store.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

namespace {

std::atomic<std::size_t> g_allocations{0};

struct TraceCollector {
  vlaforge::runtime::TraceEvent events[256]{};
  std::size_t count = 0;
};

void CollectTrace(void* context,
                  const vlaforge::runtime::TraceEvent* event) {
  auto* collector = static_cast<TraceCollector*>(context);
  if (collector->count < 256) {
    collector->events[collector->count++] = *event;
  }
}

vlaforge::runtime::Status PublishAction(
    void* context, const vlaforge::runtime::CommittedAction*) {
  auto* count = static_cast<std::uint64_t*>(context);
  ++*count;
  return vlaforge::runtime::Status::Ok();
}

}  // namespace

void* operator new(std::size_t size) {
  ++g_allocations;
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  return ::operator new(size);
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

int main() {
  using vlaforge::runtime::ActionQueue;
  using vlaforge::runtime::CommittedAction;
  using vlaforge::runtime::Epoch;
  using vlaforge::runtime::PendingAction;
  using vlaforge::runtime::StateSlotDescriptor;
  using vlaforge::runtime::StateSnapshot;
  using vlaforge::runtime::StateStore;
  using vlaforge::runtime::StaticArena;
  using vlaforge::runtime::StatusCode;
  using vlaforge::runtime::TraceSink;
  using vlaforge::runtime::Transaction;

  TraceCollector traces;
  const TraceSink trace{&traces, &CollectTrace};
  StaticArena state_arena(128, 64);
  const StateSlotDescriptor descriptors[] = {
      {0, 3, 16, 16, 0, true},
      {1, 2, 8, 8, 48, false},
  };
  StateStore store(state_arena, descriptors, 2, trace);
  Transaction transaction(2);
  std::uint64_t published_callbacks = 0;
  ActionQueue actions(&PublishAction, &published_callbacks, trace);
  if (!store.initialization_status().ok()) {
    return 1;
  }

  const std::int64_t initial_zero = 10;
  const std::int64_t initial_one = 20;
  if (!store.Initialize(0, Epoch{0, 0, 0, 0}, &initial_zero,
                        sizeof(initial_zero))
           .ok() ||
      !store.Initialize(1, Epoch{0, 0, 0, 0}, &initial_one,
                        sizeof(initial_one))
           .ok()) {
    return 2;
  }

  StateSnapshot snapshot;
  if (!store
           .ReadLatest(0, 0, std::numeric_limits<std::uint64_t>::max(),
                       false, 10, &snapshot)
           .ok() ||
      snapshot.logical_version != 0 ||
      *static_cast<const std::int64_t*>(snapshot.data) != 10) {
    return 3;
  }

  const Epoch first_tick{0, 0, 0, 0};
  const Epoch first_state_epoch{0, 1, 20, 0};
  const std::int64_t first_value = 11;
  const std::int64_t action_value = 77;
  if (!store.Begin(&transaction, first_tick, 11).ok() ||
      !store
           .Stage(&transaction, 0, first_state_epoch, &first_value,
                  sizeof(first_value), 12)
           .ok()) {
    return 4;
  }
  const auto duplicate =
      store.Stage(&transaction, 0, first_state_epoch, &first_value,
                  sizeof(first_value), 12);
  if (duplicate.code != StatusCode::kAlreadyExists) {
    return 5;
  }
  CommittedAction committed;
  if (!store
           .Commit(&transaction,
                   PendingAction{first_tick, &action_value,
                                 sizeof(action_value)},
                   true, 13, &committed)
           .ok() ||
      !committed.valid || committed.transaction_id != 0) {
    return 6;
  }
  if (store.Commit(&transaction, PendingAction{}, true, 13, &committed).code !=
      StatusCode::kFailedPrecondition) {
    return 7;
  }
  if (!actions.Publish(committed, 14).ok() ||
      actions.publish_count() != 1 || published_callbacks != 1) {
    return 8;
  }
  if (actions.Publish(CommittedAction{}, 14).code !=
      StatusCode::kFailedPrecondition) {
    return 9;
  }

  const std::size_t allocations_before_ticks = g_allocations.load();
  std::int64_t latest_value = first_value;
  for (std::uint64_t sequence = 2; sequence < 12; ++sequence) {
    const Epoch tick{0, sequence - 1, sequence * 20, 0};
    const Epoch state_epoch{0, sequence, sequence * 20, 0};
    latest_value = static_cast<std::int64_t>(100 + sequence);
    if (!store.Begin(&transaction, tick, 20).ok()) {
      return 10;
    }
    if (sequence == 2) {
      StateSnapshot in_transaction_snapshot;
      if (!store
               .ReadLatest(
                   0, 0, std::numeric_limits<std::uint64_t>::max(),
                   false, 25, &in_transaction_snapshot)
               .ok() ||
          traces.count == 0 ||
          traces.events[traces.count - 1].transaction_id !=
              transaction.id()) {
        return 10;
      }
    }
    if (
        !store
             .Stage(&transaction, 0, state_epoch, &latest_value,
                    sizeof(latest_value), 21)
             .ok() ||
        !store
             .Commit(&transaction,
                     PendingAction{tick, &action_value,
                                   sizeof(action_value)},
                     true, 22, &committed)
             .ok() ||
        !actions.Publish(committed, 23).ok()) {
      return 10;
    }
  }
  if (g_allocations.load() != allocations_before_ticks) {
    return 11;
  }
  if (!store
           .ReadLatest(0, 0, std::numeric_limits<std::uint64_t>::max(),
                       false, 24, &snapshot)
           .ok() ||
      *static_cast<const std::int64_t*>(snapshot.data) != latest_value ||
      snapshot.logical_version != 11) {
    return 12;
  }
  if (store.ReadLatest(0, 0, 1, true, 24, &snapshot).code !=
      StatusCode::kNotFound) {
    return 13;
  }

  const std::int64_t rejected_value = 999;
  if (!store.Begin(&transaction, Epoch{0, 12, 240, 0}, 30).ok() ||
      !store
           .Stage(&transaction, 0, Epoch{0, 13, 260, 0},
                  &rejected_value, sizeof(rejected_value), 31)
           .ok()) {
    return 14;
  }
  if (store
          .Commit(&transaction,
                  PendingAction{Epoch{0, 12, 240, 0}, &action_value,
                                sizeof(action_value)},
                  false, 32, &committed)
          .code != StatusCode::kValidationFailed) {
    return 15;
  }
  if (!store
           .ReadLatest(0, 0, std::numeric_limits<std::uint64_t>::max(),
                       false, 33, &snapshot)
           .ok() ||
      *static_cast<const std::int64_t*>(snapshot.data) != latest_value) {
    return 16;
  }

  if (!store.Begin(&transaction, Epoch{0, 13, 260, 0}, 34).ok() ||
      !store.Abort(&transaction, 35).ok() ||
      store.Abort(&transaction, 35).code !=
          StatusCode::kFailedPrecondition) {
    return 17;
  }
  if (!store.ResetEpisode(1, 36).ok() || store.episode() != 1 ||
      store.ReadLatest(0, 1, std::numeric_limits<std::uint64_t>::max(),
                       false, 37, &snapshot)
              .code != StatusCode::kNotFound) {
    return 18;
  }
  const std::int64_t episode_one_value = 7;
  if (!store
           .Initialize(0, Epoch{0, 0, 0, 1}, &episode_one_value,
                       sizeof(episode_one_value))
           .ok()) {
    return 19;
  }

  if (traces.count == 0 ||
      actions.publish_count() != 11 ||
      published_callbacks != 11) {
    return 20;
  }
  return 0;
}
