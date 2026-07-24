#ifndef VLAFORGE_RUNTIME_TRACE_H_
#define VLAFORGE_RUNTIME_TRACE_H_

#include <cstdint>

#include "vlaforge/runtime/epoch.h"

namespace vlaforge::runtime {

enum class TraceKind : std::uint32_t {
  kTransactionBegin = 0,
  kStateRead = 1,
  kStateStage = 2,
  kStateCommit = 3,
  kTransactionCommit = 4,
  kTransactionAbort = 5,
  kActionPending = 6,
  kActionCommit = 7,
  kActionPublish = 8,
  kReset = 9,
};

struct TraceEvent final {
  TraceKind kind = TraceKind::kTransactionBegin;
  std::uint32_t task_id = 0;
  std::uint32_t state_id = 0;
  std::uint64_t logical_version = 0;
  std::uint64_t transaction_id = 0;
  Epoch epoch{};
};

using TraceEmitFn = void (*)(void* context, const TraceEvent* event);

struct TraceSink final {
  void* context = nullptr;
  TraceEmitFn emit = nullptr;
};

inline void EmitTrace(const TraceSink& sink,
                      const TraceEvent& event) noexcept {
  if (sink.emit != nullptr) {
    sink.emit(sink.context, &event);
  }
}

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_TRACE_H_
