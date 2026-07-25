#ifndef VLAFORGE_RUNTIME_TRACE_H_
#define VLAFORGE_RUNTIME_TRACE_H_

#include <cstdint>

namespace vlaforge::runtime {

enum class TraceKind : std::uint32_t {
  kTransactionBegin = 0,
  kStateRead = 1,
  kStateStage = 2,
  kStateCommit = 3,
  kTransactionCommit = 4,
  kTransactionAbort = 5,
  kInput = 6,
  kRegion = 7,
  kCacheHit = 8,
  kCacheMiss = 9,
  kValidation = 10,
  kOutputPending = 11,
  kOutputGroupPending = 12,
  kOutputGroupCommit = 13,
  kReset = 14,
};

struct TraceEvent final {
  TraceKind kind = TraceKind::kTransactionBegin;
  std::uint32_t task_id = 0;
  std::uint32_t subject_id = 0;
  std::uint64_t logical_version = 0;
  std::uint64_t transaction_id = 0;
  std::uint64_t episode = 0;
  std::uint64_t run = 0;
  std::uint64_t revision = 0;
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
