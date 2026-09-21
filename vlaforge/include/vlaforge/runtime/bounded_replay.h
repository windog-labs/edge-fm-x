#ifndef VLAFORGE_RUNTIME_BOUNDED_REPLAY_H_
#define VLAFORGE_RUNTIME_BOUNDED_REPLAY_H_

#include "vlaforge/runtime/execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VLAFORGE_GRAPH_BACKEND_ABI_VERSION 1u
#define VLAFORGE_BOUNDED_REPLAY_ABI_VERSION 1u

typedef struct VLAForgeGraphExecutable VLAForgeGraphExecutable;
typedef struct VLAForgeBoundedReplay VLAForgeBoundedReplay;

/* Graph providers own their capture allocator pool through destroy().
 * abort() must end an active capture. Failure means process state may be
 * poisoned: ordinary execution is forbidden, even if destroy() returns.
 * destroy(graph, nonzero) must quarantine resources unsafe to reclaim when
 * completion is unproven, including a failure in the runtime's own drain. */
typedef struct VLAForgeGraphBackendApi {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeStatus (*create)(const VLAForgeExecutionContextView*,
                           VLAForgeGraphExecutable**);
  VLAForgeStatus (*begin)(VLAForgeGraphExecutable*);
  VLAForgeStatus (*end)(VLAForgeGraphExecutable*);
  VLAForgeStatus (*abort)(VLAForgeGraphExecutable*);
  VLAForgeStatus (*launch)(VLAForgeGraphExecutable*);
  void (*destroy)(VLAForgeGraphExecutable*, uint32_t execution_poisoned);
} VLAForgeGraphBackendApi;

typedef enum VLAForgeReplayMode {
  VLAFORGE_REPLAY_ORDINARY = 0,
  VLAFORGE_REPLAY_PREFER = 1,
  VLAFORGE_REPLAY_REQUIRE = 2
} VLAForgeReplayMode;

typedef enum VLAForgeReplayPhase {
  VLAFORGE_REPLAY_PHASE_ORDINARY = 0,
  VLAFORGE_REPLAY_PHASE_WARMUP = 1,
  VLAFORGE_REPLAY_PHASE_CAPTURE = 2
} VLAForgeReplayPhase;

typedef enum VLAForgeReplayState {
  VLAFORGE_REPLAY_UNPREPARED = 0,
  VLAFORGE_REPLAY_READY = 1,
  VLAFORGE_REPLAY_FALLBACK = 2,
  VLAFORGE_REPLAY_POISONED = 3
} VLAForgeReplayState;

typedef VLAForgeStatus (*VLAForgeReplayPrepareFn)(void* user_data);
typedef VLAForgeStatus (*VLAForgeReplayStepFn)(
    void* user_data, uint32_t step, VLAForgeReplayPhase phase);

typedef struct VLAForgeReplayRegion {
  VLAForgeRegionExecutable* executable;
  const VLAForgeRegionExecutableValueApi* api;
  const VLAForgeRegionExecutionExtensionApi* execution_extension;
} VLAForgeReplayRegion;

/* prepare() runs only outside capture and must reset input/carry CONTENTS
 * without moving their storage. step() may enqueue registered Region runs and
 * same-context device copies, but no synchronization, allocation of boundary
 * storage, rebind, RNG, transaction, host I/O or input-dependent control flow.
 * Boundary descriptors, addresses and shapes are immutable until destruction.
 * Callbacks and Region lists are compiler/trusted-host contracts, not a sandbox. */
typedef struct VLAForgeBoundedReplayOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  VLAForgeReplayMode mode;
  uint32_t steps;
  uint32_t warmup_runs;
  const char* capture_rejection_reason;
  VLAForgeReplayPrepareFn prepare;
  VLAForgeReplayStepFn step;
  void* user_data;
  const VLAForgeReplayRegion* regions;
  uint32_t region_count;
} VLAForgeBoundedReplayOptions;

typedef struct VLAForgeBoundedReplayInfo {
  uint32_t struct_size;
  VLAForgeReplayState state;
  uint32_t captured_steps;
  uint64_t replay_count;
  uint64_t ordinary_count;
  const char* reason;
} VLAForgeBoundedReplayInfo;

/* The replay object borrows context, Regions, weights and boundary storage.
 * Destroy it BEFORE those resources. run() is serialized and completion-safe;
 * outputs are usable only after OK. No ordinary fallback follows poisoned
 * capture/launch/drain. Fatal allocator corruption may require process exit. */
VLAForgeStatus vlaforge_bounded_replay_create(
    VLAForgeExecutionContext* context,
    const VLAForgeGraphBackendApi* graph_api,
    const VLAForgeBoundedReplayOptions* options,
    VLAForgeBoundedReplay** replay);
VLAForgeStatus vlaforge_bounded_replay_run(VLAForgeBoundedReplay* replay);
VLAForgeStatus vlaforge_bounded_replay_get_info(
    const VLAForgeBoundedReplay* replay, VLAForgeBoundedReplayInfo* info);
void vlaforge_bounded_replay_destroy(VLAForgeBoundedReplay* replay);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_RUNTIME_BOUNDED_REPLAY_H_
