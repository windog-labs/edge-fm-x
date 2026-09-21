#include "vlaforge/runtime/bounded_replay.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

struct VLAForgeBoundedReplay {
  VLAForgeExecutionContext* context = nullptr;
  const VLAForgeGraphBackendApi* graph_api = nullptr;
  VLAForgeGraphExecutable* graph = nullptr;
  VLAForgeBoundedReplayOptions options{};
  std::vector<VLAForgeReplayRegion> regions;
  VLAForgeReplayState state = VLAFORGE_REPLAY_UNPREPARED;
  std::uint64_t replay_count = 0;
  std::uint64_t ordinary_count = 0;
  bool running = false;
  std::array<char, 512> reason{};
  std::array<char, 512> error{};

  VLAForgeStatus Save(VLAForgeStatus status) noexcept {
    if (status.code == VLAFORGE_STATUS_OK) { return status; }
    std::snprintf(error.data(), error.size(), "%.*s",
                  static_cast<int>(status.message_size > 511 ? 511 : status.message_size),
                  status.message == nullptr ? "execution error" : status.message);
    return vlaforge_status_error(status.code, error.data());
  }

  VLAForgeStatus Poison(const char* message) noexcept {
    state = VLAFORGE_REPLAY_POISONED;
    std::snprintf(reason.data(), reason.size(), "%s", message);
    vlaforge_execution_context_poison(context);
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, reason.data());
  }

  VLAForgeStatus Drain() noexcept {
    auto status = vlaforge_execution_context_synchronize(context);
    if (status.code != VLAFORGE_STATUS_OK) {
      return Poison("replay context drain failed; process restart may be required");
    }
    for (const auto& region : regions) {
      status = region.api->synchronize(region.executable);
      if (status.code != VLAFORGE_STATUS_OK) {
        return Poison("replay Region drain failed; process restart may be required");
      }
    }
    return vlaforge_status_ok();
  }

  VLAForgeStatus Prepare() noexcept {
    try {
      auto status = Save(options.prepare(options.user_data));
      if (status.code != VLAFORGE_STATUS_OK) {
        const auto drained = Drain();
        return drained.code == VLAFORGE_STATUS_OK ? status : drained;
      }
      return status;
    } catch (...) {
      const auto drained = Drain();
      return drained.code == VLAFORGE_STATUS_OK
          ? vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR,
                                   "replay prepare callback threw")
          : drained;
    }
  }

  VLAForgeStatus Body(VLAForgeReplayPhase phase) noexcept {
    try {
      for (std::uint32_t step = 0; step < options.steps; ++step) {
        auto status = Save(options.step(options.user_data, step, phase));
        if (status.code != VLAFORGE_STATUS_OK) { return status; }
      }
      return vlaforge_status_ok();
    } catch (...) {
      return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR,
                                   "replay step callback threw");
    }
  }

  VLAForgeStatus Ordinary(VLAForgeReplayPhase phase) noexcept {
    auto status = Prepare();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    status = Body(phase);
    const auto drained = Drain();
    return drained.code == VLAFORGE_STATUS_OK ? status : drained;
  }

  VLAForgeStatus Reject(const char* message) noexcept {
    state = VLAFORGE_REPLAY_FALLBACK;
    std::snprintf(reason.data(), reason.size(), "%s", message);
    return options.mode == VLAFORGE_REPLAY_REQUIRE
        ? vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, reason.data())
        : vlaforge_status_ok();
  }

  VLAForgeStatus Capture() noexcept {
    if (reason[0] != '\0') {
      state = VLAFORGE_REPLAY_FALLBACK;
      return options.mode == VLAFORGE_REPLAY_REQUIRE
          ? vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, reason.data())
          : vlaforge_status_ok();
    }
    if (graph_api == nullptr) { return Reject("graph backend unavailable"); }
    for (std::uint32_t run = 0; run < options.warmup_runs; ++run) {
      auto status = Ordinary(VLAFORGE_REPLAY_PHASE_WARMUP);
      if (status.code != VLAFORGE_STATUS_OK) { return status; }
    }
    auto status = Prepare();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    status = Drain();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    VLAForgeExecutionContextView view{};
    view.struct_size = sizeof(view);
    status = vlaforge_execution_context_get_view(context, &view);
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    status = Save(graph_api->create(&view, &graph));
    if (status.code != VLAFORGE_STATUS_OK) {
      if (graph != nullptr) { graph_api->destroy(graph, 0u); graph = nullptr; }
      return Reject(error.data());
    }
    if (graph == nullptr) { return Reject("graph backend returned a null executable"); }
    status = Save(graph_api->begin(graph));
    if (status.code != VLAFORGE_STATUS_OK) {
      // A begin failure can leave allocator/generator state partially changed.
      return Poison("graph capture begin failed; process restart required");
    }
    status = Body(VLAFORGE_REPLAY_PHASE_CAPTURE);
    if (status.code != VLAFORGE_STATUS_OK) {
      const auto aborted = graph_api->abort(graph);
      if (aborted.code != VLAFORGE_STATUS_OK) {
        std::array<char, 512> detail{};
        // Body() already owns the primary message; abort may replace provider diagnostics.
        std::snprintf(detail.data(), detail.size(),
                      "graph capture abort failed; allocator state may be poisoned; exit process; "
                      "capture: %.*s; abort: %.*s",
                      static_cast<int>(status.message_size > 160 ? 160 : status.message_size),
                      status.message == nullptr ? "unknown" : status.message,
                      static_cast<int>(aborted.message_size > 160 ? 160 : aborted.message_size),
                      aborted.message == nullptr ? "unknown" : aborted.message);
        return Poison(detail.data());
      }
      graph_api->destroy(graph, 0u);
      graph = nullptr;
      const auto drained = Drain();
      if (drained.code != VLAFORGE_STATUS_OK) { return drained; }
      // Only an explicit unsupported-operation status authorizes fallback.
      if (status.code != VLAFORGE_STATUS_FAILED_PRECONDITION) { return status; }
      return Reject(status.message);
    }
    status = graph_api->end(graph);
    if (status.code != VLAFORGE_STATUS_OK) {
      return Poison("graph capture end failed; allocator state may be poisoned; exit process");
    }
    status = Drain();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    state = VLAFORGE_REPLAY_READY;
    return vlaforge_status_ok();
  }
};

extern "C" VLAForgeStatus vlaforge_bounded_replay_create(
    VLAForgeExecutionContext* context, const VLAForgeGraphBackendApi* graph_api,
    const VLAForgeBoundedReplayOptions* options, VLAForgeBoundedReplay** output) {
  if (output == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "replay output is null");
  }
  *output = nullptr;
  auto status = vlaforge_execution_context_status(context);
  if (status.code != VLAFORGE_STATUS_OK) { return status; }
  if (options == nullptr || options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_BOUNDED_REPLAY_ABI_VERSION ||
      options->steps == 0 || options->steps > 65536u ||
      options->mode < VLAFORGE_REPLAY_ORDINARY || options->mode > VLAFORGE_REPLAY_REQUIRE ||
      (options->mode != VLAFORGE_REPLAY_ORDINARY && options->warmup_runs == 0) ||
      options->prepare == nullptr || options->step == nullptr ||
      (options->region_count != 0 && options->regions == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "invalid bounded replay options");
  }
  if (graph_api != nullptr &&
      (graph_api->struct_size < sizeof(*graph_api) ||
       graph_api->abi_version != VLAFORGE_GRAPH_BACKEND_ABI_VERSION ||
       graph_api->create == nullptr || graph_api->begin == nullptr ||
       graph_api->end == nullptr || graph_api->abort == nullptr ||
       graph_api->launch == nullptr || graph_api->destroy == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI, "invalid graph backend ABI");
  }
  try {
    auto replay = std::make_unique<VLAForgeBoundedReplay>();
    replay->context = context;
    replay->graph_api = graph_api;
    replay->options = *options;
    if (options->capture_rejection_reason != nullptr) {
      std::snprintf(replay->reason.data(), replay->reason.size(), "%s",
                    options->capture_rejection_reason);
    }
    VLAForgeExecutionContextView view{};
    view.struct_size = sizeof(view);
    status = vlaforge_execution_context_get_view(context, &view);
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    for (std::uint32_t i = 0; i < options->region_count; ++i) {
      const auto& region = options->regions[i];
      if (region.executable == nullptr ||
          vlaforge_region_executable_value_api_validate(region.api).code != VLAFORGE_STATUS_OK ||
          vlaforge_region_execution_extension_api_validate(region.execution_extension).code != VLAFORGE_STATUS_OK) {
        return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "invalid replay Region registration");
      }
      status = region.execution_extension->bind_context(region.executable, &view);
      if (status.code != VLAFORGE_STATUS_OK) { return status; }
      replay->regions.push_back(region);
    }
    *output = replay.release();
    return vlaforge_status_ok();
  } catch (...) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY, "bounded replay allocation failed");
  }
}

extern "C" VLAForgeStatus vlaforge_bounded_replay_run(VLAForgeBoundedReplay* replay) {
  if (replay == nullptr || replay->running) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "invalid or concurrent replay run");
  }
  auto status = vlaforge_execution_context_status(replay->context);
  if (status.code != VLAFORGE_STATUS_OK) { return status; }
  replay->running = true;
  struct RunGuard {
    bool& running;
    ~RunGuard() { running = false; }
  } guard{replay->running};
  if (replay->options.mode != VLAFORGE_REPLAY_ORDINARY &&
      replay->state == VLAFORGE_REPLAY_UNPREPARED) {
    status = replay->Capture();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
  }
  if (replay->state == VLAFORGE_REPLAY_READY) {
    status = replay->Prepare();
    if (status.code != VLAFORGE_STATUS_OK) { return status; }
    status = replay->graph_api->launch(replay->graph);
    if (status.code != VLAFORGE_STATUS_OK) {
      return replay->Poison("graph launch failed; no ordinary fallback; restart process");
    }
    status = replay->Drain();
    if (status.code == VLAFORGE_STATUS_OK) { ++replay->replay_count; }
    return status;
  }
  if (replay->options.mode == VLAFORGE_REPLAY_REQUIRE) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, replay->reason.data());
  }
  status = replay->Ordinary(VLAFORGE_REPLAY_PHASE_ORDINARY);
  if (status.code == VLAFORGE_STATUS_OK) { ++replay->ordinary_count; }
  return status;
}

extern "C" VLAForgeStatus vlaforge_bounded_replay_get_info(
    const VLAForgeBoundedReplay* replay, VLAForgeBoundedReplayInfo* info) {
  if (replay == nullptr || info == nullptr || info->struct_size < sizeof(*info)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "invalid replay info");
  }
  *info = {sizeof(*info), replay->state,
            replay->state == VLAFORGE_REPLAY_READY ? replay->options.steps : 0u,
            replay->replay_count, replay->ordinary_count, replay->reason.data()};
  return vlaforge_status_ok();
}

extern "C" void vlaforge_bounded_replay_destroy(VLAForgeBoundedReplay* replay) {
  if (replay == nullptr) { return; }
  if (replay->state != VLAFORGE_REPLAY_POISONED) { (void)replay->Drain(); }
  if (replay->graph != nullptr) {
    replay->graph_api->destroy(replay->graph,
        replay->state == VLAFORGE_REPLAY_POISONED ? 1u : 0u);
  }
  delete replay;
}
