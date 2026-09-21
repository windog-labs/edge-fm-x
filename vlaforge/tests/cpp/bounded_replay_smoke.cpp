#include "vlaforge/runtime/bounded_replay.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

struct VLAForgeGraphExecutable {};

namespace {

void Require(bool value, const char* message) {
  if (!value) { throw std::runtime_error(message); }
}

void Check(VLAForgeStatus status) {
  Require(status.code == VLAFORGE_STATUS_OK,
          status.message == nullptr ? "unexpected error" : status.message);
}

struct Counters {
  std::array<int, 3> steps{};
  int prepares = 0;
  int creates = 0;
  int begins = 0;
  int ends = 0;
  int aborts = 0;
  int launches = 0;
  int destroys = 0;
  int fail = 0;
  VLAForgeBoundedReplay* replay = nullptr;
  VLAForgeExecutionContext* context = nullptr;
} counts;

VLAForgeStatus Error(const char* message) {
  return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, message);
}

VLAForgeStatus Prepare(void*) {
  ++counts.prepares;
  return counts.fail == 1 ? Error("prepare failed") : vlaforge_status_ok();
}

VLAForgeStatus Step(void*, std::uint32_t step, VLAForgeReplayPhase phase) {
  ++counts.steps.at(phase);
  if (counts.fail == 8 && phase == VLAFORGE_REPLAY_PHASE_CAPTURE) {
    vlaforge_execution_context_poison(counts.context);
  }
  Require(vlaforge_bounded_replay_run(counts.replay).code ==
              VLAFORGE_STATUS_FAILED_PRECONDITION, "recursive run accepted");
  if (counts.fail == 2 && step == 1) { throw std::runtime_error("step threw"); }
  if ((counts.fail == 3 || counts.fail == 6) && phase == VLAFORGE_REPLAY_PHASE_CAPTURE) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "capture operation unsupported");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Create(const VLAForgeExecutionContextView*, VLAForgeGraphExecutable** graph) {
  ++counts.creates;
  *graph = new VLAForgeGraphExecutable();
  return vlaforge_status_ok();
}
VLAForgeStatus Begin(VLAForgeGraphExecutable*) {
  ++counts.begins;
  return counts.fail == 4 ? Error("begin failed") : vlaforge_status_ok();
}
VLAForgeStatus End(VLAForgeGraphExecutable*) {
  ++counts.ends;
  return counts.fail == 5 ? Error("end failed") : vlaforge_status_ok();
}
VLAForgeStatus Abort(VLAForgeGraphExecutable*) {
  ++counts.aborts;
  return counts.fail == 6 ? Error("abort failed") : vlaforge_status_ok();
}
VLAForgeStatus Launch(VLAForgeGraphExecutable*) {
  ++counts.launches;
  return counts.fail == 7 ? Error("launch failed") : vlaforge_status_ok();
}
void Destroy(VLAForgeGraphExecutable* graph, std::uint32_t poisoned) {
  Require((poisoned != 0u) == (counts.fail >= 4), "graph destruction poison flag mismatch");
  ++counts.destroys;
  delete graph;
}
const VLAForgeGraphBackendApi kStateOnlyBackend = {
    sizeof(kStateOnlyBackend), VLAFORGE_GRAPH_BACKEND_ABI_VERSION,
    &Create, &Begin, &End, &Abort, &Launch, &Destroy};

void RunCase(VLAForgeReplayMode mode, const VLAForgeGraphBackendApi* backend,
             const char* rejection, int failure) {
  counts = {};
  counts.fail = failure;
  const VLAForgeExecutionContextOptions context_options{
      sizeof(context_options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {VLAFORGE_DEVICE_CPU, 0}};
  VLAForgeExecutionContext* raw_context = nullptr;
  Check(vlaforge_execution_context_create(&context_options, &raw_context));
  std::unique_ptr<VLAForgeExecutionContext, decltype(&vlaforge_execution_context_destroy)>
      context(raw_context, vlaforge_execution_context_destroy);
  counts.context = context.get();
  const VLAForgeBoundedReplayOptions options{
      sizeof(options), VLAFORGE_BOUNDED_REPLAY_ABI_VERSION,
      mode, 4, 2, rejection, &Prepare, &Step, nullptr, nullptr, 0};
  VLAForgeBoundedReplay* replay = nullptr;
  Check(vlaforge_bounded_replay_create(context.get(), backend, &options, &replay));
  counts.replay = replay;
  const auto status = vlaforge_bounded_replay_run(replay);
  VLAForgeBoundedReplayInfo info{};
  info.struct_size = sizeof(info);
  Check(vlaforge_bounded_replay_get_info(replay, &info));
  if (failure >= 4) {
    Require(status.code == VLAFORGE_STATUS_BACKEND_ERROR &&
                info.state == VLAFORGE_REPLAY_POISONED, "fatal graph error not poisoned");
    if (failure == 6) {
      Require(std::strstr(info.reason, "capture operation unsupported") != nullptr &&
                  std::strstr(info.reason, "abort failed") != nullptr,
              "capture cause lost when graph abort also fails");
    }
    Require(vlaforge_execution_context_status(context.get()).code ==
                VLAFORGE_STATUS_FAILED_PRECONDITION, "poisoned context still usable");
    Require(vlaforge_execution_context_copy(context.get(), nullptr, nullptr, 0).code ==
                VLAFORGE_STATUS_FAILED_PRECONDITION, "poisoned copy accepted");
    Require(vlaforge_bounded_replay_run(replay).code ==
                VLAFORGE_STATUS_FAILED_PRECONDITION, "poisoned replay retried");
  } else if (failure == 1 || failure == 2) {
    Require(status.code == VLAFORGE_STATUS_BACKEND_ERROR, "callback error hidden");
    Check(vlaforge_execution_context_status(context.get()));
  } else if (mode == VLAFORGE_REPLAY_REQUIRE && (rejection != nullptr || backend == nullptr)) {
    Require(status.code == VLAFORGE_STATUS_FAILED_PRECONDITION, "required graph fell back");
    Require(counts.prepares == 0 && counts.creates == 0, "preflight rejection executed work");
  } else {
    Check(status);
    Check(vlaforge_bounded_replay_run(replay));
    Check(vlaforge_bounded_replay_get_info(replay, &info));
    if (mode == VLAFORGE_REPLAY_ORDINARY || rejection != nullptr || backend == nullptr || failure == 3) {
      Require(info.ordinary_count == 2 && counts.steps[0] == 8, "ordinary step count mismatch");
      if (mode != VLAFORGE_REPLAY_ORDINARY) {
        Require(info.state == VLAFORGE_REPLAY_FALLBACK && std::strlen(info.reason) != 0,
                "fallback reason missing");
      }
      if (failure == 3) { Require(counts.aborts == 1, "valid partial graph not aborted"); }
    } else {
      Require(info.state == VLAFORGE_REPLAY_READY && info.captured_steps == 4 &&
                  info.replay_count == 2 && counts.prepares == 5 &&
                  counts.steps[1] == 8 && counts.steps[2] == 4 &&
                  counts.creates == 1 && counts.begins == 1 && counts.ends == 1 &&
                  counts.launches == 2, "full loop capture lifecycle mismatch");
    }
  }
  vlaforge_bounded_replay_destroy(replay);
  Require(counts.destroys == counts.creates, "graph not destroyed exactly once");
}

void CopyCase() {
  VLAForgeExecutionContext* context = nullptr;
  VLAForgeExecutionContextOptions options{
      sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION, {VLAFORGE_DEVICE_CPU, 0}};
  Check(vlaforge_execution_context_create(&options, &context));
  std::array<float, 4> source{1, 2, 3, 4}, destination{};
  VLAForgeTensorView src{source.data(), sizeof(source), nullptr, 0, VLAFORGE_DTYPE_F32,
                         {VLAFORGE_DEVICE_CPU, 0}};
  VLAForgeTensorView dst{destination.data(), sizeof(destination), nullptr, 0, VLAFORGE_DTYPE_F32,
                         {VLAFORGE_DEVICE_CPU, 0}};
  Check(vlaforge_execution_context_copy(context, &dst, &src, sizeof(source)));
  Require(source == destination, "CPU context copy mismatch");
  Require(vlaforge_execution_context_copy(context, &dst, &src, sizeof(source) + 1).code ==
              VLAFORGE_STATUS_INVALID_ARGUMENT, "oversized copy accepted");
  dst.data = source.data() + 1;
  Require(vlaforge_execution_context_copy(context, &dst, &src, sizeof(float) * 2).code ==
              VLAFORGE_STATUS_INVALID_ARGUMENT, "overlapping copy accepted");
  vlaforge_execution_context_poison(context);
  VLAForgeExecutionContextView view{};
  view.struct_size = sizeof(view);
  Require(vlaforge_execution_context_get_view(context, &view).code ==
              VLAFORGE_STATUS_FAILED_PRECONDITION, "poisoned view exported");
  vlaforge_execution_context_destroy(context);
}

}  // namespace

int main() {
  try {
    CopyCase();
    RunCase(VLAFORGE_REPLAY_ORDINARY, nullptr, nullptr, 0);
    RunCase(VLAFORGE_REPLAY_PREFER, nullptr, nullptr, 0);
    RunCase(VLAFORGE_REPLAY_REQUIRE, nullptr, nullptr, 0);
    RunCase(VLAFORGE_REPLAY_PREFER, &kStateOnlyBackend, "dynamic boundary", 0);
    RunCase(VLAFORGE_REPLAY_REQUIRE, &kStateOnlyBackend, "dynamic boundary", 0);
    for (int failure = 0; failure <= 8; ++failure) {
      RunCase(VLAFORGE_REPLAY_PREFER, &kStateOnlyBackend, nullptr, failure);
    }
    std::puts("BOUNDED_REPLAY state-machine and CPU-copy tests passed (no CUDA claims)");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "bounded replay smoke: %s\n", error.what());
    return 1;
  }
}
