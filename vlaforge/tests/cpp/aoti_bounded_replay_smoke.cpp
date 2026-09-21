#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
#include "vlaforge/backends/libtorch_graph.h"
#include "vlaforge/backends/torchscript_region_executable.h"
#include <torch/script.h>
#else
#include "vlaforge/backends/aoti_graph.h"
#include "vlaforge/backends/aoti_region_executable.h"
#endif

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
const auto* RegionApi() { return vlaforge_torchscript_region_executable_value_api(); }
const auto* ExtensionApi() { return vlaforge_torchscript_region_execution_extension_api(); }
const auto* GraphApi() { return vlaforge_libtorch_graph_backend_api(); }
constexpr char kVariant[] = "torchscript-aten-context/1";
#else
const auto* RegionApi() { return vlaforge_aoti_region_executable_value_api(); }
const auto* ExtensionApi() { return vlaforge_aoti_region_execution_extension_api(); }
const auto* GraphApi() { return vlaforge_aoti_graph_backend_api(); }
constexpr char kVariant[] = "";
#endif

void Check(VLAForgeStatus status) {
  if (status.code != VLAFORGE_STATUS_OK) {
    throw std::runtime_error(std::string(status.message, status.message_size));
  }
}
void Require(bool value, const char* message) {
  if (!value) { throw std::runtime_error(message); }
}

VLAForgeValueView View(at::Tensor& tensor) {
  VLAForgeValueView view{};
  view.struct_size = sizeof(view);
  view.kind = VLAFORGE_VALUE_TENSOR;
  view.value.tensor = {
      sizeof(VLAForgeBoundTensor),
      {tensor.data_ptr(), static_cast<std::uint64_t>(tensor.nbytes()),
       tensor.sizes().data(), static_cast<std::uint32_t>(tensor.dim()),
       VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, tensor.get_device()}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 4u};
  return view;
}

struct Loop {
  const VLAForgeRegionExecutableValueApi* api = RegionApi();
  const VLAForgeRegionExecutionExtensionApi* extension = ExtensionApi();
  std::unique_ptr<VLAForgeExecutionContext, decltype(&vlaforge_execution_context_destroy)>
      context{nullptr, &vlaforge_execution_context_destroy};
  VLAForgeExecutionContextView context_view{};
  using Region = std::unique_ptr<VLAForgeRegionExecutable, VLAForgeRegionDestroyFn>;
  std::array<Region, 2> regions{Region(nullptr, api->destroy), Region(nullptr, api->destroy)};
  std::array<VLAForgeReplayRegion, 2> registrations{};
  at::Tensor seed, carry, intermediate, output, gain;
  VLAForgeValueView seed_view{}, carry_view{}, intermediate_view{}, output_view{}, gain_view{};
  std::array<int, 3> steps{};
  int prepares = 0;
  int pending_checks = 0;
  bool reject_capture = false;
  bool invalidate_capture = false;

  explicit Loop(const std::string& artifact_path, const std::string& target) {
    VLAForgeExecutionContext* raw = nullptr;
    VLAForgeExecutionContextOptions options{
        sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION, {VLAFORGE_DEVICE_CUDA, 0}};
    Check(vlaforge_execution_context_create(&options, &raw));
    context.reset(raw);
    context_view.struct_size = sizeof(context_view);
    Check(vlaforge_execution_context_get_view(context.get(), &context_view));
    Require(context_view.native_stream != c10::cuda::getDefaultCUDAStream(0).stream(),
            "default stream is not a runtime-owned replay stream");
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      VLAForgeRegionExecutable* executable = nullptr;
      VLAForgeRegionCreateOptions create{
          sizeof(create), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION, i,
          {VLAFORGE_DEVICE_CUDA, 0}};
      Check(api->create(&create, &executable));
      regions[i].reset(executable);
      Check(extension->bind_context(executable, &context_view));
      const VLAForgeArtifactDescriptor artifact{
          sizeof(artifact), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
          artifact_path.data(), artifact_path.size(), nullptr, 0,
          nullptr, 0, target.data(), target.size(), kVariant, sizeof(kVariant) - 1u};
      Check(api->load(executable, &artifact));
      registrations[i] = {executable, api, extension};
    }
    const auto tensor_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    seed = at::zeros({4, 4}, tensor_options);
    carry = at::empty_like(seed);
    intermediate = at::empty_like(seed);
    output = at::empty_like(seed);
    gain = at::full({}, 0.1, tensor_options);
    seed_view = View(seed);
    carry_view = View(carry);
    intermediate_view = View(intermediate);
    output_view = View(output);
    gain_view = View(gain);
    Check(api->bind_input(regions[0].get(), 0, &carry_view));
    Check(api->bind_input(regions[0].get(), 1, &gain_view));
    Check(api->bind_output(regions[0].get(), 0, &intermediate_view));
    Check(api->bind_input(regions[1].get(), 0, &intermediate_view));
    Check(api->bind_input(regions[1].get(), 1, &gain_view));
    Check(api->bind_output(regions[1].get(), 0, &output_view));
    c10::cuda::device_synchronize();
  }

  static VLAForgeStatus Prepare(void* pointer) {
    auto& loop = *static_cast<Loop*>(pointer);
    ++loop.prepares;
    return vlaforge_execution_context_copy(loop.context.get(),
        &loop.carry_view.value.tensor.tensor, &loop.seed_view.value.tensor.tensor,
        loop.seed.nbytes());
  }

  static VLAForgeStatus Step(void* pointer, std::uint32_t step, VLAForgeReplayPhase phase) {
    auto& loop = *static_cast<Loop*>(pointer);
    ++loop.steps.at(phase);
    for (auto& region : loop.regions) {
      const auto status = loop.api->run(region.get());
      if (status.code != VLAFORGE_STATUS_OK) { return status; }
      Require(loop.extension->bind_context(region.get(), &loop.context_view).code ==
                  VLAFORGE_STATUS_FAILED_PRECONDITION,
              "pending Region allowed context rebind inside bounded loop");
#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
      Require(loop.api->bind_input(region.get(), 1, &loop.gain_view).code ==
                  VLAFORGE_STATUS_FAILED_PRECONDITION,
              "pending TorchScript Region allowed tensor rebind");
      Require(loop.api->bind_output(region.get(), 0, &loop.output_view).code ==
                  VLAFORGE_STATUS_FAILED_PRECONDITION,
              "pending TorchScript Region allowed output rebind");
      Require(loop.api->load(region.get(), nullptr).code == VLAFORGE_STATUS_FAILED_PRECONDITION,
              "pending TorchScript Region allowed destructive reload");
#endif
      ++loop.pending_checks;
    }
    if (phase == VLAFORGE_REPLAY_PHASE_CAPTURE && step == 1) {
      if (loop.invalidate_capture) {
        const auto error = cudaStreamSynchronize(
            static_cast<cudaStream_t>(loop.context_view.native_stream));
        Require(error != cudaSuccess, "raw synchronize did not invalidate CUDA capture");
        return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "intentional CUDA invalidation");
      }
      if (loop.reject_capture) {
        return loop.api->synchronize(loop.regions[0].get());
      }
    }
    return vlaforge_execution_context_copy(loop.context.get(),
        &loop.carry_view.value.tensor.tensor, &loop.output_view.value.tensor.tensor,
        loop.output.nbytes());
  }

  VLAForgeBoundedReplay* MakeReplay(VLAForgeReplayMode mode, const char* rejection = nullptr) {
    steps = {};
    prepares = pending_checks = 0;
    const VLAForgeBoundedReplayOptions options{
        sizeof(options), VLAFORGE_BOUNDED_REPLAY_ABI_VERSION, mode,
        4, 2, rejection, &Prepare, &Step, this,
        registrations.data(), static_cast<std::uint32_t>(registrations.size())};
    VLAForgeBoundedReplay* replay = nullptr;
    Check(vlaforge_bounded_replay_create(context.get(), GraphApi(),
                                        &options, &replay));
    return replay;
  }

  void UpdateSeed(int run) {
    const auto stream = c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(context_view.native_stream), 0);
    const c10::cuda::CUDAStreamGuard guard(stream);
    seed.copy_(at::arange(16, seed.options()).reshape({4, 4}) / 100.0 + 0.01 * run);
    Check(vlaforge_execution_context_synchronize(context.get()));
  }

  double Validate() {
#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
    auto expected = seed.clone();
    for (int i = 0; i < 8; ++i) {
      expected = (expected.sin() + expected.square()) * gain;
    }
    expected = expected.cpu();
#else
    auto expected = seed.cpu();
    for (int i = 0; i < 8; ++i) {
      expected = (expected.sin() + expected.square()) * 0.1;
    }
#endif
    const auto actual = output.cpu();
    const double error = (actual - expected).abs().max().item<double>();
    Require(at::allclose(actual, expected, 1e-5, 1e-12), "full N-step output mismatch");
#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
    Require(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
            "TorchScript full N-step bytes differ from eager GPU");
#endif
    Require(at::equal(carry.cpu(), actual), "loop carry was not completed");
    for (auto& region : regions) {
      Check(extension->bind_context(region.get(), &context_view));
    }
    return error;
  }
};

using Replay = std::unique_ptr<VLAForgeBoundedReplay, decltype(&vlaforge_bounded_replay_destroy)>;

void Case(Loop& loop, VLAForgeReplayMode mode, const char* rejection, bool callback_rejection) {
  loop.reject_capture = callback_rejection;
  Replay replay(loop.MakeReplay(mode, rejection), &vlaforge_bounded_replay_destroy);
  double max_error = 0;
  for (int run = 0; run < 3; ++run) {
    loop.UpdateSeed(run + 1);
    const auto previous = c10::cuda::getCurrentCUDAStream(0);
    Check(vlaforge_bounded_replay_run(replay.get()));
    Require(previous == c10::cuda::getCurrentCUDAStream(0), "replay leaked current stream");
    max_error = std::max(max_error, loop.Validate());
  }
  VLAForgeBoundedReplayInfo info{};
  info.struct_size = sizeof(info);
  Check(vlaforge_bounded_replay_get_info(replay.get(), &info));
  if (mode == VLAFORGE_REPLAY_ORDINARY || rejection != nullptr || callback_rejection) {
    Require(info.ordinary_count == 3 && info.replay_count == 0 && loop.steps[0] == 12,
            "ordinary fallback invocation mismatch");
    if (mode != VLAFORGE_REPLAY_ORDINARY) {
      Require(info.state == VLAFORGE_REPLAY_FALLBACK && info.reason[0] != '\0',
              "fallback status or reason missing");
    }
    if (rejection != nullptr) {
      Require(loop.steps[1] == 0 && loop.steps[2] == 0, "preflight fallback attempted capture");
    }
  } else {
    Require(info.state == VLAFORGE_REPLAY_READY && info.captured_steps == 4 &&
                info.replay_count == 3 && info.ordinary_count == 0 &&
                loop.steps[0] == 0 && loop.steps[1] == 8 && loop.steps[2] == 4,
            "graph did not capture one complete four-step loop");
  }
  std::printf("BOUNDED_REPLAY mode=%d callback_rejection=%d state=%d capture_steps=%u "
              "warmup_calls=%d capture_calls=%d ordinary_calls=%d replays=%llu "
              "pending_checks=%d max_abs=%.12g reason=%s\n",
              mode, callback_rejection, info.state, info.captured_steps,
              loop.steps[1], loop.steps[2], loop.steps[0],
              static_cast<unsigned long long>(info.replay_count), loop.pending_checks,
              max_error, info.reason);
}

int Run(const std::string& path, const std::string& target, bool fatal) {
  const c10::cuda::CUDAGuard guard(0);
  Loop loop(path, target);
  if (fatal) {
    loop.UpdateSeed(1);
    loop.invalidate_capture = true;
    auto* replay = loop.MakeReplay(VLAFORGE_REPLAY_PREFER);
    const auto status = vlaforge_bounded_replay_run(replay);
    VLAForgeBoundedReplayInfo info{};
    info.struct_size = sizeof(info);
    Check(vlaforge_bounded_replay_get_info(replay, &info));
    const bool rejected = status.code != VLAFORGE_STATUS_OK &&
        info.state == VLAFORGE_REPLAY_POISONED && info.ordinary_count == 0 &&
        vlaforge_execution_context_status(loop.context.get()).code == VLAFORGE_STATUS_FAILED_PRECONDITION &&
        vlaforge_execution_context_copy(loop.context.get(), nullptr, nullptr, 0).code == VLAFORGE_STATUS_FAILED_PRECONDITION &&
        vlaforge_bounded_replay_run(replay).code == VLAFORGE_STATUS_FAILED_PRECONDITION;
    std::printf("BOUNDED_REPLAY fatal_poisoned=%d ordinary_count=%llu reason=%s\n", rejected,
                static_cast<unsigned long long>(info.ordinary_count), info.reason);
    // Fatal LibTorch capture failure has no supported pool cleanup. Destroy
    // runtime wrappers, quarantine native resources, then terminate without
    // invoking tensor/Region/static allocator destructors.
    vlaforge_bounded_replay_destroy(replay);
    vlaforge_execution_context_destroy(loop.context.release());
    std::fflush(stdout);
    std::_Exit(rejected ? 78 : 1);
  }
  Case(loop, VLAFORGE_REPLAY_ORDINARY, nullptr, false);
  Case(loop, VLAFORGE_REPLAY_PREFER, "preflight: dynamic scheduler is not capturable", false);
  Case(loop, VLAFORGE_REPLAY_REQUIRE, nullptr, false);
  Case(loop, VLAFORGE_REPLAY_PREFER, nullptr, true);
  // New graphs after valid partial-capture cleanup and graph destruction must
  // remain executable with the same context, Regions, weights and addresses.
  Case(loop, VLAFORGE_REPLAY_REQUIRE, nullptr, false);
  Case(loop, VLAFORGE_REPLAY_ORDINARY, nullptr, false);
#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
  std::puts("BOUNDED_REPLAY real TorchScript CUDA two-region N4 audit passed, full eager bytes exact");
#else
  std::puts("BOUNDED_REPLAY real AOTI CUDA two-region N4 audit passed");
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 || (std::string(argv[3]) != "success" && std::string(argv[3]) != "fatal")) {
    std::fprintf(stderr, "usage: %s AUDIT_ARTIFACT TARGET success|fatal\n", argv[0]);
    return 2;
  }
  try {
#ifdef VLAFORGE_REPLAY_TEST_TORCHSCRIPT
    torch::jit::Module module("ReplayProbe");
    module.define("def forward(self, values, gain):\n  return (values.sin() + values.square()) * gain\n");
    module.save(argv[1]);
#endif
    return Run(argv[1], argv[2], std::string(argv[3]) == "fatal");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "AOTI bounded replay smoke: %s\n", error.what());
    return 1;
  }
}
