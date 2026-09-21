#include "vlaforge/backends/libtorch_rng.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

bool Exact(const at::Tensor& left, const at::Tensor& right) {
  const auto a = left.cpu().contiguous();
  const auto b = right.cpu().contiguous();
  return a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
         std::memcmp(a.data_ptr(), b.data_ptr(), a.nbytes()) == 0;
}

void FileExact(const std::string& path, const at::Tensor& value) {
  const auto cpu = value.cpu().contiguous();
  std::ifstream stream(path, std::ios::binary);
  std::vector<char> bytes(cpu.nbytes());
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  Require(stream.gcount() == static_cast<std::streamsize>(bytes.size()) &&
          stream.peek() == std::ifstream::traits_type::eof(), "reference size differs");
  Require(std::memcmp(cpu.data_ptr(), bytes.data(), bytes.size()) == 0,
          "complete reference bytes differ");
}
}

int main(int argc, char** argv) {
  try {
    Require(argc == 2, "expected reference input root");
    const c10::cuda::CUDAGuard guard(0);
    auto global = at::cuda::detail::getDefaultCUDAGenerator(0);
    const std::vector<std::uint64_t> seeds = {42, 43, 0, std::numeric_limits<std::uint64_t>::max()};
    for (const auto seed : seeds) {
      global.set_current_seed(seed);
      vlaforge::CudaRngProvider provider(0, seed);
      Require(Exact(provider.State(), global.get_state()), "initial state differs");
      for (int request = 0; request < 3; ++request) {
        const auto global_before = global.get_state().clone();
        std::vector<at::Tensor> states{provider.State()};
        const auto initial = provider.Normal({1, 16, 7});
        states.push_back(provider.State());
        const auto template_tensor = at::cat({initial, initial}, 0);
        std::vector<at::Tensor> steps;
        for (int step = 0; step < 10; ++step) {
          steps.push_back(provider.NormalLike(template_tensor));
          states.push_back(provider.State());
        }
        Require(Exact(global_before, global.get_state()), "provider altered default RNG");
        const auto expected_initial = at::randn({1, 16, 7},
            at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
        Require(Exact(initial, expected_initial), "initial draw differs");
        Require(Exact(states[1], global.get_state()), "state after initial draw differs");
        const auto expected_template = at::cat({expected_initial, expected_initial}, 0);
        for (int step = 0; step < 10; ++step) {
          Require(Exact(steps[step], at::randn_like(expected_template)), "step draw differs");
          Require(Exact(states[step + 2], global.get_state()), "step state differs");
        }
        if (request == 0 && (seed == 42 || seed == 43)) {
          const auto root = std::string(argv[1]) + (seed == 42 ? "/sample-0/" : "/sample-1/");
          FileExact(root + "initial_noise.bin", initial);
          FileExact(root + "step_noise.bin", at::stack(steps));
          FileExact(root + "rng_states.bin", at::stack(states));
          FileExact(root + "rng_before.bin", states[0]);
        }
      }
    }
    const auto global_before = global.get_state().clone();
    vlaforge::CudaRngProvider first(0, 42), second(0, 43), control(0, 42);
    const auto saved = first.State();
    const auto start = first.Normal({3, 11});
    Require(Exact(start, control.Normal({3, 11})), "first draw differs");
    const auto middle = first.State();
    second.Normal({257, 13});
    Require(Exact(first.Normal({17}), control.Normal({17})), "interleaved provider affected state");
    first.Restore(middle);
    control.Restore(middle);
    Require(Exact(first.Normal({17}), control.Normal({17})), "restore differs");
    first.Reset(42);
    Require(Exact(first.State(), saved), "seed reset differs");
    Require(Exact(first.Normal({3, 11}), start), "seed reset draw differs");
    auto detached = first.State();
    detached.zero_();
    Require(!Exact(first.State(), detached), "snapshot aliases provider state");
    const auto unchanged = first.State();
    for (const auto& invalid : {at::zeros({1}, at::kByte),
                               at::zeros({16}, at::kFloat),
                               at::zeros({2, 8}, at::kByte)}) {
      bool rejected = false;
      try { first.Restore(invalid); } catch (const c10::Error&) { rejected = true; }
      Require(rejected && Exact(first.State(), unchanged), "invalid restore changed state");
    }
    bool rejected = false;
    try { first.NormalLike(at::zeros({3})); } catch (const c10::Error&) { rejected = true; }
    Require(rejected && Exact(first.State(), unchanged), "invalid draw changed state");
    Require(Exact(global_before, global.get_state()), "lifecycle changed default RNG");
    Require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution failed");
    std::cout << "private_rng_passed: four seeds, three consecutive requests, all draws/states, "
                 "frozen Python tapes, reset/restore/isolation/rejection\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
