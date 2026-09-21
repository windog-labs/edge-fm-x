#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#include "vlaforge/runtime/device_copy.h"
#include "vlaforge/runtime/static_arena.h"

namespace {
constexpr VLAForgeDevice kCpu{VLAFORGE_DEVICE_CPU, 0};
constexpr VLAForgeDevice kCuda{VLAFORGE_DEVICE_CUDA, 0};

void CUDART_CB Delay(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
}

bool TestCopy(bool host_source, void* source, void* destination,
              cudaStream_t consumer, std::int32_t* observed) {
  const std::int32_t expected = host_source ? 71 : 37;
  if (cudaLaunchHostFunc(nullptr, &Delay, nullptr) != cudaSuccess) return false;
  const auto status = vlaforge::runtime::CopyBytes(
      destination, kCuda, host_source ? &expected : source,
      host_source ? kCpu : kCuda, sizeof(expected));
  const auto completion = cudaStreamQuery(nullptr);
  if (!status.ok() || completion != cudaSuccess) {
    std::cerr << "copy returned before completion: " << status.message
              << " stream=" << static_cast<int>(completion) << '\n';
    (void)cudaStreamSynchronize(nullptr);
    return false;
  }
  return cudaMemcpyAsync(observed, destination, sizeof(expected),
                         cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
      cudaStreamSynchronize(consumer) == cudaSuccess && *observed == expected;
}
}  // namespace

int main() {
  using vlaforge::runtime::StaticArena;
  StaticArena source(sizeof(std::int32_t), 64, kCuda);
  StaticArena destination(sizeof(std::int32_t), 64, kCuda);
  const std::int32_t initial = 37;
  if (cudaMemcpy(source.data(), &initial, sizeof(initial),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaStreamSynchronize(nullptr) != cudaSuccess) return 1;
  cudaStream_t consumer = nullptr;
  std::int32_t* observed = nullptr;
  if (cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) != cudaSuccess ||
      cudaMallocHost(reinterpret_cast<void**>(&observed), sizeof(*observed)) !=
          cudaSuccess) return 2;
  const bool passed = TestCopy(false, source.data(), destination.data(),
                               consumer, observed) &&
      TestCopy(true, source.data(), destination.data(), consumer, observed);
  (void)cudaStreamDestroy(consumer);
  (void)cudaFreeHost(observed);
  return passed ? 0 : 3;
}
