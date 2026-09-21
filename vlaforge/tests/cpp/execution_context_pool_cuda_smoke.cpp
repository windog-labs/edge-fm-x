#include "vlaforge/runtime/execution_context.h"

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>
#include <set>
#include <vector>

int main() {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const c10::cuda::CUDAGuard guard(0);
  const VLAForgeExecutionContextOptions options{
      sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {VLAFORGE_DEVICE_CUDA, 0}};
  for (const int simultaneous : {1, 3}) {
    std::int64_t first_allocated = 0, first_reserved = 0;
    for (int cycle = 0; cycle < 5; ++cycle) {
      std::vector<VLAForgeExecutionContext*> owners;
      std::set<void*> active_streams;
      for (int index = 0; index < simultaneous; ++index) {
        VLAForgeExecutionContext* context = nullptr;
        if (vlaforge_execution_context_create(&options, &context).code != VLAFORGE_STATUS_OK) return 2;
        owners.push_back(context);
        VLAForgeExecutionContextView view{};
        view.struct_size = sizeof(view);
        if (vlaforge_execution_context_get_view(context, &view).code != VLAFORGE_STATUS_OK) return 3;
        if (!active_streams.insert(view.native_stream).second) return 4;
        {
          const c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromExternal(
              static_cast<cudaStream_t>(view.native_stream), 0));
          auto input = at::ones({32, 32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
          auto output = at::mm(input, input);
          if (!output.eq(32).all().item<bool>()) return 5;
          // Exercise both official stream-keyed caches even when GEMM selects
          // only one of them for this small shape.
          (void)at::cuda::getCurrentCUDABlasHandle();
          (void)at::cuda::getCUDABlasLtWorkspace();
        }
      }
      for (auto* context : owners) {
        if (vlaforge_execution_context_synchronize(context).code != VLAFORGE_STATUS_OK) return 6;
        vlaforge_execution_context_destroy(context);
      }
      if (cudaDeviceSynchronize() != cudaSuccess) return 7;
      const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
      const auto allocated = stats.allocated_bytes[0].current;
      const auto reserved = stats.reserved_bytes[0].current;
      std::printf("{\"simultaneous\":%d,\"cycle\":%d,\"allocated_bytes\":%lld,\"reserved_bytes\":%lld}\n",
                  simultaneous, cycle, static_cast<long long>(allocated), static_cast<long long>(reserved));
      if (cycle == 0) {
        first_allocated = allocated;
        first_reserved = reserved;
      } else if (allocated != first_allocated || reserved != first_reserved) {
        return 9;
      }
    }
  }
  return 0;
}
