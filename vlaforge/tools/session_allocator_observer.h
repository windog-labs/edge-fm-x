#ifndef VLAFORGE_TOOLS_SESSION_ALLOCATOR_OBSERVER_H_
#define VLAFORGE_TOOLS_SESSION_ALLOCATOR_OBSERVER_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/version.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <string>

namespace vlaforge_benchmark {

// Diagnostic snapshots only: no counter resets, emptyCache, or allocator changes.
inline bool ObserveAllocator(const std::string& root, const char* phase, int ordinal) noexcept {
  try {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
    namespace allocator = c10::cuda::CUDACachingAllocator;
    if (allocator::name() != "native") return false;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return false;
    std::ofstream out(root + "/allocator-snapshots.jsonl", std::ios::app);
    out << "{\"schema\":\"vlaforge.libtorch_allocator_snapshot/1\",\"phase\":\"" << phase
        << "\",\"torch_release\":\"" << TORCH_VERSION
        << "\",\"allocator\":\"native\",\"ordinal\":" << ordinal
        << ",\"device_free_bytes\":" << free_bytes << ",\"device_total_bytes\":" << total_bytes;
    const bool initialized = allocator::get()->initialized();
    out << ",\"initialized\":" << (initialized ? "true" : "false");
    if (!initialized) {
      out << ",\"counters\":null}\n";
      return out.good();
    }
    const auto stats = allocator::getDeviceStats(ordinal);
    out << ",\"counters\":{";
    bool first = true;
    const auto stat = [&](const char* name, const auto& values) {
      const auto& value = values[0];
      if (!first) out << ',';
      first = false;
      out << '\"' << name << "\":{\"current\":" << value.current
          << ",\"peak\":" << value.peak << ",\"allocated\":" << value.allocated
          << ",\"freed\":" << value.freed << '}';
    };
    stat("allocation", stats.allocation);
    stat("segment", stats.segment);
    stat("active", stats.active);
    stat("allocated_bytes", stats.allocated_bytes);
    stat("reserved_bytes", stats.reserved_bytes);
    stat("active_bytes", stats.active_bytes);
    stat("requested_bytes", stats.requested_bytes);
    out << ",\"num_alloc_retries\":" << stats.num_alloc_retries
        << ",\"num_ooms\":" << stats.num_ooms
        << ",\"num_sync_all_streams\":" << stats.num_sync_all_streams
        << ",\"num_device_alloc\":" << stats.num_device_alloc
        << ",\"num_device_free\":" << stats.num_device_free << "}}\n";
    return out.good();
#else
    (void)root; (void)phase; (void)ordinal;
    return false;
#endif
  } catch (...) {
    return false;
  }
}

}  // namespace vlaforge_benchmark
#endif
