#pragma once
namespace c10::cuda::CUDACachingAllocator {
inline void releasePool(int device, std::pair<int, int>) {
  ++release_calls;
  if (device != current_device || fail_stage == 3) throw std::runtime_error("injected allocator release failure");
}
}
