#pragma once
namespace c10::cuda {
struct CUDAGuard {
  int previous;
  explicit CUDAGuard(int device) : previous(current_device) {
    if (fail_stage == 5) throw std::runtime_error("injected device failure");
    current_device = device;
  }
  ~CUDAGuard() { current_device = previous; }
};
}
