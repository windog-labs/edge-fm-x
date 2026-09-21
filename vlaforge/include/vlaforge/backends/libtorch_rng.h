#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <mutex>

namespace vlaforge {

// One owner per request stream. State is an opaque LibTorch-version-bound
// CPU byte tensor; restoring it never changes the process default generator.
class CudaRngProvider {
 public:
  explicit CudaRngProvider(c10::DeviceIndex device, std::uint64_t seed)
      : device_(device), generator_(at::cuda::detail::createCUDAGenerator(device)) {
    Reset(seed);
  }
  CudaRngProvider(const CudaRngProvider&) = delete;
  CudaRngProvider& operator=(const CudaRngProvider&) = delete;

  void Reset(std::uint64_t seed) {
    const std::lock_guard<std::mutex> lock(mutex_);
    generator_.set_current_seed(seed);
  }

  at::Tensor State() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return generator_.get_state().clone();
  }

  void Restore(const at::Tensor& state) {
    const std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state.device().is_cpu() && state.scalar_type() == at::kByte &&
                state.dim() == 1 && state.is_contiguous() &&
                state.nbytes() == generator_.get_state().nbytes(),
                "RNG state must be a complete CPU byte tensor for this provider");
    // Validate with a separate generator so a rejected state cannot alter ours.
    auto replacement = at::cuda::detail::createCUDAGenerator(device_);
    replacement.set_state(state.clone());
    generator_ = std::move(replacement);
  }

  at::Tensor Normal(at::IntArrayRef shape, at::ScalarType dtype = at::kFloat) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const c10::cuda::CUDAGuard guard(device_);
    return at::randn(shape, generator_, at::TensorOptions().device(at::kCUDA, device_).dtype(dtype));
  }

  at::Tensor NormalLike(const at::Tensor& value) {
    const std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(value.is_cuda() && value.get_device() == device_,
                "RNG template must be on the provider device");
    const c10::cuda::CUDAGuard guard(device_);
    return at::randn_like(value, generator_);
  }

 private:
  const c10::DeviceIndex device_;
  at::Generator generator_;
  mutable std::mutex mutex_;
};

}  // namespace vlaforge
