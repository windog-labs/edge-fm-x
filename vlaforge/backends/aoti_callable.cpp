#include "aoti_callable.h"

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#define VLAFORGE_HAS_AOTI_PACKAGE_LOADER 1
#else
#define VLAFORGE_HAS_AOTI_PACKAGE_LOADER 0
#endif

#include <stdexcept>
#include <string_view>

namespace vlaforge::backends {
namespace {

bool IsRawSharedLibrary(std::string_view path) {
  constexpr std::string_view kSharedLibrarySuffix = ".so";
  return path.size() >= kSharedLibrarySuffix.size() &&
      path.substr(path.size() - kSharedLibrarySuffix.size()) ==
      kSharedLibrarySuffix;
}

}  // namespace

struct AotiCallable::Impl {
  VLAForgeDeviceKind device_kind = VLAFORGE_DEVICE_CPU;
  int device_ordinal = 0;
#if VLAFORGE_HAS_AOTI_PACKAGE_LOADER
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> package_loader;
#endif
  std::unique_ptr<torch::inductor::AOTIModelContainerRunnerCpu> cpu_runner;
  std::unique_ptr<torch::inductor::AOTIModelContainerRunnerCuda> cuda_runner;

  [[nodiscard]] bool loaded() const noexcept {
#if VLAFORGE_HAS_AOTI_PACKAGE_LOADER
    if (package_loader != nullptr) {
      return true;
    }
#endif
    return cpu_runner != nullptr || cuda_runner != nullptr;
  }
};

AotiCallable::AotiCallable(
    VLAForgeDeviceKind device_kind, int device_ordinal)
    : impl_(std::make_unique<Impl>()) {
  impl_->device_kind = device_kind;
  impl_->device_ordinal = device_ordinal;
}

AotiCallable::~AotiCallable() = default;

void AotiCallable::Load(const std::string& path) {
  if (impl_->loaded()) {
    throw std::runtime_error("AOTI callable is already loaded");
  }
  if (IsRawSharedLibrary(path)) {
    if (impl_->device_kind == VLAFORGE_DEVICE_CUDA) {
      impl_->cuda_runner = std::make_unique<
          torch::inductor::AOTIModelContainerRunnerCuda>(
              path, 1u, "cuda:" + std::to_string(impl_->device_ordinal));
    } else {
      impl_->cpu_runner = std::make_unique<
          torch::inductor::AOTIModelContainerRunnerCpu>(path, 1u);
    }
    return;
  }
#if VLAFORGE_HAS_AOTI_PACKAGE_LOADER
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
  impl_->package_loader =
      std::make_unique<torch::inductor::AOTIModelPackageLoader>(
          path, "model", true, 1u,
          impl_->device_kind == VLAFORGE_DEVICE_CUDA
              ? impl_->device_ordinal
              : -1);
#else
  impl_->package_loader =
      std::make_unique<torch::inductor::AOTIModelPackageLoader>(
          path, "model");
#endif
#else
  throw std::runtime_error(
      "this LibTorch version supports raw AOTI shared libraries only");
#endif
}

bool AotiCallable::loaded() const noexcept {
  return impl_->loaded();
}

std::vector<at::Tensor> AotiCallable::Run(
    std::vector<at::Tensor>& inputs) {
#if VLAFORGE_HAS_AOTI_PACKAGE_LOADER
  if (impl_->package_loader != nullptr) {
    return impl_->package_loader->run(inputs);
  }
#endif
  if (impl_->cuda_runner != nullptr) {
    return impl_->cuda_runner->run(inputs);
  }
  if (impl_->cpu_runner != nullptr) {
    return impl_->cpu_runner->run(inputs);
  }
  throw std::runtime_error("AOTI callable is not loaded");
}

}  // namespace vlaforge::backends
