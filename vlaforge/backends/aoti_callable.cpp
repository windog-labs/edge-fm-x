#include "aoti_callable.h"
#include "aoti_extracted_package.h"
#include "aoti_materialized_package.h"

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

#include <filesystem>
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
  std::unique_ptr<AotiExtractedPackage> extracted_package;
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

void AotiCallable::Load(const std::string& path,
                        const std::string& extraction_root,
                        const std::string& sha256,
                        std::uint64_t size_bytes) {
  if (impl_->loaded()) {
    throw std::runtime_error("AOTI callable is already loaded");
  }
  std::unique_ptr<AotiExtractedPackage> extracted;
  std::string library_path = path;
  std::string weight_blob;
  if (std::filesystem::path(path).extension() == ".vfaoti") {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10 && TORCH_VERSION_PATCH == 0
    const AotiMaterializedPackage materialized(path, sha256, size_bytes,
        impl_->device_kind == VLAFORGE_DEVICE_CUDA ? "cuda" : "cpu");
    library_path = materialized.library();
    weight_blob = materialized.weight_blob();
#else
    throw std::runtime_error("materialized AOTI requires audited LibTorch 2.10.0");
#endif
  } else if (!extraction_root.empty() && !IsRawSharedLibrary(path)) {
    extracted = std::make_unique<AotiExtractedPackage>(
        path, extraction_root, sha256, size_bytes);
    library_path = extracted->library();
    weight_blob = extracted->weight_blob();
  }
  if (IsRawSharedLibrary(library_path)) {
    if (impl_->device_kind == VLAFORGE_DEVICE_CUDA) {
      const auto cubin_dir =
          std::filesystem::path(library_path).parent_path().string();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
      // The Session serializes this callable. Runner-pool completion queries
      // are not valid between repeated calls inside external CUDA capture.
      impl_->cuda_runner = std::make_unique<
          torch::inductor::AOTIModelContainerRunnerCuda>(
              library_path, 1u, "cuda:" + std::to_string(impl_->device_ordinal),
              cubin_dir, true);
#else
      impl_->cuda_runner = std::make_unique<
          torch::inductor::AOTIModelContainerRunnerCuda>(
              library_path, 1u, "cuda:" + std::to_string(impl_->device_ordinal),
              cubin_dir);
#endif
    } else {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
      if (extracted) {
        impl_->cpu_runner = std::make_unique<
            torch::inductor::AOTIModelContainerRunnerCpu>(library_path, 1u, true);
      } else
#endif
      {
        impl_->cpu_runner = std::make_unique<
            torch::inductor::AOTIModelContainerRunnerCpu>(library_path, 1u);
      }
    }
    try {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10
      if (!weight_blob.empty()) {
        auto* runner = impl_->cuda_runner
            ? static_cast<torch::inductor::AOTIModelContainerRunner*>(impl_->cuda_runner.get())
            : static_cast<torch::inductor::AOTIModelContainerRunner*>(impl_->cpu_runner.get());
        runner->update_constant_buffer_from_blob(weight_blob);
      }
#endif
    } catch (...) {
      impl_->cuda_runner.reset();
      impl_->cpu_runner.reset();
      throw;
    }
    impl_->extracted_package = std::move(extracted);
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
    std::vector<at::Tensor>& inputs, void* stream_handle) {
#if VLAFORGE_HAS_AOTI_PACKAGE_LOADER
  if (impl_->package_loader != nullptr) {
    return impl_->package_loader->run(inputs, stream_handle);
  }
#endif
  if (impl_->cuda_runner != nullptr) {
    return impl_->cuda_runner->run(inputs, stream_handle);
  }
  if (impl_->cpu_runner != nullptr) {
    return impl_->cpu_runner->run(inputs);
  }
  throw std::runtime_error("AOTI callable is not loaded");
}

}  // namespace vlaforge::backends
