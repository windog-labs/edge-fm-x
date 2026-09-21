#include "vlaforge/runtime/execution_context.h"

#include <new>
#include <cstring>
#include <cstdint>
#include <mutex>

#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
#include <cuda_runtime_api.h>
#endif

struct VLAForgeExecutionContext {
  VLAForgeExecutionContextView view{};
  bool poisoned = false;
  VLAForgeExecutionContext* next_idle = nullptr;
};

namespace {

bool SupportedDevice(VLAForgeDevice device) noexcept {
  return (device.kind == VLAFORGE_DEVICE_CPU && device.ordinal == 0) ||
      (device.kind == VLAFORGE_DEVICE_CUDA && device.ordinal >= 0);
}

#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
// External-stream allocators and BLAS workspace caches use the stream identity.
// Reuse drained, exclusively leased streams instead of stranding those caches
// after every Session. Idle streams live to process exit, bounded by the peak
// number of simultaneous leases per device, not the number of Session lifetimes.
class StreamPool final {
 public:
  VLAForgeExecutionContext* Acquire(int ordinal) noexcept {
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      auto** next = &idle_;
      while (*next != nullptr) {
        auto* context = *next;
        if (context->view.device.ordinal == ordinal) {
          *next = context->next_idle;
          context->next_idle = nullptr;
          return context;
        }
        next = &context->next_idle;
      }
    } catch (...) {
      // A failed pool operation must never expose a lease held by another user.
    }
    return nullptr;
  }

  void Release(VLAForgeExecutionContext* context) noexcept {
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      context->next_idle = idle_;
      idle_ = context;
    } catch (...) {
      // Retain this stream rather than free storage with unknown ownership.
    }
  }

 private:
  std::mutex mutex_;
  VLAForgeExecutionContext* idle_ = nullptr;
};

StreamPool* ReusableStreams() noexcept {
  // A global owner may destroy its Session after ordinary function-static
  // objects. Keep the pool mutex alive for those late releases as well.
  static auto* pool = new (std::nothrow) StreamPool();
  return pool;
}

class DeviceGuard final {
 public:
  explicit DeviceGuard(int ordinal) noexcept {
    status_ = cudaGetDevice(&previous_);
    if (status_ == cudaSuccess && previous_ != ordinal) {
      status_ = cudaSetDevice(ordinal);
      restore_ = status_ == cudaSuccess;
    }
  }
  ~DeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_);
    }
  }
  cudaError_t status() const noexcept { return status_; }

 private:
  int previous_ = 0;
  bool restore_ = false;
  cudaError_t status_ = cudaSuccess;
};

VLAForgeStatus CudaStatus(cudaError_t status) noexcept {
  return status == cudaSuccess
      ? vlaforge_status_ok()
      : vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR,
                              cudaGetErrorString(status));
}
#endif

}  // namespace

extern "C" VLAForgeStatus vlaforge_execution_context_create(
    const VLAForgeExecutionContextOptions* options,
    VLAForgeExecutionContext** output) {
  if (output == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "execution context output is null");
  }
  *output = nullptr;
  if (options == nullptr || options->struct_size < sizeof(*options) ||
      !SupportedDevice(options->device)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid execution context options");
  }
  if (options->abi_version != VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported execution context ABI");
  }
#if !defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (options->device.kind == VLAFORGE_DEVICE_CUDA) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "CUDA context requires CUDA-enabled runtime");
  }
#endif
  auto* context = new (std::nothrow) VLAForgeExecutionContext();
  if (context == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "execution context allocation failed");
  }
  context->view = {sizeof(VLAForgeExecutionContextView),
                   VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
                   options->device, nullptr};
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (options->device.kind == VLAFORGE_DEVICE_CUDA) {
    const DeviceGuard guard(options->device.ordinal);
    auto status = guard.status();
    if (status == cudaSuccess) {
      auto* pool = ReusableStreams();
      if (pool == nullptr) {
        delete context;
        return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                     "CUDA stream pool allocation failed");
      }
      if (auto* reusable = pool->Acquire(options->device.ordinal)) {
        delete context;
        *output = reusable;
        return vlaforge_status_ok();
      }
    }
    cudaStream_t stream = nullptr;
    if (status == cudaSuccess) {
      status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    if (status != cudaSuccess) {
      delete context;
      return CudaStatus(status);
    }
    context->view.native_stream = stream;
  }
#endif
  *output = context;
  return vlaforge_status_ok();
}

extern "C" VLAForgeStatus vlaforge_execution_context_get_view(
    const VLAForgeExecutionContext* context,
    VLAForgeExecutionContextView* view) {
  if (context == nullptr || view == nullptr ||
      view->struct_size < sizeof(*view)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid execution context view output");
  }
  if (context->poisoned) {
    return vlaforge_execution_context_status(context);
  }
  *view = context->view;
  return vlaforge_status_ok();
}

extern "C" VLAForgeStatus vlaforge_execution_context_view_validate(
    const VLAForgeExecutionContextView* view) {
  if (view == nullptr || view->struct_size < sizeof(*view)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid execution context view");
  }
  if (view->abi_version != VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported execution context view ABI");
  }
  if (!SupportedDevice(view->device) ||
      (view->device.kind == VLAFORGE_DEVICE_CPU &&
       view->native_stream != nullptr) ||
      (view->device.kind == VLAFORGE_DEVICE_CUDA &&
       view->native_stream == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "execution context device/stream mismatch");
  }
  return vlaforge_status_ok();
}

extern "C" VLAForgeStatus vlaforge_execution_context_synchronize(
    VLAForgeExecutionContext* context) {
  if (context == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "execution context is null");
  }
  if (context->poisoned) {
    return vlaforge_execution_context_status(context);
  }
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (context->view.device.kind == VLAFORGE_DEVICE_CUDA) {
    const DeviceGuard guard(context->view.device.ordinal);
    if (guard.status() != cudaSuccess) {
      context->poisoned = true;
      return CudaStatus(guard.status());
    }
    const auto drained = cudaStreamSynchronize(
        static_cast<cudaStream_t>(context->view.native_stream));
    if (drained != cudaSuccess) { context->poisoned = true; }
    return CudaStatus(drained);
  }
#endif
  return vlaforge_status_ok();
}

extern "C" void vlaforge_execution_context_destroy(
    VLAForgeExecutionContext* context) {
  if (context == nullptr) {
    return;
  }
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (context->view.device.kind == VLAFORGE_DEVICE_CUDA) {
    if (context->poisoned) {
      // A capture or asynchronous failure can leave borrowers using the stream.
      return;
    }
    const DeviceGuard guard(context->view.device.ordinal);
    if (guard.status() == cudaSuccess) {
      const auto stream =
          static_cast<cudaStream_t>(context->view.native_stream);
      if (cudaStreamSynchronize(stream) == cudaSuccess) {
        if (auto* pool = ReusableStreams()) {
          pool->Release(context);
        }
        return;
      }
    }
    context->poisoned = true;
    return;
  }
#endif
  delete context;
}

extern "C" VLAForgeStatus vlaforge_execution_context_status(
    const VLAForgeExecutionContext* context) {
  if (context == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "execution context is null");
  }
  return context->poisoned
      ? vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "execution context poisoned; process restart may be required")
      : vlaforge_status_ok();
}

extern "C" void vlaforge_execution_context_poison(
    VLAForgeExecutionContext* context) {
  if (context != nullptr) {
    context->poisoned = true;
  }
}

extern "C" VLAForgeStatus vlaforge_execution_context_copy(
    VLAForgeExecutionContext* context, const VLAForgeTensorView* destination,
    const VLAForgeTensorView* source, std::uint64_t size_bytes) {
  const auto status = vlaforge_execution_context_status(context);
  if (status.code != VLAFORGE_STATUS_OK) { return status; }
  if (destination == nullptr || source == nullptr ||
      size_bytes > destination->size_bytes || size_bytes > source->size_bytes ||
      (size_bytes != 0 && (destination->data == nullptr || source->data == nullptr)) ||
      destination->device.kind != context->view.device.kind ||
      source->device.kind != context->view.device.kind ||
      destination->device.ordinal != context->view.device.ordinal ||
      source->device.ordinal != context->view.device.ordinal) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid same-context copy");
  }
  if (size_bytes == 0 || destination->data == source->data) {
    return vlaforge_status_ok();
  }
  const auto dst = reinterpret_cast<std::uintptr_t>(destination->data);
  const auto src = reinterpret_cast<std::uintptr_t>(source->data);
  if ((dst > src ? dst - src : src - dst) < size_bytes) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "same-context copy ranges overlap");
  }
  if (context->view.device.kind == VLAFORGE_DEVICE_CPU) {
    std::memcpy(destination->data, source->data, size_bytes);
    return vlaforge_status_ok();
  }
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  const DeviceGuard guard(context->view.device.ordinal);
  if (guard.status() != cudaSuccess) { return CudaStatus(guard.status()); }
  return CudaStatus(cudaMemcpyAsync(
      destination->data, source->data, size_bytes, cudaMemcpyDeviceToDevice,
      static_cast<cudaStream_t>(context->view.native_stream)));
#else
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "CUDA copy requires CUDA-enabled runtime");
#endif
}

extern "C" VLAForgeStatus vlaforge_region_execution_extension_api_validate(
    const VLAForgeRegionExecutionExtensionApi* api) {
  if (api == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "Region execution extension is null");
  }
  if (api->struct_size < sizeof(*api) ||
      api->abi_version != VLAFORGE_REGION_EXECUTION_EXTENSION_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_UNSUPPORTED_ABI,
                                 "unsupported Region execution extension ABI");
  }
  if (api->bind_context == nullptr ||
      (api->capabilities & VLAFORGE_REGION_EXECUTION_CAP_SHARED_CONTEXT) == 0u) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "incomplete Region execution extension");
  }
  return vlaforge_status_ok();
}
