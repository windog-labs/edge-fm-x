#include "vlaforge/backends/libtorch_graph.h"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
#include "libtorch_graph_cleanup.h"
#include <ATen/Context.h>
#include <ATen/cuda/MemPool.h>
#endif

#include <array>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>

struct VLAForgeGraphExecutable {
  VLAForgeExecutionContextView context{};
#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
  std::unique_ptr<at::cuda::MemPool> pool;
  std::unique_ptr<vlaforge::backends::CheckedCUDAGraph> graph;
#else
  std::unique_ptr<at::cuda::CUDAGraph> graph;
#endif
  bool active = false;
  bool ready = false;
  bool fatal = false;
  std::array<char, 512> error{};

  c10::cuda::CUDAStream Stream() const {
    return c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(context.native_stream), context.device.ordinal);
  }

  VLAForgeStatus Error(const char* message, bool poison) noexcept {
    fatal = fatal || poison;
    std::snprintf(error.data(), error.size(), "%s", message);
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }

#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
  void Quarantine(const char* reason) noexcept {
    std::fprintf(stderr, "[VLAFORGE-GRAPH-QUARANTINE] device=%d graph=%p pool=%p reason=%s; "
                         "owned graph and pool retained; destroy ABI has no returned status\n",
                 context.device.ordinal, static_cast<void*>(graph.get()),
                 static_cast<void*>(pool.get()), reason);
    (void)graph.release();
    (void)pool.release();
  }
#endif
};

namespace {

VLAForgeStatus Create(const VLAForgeExecutionContextView* view,
                      VLAForgeGraphExecutable** output) {
  if (output == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "null graph output");
  }
  *output = nullptr;
  auto status = vlaforge_execution_context_view_validate(view);
  if (status.code != VLAFORGE_STATUS_OK) { return status; }
  if (view->device.kind != VLAFORGE_DEVICE_CUDA) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "LibTorch graph requires CUDA");
  }
  try {
    auto executable = std::make_unique<VLAForgeGraphExecutable>();
    executable->context = *view;
    const c10::cuda::CUDAStreamGuard guard(executable->Stream());
#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    AT_CUDA_CHECK(cudaStreamIsCapturing(executable->Stream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "graph creation requires a noncapturing stream");
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    executable->pool = std::make_unique<at::cuda::MemPool>();
    executable->graph = std::make_unique<vlaforge::backends::CheckedCUDAGraph>(true);
#else
    executable->graph = std::make_unique<at::cuda::CUDAGraph>(true);
#endif
    *output = executable.release();
    return vlaforge_status_ok();
  } catch (...) {
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "LibTorch graph allocation failed");
  }
}

VLAForgeStatus Begin(VLAForgeGraphExecutable* executable) {
  if (executable == nullptr || executable->active || executable->ready || executable->fatal) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "invalid graph begin state");
  }
  try {
    const c10::cuda::CUDAStreamGuard guard(executable->Stream());
    // Mark active before entering LibTorch because a failed begin can already
    // have changed generator and caching-allocator state.
    executable->active = true;
#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
    executable->graph->capture_begin(executable->pool->id());
#else
    executable->graph->capture_begin();
#endif
    return vlaforge_status_ok();
  } catch (const std::exception& error) {
    return executable->Error(error.what(), true);
  } catch (...) {
    return executable->Error("unknown LibTorch graph begin exception", true);
  }
}

VLAForgeStatus End(VLAForgeGraphExecutable* executable) {
  if (executable == nullptr || !executable->active || executable->fatal) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "invalid graph end state");
  }
  try {
    const c10::cuda::CUDAStreamGuard guard(executable->Stream());
    executable->graph->capture_end();
    executable->active = false;
    executable->graph->instantiate();
    executable->ready = true;
    return vlaforge_status_ok();
  } catch (const std::exception& error) {
    return executable->Error(error.what(), true);
  } catch (...) {
    return executable->Error("unknown LibTorch graph end exception", true);
  }
}

VLAForgeStatus Abort(VLAForgeGraphExecutable* executable) {
  if (executable == nullptr || executable->fatal) {
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "graph abort cannot recover poisoned allocator state");
  }
  try {
    const c10::cuda::CUDAStreamGuard guard(executable->Stream());
    if (executable->active) {
      // A valid partial capture can end normally, balancing allocator and RNG
      // state. An invalidated capture must not be treated as this safe case.
      executable->graph->capture_end();
      executable->active = false;
    }
#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
    executable->Stream().synchronize();
    executable->graph->ResetChecked();
#else
    executable->graph->reset();
#endif
    executable->ready = false;
    return vlaforge_status_ok();
  } catch (const std::exception& error) {
    return executable->Error(error.what(), true);
  } catch (...) {
    return executable->Error("unknown LibTorch graph abort exception", true);
  }
}

VLAForgeStatus Launch(VLAForgeGraphExecutable* executable) {
  if (executable == nullptr || !executable->ready || executable->fatal) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "graph is not replayable");
  }
  try {
    const c10::cuda::CUDAStreamGuard guard(executable->Stream());
    executable->graph->replay();
    return vlaforge_status_ok();
  } catch (const std::exception& error) {
    return executable->Error(error.what(), true);
  } catch (...) {
    return executable->Error("unknown LibTorch graph replay exception", true);
  }
}

void Destroy(VLAForgeGraphExecutable* executable, std::uint32_t execution_poisoned) {
  if (executable == nullptr) { return; }
#if defined(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
  if (execution_poisoned != 0u || executable->active || executable->fatal) {
    executable->Quarantine("poisoned execution or incomplete capture");
  } else {
    try {
      const c10::cuda::CUDAStreamGuard guard(executable->Stream());
      executable->Stream().synchronize();
      executable->graph->ResetChecked();
      // Unregister captured generator references before releasing the sole
      // keeper. MemPool destruction reclaims only this private allocator pool.
      executable->graph.reset();
      executable->pool.reset();
    } catch (const std::exception& error) {
      executable->Quarantine(error.what());
    } catch (...) {
      executable->Quarantine("unknown checked graph destruction failure");
    }
  }
#else
  if (execution_poisoned != 0u || executable->active || executable->fatal) {
    // LibTorch documents incomplete cleanup after capture-end failure. Keep
    // its pool predicate/generator references alive until process termination.
    (void)executable->graph.release();
  }
#endif
  delete executable;
}

const VLAForgeGraphBackendApi kApi = {
    sizeof(kApi), VLAFORGE_GRAPH_BACKEND_ABI_VERSION,
    &Create, &Begin, &End, &Abort, &Launch, &Destroy};

}  // namespace

extern "C" const VLAForgeGraphBackendApi* vlaforge_libtorch_graph_backend_api(void) {
  return &kApi;
}
