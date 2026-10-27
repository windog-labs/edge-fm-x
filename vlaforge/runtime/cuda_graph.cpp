#include "vlaforge/runtime/cuda_graph.h"

#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <new>

namespace {

struct CudaGraph final {
  VLAForgeExecutionContextView context{};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;
  bool active = false;
  bool fatal = false;
  std::array<char, 512> error{};

  cudaStream_t stream() const noexcept {
    return static_cast<cudaStream_t>(context.native_stream);
  }

  VLAForgeStatus Error(cudaError_t code) noexcept {
    fatal = true;
    std::snprintf(error.data(), error.size(), "native CUDA graph: %s", cudaGetErrorString(code));
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, error.data());
  }
};

class DeviceGuard final {
 public:
  explicit DeviceGuard(int device) noexcept {
    status_ = cudaGetDevice(&previous_);
    if (status_ == cudaSuccess && previous_ != device) {
      status_ = cudaSetDevice(device);
      restore_ = status_ == cudaSuccess;
    }
  }
  ~DeviceGuard() { if (restore_) { (void)cudaSetDevice(previous_); } }
  cudaError_t status() const noexcept { return status_; }

 private:
  int previous_ = 0;
  bool restore_ = false;
  cudaError_t status_ = cudaSuccess;
};

CudaGraph* Unwrap(VLAForgeGraphExecutable* graph) noexcept {
  return reinterpret_cast<CudaGraph*>(graph);
}

VLAForgeStatus InvalidState() noexcept {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "invalid native CUDA graph state");
}

VLAForgeStatus Create(const VLAForgeExecutionContextView* view, VLAForgeGraphExecutable** output) {
  if (output == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "null graph output");
  }
  *output = nullptr;
  const auto status = vlaforge_execution_context_view_validate(view);
  if (status.code != VLAFORGE_STATUS_OK) { return status; }
  if (view->device.kind != VLAFORGE_DEVICE_CUDA) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "native CUDA graph requires CUDA");
  }
  auto* graph = new (std::nothrow) CudaGraph();
  if (graph == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY, "native CUDA graph allocation failed");
  }
  graph->context = *view;
  *output = reinterpret_cast<VLAForgeGraphExecutable*>(graph);
  return vlaforge_status_ok();
}

VLAForgeStatus Begin(VLAForgeGraphExecutable* opaque) {
  auto* graph = Unwrap(opaque);
  if (graph == nullptr || graph->active || graph->fatal || graph->graph != nullptr) { return InvalidState(); }
  const DeviceGuard guard(graph->context.device.ordinal);
  if (guard.status() != cudaSuccess) { return graph->Error(guard.status()); }
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  auto status = cudaStreamIsCapturing(graph->stream(), &capture);
  if (status != cudaSuccess) { return graph->Error(status); }
  if (capture != cudaStreamCaptureStatusNone) { return InvalidState(); }
  // A failed begin can leave stream state uncertain; never silently fall back.
  graph->active = true;
  status = cudaStreamBeginCapture(graph->stream(), cudaStreamCaptureModeGlobal);
  return status == cudaSuccess ? vlaforge_status_ok() : graph->Error(status);
}

VLAForgeStatus End(VLAForgeGraphExecutable* opaque) {
  auto* graph = Unwrap(opaque);
  if (graph == nullptr || !graph->active || graph->fatal) { return InvalidState(); }
  const DeviceGuard guard(graph->context.device.ordinal);
  if (guard.status() != cudaSuccess) { return graph->Error(guard.status()); }
  auto status = cudaStreamEndCapture(graph->stream(), &graph->graph);
  if (status != cudaSuccess) { return graph->Error(status); }
  graph->active = false;
  if (graph->graph == nullptr) { return InvalidState(); }
  status = cudaGraphInstantiateWithFlags(&graph->instance, graph->graph, 0u);
  return status == cudaSuccess ? vlaforge_status_ok() : graph->Error(status);
}

VLAForgeStatus Abort(VLAForgeGraphExecutable* opaque) {
  auto* graph = Unwrap(opaque);
  if (graph == nullptr || graph->fatal || graph->instance != nullptr) { return InvalidState(); }
  const DeviceGuard guard(graph->context.device.ordinal);
  if (guard.status() != cudaSuccess) { return graph->Error(guard.status()); }
  if (graph->active) {
    const auto status = cudaStreamEndCapture(graph->stream(), &graph->graph);
    if (status != cudaSuccess) { return graph->Error(status); }
    graph->active = false;
  }
  if (graph->graph != nullptr) {
    const auto status = cudaGraphDestroy(graph->graph);
    if (status != cudaSuccess) { return graph->Error(status); }
    graph->graph = nullptr;
  }
  return vlaforge_status_ok();
}

VLAForgeStatus Launch(VLAForgeGraphExecutable* opaque) {
  auto* graph = Unwrap(opaque);
  if (graph == nullptr || graph->active || graph->fatal || graph->instance == nullptr) { return InvalidState(); }
  const DeviceGuard guard(graph->context.device.ordinal);
  if (guard.status() != cudaSuccess) { return graph->Error(guard.status()); }
  const auto status = cudaGraphLaunch(graph->instance, graph->stream());
  return status == cudaSuccess ? vlaforge_status_ok() : graph->Error(status);
}

void Destroy(VLAForgeGraphExecutable* opaque, std::uint32_t execution_poisoned) {
  auto* graph = Unwrap(opaque);
  if (graph == nullptr) { return; }
  bool retain = execution_poisoned != 0u || graph->fatal || graph->active;
  if (!retain) {
    const DeviceGuard guard(graph->context.device.ordinal);
    retain = guard.status() != cudaSuccess || cudaStreamSynchronize(graph->stream()) != cudaSuccess;
    if (!retain && graph->instance != nullptr) {
      retain = cudaGraphExecDestroy(graph->instance) != cudaSuccess;
      if (!retain) { graph->instance = nullptr; }
    }
    if (!retain && graph->graph != nullptr) {
      retain = cudaGraphDestroy(graph->graph) != cudaSuccess;
      if (!retain) { graph->graph = nullptr; }
    }
  }
  if (retain) {
    std::fprintf(stderr, "VLAForge native CUDA graph: quarantining resources with unproven completion\n");
    return;
  }
  delete graph;
}

const VLAForgeGraphBackendApi kApi{
    sizeof(VLAForgeGraphBackendApi), VLAFORGE_GRAPH_BACKEND_ABI_VERSION,
    &Create, &Begin, &End, &Abort, &Launch, &Destroy,
};

}  // namespace
#endif

extern "C" const VLAForgeGraphBackendApi* vlaforge_cuda_graph_backend_api(void) {
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  return &kApi;
#else
  return nullptr;
#endif
}
