#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 10
#error "Scoped graph reclaim requires the audited LibTorch 2.10 CUDAGraph lifecycle"
#endif

namespace vlaforge::backends {

// LibTorch's reset warns and clears ownership even when CUDA destroy fails.
// Keep each remaining owner intact until its checked operation succeeds. This
// private, version-gated adapter must be audited again for every Torch minor.
class CheckedCUDAGraph final : public at::cuda::CUDAGraph {
 public:
  using at::cuda::CUDAGraph::CUDAGraph;

  void ResetChecked() {
    if (!has_graph_exec_ && !has_graph_ && !capture_ended_) { return; }
    const c10::cuda::CUDAGuard guard(capture_dev_);
    if (has_graph_exec_) {
      AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
      has_graph_exec_ = false;
    }
    if (has_graph_) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
      has_graph_ = false;
    }
    if (capture_ended_) {
      c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
      capture_ended_ = false;
    }
  }
};

}  // namespace vlaforge::backends
