#pragma once
#include <stdexcept>
#include <utility>

inline int fail_stage = 0;
inline int exec_calls = 0, graph_calls = 0, release_calls = 0;
inline int base_destructors = 0, unsafe_cleanup = 0, scoped_reclaims = 0;
inline int current_device = 5;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
inline int cudaGraphExecDestroy(cudaGraphExec_t) { ++exec_calls; return fail_stage == 1 ? 1 : 0; }
inline int cudaGraphDestroy(cudaGraph_t) { ++graph_calls; return fail_stage == 2 ? 1 : 0; }
namespace at::cuda {
class CUDAGraph {
 public:
  explicit CUDAGraph(bool) {}
  ~CUDAGraph() {
    ++base_destructors;
    if (has_graph_ || has_graph_exec_ || capture_ended_) ++unsafe_cleanup;
  }
  bool owns_exec() const { return has_graph_exec_; }
  bool owns_graph() const { return has_graph_; }
  bool owns_pool_ref() const { return capture_ended_; }
 protected:
  cudaGraph_t graph_ = reinterpret_cast<void*>(1);
  cudaGraphExec_t graph_exec_ = reinterpret_cast<void*>(2);
  bool has_graph_ = true, capture_ended_ = true, has_graph_exec_ = true;
  int capture_dev_ = 0;
  std::pair<int, int> mempool_id_{0, 1};
};
}
