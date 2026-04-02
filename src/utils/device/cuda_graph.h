#pragma once

#include "utils/non_copyable.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace edge_fm {

/**
 * Generic RAII wrapper for CUDA graph capture-once / launch-many workflow.
 *
 * After capture, the caller can track specific memcpy nodes and update
 * them each step via opaque integer handles.
 *
 * This class knows nothing about model structure — all model-specific
 * mapping is done by the caller.
 */
class CudaGraphRunner : public NonCopyable {
public:
    CudaGraphRunner() = default;
    ~CudaGraphRunner();

    CudaGraphRunner(CudaGraphRunner&& other) noexcept;
    CudaGraphRunner& operator=(CudaGraphRunner&& other) noexcept;

    void begin_capture(cudaStream_t stream);
    void end_capture(cudaStream_t stream);
    bool launch(cudaStream_t stream);
    bool is_captured() const { return exec_ != nullptr; }
    void reset();

    // ------ generic dynamic-node tracking (call after end_capture) ------

    /// Find a D2D memcpy node whose destination == @p dst_ptr.
    /// Returns an opaque handle (>= 0), or -1 if not found.
    int track_memcpy_node(void* dst_ptr);

    /// Update the destination pointer of a tracked memcpy node.
    void update_memcpy_dst(int handle, void* new_dst);

private:
    cudaGraphExec_t exec_  = nullptr;
    cudaGraph_t     graph_ = nullptr;

    // Lazy node enumeration cache.
    std::vector<cudaGraphNode_t> all_nodes_;
    bool nodes_scanned_ = false;
    void ensure_nodes_scanned();

    struct TrackedMemcpy {
        cudaGraphNode_t   node;
        cudaMemcpy3DParms params;
    };
    std::vector<TrackedMemcpy> tracked_memcpy_;
};

} // namespace edge_fm
