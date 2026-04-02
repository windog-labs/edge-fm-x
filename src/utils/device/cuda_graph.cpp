#include "utils/device/cuda_graph.h"
#include "utils/device/cuda_utils.h"
#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <utility>

namespace edge_fm {

// ---------- dtor / move --------------------------------------------------

CudaGraphRunner::~CudaGraphRunner() { reset(); }

CudaGraphRunner::CudaGraphRunner(CudaGraphRunner&& other) noexcept
    : exec_(std::exchange(other.exec_, nullptr)),
      graph_(std::exchange(other.graph_, nullptr)),
      all_nodes_(std::move(other.all_nodes_)),
      nodes_scanned_(other.nodes_scanned_),
      tracked_memcpy_(std::move(other.tracked_memcpy_))
{
    other.nodes_scanned_ = false;
}

CudaGraphRunner& CudaGraphRunner::operator=(CudaGraphRunner&& other) noexcept {
    if (this != &other) {
        reset();
        exec_  = std::exchange(other.exec_, nullptr);
        graph_ = std::exchange(other.graph_, nullptr);
        all_nodes_       = std::move(other.all_nodes_);
        nodes_scanned_   = other.nodes_scanned_;
        tracked_memcpy_  = std::move(other.tracked_memcpy_);
        other.nodes_scanned_ = false;
    }
    return *this;
}

// ---------- capture ------------------------------------------------------

void CudaGraphRunner::begin_capture(cudaStream_t stream) {
    CUDA_CHECK_THROW(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "CudaGraphRunner::begin_capture failed");
}

void CudaGraphRunner::end_capture(cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    CUDA_CHECK_THROW(
        cudaStreamEndCapture(stream, &graph),
        "CudaGraphRunner::end_capture failed");

    // Replace previous graph / exec.
    if (graph_ != nullptr) {
        CUDA_CHECK_THROW(cudaGraphDestroy(graph_), "CudaGraphRunner::end_capture cudaGraphDestroy");
    }
    graph_ = graph;
    nodes_scanned_ = false;
    all_nodes_.clear();
    tracked_memcpy_.clear();

    if (exec_ != nullptr) {
        CUDA_CHECK_THROW(cudaGraphExecDestroy(exec_), "CudaGraphRunner::end_capture cudaGraphExecDestroy");
    }
    exec_ = nullptr;

    cudaError_t err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        throw DeviceError(
            std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(err));
    }
}

// ---------- launch -------------------------------------------------------

bool CudaGraphRunner::launch(cudaStream_t stream) {
    if (exec_ == nullptr) return false;
    CUDA_CHECK_THROW(
        cudaGraphLaunch(exec_, stream),
        "CudaGraphRunner::launch failed");
    return true;
}

// ---------- reset --------------------------------------------------------

void CudaGraphRunner::reset() {
    tracked_memcpy_.clear();
    all_nodes_.clear();
    nodes_scanned_ = false;
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

// ---------- lazy node scan -----------------------------------------------

void CudaGraphRunner::ensure_nodes_scanned() {
    if (nodes_scanned_ || graph_ == nullptr) return;
    size_t n = 0;
    CUDA_CHECK_THROW(cudaGraphGetNodes(graph_, nullptr, &n),
                     "cudaGraphGetNodes (count)");
    all_nodes_.resize(n);
    CUDA_CHECK_THROW(cudaGraphGetNodes(graph_, all_nodes_.data(), &n),
                     "cudaGraphGetNodes (fetch)");
    nodes_scanned_ = true;
}

// ---------- track memcpy -------------------------------------------------

int CudaGraphRunner::track_memcpy_node(void* dst_ptr) {
    ensure_nodes_scanned();
    for (auto& nd : all_nodes_) {
        cudaGraphNodeType type;
        CUDA_CHECK_THROW(cudaGraphNodeGetType(nd, &type),
                         "CudaGraphRunner::track_memcpy_node cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeMemcpy) continue;

        cudaMemcpy3DParms mp = {};
        CUDA_CHECK_THROW(cudaGraphMemcpyNodeGetParams(nd, &mp),
                         "CudaGraphRunner::track_memcpy_node cudaGraphMemcpyNodeGetParams");
        if (mp.kind != cudaMemcpyDeviceToDevice) continue;
        if (mp.dstPtr.ptr != dst_ptr) continue;

        int handle = static_cast<int>(tracked_memcpy_.size());
        tracked_memcpy_.push_back({nd, mp});
        return handle;
    }
    return -1;
}

// ---------- update memcpy ------------------------------------------------

void CudaGraphRunner::update_memcpy_dst(int handle, void* new_dst) {
    auto& tm = tracked_memcpy_.at(handle);
    tm.params.dstPtr.ptr = new_dst;
    CUDA_CHECK_THROW(
        cudaGraphExecMemcpyNodeSetParams(exec_, tm.node, &tm.params),
        "CudaGraphRunner::update_memcpy_dst");
}

} // namespace edge_fm
