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
      tracked_memcpy_(std::move(other.tracked_memcpy_)),
      tracked_kernels_(std::move(other.tracked_kernels_))
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
        tracked_kernels_ = std::move(other.tracked_kernels_);
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
    tracked_kernels_.clear();

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
    tracked_kernels_.clear();
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

// ---------- track kernel -------------------------------------------------

int CudaGraphRunner::track_kernel_node(void* arg0_ptr) {
    ensure_nodes_scanned();
    for (auto& nd : all_nodes_) {
        cudaGraphNodeType type;
        CUDA_CHECK_THROW(cudaGraphNodeGetType(nd, &type),
                         "CudaGraphRunner::track_kernel_node cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeKernel) continue;

        cudaKernelNodeParams kp = {};
        cudaError_t err = cudaGraphKernelNodeGetParams(nd, &kp);
        if (err == cudaErrorInvalidDeviceFunction) {
            // cuBLAS (and other libraries) launch kernels via the CUDA Driver
            // API from dynamically loaded cubins. The Runtime API cannot
            // resolve these function pointers, so GetParams returns
            // cudaErrorInvalidDeviceFunction. This is expected — clear the
            // sticky error and skip.
            (void)cudaGetLastError();
            continue;
        }
        if (err != cudaSuccess) {
            CUDA_CHECK_THROW(err,
                "CudaGraphRunner::track_kernel_node cudaGraphKernelNodeGetParams");
        }
        if (kp.kernelParams == nullptr) continue;

        void* val = *static_cast<void**>(kp.kernelParams[0]);
        if (val != arg0_ptr) continue;

        TrackedKernel tk;
        tk.node = nd;
        tk.params = kp;
        tk.arg0_value = val;

        // Build our own kernelParams[] array: arg-0 points to our mutable
        // copy; remaining entries re-use the graph-internal pointers (they
        // are stable while graph_ is alive and the args are unchanging).
        constexpr int kMaxArgs = 16;
        tk.arg_ptrs.resize(kMaxArgs, nullptr);
        tk.arg_ptrs[0] = &tk.arg0_value;
        for (int a = 1; a < kMaxArgs; ++a)
            tk.arg_ptrs[a] = kp.kernelParams[a];
        tk.params.kernelParams = tk.arg_ptrs.data();

        int handle = static_cast<int>(tracked_kernels_.size());
        tracked_kernels_.push_back(std::move(tk));
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

// ---------- update kernel ------------------------------------------------

void CudaGraphRunner::update_kernel_arg0(int handle, void* new_arg0) {
    auto& tk = tracked_kernels_.at(handle);
    tk.arg0_value = new_arg0;
    // tk.arg_ptrs[0] still points to &tk.arg0_value, so params is up-to-date.
    CUDA_CHECK_THROW(
        cudaGraphExecKernelNodeSetParams(exec_, tk.node, &tk.params),
        "CudaGraphRunner::update_kernel_arg0");
}

} // namespace edge_fm
