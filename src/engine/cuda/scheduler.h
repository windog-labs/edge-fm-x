#pragma once

#include "engine/scheduler.h"

#include <cuda_runtime.h>

namespace edge_fm {

inline cudaStream_t cuda_stream(const Context& context) {
    return reinterpret_cast<cudaStream_t>(context.stream_handle());
}

class CudaScheduler : public Scheduler {
public:
    explicit CudaScheduler(std::shared_ptr<KVManager> kv_manager);
    ~CudaScheduler() override;

    cudaStream_t stream() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace edge_fm
