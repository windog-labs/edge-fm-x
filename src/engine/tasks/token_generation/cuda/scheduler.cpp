#include "engine/tasks/token_generation/cuda/scheduler.h"

#include "utils/device/cuda_utils.h"

#include <utility>

namespace edge_fm {

CudaScheduler::CudaScheduler(std::shared_ptr<KVManager> kv_manager, int32_t max_new_tokens)
    : Scheduler(std::move(kv_manager), max_new_tokens)
{
    CUDA_CHECK_THROW_EX(cudaStreamCreate(&stream_), "Failed to create CUDA stream", DeviceError);
    set_stream_handle(reinterpret_cast<EngineStreamHandle>(stream_));
}

CudaScheduler::~CudaScheduler() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

} // namespace edge_fm
