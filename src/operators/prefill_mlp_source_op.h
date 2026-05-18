#pragma once

#include "engine/engine.h"
#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <memory>

namespace edge_fm {

class PrefillMlpSourceOp {
public:
    explicit PrefillMlpSourceOp(const EngineConfig& config);
    ~PrefillMlpSourceOp();

    bool try_forward(
        int32_t layer_id,
        const Tensor& input,
        const Tensor& gate_up_weight,
        const Tensor& down_weight,
        Tensor& output,
        cudaStream_t stream);

    void reset_runtime_caches();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_fm
