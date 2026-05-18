#pragma once

#include "engine/engine.h"
#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace edge_fm {

class PrefillLinearSourceOp {
public:
    explicit PrefillLinearSourceOp(const EngineConfig& config);
    ~PrefillLinearSourceOp();

    bool try_forward(
        const std::string& role,
        int32_t layer_id,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output,
        cudaStream_t stream);

    void reset_runtime_caches();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_fm
