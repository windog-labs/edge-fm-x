#pragma once

#include "engine/engine.h"
#include "layers/activation.h"
#include "layers/linear.h"
#include "operators/prefill_mlp_source_op.h"

#include <cuda_runtime.h>
#include <memory>

namespace edge_fm {

class GatedMlpLayer {
public:
    explicit GatedMlpLayer(const EngineConfig& config);
    ~GatedMlpLayer();

    void forward(
        int32_t layer_id,
        const Tensor& input,
        FusedGateUpLinearLayer& gate_up,
        LinearLayer& down,
        ActivationLayer& activation,
        Tensor& activation_input,
        Tensor& intermediate,
        Tensor& output,
        cudaStream_t stream,
        ModelStage stage);

    void reset_operator_impl_cache();

private:
    std::unique_ptr<PrefillMlpSourceOp> prefill_source_op_;
};

} // namespace edge_fm
