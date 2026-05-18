#include "layers/gated_mlp.h"

namespace edge_fm {

GatedMlpLayer::GatedMlpLayer(const EngineConfig& config)
    : prefill_source_op_(std::make_unique<PrefillMlpSourceOp>(config))
{
}

GatedMlpLayer::~GatedMlpLayer() = default;

void GatedMlpLayer::reset_operator_impl_cache()
{
    if (prefill_source_op_ != nullptr) {
        prefill_source_op_->reset_runtime_caches();
    }
}

void GatedMlpLayer::forward(
    int32_t layer_id,
    const Tensor& input,
    FusedGateUpLinearLayer& gate_up,
    LinearLayer& down,
    ActivationLayer& activation,
    Tensor& activation_input,
    Tensor& intermediate,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    if (stage == ModelStage::Prefill && prefill_source_op_ != nullptr) {
        const Tensor* gate_up_weight = gate_up.weight_tensor(ModelStage::Prefill);
        const Tensor* down_weight = down.weight_tensor(ModelStage::Prefill);
        if (gate_up_weight != nullptr && down_weight != nullptr &&
            prefill_source_op_->try_forward(
                layer_id,
                input,
                *gate_up_weight,
                *down_weight,
                output,
                stream))
        {
            return;
        }
    }

    const int64_t seq_len = input.shape().empty() ? 0 : input.shape()[0];
    Tensor gate_up_flat = Tensor::view(
        activation_input.data_ptr(),
        activation_input.shape(),
        activation_input.dtype(),
        std::get<0>(activation_input.device()),
        std::get<1>(activation_input.device()));

    bool swiglu_fused = false;
    if (stage == ModelStage::Decode) {
        swiglu_fused = gate_up.try_forward_decode_swiglu_fused(input, intermediate, stream);
    } else if (seq_len >= 64) {
        swiglu_fused = gate_up.try_forward_prefill_swiglu_fused(input, intermediate, stream);
    }

    if (!swiglu_fused) {
        gate_up.forward_fp16_bf16(input, gate_up_flat, stream, stage);
        activation.forward_silu_and_mul_up_gate(activation_input, intermediate, stream, stage);
    }
    down.forward_fp16_bf16(intermediate, output, stream, stage);
}

} // namespace edge_fm
