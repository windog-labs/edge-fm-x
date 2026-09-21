"""Pure-Tensor lowering of explicitly verified SmolVLA output statistics.

This is a new output Region, not a claim that legacy normalized bundles already
performed inverse normalization. Tensor dimensions and statistics are supplied
through the processor interface; no robot or action horizon is selected here.
"""

from vlaforge.adapters.smolvla.smolvla_processing import SmolVLAStatistics


def mean_std_output_module(statistics: SmolVLAStatistics, *, feature: str):
    import torch

    if not isinstance(statistics, SmolVLAStatistics) or set(statistics.feature_shapes) != {feature}:
        raise ValueError("output lowering requires exactly the declared consumed feature")
    values = statistics.stats[feature]

    class Output(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", values["mean"])
            self.register_buffer("std", values["std"])
            self.feature_shape = tuple(statistics.feature_shapes[feature])

        def forward(self, normalized, incoming_accepted):
            if incoming_accepted.dtype != torch.bool or tuple(incoming_accepted.shape) != (1,):
                raise ValueError("incoming output acceptance must be bool[1]")
            if tuple(normalized.shape[-len(self.feature_shape):]) != self.feature_shape:
                raise ValueError("normalized output shape does not match selected statistics")
            if not normalized.is_floating_point():
                raise TypeError("normalized output must have a floating dtype")
            native = normalized * self.std.to(normalized) + self.mean.to(normalized)
            accepted = incoming_accepted & torch.isfinite(normalized).all() & torch.isfinite(native).all()
            return native, accepted.reshape(1)

    return Output().eval()


def lower_official_unnormalizer(processor, statistics: SmolVLAStatistics):
    """Require actual upstream processor consumption before lowering its math."""
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep

    if type(processor) is not UnnormalizerProcessorStep:
        raise TypeError("unsupported output processor requires an explicit Adapter lowerer")
    statistics.require_consumed(processor)
    if set(processor.features) != {"action"}:
        raise ValueError("this output interface accepts only the declared action feature")
    return mean_std_output_module(statistics, feature="action")
