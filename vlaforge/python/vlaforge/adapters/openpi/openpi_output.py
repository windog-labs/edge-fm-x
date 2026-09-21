"""Extensible lowering of pinned OpenPI output processors into pure tensors.

The returned module consumes batched normalized state/actions. It does not run
the model or claim robot calibration. Registration is by processor type, never
model name; an unsupported processor requires an explicit extension.
"""

from __future__ import annotations


def tensor_unnormalize(norm_stats, *, use_quantiles):
    import numpy as np
    import torch

    if type(use_quantiles) is not bool:
        raise TypeError("normalization mode must be explicit")
    if norm_stats is None:
        norm_stats = {}
    if set(norm_stats) - {"state", "actions"}:
        raise ValueError("output normalization has unsupported tensor keys")

    class Unnormalize(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.names = tuple(norm_stats)
            self.quantiles = use_quantiles
            for name, stats in norm_stats.items():
                dimensions = set()
                for field in (("q01", "q99") if use_quantiles else ("mean", "std")):
                    value = getattr(stats, field)
                    if value is None:
                        raise ValueError("required normalization statistics are missing")
                    array = np.asarray(value)
                    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
                        raise TypeError("normalization statistics require float32 or float64")
                    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
                        raise ValueError("normalization requires finite nonempty vectors")
                    dimensions.add(array.shape)
                    self.register_buffer(name + "_" + field, torch.from_numpy(array.copy()))
                if len(dimensions) != 1:
                    raise ValueError("normalization statistics have inconsistent dimensions")

        def one(self, value, name):
            if name not in self.names:
                return value
            if self.quantiles:
                low, high = getattr(self, name + "_q01"), getattr(self, name + "_q99")
                dim = low.shape[-1]
                if dim < value.shape[-1]:
                    first = (value[..., :dim] + 1.0) / 2.0 * (high - low + 1e-6) + low
                    return torch.cat((first, value[..., dim:]), dim=-1)
                return (value + 1.0) / 2.0 * (high - low + 1e-6) + low
            mean, std = getattr(self, name + "_mean"), getattr(self, name + "_std")
            padding = value.shape[-1] - mean.shape[-1]
            if padding < 0:
                raise ValueError("normalization vectors exceed the declared tensor width")
            mean = torch.nn.functional.pad(mean, (0, padding), value=0.0)
            std = torch.nn.functional.pad(std, (0, padding), value=1.0)
            return value * (std + 1e-6) + mean

        def forward(self, state, actions):
            return self.one(state, "state"), self.one(actions, "actions")

    return Unnormalize().eval()


def tensor_absolute_actions(mask):
    import torch

    if mask is not None and (not mask or any(type(item) is not bool for item in mask)):
        raise ValueError("absolute-action mask requires explicit boolean values")

    class Absolute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.tensor(mask or (), dtype=torch.bool))

        def forward(self, state, actions):
            dims = self.mask.shape[0]
            if dims == 0:
                return state, actions
            base = torch.where(self.mask, state[..., :dims], 0)
            first = (actions[..., :dims] + base.unsqueeze(-2)).to(actions.dtype)
            return state, torch.cat((first, actions[..., dims:]), dim=-1)

    return Absolute().eval()


def tensor_aloha_actions(*, adapt_to_pi, joint_flip_mask):
    import numpy as np
    import torch

    if type(adapt_to_pi) is not bool:
        raise TypeError("robot-space adaptation must be explicit")
    flips = np.asarray(joint_flip_mask)
    if flips.shape != (14,) or flips.dtype != np.dtype("int64"):
        raise ValueError("pinned ALOHA output mask contract differs")

    class Aloha(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapt = adapt_to_pi
            self.register_buffer("flips", torch.from_numpy(flips.copy()))
            self.register_buffer("gripper_denominator", torch.tensor([1.4910 - (-0.6213)], dtype=torch.float64))

        def forward(self, state, actions):
            actions = actions[..., :14]
            if self.adapt:
                # NumPy promotes an int64 joint mask times floating actions to float64.
                actions = self.flips * actions.to(torch.float64)
                # CUDA scalar division uses a reciprocal; Tensor division retains NumPy rounding.
                left = ((actions[..., 6:7] + 0.5476) - (-0.6213)) / self.gripper_denominator
                right = ((actions[..., 13:14] + 0.5476) - (-0.6213)) / self.gripper_denominator
                actions = torch.cat((actions[..., :6], left, actions[..., 7:13], right), dim=-1)
            return state, actions

    return Aloha().eval()


def lower_official_output_transform(transform, *, extra_lowerers=None):
    """Lower a verified upstream processor tree; reject unregistered semantics."""
    import torch
    from openpi import transforms
    from openpi.policies import aloha_policy

    registry = {
        transforms.Unnormalize: lambda item: tensor_unnormalize(item.norm_stats, use_quantiles=item.use_quantiles),
        transforms.AbsoluteActions: lambda item: tensor_absolute_actions(item.mask),
        aloha_policy.AlohaOutputs: lambda item: tensor_aloha_actions(
            adapt_to_pi=item.adapt_to_pi, joint_flip_mask=aloha_policy._joint_flip_mask()
        ),
    }
    for kind, lowerer in (extra_lowerers or {}).items():
        if kind in registry:
            raise ValueError("extensions cannot replace a pinned output processor")
        registry[kind] = lowerer
    if type(transform) is not transforms.CompositeTransform:
        raise TypeError("official output lowering requires a concrete processor sequence")
    modules = []
    for item in transform.transforms:
        if type(item) not in registry:
            raise TypeError(f"output processor needs an explicit lowerer: {type(item).__qualname__}")
        modules.append(registry[type(item)](item))

    class Output(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = torch.nn.ModuleList(modules)

        def forward(self, state, actions):
            for stage in self.stages:
                state, actions = stage(state, actions)
            return actions

    return Output().eval()


def checked_output_module(module):
    """Expose native actions and an explicit predicate for atomic publication."""
    import torch

    class Checked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = module

        def forward(self, state, actions, incoming_accepted):
            if incoming_accepted.dtype != torch.bool or tuple(incoming_accepted.shape) != (1,):
                raise ValueError("incoming output acceptance must be bool[1]")
            native = self.output(state, actions)
            accepted = incoming_accepted & torch.isfinite(actions).all() & torch.isfinite(native).all()
            return native, accepted.reshape(1)

    return Checked().eval()
