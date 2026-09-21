"""Explicit lower-precision float Linear replacement without INT8 scale fitting."""

from __future__ import annotations

from copy import deepcopy

from vlaforge.deployment.int8_linear import _buffer_identity, _tensor_identity
from vlaforge.numerical_context import NumericalContext


class HalfLinearLowering:
    """Owned immutable half-precision buffers plus an explicit lowering record."""

    def __init__(self, module, record):
        self.module = module
        self._record = deepcopy(record)
        self._buffers = {
            name: _buffer_identity(value) for name, value in module.named_buffers()
        }

    def to_data(self):
        actual = {
            name: _buffer_identity(value) for name, value in self.module.named_buffers()
        }
        if actual != self._buffers:
            raise ValueError("half lowering buffers changed after construction")
        return deepcopy({**self._record, "buffers": self._buffers})


def lower_scheduled_half_linear(
    step_keys,
    *,
    site,
    region_artifact_sha256,
    numerical_context,
    weight,
    bias=None,
    input_dtype="float32",
    input_shape=(1, 50, 720),
    precision="float16",
):
    """Build a static replacement with real half-precision Linear compute.

    The explicit int64 [1] schedule input remains in the public ABI so a
    rewritten Region keeps its original call signature. Invalid indices yield
    nonfinite output instead of silently using a different scheduler. This is a
    real lower-precision candidate, not a claim of quality, speed, memory or
    lossless acceptance.
    """
    import torch

    if not isinstance(numerical_context, NumericalContext) or numerical_context.partial:
        raise ValueError("half lowering requires a complete NumericalContext")
    if not isinstance(step_keys, (tuple, list)) or not step_keys:
        raise ValueError("half lowering requires explicit scheduler step keys")
    if precision not in ("float16", "bfloat16"):
        raise ValueError("half lowering requires float16 or bfloat16")
    if input_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("half replacement source requires a supported float dtype")
    if not input_shape or any(type(dim) is not int or dim < 1 for dim in input_shape):
        raise ValueError("half replacement requires a static nonempty shape")
    dtype = getattr(torch, input_dtype)
    half = getattr(torch, precision)
    if (
        type(weight) not in (torch.Tensor, torch.nn.Parameter)
        or weight.layout != torch.strided
        or weight.ndim != 2
        or weight.dtype != dtype
        or weight.device.type not in ("cpu", "cuda")
        or weight.shape[1] != input_shape[-1]
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("finite source weight must match the static input profile")
    if bias is not None and (
        type(bias) not in (torch.Tensor, torch.nn.Parameter)
        or bias.layout != torch.strided
        or bias.dtype != dtype
        or bias.shape != (weight.shape[0],)
        or bias.device != weight.device
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("finite source bias must match the Linear output columns")
    owned_weight = weight.detach().to(device="cpu", copy=True).contiguous()
    owned_bias = (
        bias.detach().to(device="cpu", copy=True).contiguous()
        if bias is not None
        else None
    )
    device = weight.device
    columns = owned_weight.shape[0]

    class HalfLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight_half", owned_weight.to(half).to(device=device, copy=True)
            )
            if owned_bias is not None:
                self.register_buffer(
                    "bias_half", owned_bias.to(half).to(device=device, copy=True)
                )
            else:
                self.bias_half = None

        def forward(self, value, step_index):
            if (
                value.shape != tuple(input_shape)
                or value.dtype != dtype
                or value.device != self.weight_half.device
                or step_index.dtype != torch.int64
                or step_index.shape != (1,)
                or step_index.device != self.weight_half.device
            ):
                raise ValueError("actual input profile differs from the locked replacement")
            valid = (step_index >= 0) & (step_index < len(step_keys))
            safe = torch.where(valid, step_index, torch.zeros_like(step_index))
            matrix = value.to(half)
            result = torch.ops.aten.linear.default(matrix, self.weight_half, self.bias_half)
            output = result.to(dtype).reshape(*value.shape[:-1], columns)
            return torch.where(valid.reshape(()), output, torch.full_like(output, float("nan")))

    module = HalfLinear().eval()
    numerical_context.require_current()
    return HalfLinearLowering(module, {
        "schema": "vlaforge.scheduled_half_linear_lowering/1",
        "precision": precision,
        "site": site,
        "region_artifact_sha256": region_artifact_sha256,
        "step_keys": list(step_keys),
        "input_dtype": input_dtype,
        "input_shape": list(input_shape),
        "source_weight": _tensor_identity(owned_weight),
        "source_bias": _tensor_identity(owned_bias) if owned_bias is not None else None,
        "linear_implementation": "aten.linear.default after explicit half cast",
        "accumulator_dtype": "same_as_half_compute_or_backend_reduction",
        "invalid_step_result": "all_nonfinite_output_for_invocation_commit_rejection",
        "implicit_step_counter": False,
        "module_constructed": True,
        "region_rewritten": False,
        "schedule_binding_verified": False,
        "real_low_precision_kernel_verified": False,
        "full_model_output_verified": False,
        "held_out_validation_complete": False,
        "lossless_verified": False,
        "torch_version": str(torch.__version__),
        "module_device": str(device),
    })
