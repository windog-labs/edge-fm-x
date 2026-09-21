"""Explicit calibrated INT8 GEMM modules, separate from same-precision export."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict

from vlaforge.analysis.precision_calibration import PrecisionPlan


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _tensor_identity(tensor):
    import torch

    owned = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
    owned.copy_(tensor.detach())
    raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(owned.dtype),
        "shape": list(owned.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _buffer_identity(tensor):
    return {
        **_tensor_identity(tensor),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
    }


class Int8LinearLowering:
    """Owned immutable buffers plus an explicit, not-yet-deployed lowering record."""

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
            raise ValueError("lowered buffers changed after weight quantization")
        return deepcopy({**self._record, "buffers": self._buffers})


def lower_int8_linear(
    plan,
    calibration_report,
    *,
    site,
    step,
    region_artifact_sha256,
    numerical_context,
    weight,
    bias=None,
):
    """Construct one static step/site module using ATen's actual integer GEMM.

    This does not rewrite a Region or select a model deployment. The caller must
    bind the actual source Linear/weight relationship, export on its target,
    verify integer-kernel execution, and validate held-out complete trajectories.
    A report or source digest is provenance, not authentication of caller data.
    """
    import torch

    from vlaforge.numerical_context import NumericalContext

    if not isinstance(plan, PrecisionPlan):
        raise TypeError("INT8 lowering requires a typed PrecisionPlan")
    if str(torch.__version__).split("+", 1)[0] != "2.10.0":
        raise ValueError(
            "this integer-GEMM backend requires the verified Torch 2.10.0 API"
        )
    if (
        not isinstance(numerical_context, NumericalContext)
        or numerical_context.partial
        or _digest(numerical_context.to_dict()) != plan.numerical_context_sha256
    ):
        raise ValueError("actual numerical context must match the calibrated plan")
    numerical_context.require_current()
    if _digest(calibration_report) != plan.calibration_sha256:
        raise ValueError("calibration report differs from the fitted plan")
    if (
        calibration_report.get("profile_sha256") != plan.profile_sha256
        or calibration_report.get("numerical_context_sha256")
        != plan.numerical_context_sha256
        or calibration_report.get("step_keys") != list(plan.step_keys)
        or calibration_report.get("sites") != [asdict(x) for x in plan.sites]
        or _digest(calibration_report.get("split")) != plan.split_sha256
        or calibration_report.get("held_out_used_for_fit") is not False
    ):
        raise ValueError("calibration identities differ from the fitted plan")
    declaration = next((item for item in plan.sites if item.name == site), None)
    if (
        declaration is None
        or not declaration.quantize
        or declaration.artifact_sha256 != region_artifact_sha256
    ):
        raise ValueError("lowering must bind a selected site and exact Region artifact")
    if (declaration.stage == "context" and step is not None) or (
        declaration.stage == "iteration"
        and (type(step) is not int or not 0 <= step < len(plan.step_keys))
    ):
        raise ValueError("step must match the site's explicit schedule stage")
    scale = next(
        item
        for item in plan.scales
        if item.site == site and (step is None or step in item.steps)
    )
    quantized_sites = {item.name for item in plan.sites if item.quantize}
    rows = [
        row
        for row in calibration_report["observations"]
        if (
            row["site"] in quantized_sites
            if plan.strategy == "global"
            else row["site"] == site
        )
        and (
            plan.strategy in ("global", "site")
            or row["step"] is None
            or row["step"] in scale.steps
        )
    ]
    train_ids = {row["sample_id"] for row in calibration_report["split"]["calibration"]}
    if (
        not rows
        or any(row["sample_id"] not in train_ids for row in rows)
        or len(rows) != scale.observations
        or sum(row["elements"] for row in rows) != scale.elements
        or min(row["minimum"] for row in rows) != scale.minimum
        or max(row["maximum"] for row in rows) != scale.maximum
    ):
        raise ValueError("selected scale does not reproduce its calibration group")
    profiles = {
        (row["dtype"], tuple(row["shape"]))
        for row in calibration_report["observations"]
        if row["site"] == site
    }
    if len(profiles) != 1:
        raise ValueError("selected activation requires one static dtype and shape")
    dtype_name, shape = next(iter(profiles))
    if (
        dtype_name not in ("float16", "bfloat16", "float32")
        or len(shape) < 2
        or any(type(dim) is not int or dim < 1 for dim in shape)
    ):
        raise ValueError(
            "INT8 Linear requires a static supported floating input profile"
        )
    dtype = getattr(torch, dtype_name)
    if (
        type(weight) not in (torch.Tensor, torch.nn.Parameter)
        or weight.layout != torch.strided
        or weight.ndim != 2
        or weight.dtype != dtype
        or weight.device.type not in ("cpu", "cuda")
        or weight.shape[1] != shape[-1]
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("finite weight must match the complete input dtype/shape")
    columns, inner = weight.shape
    if inner % 8 or columns < 8 or columns % 8 or math.prod(shape[:-1]) < 16:
        raise ValueError("integer GEMM profile requires M>=16 and K/N multiples of 8")
    if inner * 127 * 127 > torch.iinfo(torch.int32).max:
        raise ValueError("integer GEMM accumulation could overflow int32")
    if bias is not None and (
        type(bias) not in (torch.Tensor, torch.nn.Parameter)
        or bias.layout != torch.strided
        or bias.dtype != dtype
        or bias.shape != (columns,)
        or bias.device != weight.device
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("finite bias must match the original Linear weight")
    owned_weight = weight.detach().to(device="cpu", copy=True).contiguous()
    owned_bias = (
        bias.detach().to(device="cpu", copy=True).contiguous()
        if bias is not None
        else None
    )
    if not bool(torch.isfinite(owned_weight).all()) or (
        owned_bias is not None and not bool(torch.isfinite(owned_bias).all())
    ):
        raise ValueError("owned weight/bias snapshot must be finite")
    source_weight = _tensor_identity(owned_weight)
    source_bias = _tensor_identity(owned_bias) if owned_bias is not None else None
    original = owned_weight.float()
    weight_scale = (original.abs().amax(dim=1) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (
        torch.round(original / weight_scale[:, None]).clamp(-127, 127).to(torch.int8)
    )
    device = weight.device

    class IntegerLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight_int8", quantized.T.contiguous().to(device, copy=True)
            )
            self.register_buffer("weight_scale", weight_scale.to(device, copy=True))
            self.register_buffer(
                "activation_scale",
                torch.tensor(scale.scale, dtype=torch.float32, device=device),
            )
            if owned_bias is not None:
                self.register_buffer(
                    "bias",
                    owned_bias.to(device=device, dtype=torch.float32, copy=True),
                )
            else:
                self.bias = None

        def forward(self, value):
            return self._evaluate(value, self.activation_scale)

        def _evaluate(self, value, activation_scale):
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device != self.weight_int8.device
            ):
                raise ValueError("actual input differs from the locked Linear profile")
            matrix = value.reshape(-1, inner).float()
            packed = (
                torch.round(matrix / activation_scale)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            accumulated = torch.ops.aten._int_mm.default(packed, self.weight_int8)
            output = accumulated.float() * activation_scale
            output = output * self.weight_scale
            if self.bias is not None:
                output = output + self.bias
            # Integer casts must not hide nonfinite inputs from model validators.
            output = torch.where(
                torch.isfinite(matrix).all(dim=1, keepdim=True), output, float("nan")
            )
            return output.to(dtype).reshape(*shape[:-1], columns)

    module = IntegerLinear().eval()
    record = {
        "schema": "vlaforge.int8_linear_lowering/1",
        "precision_plan_sha256": plan.sha256,
        "site": asdict(declaration),
        "step": step,
        "step_key": plan.step_keys[step] if step is not None else None,
        "activation_scale": asdict(scale),
        "input_dtype": dtype_name,
        "input_shape": list(shape),
        "source_weight": source_weight,
        "source_bias": source_bias,
        "weight_quantization": "symmetric_int8_per_output_channel_absolute_max_f32",
        "activation_quantization": "fitted_symmetric_int8_f32",
        "quant_min": -127,
        "quant_max": 127,
        "rounding": "nearest_ties_to_even",
        "weight_scale_floor": float(torch.finfo(torch.float32).tiny),
        "gemm": "aten._int_mm.default",
        "accumulator_dtype": "int32",
        "dequantization": "float32(accumulator) * activation_scale * weight_scale + float32(bias)",
        "torch_version": str(torch.__version__),
        "module_device": str(device),
        "module_constructed": True,
        "region_rewritten": False,
        "backend_lowered": False,
        "real_low_precision_kernel_verified": False,
        "held_out_validation_complete": False,
        "lossless_verified": False,
        "full_model_output_verified": False,
    }
    numerical_context.require_current()
    return Int8LinearLowering(module, record)


def lower_scheduled_int8_linear(
    plan,
    calibration_report,
    *,
    site,
    region_artifact_sha256,
    numerical_context,
    weight,
    bias=None,
):
    """Lower an iterative site with scales selected by an explicit device step.

    The second input is an int64 [1] Tensor, not a Python call counter. Invalid
    indices produce nonfinite output so the invocation commit gate rejects it.
    This does not prove that the caller's index follows the calibrated schedule.
    """
    import torch

    if not isinstance(plan, PrecisionPlan):
        raise TypeError("scheduled lowering requires a typed PrecisionPlan")
    declaration = next((item for item in plan.sites if item.name == site), None)
    if declaration is None or declaration.stage != "iteration":
        raise ValueError("scheduled lowering requires an iterative site")
    lowerings = [
        lower_int8_linear(
            plan,
            calibration_report,
            site=site,
            step=step,
            region_artifact_sha256=region_artifact_sha256,
            numerical_context=numerical_context,
            weight=weight,
            bias=bias,
        )
        for step in range(len(plan.step_keys))
    ]
    records = [item.to_data() for item in lowerings]
    for record in records[1:]:
        for key in ("source_weight", "source_bias"):
            if record[key] != records[0][key]:
                raise ValueError("source Linear state changed while fitting schedule")
    base = lowerings[0].module
    scale_table = torch.stack(
        [item.module.activation_scale for item in lowerings]
    ).detach().clone()
    steps = len(plan.step_keys)

    class ScheduledIntegerLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.register_buffer("scale_table", scale_table)

        def forward(self, value, step_index):
            if (
                step_index.dtype != torch.int64
                or step_index.shape != (1,)
                or step_index.device != self.scale_table.device
            ):
                raise ValueError("step index requires int64 [1] on the input device")
            valid = (step_index >= 0) & (step_index < steps)
            safe_index = torch.where(valid, step_index, torch.zeros_like(step_index))
            scale = torch.index_select(self.scale_table, 0, safe_index).reshape(())
            output = self.base._evaluate(value, scale)
            return torch.where(valid.reshape(()), output, float("nan"))

    module = ScheduledIntegerLinear().eval()
    numerical_context.require_current()
    return Int8LinearLowering(module, {
        "schema": "vlaforge.scheduled_int8_linear_lowering/1",
        "precision_plan_sha256": plan.sha256,
        "site": asdict(declaration),
        "step_keys": list(plan.step_keys),
        "step_lowerings": records,
        "step_input": {"dtype": "int64", "shape": [1]},
        "invalid_step_result": "all_nonfinite_output_for_invocation_commit_rejection",
        "implicit_step_counter": False,
        "module_constructed": True,
        "region_rewritten": False,
        "schedule_binding_verified": False,
        "real_low_precision_kernel_verified": False,
        "full_model_output_verified": False,
        "held_out_validation_complete": False,
        "lossless_verified": False,
    })
