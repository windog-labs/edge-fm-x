#!/usr/bin/env python3
"""
Horizon J6M rewrite helpers used before whole-graph compilation.

The functions in this file are intentionally independent from the C++ runtime.
They prepare a reproducible manifest for Horizon tooling and, when a Python
model object is supplied, monkey-patch the small set of SmolVLA operators that
are known to be fragile for full-int16 J6M compilation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable


J6M_INT16_MIN = -32768
J6M_INT16_MAX = 32767
J6M_MASK_FILL_VALUE = -32760.0
J6M_GELU_CLAMP = 8.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON at {path}, got {type(data).__name__}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _normalized_model_name(compile_spec: dict[str, Any]) -> str:
    raw = str(compile_spec.get("model_name") or compile_spec.get("model_variant") or "")
    out = []
    for ch in raw.lower():
        out.append(ch if ch.isalnum() else "_")
    normalized = "_".join(part for part in "".join(out).split("_") if part)
    if normalized in {"smolvla", "smol_vla", "lerobot_smolvla", "lerobot_smol_vla"}:
        return "smolvla"
    return normalized


def _runtime_hw_profile(compile_spec: dict[str, Any]) -> str:
    graph_tuning = compile_spec.get("graph_tuning", {})
    constraints = graph_tuning.get("target_hw_constraints", {}) if isinstance(graph_tuning, dict) else {}
    engine_config = compile_spec.get("engine_config", {})
    runtime = engine_config.get("runtime", {}) if isinstance(engine_config, dict) else {}
    value = (
        constraints.get("resolved_hw_profile")
        or runtime.get("hw_profile")
        or runtime.get("device")
        or ""
    )
    return str(value).lower()


def should_enable_j6m_rewrite(compile_spec: dict[str, Any], mode: str = "auto") -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    rewrite_cfg = compile_spec.get("horizon_rewrite", {})
    if isinstance(rewrite_cfg, dict) and bool(rewrite_cfg.get("enabled", False)):
        return True
    hw_profile = _runtime_hw_profile(compile_spec)
    return "j6m" in hw_profile or (
        _normalized_model_name(compile_spec) == "smolvla" and "horizon" in hw_profile
    )


def resolve_lerobot_root(compile_spec: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    rewrite_cfg = compile_spec.get("horizon_rewrite", {})
    if isinstance(rewrite_cfg, dict):
        configured = rewrite_cfg.get("lerobot_root")
        if configured:
            return Path(str(configured)).expanduser().resolve()
        smolvla_cfg = rewrite_cfg.get("smolvla")
        if isinstance(smolvla_cfg, dict) and smolvla_cfg.get("lerobot_root"):
            return Path(str(smolvla_cfg["lerobot_root"])).expanduser().resolve()
    env_value = os.environ.get("LEROBOT_ROOT") or os.environ.get("EDGE_FM_LEROBOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/DATA/repos/public/lerobot").expanduser().resolve()


def smolvla_source_files(lerobot_root: Path) -> dict[str, dict[str, Any]]:
    base = lerobot_root / "src" / "lerobot" / "policies" / "smolvla"
    files = {
        "modeling_smolvla": base / "modeling_smolvla.py",
        "smolvlm_with_expert": base / "smolvlm_with_expert.py",
        "configuration_smolvla": base / "configuration_smolvla.py",
        "processor_smolvla": base / "processor_smolvla.py",
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in files.items()
    }


def default_operator_rewrites(model_name: str) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = [
        {
            "id": "full_int16_quantization_contract",
            "kind": "quantization_contract",
            "status": "planned",
            "description": "Compile for signed int16 activation/weight ranges and emit scale diagnostics before HBM build.",
            "int16_range": [J6M_INT16_MIN, J6M_INT16_MAX],
        },
        {
            "id": "attention_mask_bool_to_bounded_bias",
            "kind": "mask",
            "status": "planned",
            "description": "Keep mask construction boolean and use a bounded negative fill instead of fp32 finfo.min/-inf.",
            "bounded_mask_fill": J6M_MASK_FILL_VALUE,
        },
        {
            "id": "rope_explicit_sincos_fp32",
            "kind": "position_encoding",
            "status": "planned",
            "description": "Rewrite RoPE to explicit fp32 inv-frequency/sin/cos tensors before casting back to model dtype.",
        },
        {
            "id": "gelu_tanh_int16_safe_piecewise",
            "kind": "activation",
            "status": "planned",
            "description": "Replace tanh GELU modules/callables with a piecewise equivalent that avoids cubic/tanh overflow.",
            "gelu_input_clamp": J6M_GELU_CLAMP,
        },
        {
            "id": "multimodal_input_normalization_contract",
            "kind": "input_normalization",
            "status": "planned",
            "description": "Record visual/state/action normalization expectations so calibration uses already-normalized inputs.",
        },
    ]
    if model_name == "smolvla":
        rewrites.extend(
            [
                {
                    "id": "smolvla_make_att_2d_masks_int16_cumsum",
                    "kind": "mask",
                    "status": "planned",
                    "source_symbols": ["make_att_2d_masks"],
                    "description": "Use int16 cumsum and bool pad masks for SmolVLA block-causal masks.",
                },
                {
                    "id": "smolvla_eager_attention_bounded_mask_fill",
                    "kind": "attention",
                    "status": "planned",
                    "source_symbols": ["SmolVLMWithExpertModel.eager_attention_forward"],
                    "description": "Replace torch.finfo(float32).min mask fill with a bounded int16-safe value.",
                },
                {
                    "id": "smolvla_flow_matching_loop_bins",
                    "kind": "flow_matching",
                    "status": "planned",
                    "source_symbols": ["VLAFlowMatching.sample_actions", "VLAFlowMatching.denoise_step"],
                    "description": "Export per-step x_t/v_t/time tensors and calibrate the denoise loop step-by-step.",
                },
            ]
        )
    return rewrites


def build_scale_check_config(
    compile_spec: dict[str, Any],
    output_dir: Path,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema": "edgefm_horizon_scale_check_config_v1",
        "stage": stage,
        "model_name": _normalized_model_name(compile_spec),
        "int16_range": [J6M_INT16_MIN, J6M_INT16_MAX],
        "mask_fill_value": J6M_MASK_FILL_VALUE,
        "checks": [
            "parameter_absmax",
            "buffer_absmax",
            "activation_absmax_with_calibration",
            "nonfinite_values",
            "outlier_scale_ratio",
        ],
        "recommended_artifacts": {
            "parameter_report": str(output_dir / "scale_parameter_report.json"),
            "activation_report": str(output_dir / "scale_activation_report.json"),
            "anomaly_report": str(output_dir / "scale_anomalies.json"),
        },
        "notes": [
            "Run activation checks with representative normalized multimodal inputs.",
            "For SmolVLA, calibrate prefix and every flow-matching denoise step separately.",
        ],
    }


def build_flow_matching_export_plan(
    compile_spec: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    model_cfg = compile_spec.get("model_config", {})
    rewrite_cfg = compile_spec.get("horizon_rewrite", {})
    num_steps = 10
    chunk_size = 50
    max_action_dim = 32
    if isinstance(model_cfg, dict):
        num_steps = int(model_cfg.get("num_steps", num_steps))
        chunk_size = int(model_cfg.get("chunk_size", chunk_size))
        max_action_dim = int(model_cfg.get("max_action_dim", max_action_dim))
    if isinstance(rewrite_cfg, dict):
        smolvla_cfg = rewrite_cfg.get("smolvla", {})
        if isinstance(smolvla_cfg, dict):
            num_steps = int(smolvla_cfg.get("num_steps", num_steps))
            chunk_size = int(smolvla_cfg.get("chunk_size", chunk_size))
            max_action_dim = int(smolvla_cfg.get("max_action_dim", max_action_dim))

    bin_dir = output_dir / "flow_matching_bins"
    return {
        "schema": "edgefm_horizon_flow_matching_export_plan_v1",
        "enabled": _normalized_model_name(compile_spec) == "smolvla",
        "num_steps": num_steps,
        "dt": -1.0 / float(num_steps),
        "logical_shapes": {
            "x_t": ["batch", chunk_size, max_action_dim],
            "v_t": ["batch", chunk_size, max_action_dim],
            "time": ["batch"],
        },
        "bin_dir": str(bin_dir),
        "per_step_files": [
            {
                "step": step,
                "time": 1.0 - float(step) / float(num_steps),
                "x_t": str(bin_dir / f"step_{step:02d}_x_t.bin"),
                "v_t": str(bin_dir / f"step_{step:02d}_v_t.bin"),
                "time_tensor": str(bin_dir / f"step_{step:02d}_time.bin"),
            }
            for step in range(num_steps)
        ],
        "calibration": {
            "prefix_cache": "calibrate_once_before_loop",
            "denoise_steps": "calibrate_each_step_or_bucket_steps_with_matching_time_tensor",
            "known_risk": "LLM output and early flow-matching layers tend to dominate drift; inspect these first.",
        },
    }


def build_j6m_rewrite_manifest(
    compile_spec_path: Path,
    compile_spec: dict[str, Any],
    output_dir: Path,
    stage: str,
    lerobot_root: Path | None,
    applied_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_name = _normalized_model_name(compile_spec)
    source_root = lerobot_root or resolve_lerobot_root(compile_spec)
    scale_config_path = output_dir / "scale_check_config.json"
    flow_plan_path = output_dir / "flow_matching_export_plan.json"
    return {
        "schema": "edgefm_horizon_j6m_rewrite_manifest_v1",
        "compile_spec_path": str(compile_spec_path),
        "stage": stage,
        "model_name": model_name,
        "target": {
            "backend": "horizon",
            "hardware": "j6m",
            "quantization": "int16",
            "public_tensor_policy": "copy_in_copy_out",
        },
        "source": {
            "lerobot_root": str(source_root),
            "smolvla_files": smolvla_source_files(source_root) if model_name == "smolvla" else {},
        },
        "operator_rewrites": default_operator_rewrites(model_name),
        "multimodal_normalization": {
            "visual": "identity in LeRobot processor, then image resize/pad and SigLIP preprocessing",
            "state": "mean_std; calibration must use normalized state",
            "action": "mean_std; flow-matching noise/action tensors are in normalized action space",
            "language": "tokenizer padding must match compile-time tokenizer_max_length/pad_language_to",
        },
        "scale_diagnostics": {
            "config_path": str(scale_config_path),
            "int16_range": [J6M_INT16_MIN, J6M_INT16_MAX],
            "anomaly_policy": "fail_compile_on_nonfinite_or_scale_outside_int16_range",
        },
        "flow_matching": {
            "plan_path": str(flow_plan_path),
            "enabled": model_name == "smolvla",
        },
        "applied_summary": applied_summary or {"applied": False, "reason": "model object was not supplied"},
    }


def _require_torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("J6M model rewrites require PyTorch") from exc
    return torch, nn


def make_att_2d_masks_j6m(pad_masks: Any, att_masks: Any):
    torch, _ = _require_torch()
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks.to(torch.int16), dim=1, dtype=torch.int16)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_bool = pad_masks.to(torch.bool)
    pad_2d_masks = pad_bool[:, None, :] & pad_bool[:, :, None]
    return att_2d_masks & pad_2d_masks


def safe_gelu_tanh_j6m(x: Any):
    torch, _ = _require_torch()
    original_dtype = x.dtype
    x32 = x.to(torch.float32)
    middle = torch.clamp(x32, min=-J6M_GELU_CLAMP, max=J6M_GELU_CLAMP)
    inner = math.sqrt(2.0 / math.pi) * (middle + 0.044715 * middle * middle * middle)
    y_middle = 0.5 * middle * (1.0 + torch.tanh(inner))
    y = torch.where(x32 > J6M_GELU_CLAMP, x32, y_middle)
    y = torch.where(x32 < -J6M_GELU_CLAMP, torch.zeros_like(y), y)
    return y.to(original_dtype)


def safe_silu_j6m(x: Any):
    torch, _ = _require_torch()
    original_dtype = x.dtype
    x32 = x.to(torch.float32)
    middle = torch.clamp(x32, min=-J6M_GELU_CLAMP, max=J6M_GELU_CLAMP)
    y_middle = middle * torch.sigmoid(middle)
    y = torch.where(x32 > J6M_GELU_CLAMP, x32, y_middle)
    y = torch.where(x32 < -J6M_GELU_CLAMP, torch.zeros_like(y), y)
    return y.to(original_dtype)


def create_sinusoidal_pos_embedding_j6m(
    time: Any,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | None = "cpu",
):
    torch, _ = _require_torch()
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    target_device = time.device if device is None else device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=target_device)
    log_period = math.log(min_period) + fraction * math.log(max_period / min_period)
    period = torch.exp(log_period)
    sin_input = (2.0 * math.pi) * time.to(torch.float32)[:, None] / period[None, :]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def apply_rope_j6m(x: Any, positions: Any, max_wavelength: float = 10_000):
    torch, _ = _require_torch()
    d_half = x.shape[-1] // 2
    original_dtype = x.dtype
    x32 = x.to(torch.float32)
    inv_freq = torch.exp(
        -math.log(float(max_wavelength))
        * torch.arange(d_half, dtype=torch.float32, device=x.device)
        * (2.0 / float(x.shape[-1]))
    )
    radians = positions[..., None].to(torch.float32) * inv_freq[None, None, :]
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x32.split(d_half, dim=-1)
    first = x1 * cos - x2 * sin
    second = x2 * cos + x1 * sin
    return torch.cat([first, second], dim=-1).to(original_dtype)


class J6MSafeGELU:
    def __new__(cls):
        _, nn = _require_torch()

        class _Module(nn.Module):
            def forward(self, x):
                return safe_gelu_tanh_j6m(x)

        return _Module()


def eager_attention_forward_j6m(
    self,
    attention_mask: Any,
    batch_size: int,
    head_dim: int,
    query_states: Any,
    key_states: Any,
    value_states: Any,
):
    torch, nn = _require_torch()
    num_att_heads = self.num_attention_heads
    num_key_value_heads = self.num_key_value_heads
    num_key_value_groups = num_att_heads // num_key_value_heads
    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )
    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
    key_states = key_states.to(dtype=torch.float32).transpose(1, 2)
    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5
    mask_value = torch.tensor(J6M_MASK_FILL_VALUE, dtype=att_weights.dtype, device=att_weights.device)
    masked_att_weights = torch.where(attention_mask[:, None, :, :].to(torch.bool), att_weights, mask_value)
    probs = nn.functional.softmax(masked_att_weights, dim=-1).to(dtype=value_states.dtype)
    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    return att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)


def _replace_child_modules(root: Any, predicate: Callable[[str, Any], bool], factory: Callable[[Any], Any]) -> int:
    replaced = 0
    for child_name, child in list(root.named_children()):
        if predicate(child_name, child):
            setattr(root, child_name, factory(child))
            replaced += 1
        else:
            replaced += _replace_child_modules(child, predicate, factory)
    return replaced


def _patch_activation_callables(root: Any) -> int:
    replaced = 0
    for module in root.modules():
        activation = getattr(module, "activation_fn", None)
        if activation is None:
            continue
        name = getattr(activation, "__name__", activation.__class__.__name__).lower()
        if "gelu" in name:
            setattr(module, "activation_fn", safe_gelu_tanh_j6m)
            replaced += 1
    return replaced


def _root_module(model_or_policy: Any) -> Any:
    if hasattr(model_or_policy, "model"):
        return model_or_policy.model
    return model_or_policy


def apply_smolvla_j6m_rewrites(model_or_policy: Any, lerobot_root: str | Path | None = None) -> dict[str, Any]:
    torch, nn = _require_torch()
    if lerobot_root:
        root = str(Path(lerobot_root).expanduser().resolve() / "src")
        if root not in sys.path:
            sys.path.insert(0, root)

    applied: dict[str, Any] = {
        "applied": True,
        "module_patches": [],
        "replaced_gelu_modules": 0,
        "replaced_activation_callables": 0,
    }

    try:
        import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla

        modeling_smolvla.make_att_2d_masks = make_att_2d_masks_j6m
        modeling_smolvla.create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_j6m
        applied["module_patches"].extend(
            [
                "modeling_smolvla.make_att_2d_masks",
                "modeling_smolvla.create_sinusoidal_pos_embedding",
            ]
        )
    except Exception as exc:  # pragma: no cover - depends on optional LeRobot install
        applied.setdefault("warnings", []).append(f"Could not patch modeling_smolvla globals: {exc}")

    try:
        import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert

        smolvlm_with_expert.apply_rope = apply_rope_j6m
        smolvlm_with_expert.SmolVLMWithExpertModel.eager_attention_forward = eager_attention_forward_j6m
        applied["module_patches"].extend(
            [
                "smolvlm_with_expert.apply_rope",
                "SmolVLMWithExpertModel.eager_attention_forward",
            ]
        )
    except Exception as exc:  # pragma: no cover - depends on optional LeRobot install
        applied.setdefault("warnings", []).append(f"Could not patch smolvlm_with_expert globals: {exc}")

    root_module = _root_module(model_or_policy)

    def is_gelu(_name: str, child: Any) -> bool:
        class_name = child.__class__.__name__.lower()
        return isinstance(child, nn.GELU) or "gelu" in class_name

    applied["replaced_gelu_modules"] = _replace_child_modules(
        root_module,
        is_gelu,
        lambda _child: J6MSafeGELU(),
    )
    applied["replaced_activation_callables"] = _patch_activation_callables(root_module)
    return applied


def collect_parameter_scale_diagnostics(model: Any) -> dict[str, Any]:
    torch, _ = _require_torch()
    records: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    with torch.no_grad():
        for source, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
            for name, tensor in iterator:
                if tensor is None or not tensor.is_floating_point():
                    continue
                detached = tensor.detach()
                finite = bool(torch.isfinite(detached).all().item())
                absmax = float(detached.abs().max().item()) if detached.numel() else 0.0
                scale = absmax / float(J6M_INT16_MAX) if absmax > 0.0 else 1.0
                record = {
                    "name": name,
                    "source": source,
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                    "finite": finite,
                    "absmax": absmax,
                    "suggested_int16_scale": scale,
                }
                records.append(record)
                if not finite or not math.isfinite(absmax) or absmax > float(J6M_INT16_MAX):
                    anomalies.append(record)
    return {
        "schema": "edgefm_horizon_parameter_scale_report_v1",
        "int16_range": [J6M_INT16_MIN, J6M_INT16_MAX],
        "records": records,
        "anomalies": anomalies,
    }


def export_flow_matching_step_bins(
    step_tensors: dict[str, Any],
    output_dir: Path,
    step: int,
) -> dict[str, Any]:
    torch, _ = _require_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, tensor in step_tensors.items():
        if tensor is None:
            continue
        tensor_cpu = tensor.detach().to("cpu").contiguous()
        path = output_dir / f"step_{step:02d}_{name}.bin"
        path.write_bytes(tensor_cpu.numpy().tobytes())
        files[name] = {
            "path": str(path),
            "shape": list(tensor_cpu.shape),
            "dtype": str(tensor_cpu.dtype),
            "numel": int(tensor_cpu.numel()),
        }
    return {"step": step, "files": files}


def prepare_j6m_rewrite_artifacts(
    compile_spec_path: Path,
    compile_spec: dict[str, Any],
    output_dir: Path,
    stage: str,
    lerobot_root: str | Path | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    root = Path(lerobot_root).expanduser().resolve() if lerobot_root else resolve_lerobot_root(compile_spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    applied_summary = None
    if model is not None and _normalized_model_name(compile_spec) == "smolvla":
        applied_summary = apply_smolvla_j6m_rewrites(model, root)

    scale_config = build_scale_check_config(compile_spec, output_dir, stage)
    flow_plan = build_flow_matching_export_plan(compile_spec, output_dir)
    manifest = build_j6m_rewrite_manifest(
        compile_spec_path=compile_spec_path,
        compile_spec=compile_spec,
        output_dir=output_dir,
        stage=stage,
        lerobot_root=root,
        applied_summary=applied_summary,
    )

    scale_config_path = _write_json(output_dir / "scale_check_config.json", scale_config)
    flow_plan_path = _write_json(output_dir / "flow_matching_export_plan.json", flow_plan)
    manifest_path = _write_json(output_dir / "horizon_j6m_rewrite_manifest.json", manifest)

    parameter_report_path = None
    if model is not None:
        try:
            report = collect_parameter_scale_diagnostics(_root_module(model))
            parameter_report_path = _write_json(output_dir / "scale_parameter_report.json", report)
        except Exception as exc:  # pragma: no cover - optional model/runtime dependent
            manifest.setdefault("warnings", []).append(f"Parameter scale diagnostics failed: {exc}")
            manifest_path = _write_json(output_dir / "horizon_j6m_rewrite_manifest.json", manifest)

    return {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "scale_config_path": str(scale_config_path),
        "flow_plan_path": str(flow_plan_path),
        "parameter_report_path": str(parameter_report_path) if parameter_report_path else "",
        "applied": bool(applied_summary and applied_summary.get("applied")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare EdgeFM Horizon J6M rewrite artifacts")
    parser.add_argument("compile_spec", help="Path to EdgeFM-generated compile_spec.json")
    parser.add_argument("--output-dir", default=None, help="Directory for rewrite manifests")
    parser.add_argument("--stage", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--lerobot-root", default=None, help="Override LeRobot checkout root")
    parser.add_argument("--mode", default="auto", choices=["auto", "on", "off"])
    args = parser.parse_args()

    compile_spec_path = Path(args.compile_spec).resolve()
    compile_spec = _load_json(compile_spec_path)
    if not should_enable_j6m_rewrite(compile_spec, args.mode):
        print(json.dumps({"enabled": False, "compile_spec": str(compile_spec_path)}, indent=2))
        return 0

    output_dir = Path(args.output_dir).resolve() if args.output_dir else compile_spec_path.parent
    result = prepare_j6m_rewrite_artifacts(
        compile_spec_path=compile_spec_path,
        compile_spec=compile_spec,
        output_dir=output_dir,
        stage=args.stage,
        lerobot_root=args.lerobot_root,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
