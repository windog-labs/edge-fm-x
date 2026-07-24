"""Real-checkpoint torch.export audit for SmolVLA TensorRegions."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from vlaforge.adapters.smolvla_real import RealSmolVLAConfig
from vlaforge.frontend import (
    ModelFrontendAudit,
    RegionAuditRecord,
    capture_region,
)
from vlaforge.ir.program import TensorRegion, Value
from vlaforge.ir.types import TensorType


def audit_real_smolvla_frontend(
    config: RealSmolVLAConfig,
    *,
    report_path: str | Path | None = None,
) -> ModelFrontendAudit:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        make_att_2d_masks,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    policy_config = PreTrainedConfig.from_pretrained(
        config.policy_path, local_files_only=True
    )
    policy_config.vlm_model_name = str(config.vlm_path)
    policy_config.device = config.device
    policy_config.num_steps = config.num_steps
    policy = SmolVLAPolicy.from_pretrained(
        config.policy_path,
        config=policy_config,
        local_files_only=True,
        strict=False,
    ).eval()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(
        "pick up the block\n",
        padding="max_length",
        max_length=policy_config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    image_key = next(iter(policy_config.image_features))
    image = torch.linspace(
        0,
        1,
        3 * 256 * 256,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, 3, 256, 256)
    state = torch.linspace(
        -0.2,
        0.3,
        6,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, 6)
    language_tokens = tokens["input_ids"].to(config.device)
    language_masks = tokens["attention_mask"].to(config.device, dtype=torch.bool)

    class PrefixModule(torch.nn.Module):
        def __init__(self, source_policy: Any):
            super().__init__()
            self.policy = source_policy

        def forward(
            self,
            image_value: Any,
            state_value: Any,
            tokens_value: Any,
            masks_value: Any,
        ) -> tuple[Any, ...]:
            batch = {
                image_key: image_value,
                OBS_STATE: state_value,
                OBS_LANGUAGE_TOKENS: tokens_value,
                OBS_LANGUAGE_ATTENTION_MASK: masks_value,
            }
            images, image_masks = self.policy.prepare_images(batch)
            prepared_state = self.policy.prepare_state(batch)
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.policy.model.embed_prefix(
                    images,
                    image_masks,
                    tokens_value,
                    masks_value,
                    state=prepared_state,
                )
            )
            attention_masks = make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks
            )
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, cache = self.policy.model.vlm_with_expert.forward(
                attention_mask=attention_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=policy_config.use_cache,
                fill_kv_cache=True,
            )
            return (prefix_pad_masks,) + tuple(
                value
                for index in range(len(cache))
                for value in (
                    cache[index]["key_states"],
                    cache[index]["value_states"],
                )
            )

    prefix_module = PrefixModule(policy).eval()
    prefix_args = (image, state, language_tokens, language_masks)
    with torch.inference_mode():
        prefix_outputs = prefix_module(*prefix_args)
    prefix_inputs = tuple(
        Value(name, _tensor_type(torch, value))
        for name, value in zip(
            ("image", "state", "language_tokens", "language_masks"),
            prefix_args,
            strict=True,
        )
    )
    prefix_region = TensorRegion(
        "prepare_prefix",
        prefix_inputs,
        tuple(_tensor_type(torch, value) for value in prefix_outputs),
    )
    prefix_capture = capture_region(
        prefix_region,
        prefix_module,
        prefix_args,
        strict=False,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    records = [
        RegionAuditRecord.from_capture(
            prefix_capture,
            source_location=_source_location(policy.model.embed_prefix),
            major_compute=True,
        )
    ]
    del prefix_capture

    sample = torch.linspace(
        -1,
        1,
        policy_config.chunk_size * policy_config.max_action_dim,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, policy_config.chunk_size, policy_config.max_action_dim)
    timestep = torch.ones(1, device=config.device, dtype=torch.float32)
    prefix_pad_masks = prefix_outputs[0]
    flat_cache = prefix_outputs[1:]

    class SolverModule(torch.nn.Module):
        def __init__(self, source_model: Any, cache_layers: int):
            super().__init__()
            self.model = source_model
            self.cache_layers = cache_layers

        def forward(
            self,
            pad_masks: Any,
            sample_value: Any,
            timestep_value: Any,
            *cache_values: Any,
        ) -> Any:
            cache = {
                index: {
                    "key_states": cache_values[index * 2],
                    "value_states": cache_values[index * 2 + 1],
                }
                for index in range(self.cache_layers)
            }
            velocity = self.model.denoise_step(
                prefix_pad_masks=pad_masks,
                past_key_values=cache,
                x_t=sample_value,
                timestep=timestep_value,
            )
            return sample_value - velocity / config.num_steps

    solver_module = SolverModule(policy.model, len(flat_cache) // 2).eval()
    solver_args = (prefix_pad_masks, sample, timestep, *flat_cache)
    solver_names = (
        "prefix_pad_masks",
        "sample",
        "timestep",
        *tuple(
            f"cache_{layer}_{kind}"
            for layer in range(len(flat_cache) // 2)
            for kind in ("key", "value")
        ),
    )
    solver_region = TensorRegion(
        "solver_step",
        tuple(
            Value(name, _tensor_type(torch, value))
            for name, value in zip(solver_names, solver_args, strict=True)
        ),
        (_tensor_type(torch, sample),),
    )
    solver_capture = capture_region(
        solver_region,
        solver_module,
        solver_args,
        strict=False,
        absolute_tolerance=config.tolerance,
        relative_tolerance=0.0,
    )
    records.append(
        RegionAuditRecord.from_capture(
            solver_capture,
            source_location=_source_location(policy.model.denoise_step),
            major_compute=True,
        )
    )
    del solver_capture

    action_dim = int(policy_config.action_feature.shape[0])

    class TrimModule(torch.nn.Module):
        def forward(self, value: Any) -> Any:
            return value[:, :, :action_dim]

    trim_region = TensorRegion(
        "trim_action_chunk",
        (Value("sample", _tensor_type(torch, sample)),),
        (
            TensorType(
                (1, policy_config.chunk_size, action_dim),
                "f32",
            ),
        ),
    )
    trim_capture = capture_region(
        trim_region,
        TrimModule(),
        (sample,),
        strict=True,
    )
    records.append(
        RegionAuditRecord.from_capture(
            trim_capture,
            source_location="adapter:trim_action_chunk",
            major_compute=False,
        )
    )

    checkpoint = config.policy_path / "model.safetensors"
    report = ModelFrontendAudit(
        model="SmolVLA",
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_revision=config.lerobot_revision,
        checkpoint_digests=((checkpoint.name, _sha256(checkpoint)),),
        torch_version=torch.__version__,
        device=config.device,
        persistent_states=("action_queue", "queue_cursor"),
        persistent_state_evidence_complete=True,
        regions=tuple(records),
        notes=(
            "Prefix KV is invocation-local and flattened into explicit tensor ABI values.",
            "Solver sample is loop-carried SSA; it is not a StateSlot.",
            "Noise is an explicit input and no hidden random operator is captured.",
        ),
    )
    if report_path is not None:
        report.write(report_path)
    return report


def _tensor_type(torch: Any, value: Any) -> TensorType:
    dtypes = {
        torch.bool: "bool",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
    }
    if value.dtype not in dtypes:
        raise ValueError(f"unsupported frontend tensor dtype: {value.dtype}")
    return TensorType(tuple(int(dim) for dim in value.shape), dtypes[value.dtype])


def _source_location(function: Any) -> str:
    path = inspect.getsourcefile(function) or "<unknown>"
    try:
        _, line = inspect.getsourcelines(function)
    except (OSError, TypeError):
        line = 0
    return f"{path}:{line}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
