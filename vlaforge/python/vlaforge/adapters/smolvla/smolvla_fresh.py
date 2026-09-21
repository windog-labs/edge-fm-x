"""Explicit-input SmolVLA fresh chunks through the generic InvocationBuilder.

This is a tensor-to-tensor boundary: callers supply tokenized, normalized
observations, real camera masks and the saved noise tensor. Dataset decoding
and action denormalization belong to the separately recorded input protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from vlaforge.adapters.smolvla.smolvla_frontend import (
    _assert_tensor_tuples_exact,
    _make_prefix_module,
    _specialize_fixed_vision_positions,
    _tensor_type,
)
from vlaforge.frontend import (
    InvocationBuilder,
    InvocationProgram,
    capture_region,
    save_exported_region,
    tensor_region,
)
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value


STATE = "observation.state"
TOKENS = "observation.language.tokens"
LANGUAGE_MASK = "observation.language.attention_mask"


@dataclass(frozen=True, slots=True)
class SmolVLAFreshProgram:
    program: InvocationProgram
    input_tensors: Mapping[str, Any]
    batch_keys: Mapping[str, str]
    camera_keys: tuple[str, ...]
    region_examples: Mapping[str, tuple[Any, ...]]
    reference_action_chunk: Any
    timestep_table: Any

    def bind_inputs(
        self,
        batch: Mapping[str, Any] | None = None,
        noise: Any | None = None,
        *,
        revisions: Mapping[str, int | None] | None = None,
    ) -> dict[str, InputBinding]:
        """Borrow profile-matching tensors; revisions use the explicit port names."""
        import torch

        if (batch is None) != (noise is None):
            raise ValueError("supply both batch and saved noise, or neither")
        ports = {port.name for port in self.program.module.inputs}
        if set(revisions or {}) - ports:
            raise ValueError("revision keys must name declared input ports")
        if batch is None:
            values = self.input_tensors
        else:
            configured = (
                *self.camera_keys,
                *self.program.module.metadata["missing_camera_keys"],
            )
            if {key for key in configured if key in batch} != set(self.camera_keys):
                raise ValueError("camera availability changed the locked input profile")
            if set(self.batch_keys.values()) - batch.keys():
                raise ValueError(
                    "the locked profile requires every declared batch input"
                )
            values = {name: batch[key] for name, key in self.batch_keys.items()}
            values["noise"] = noise
        result = {}
        for port in self.program.module.inputs:
            value = values[port.name]
            _check_tensor(torch, port.name, value)
            if (
                _tensor_type(torch, value) != port.payload
                or str(value.device) != port.device
            ):
                raise ValueError(
                    f"input {port.name} changed its locked shape/dtype/device profile"
                )
            result[port.name] = InputBinding(
                TensorView(
                    value, port.payload.shape, port.payload.dtype, device=port.device
                ),
                InputStamp(revision=(revisions or {}).get(port.name)),
            )
        return result


def _check_tensor(torch: Any, name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be an explicit tensor")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _profile(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any):
    config = policy.config
    if type(config.num_steps) is not int or config.num_steps < 1:
        raise ValueError("num_steps must be a positive fixed integer")
    if getattr(config, "n_obs_steps", 1) != 1:
        raise ValueError(
            "fresh adapter requires n_obs_steps=1; observation queues are not implicit inputs"
        )
    if getattr(config, "adapt_to_pi_aloha", False):
        raise ValueError(
            "Pi-Aloha transforms are not yet declared by this fresh profile"
        )
    if getattr(config, "rtc_config", None) is not None:
        raise ValueError(
            "RTC history and tracking must be explicit before using this fresh profile"
        )
    if not config.use_cache:
        raise ValueError(
            "fresh prefix profile requires the upstream use_cache=True contract"
        )
    cameras = tuple(key for key in config.image_features if key in batch)
    if not cameras:
        raise ValueError("at least one configured actual camera must be supplied")
    mapping = {}
    for index, key in enumerate(cameras):
        if f"{key}_padding_mask" not in batch:
            raise ValueError(f"camera {key} requires its actual padding mask")
        mapping[f"image_{index}"] = key
        mapping[f"image_mask_{index}"] = f"{key}_padding_mask"
    mapping.update(
        state=STATE, instruction_tokens=TOKENS, instruction_mask=LANGUAGE_MASK
    )
    missing = set(mapping.values()) - batch.keys()
    if missing:
        raise ValueError(f"missing explicit batch inputs: {sorted(missing)}")
    values = {name: batch[key] for name, key in mapping.items()}
    values["noise"] = noise
    for name, value in values.items():
        _check_tensor(torch, name, value)
        if value.ndim == 0:
            raise ValueError(f"{name} requires a batch axis")
    if noise.ndim != 3:
        raise ValueError("saved noise must be FP32 [batch,chunk_size,max_action_dim]")
    batch_size = int(noise.shape[0])
    action_dim = int(config.action_feature.shape[0])
    if (
        noise.dtype != torch.float32
        or tuple(noise.shape)
        != (
            batch_size,
            config.chunk_size,
            config.max_action_dim,
        )
        or not 0 < action_dim <= config.max_action_dim
    ):
        raise ValueError("saved noise must be FP32 [batch,chunk_size,max_action_dim]")
    if any(
        value.device != noise.device or value.shape[0] != batch_size
        for value in values.values()
    ):
        raise ValueError("all input tensors must share the noise batch size and device")
    if values["state"].ndim not in (2, 3) or not values["state"].is_floating_point():
        raise ValueError(
            "state must be a floating [batch,state] or [batch,time,state] tensor"
        )
    tokens, masks = values["instruction_tokens"], values["instruction_mask"]
    if (
        tokens.ndim != 2
        or tokens.dtype != torch.int64
        or masks.dtype != torch.bool
        or masks.shape != tokens.shape
    ):
        raise ValueError(
            "language requires int64 [batch,tokens] and matching bool masks"
        )
    for index in range(len(cameras)):
        image, mask = values[f"image_{index}"], values[f"image_mask_{index}"]
        if (
            image.ndim not in (4, 5)
            or image.shape[-3] != 3
            or not image.is_floating_point()
        ):
            raise ValueError(
                "cameras require floating RGB [batch,3,H,W] or [batch,time,3,H,W]"
            )
        if mask.dtype != torch.bool or tuple(mask.shape) != (batch_size,):
            raise ValueError("camera padding masks must be bool [batch]")
    return cameras, mapping, values


def build_smolvla_fresh_program(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    *,
    cache_prefix: bool = True,
    make_attention_masks: Any | None = None,
    specialize_fixed_vision: bool = False,
) -> SmolVLAFreshProgram:
    """Freeze a real input profile without loading weights or manufacturing data.

    ``policy`` must already hold the strictly loaded checkpoint. Construction
    runs the official full-chunk reference once, preserving its action queues.
    It is an offline preparation operation, not a concurrent policy operation.
    Missing configured views remain missing inputs; only upstream prepare_images
    may apply its declared empty-camera padding. Every present mask is explicit.
    """
    import torch

    cameras, mapping, supplied = _profile(policy, batch, noise, torch)
    if make_attention_masks is None:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        make_attention_masks = make_att_2d_masks
    policy.eval()
    inputs = {name: value.detach().clone() for name, value in supplied.items()}
    reference_batch = {key: inputs[name].clone() for name, key in mapping.items()}
    original_queues = policy._queues
    policy._queues = {name: copy.copy(queue) for name, queue in original_queues.items()}
    try:
        with torch.inference_mode():
            reference = (
                policy.predict_action_chunk(
                    reference_batch, noise=inputs["noise"].clone()
                )
                .detach()
                .clone()
            )
    finally:
        policy._queues = original_queues
    _check_tensor(torch, "official reference chunk", reference)
    expected_shape = (
        noise.shape[0],
        policy.config.chunk_size,
        policy.config.action_feature.shape[0],
    )
    if tuple(reference.shape) != expected_shape or reference.dtype != torch.float32:
        raise ValueError(
            "official reference does not match the full FP32 action-chunk profile"
        )

    prefix = _make_prefix_module(
        torch,
        policy,
        batch_keys=tuple(mapping.values()),
        make_attention_masks=make_attention_masks,
        tokens_key=TOKENS,
        masks_key=LANGUAGE_MASK,
    )
    prefix_args = tuple(inputs[name] for name in mapping)
    with torch.inference_mode():
        prefix_outputs = prefix(*prefix_args)
    if specialize_fixed_vision:
        with torch.inference_mode():
            images, _ = policy.prepare_images(reference_batch)
        resolutions = {tuple(image.shape[-2:]) for image in images}
        if len(resolutions) != 1:
            raise ValueError(
                "fixed vision position specialization requires one prepared resolution"
            )
        height, width = resolutions.pop()
        vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
        original_embeddings = vision.embeddings
        try:
            _specialize_fixed_vision_positions(
                torch, policy, height=height, width=width
            )
            with torch.inference_mode():
                specialized = prefix(*prefix_args)
            _assert_tensor_tuples_exact(torch, prefix_outputs, specialized)
        except Exception:
            vision.embeddings = original_embeddings
            raise
        prefix_outputs = specialized
    for index, value in enumerate(prefix_outputs):
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"prefix output {index} must be nonempty and finite")
    if len(prefix_outputs) < 3 or len(prefix_outputs) % 2 != 1:
        raise ValueError(
            "prefix must return padding masks and complete key/value pairs"
        )
    steps = policy.config.num_steps
    dt = -1.0 / steps

    class InitializeIndex(torch.nn.Module):
        def forward(self, sample: Any) -> Any:
            return torch.zeros_like(sample[:, 0, 0], dtype=torch.int64)

    class SolverStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = policy.model
            self.register_buffer(
                "timesteps",
                torch.tensor(
                    [1.0 + step * dt for step in range(steps)],
                    dtype=torch.float32,
                    device=noise.device,
                ),
            )

        def forward(
            self, pad_masks: Any, sample: Any, step_index: Any, *flat_cache: Any
        ):
            cache = {
                index: {
                    "key_states": flat_cache[index * 2],
                    "value_states": flat_cache[index * 2 + 1],
                }
                for index in range(len(flat_cache) // 2)
            }
            timestep = self.timesteps.index_select(0, step_index)
            velocity = self.model.denoise_step(
                prefix_pad_masks=pad_masks,
                past_key_values=cache,
                x_t=sample,
                timestep=timestep,
            )
            return sample + dt * velocity, step_index + 1

    class FinishChunk(torch.nn.Module):
        def forward(self, sample: Any):
            return sample[:, :, : expected_shape[-1]].contiguous(), torch.isfinite(
                sample
            ).all().reshape(1)

    initialize, solver, finish = (
        InitializeIndex().eval(),
        SolverStep().eval(),
        FinishChunk().eval(),
    )
    with torch.inference_mode():
        index = initialize(inputs["noise"])
    solver_args = (prefix_outputs[0], inputs["noise"], index, *prefix_outputs[1:])
    declarations = (
        (prefix, "smolvla_fresh_prefix", tuple(mapping), prefix_args, prefix_outputs),
        (
            initialize,
            "smolvla_fresh_initialize",
            ("sample",),
            (inputs["noise"],),
            (index,),
        ),
        (
            solver,
            "smolvla_fresh_step",
            (
                "prefix_pad_masks",
                "sample",
                "step_index",
                *(
                    f"cache_{layer}_{kind}"
                    for layer in range((len(prefix_outputs) - 1) // 2)
                    for kind in ("key", "value")
                ),
            ),
            solver_args,
            (inputs["noise"], index),
        ),
        (
            finish,
            "smolvla_fresh_finish",
            ("sample",),
            (inputs["noise"],),
            (
                reference,
                torch.ones(1, dtype=torch.bool, device=noise.device),
            ),
        ),
    )
    examples_by_implementation = {}
    for implementation, name, names, arguments, outputs in declarations:
        tensor_region(
            name,
            inputs=(
                Value(key, _tensor_type(torch, value))
                for key, value in zip(names, arguments, strict=True)
            ),
            outputs=(_tensor_type(torch, value) for value in outputs),
        )(implementation)
        examples_by_implementation[id(implementation)] = arguments
    builder = InvocationBuilder(
        "smolvla_fresh_chunk",
        inputs=(
            InputPort(name, _tensor_type(torch, value), device=str(value.device))
            for name, value in inputs.items()
        ),
        outputs=(
            OutputPort(
                "action_chunk",
                _tensor_type(torch, reference),
                group="manipulation",
                device=str(reference.device),
            ),
        ),
        metadata={
            "adapter": "SmolVLA",
            "boundary": "normalized_tensors_to_fresh_normalized_action_chunk",
            "num_steps": steps,
            "camera_keys": list(cameras),
            "missing_camera_keys": [
                key for key in policy.config.image_features if key not in cameras
            ],
            "upstream_empty_cameras": int(getattr(policy.config, "empty_cameras", 0)),
            "fixed_vision_positions": specialize_fixed_vision,
            "timestep_rule": "FP32(1.0 + python_step * (-1.0 / N)); device index_select",
            "solver_update": "sample + (-1.0 / N) * velocity",
        },
    )
    symbols = {name: builder.input(name) for name in inputs}
    context = builder.call(
        prefix, *(symbols[name] for name in mapping), cache=cache_prefix
    )
    (start_index,) = builder.call(initialize, symbols["noise"])

    def step(_host_index: Any, sample: Any, device_index: Any):
        return builder.call(solver, context[0], sample, device_index, *context[1:])

    sample, _ = builder.iterate((symbols["noise"], start_index), step, steps=steps)
    chunk, accepted = builder.call(finish, sample)
    program = builder.finish({"action_chunk": chunk}, accepted=accepted)
    return SmolVLAFreshProgram(
        program,
        MappingProxyType(inputs),
        MappingProxyType(mapping),
        cameras,
        MappingProxyType(
            {
                name: examples_by_implementation[id(implementation)]
                for name, implementation in program.regions.items()
            }
        ),
        reference,
        solver.timesteps.detach().clone(),
    )


def capture_smolvla_fresh_regions(
    bundle: SmolVLAFreshProgram,
    output_dir: str | Path,
    *,
    strict: bool = False,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> tuple[Any, ...]:
    """Use the shared capture/effect audit and serializer without eager fallback."""
    root = Path(output_dir)
    captures = []
    for region in bundle.program.module.regions:
        outcome = capture_region(
            region,
            bundle.program.regions[region.name],
            bundle.region_examples[region.name],
            strict=strict,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ).require_supported()
        save_exported_region(
            outcome,
            program_path=root / f"{region.name}.pt2e",
            evidence_path=root / f"{region.name}.capture.json",
        )
        captures.append(outcome)
    return tuple(captures)
