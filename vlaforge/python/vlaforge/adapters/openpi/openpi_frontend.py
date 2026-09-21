"""Shared pi0/pi0.5 adapter for the pinned official OpenPI PyTorch models.

The exported boundary is prepared tensors to a complete normalized action chunk.
Official input/output transforms remain available for independent raw-I/O parity;
this module does not claim those Python processors are deployed in C++.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from vlaforge.adapters.openpi.openpi_checkpoint import (
    OPENPI_REVISION,
    file_digest,
    import_openpi,
)
from vlaforge.frontend import (
    InvocationBuilder,
    InvocationProgram,
    capture_region,
    tensor_region,
)
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


@dataclass(frozen=True, slots=True)
class OpenPIConfig:
    source_root: Path
    checkpoint_dir: Path
    config_name: str
    checkpoint_sha256: str
    device: str = "cpu"
    num_steps: int = 10

    def __post_init__(self) -> None:
        if (
            type(self.num_steps) is not int
            or self.num_steps < 1
            or not self.config_name
            or not self.device
        ):
            raise ValueError("a named static OpenPI inference profile is required")
        if len(self.checkpoint_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.checkpoint_sha256
        ):
            raise ValueError(
                "checkpoint_sha256 must pin the complete converted weights"
            )


@dataclass(frozen=True, slots=True)
class LoadedOpenPI:
    config: OpenPIConfig
    policy: Any
    provenance: Mapping[str, object]

    @property
    def model(self):
        return self.policy._model


@dataclass(frozen=True, slots=True)
class OpenPIInput:
    observation: Any
    tensors: Mapping[str, Any]
    image_count: int
    image_memory_formats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OpenPIFrontend:
    program: InvocationProgram
    example_args: Mapping[str, tuple[Any, ...]]
    normalized_reference: Any
    normalized_partitioned: Any


def load_openpi(config: OpenPIConfig) -> LoadedOpenPI:
    """Load all official weights and transforms, rejecting incomplete conversion."""
    directory = Path(config.checkpoint_dir)
    report = json.loads((directory / "vlaforge_conversion.json").read_text())
    if report.get("schema") != "vlaforge.openpi_conversion/1":
        raise ValueError("a strict OpenPI conversion report is required")
    if report.get("source", {}).get("revision") != OPENPI_REVISION:
        raise ValueError("checkpoint conversion source revision mismatch")
    gate = report.get("load_gate", {})
    if (
        not gate.get("strict")
        or gate.get("missing_required") != []
        or gate.get("unexpected") != []
    ):
        raise ValueError(
            "checkpoint did not pass the complete-parameter conversion gate"
        )
    weight = directory / "model.safetensors"
    actual = file_digest(weight)
    if (
        actual["sha256"] != config.checkpoint_sha256
        or report.get("checkpoint") != actual
    ):
        raise ValueError("converted checkpoint digest mismatch")
    if not report.get("assets"):
        raise ValueError("normalization assets are missing from conversion provenance")
    for name, digest in report["assets"].items():
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or file_digest(path) != digest:
            raise ValueError(f"normalization asset mismatch: {name}")

    from vlaforge.adapters.openpi.openpi_assets import verify_openpi_tokenizer

    tokenizer = verify_openpi_tokenizer()

    import safetensors.torch

    configs = import_openpi(config.source_root)
    train = configs.get_config(config.config_name)
    model_config = replace(train.model, pytorch_compile_mode=None)
    expected = report["model_config"]
    for name in (
        "pi05",
        "action_dim",
        "action_horizon",
        "paligemma_variant",
        "action_expert_variant",
        "max_token_len",
        "discrete_state_input",
        "dtype",
    ):
        if getattr(model_config, name) != expected[name]:
            raise ValueError(f"converted checkpoint profile mismatch: {name}")
    train = replace(train, model=model_config)
    pi0 = importlib.import_module("openpi.models_pytorch.pi0_pytorch")
    model = pi0.PI0Pytorch(model_config).eval()
    missing, unexpected = safetensors.torch.load_model(model, str(weight), strict=True)
    if missing or unexpected:
        raise ValueError(
            f"incomplete checkpoint: missing={missing}, unexpected={unexpected}"
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    transforms = importlib.import_module("openpi.transforms")
    normalization = importlib.import_module("openpi.shared.normalize")
    policies = importlib.import_module("openpi.policies.policy")
    data_factory = replace(
        train.data,
        assets=replace(train.data.assets, assets_dir=str(directory / "assets")),
    )
    data = data_factory.create(directory / "assets", model_config)
    if data.asset_id is None:
        raise ValueError(
            "the selected robot profile must identify normalization assets"
        )
    stats_path = f"assets/{data.asset_id}/norm_stats.json"
    if stats_path not in report["assets"]:
        raise ValueError(
            f"selected normalization assets were not verified: {stats_path}"
        )
    # This is the official load_norm_stats implementation without importing the
    # training data loader (and its unrelated LeRobot/video dependencies).
    stats = normalization.load(directory / "assets" / data.asset_id)
    policy = policies.Policy(
        model,
        transforms=[
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": config.num_steps},
        metadata=train.policy_metadata,
        is_pytorch=True,
        pytorch_device=config.device,
    )
    return LoadedOpenPI(config, policy, {**report, "tokenizer": tokenizer})


def prepare_openpi_inputs(
    loaded: LoadedOpenPI, observation: Mapping[str, object], *, noise: Any
) -> OpenPIInput:
    """Preserve the official model transforms, including pi05 state tokenization."""
    import jax
    import numpy as np
    import torch

    model_types = importlib.import_module("openpi.models.model")
    copied = jax.tree.map(lambda value: value, dict(observation))
    transformed = loaded.policy._input_transform(copied)
    batched = jax.tree.map(
        lambda value: torch.from_numpy(np.array(value)).to(loaded.config.device)[None],
        transformed,
    )
    official = model_types.Observation.from_dict(batched)
    images, masks, tokens, token_mask, state = loaded.model._preprocess_observation(
        official, train=False
    )
    noise = torch.as_tensor(noise, device=loaded.config.device)
    if noise.ndim == 2:
        noise = noise[None]
    expected = (
        state.shape[0],
        loaded.model.config.action_horizon,
        loaded.model.config.action_dim,
    )
    if tuple(noise.shape) != expected or noise.dtype != torch.float32:
        raise ValueError(f"noise must be saved float32 data with shape {expected}")
    values = {
        f"image_{index}": value.contiguous() for index, value in enumerate(images)
    }
    values.update(
        {f"image_mask_{index}": value.contiguous() for index, value in enumerate(masks)}
    )
    values.update(
        language_tokens=tokens.contiguous(),
        language_mask=token_mask.contiguous(),
        state=state.contiguous(),
        noise=noise.contiguous(),
    )
    return OpenPIInput(
        official,
        values,
        len(images),
        tuple(_image_memory_format(value) for value in images),
    )


def _image_memory_format(value):
    import torch

    if value.ndim != 4:
        raise ValueError("official image layout must be a rank-four tensor")
    if value.is_contiguous():
        return "contiguous"
    if value.is_contiguous(memory_format=torch.channels_last):
        return "channels_last"
    raise ValueError(
        "unsupported official image memory format; no silent materialization"
    )


def _restore_image_memory_format(value, memory_format):
    import torch

    if memory_format == "contiguous":
        return value.contiguous()
    if memory_format == "channels_last":
        return value.contiguous(memory_format=torch.channels_last)
    raise ValueError("unsupported recorded image memory format")


def _tensor_type(value: Any) -> TensorType:
    names = {
        "float64": "f64",
        "float32": "f32",
        "bfloat16": "bf16",
        "float16": "f16",
        "int32": "i32",
        "int64": "i64",
        "bool": "bool",
    }
    name = str(value.dtype).removeprefix("torch.")
    if name not in names:
        raise ValueError(f"unsupported OpenPI tensor dtype: {name}")
    return TensorType(tuple(value.shape), names[name])


def compose_openpi_invocation(
    *,
    ports: tuple[InputPort, ...],
    prefix: Any,
    step: Any,
    initialize: Any,
    finite: Any,
    num_steps: int,
    state_in_prefix: bool,
) -> InvocationProgram:
    """Compose declared stages with the generic builder, without executing tensors."""
    prefix_names = tuple(value.name for value in prefix.__vlaforge_region__.inputs)
    if state_in_prefix and "state" not in prefix_names:
        raise ValueError(
            "state-tokenized conditioning must declare its state dependency"
        )
    sample_type = next(port.payload for port in ports if port.name == "noise")
    device = next(port.device for port in ports if port.name == "noise")
    builder = InvocationBuilder(
        "openpi_action_chunk",
        inputs=ports,
        outputs=(
            OutputPort(
                "normalized_action_chunk", sample_type, group="action", device=device
            ),
        ),
        metadata={
            "source_revision": OPENPI_REVISION,
            "model_family": "openpi",
            "state_in_prefix": state_in_prefix,
        },
    )
    inputs = {port.name: builder.input(port.name) for port in ports}
    cached = builder.call(prefix, *(inputs[name] for name in prefix_names), cache=True)
    (time_value,) = builder.call(initialize, inputs["noise"])
    result, _ = builder.iterate(
        (inputs["noise"], time_value),
        lambda _, sample_value, time_value: builder.call(
            step, inputs["state"], sample_value, time_value, *cached
        ),
        steps=num_steps,
    )
    (accepted,) = builder.call(finite, result)
    return builder.finish(
        {"normalized_action_chunk": result},
        accepted=accepted,
        metadata={
            "measurement_boundary": "prepared-tensors-to-normalized-complete-chunk",
            "num_steps": num_steps,
        },
    )


def build_openpi_frontend(
    loaded: LoadedOpenPI,
    prepared: OpenPIInput,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> OpenPIFrontend:
    """Split real prefix/step computations and check all N steps before returning.

    Tolerances are explicit inputs fixed by the caller before running this audit.
    No eager substitution occurs when region capture subsequently fails.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    model, config = loaded.model, loaded.config
    masks_2d = importlib.import_module(
        "openpi.models_pytorch.pi0_pytorch"
    ).make_att_2d_masks
    values = prepared.tensors
    image_count = prepared.image_count
    if len(prepared.image_memory_formats) != image_count:
        raise ValueError("every preprocessed image requires its official memory format")
    prefix_names = (
        *(f"image_{index}" for index in range(image_count)),
        *(f"image_mask_{index}" for index in range(image_count)),
        "language_tokens",
        "language_mask",
        *(("state",) if model.config.discrete_state_input else ()),
    )

    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, *args):
            embeddings, padding, attention = self.model.embed_prefix(
                [
                    _restore_image_memory_format(value, memory_format)
                    for value, memory_format in zip(
                        args[:image_count], prepared.image_memory_formats, strict=True
                    )
                ],
                list(args[image_count : image_count * 2]),
                args[image_count * 2],
                args[image_count * 2 + 1],
            )
            attention = self.model._prepare_attention_masks_4d(
                masks_2d(padding, attention)
            )
            _, cache = self.model.paligemma_with_expert.forward(
                attention_mask=attention,
                position_ids=torch.cumsum(padding, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
            )
            return (padding,) + tuple(
                tensor.contiguous() for layer in cache for tensor in layer
            )

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, state, sample, time, padding, *cache_values):
            cache = DynamicCache.from_legacy_cache(
                tuple(zip(cache_values[::2], cache_values[1::2]))
            )
            # Match official denoise_step, with config mutation moved to load time.
            suffix, suffix_padding, suffix_attention, condition = (
                self.model.embed_suffix(state, sample, time.expand(state.shape[0]))
            )
            prefix_mask = padding[:, None, :].expand(
                padding.shape[0], suffix_padding.shape[1], padding.shape[1]
            )
            attention = torch.cat(
                [prefix_mask, masks_2d(suffix_padding, suffix_attention)], dim=2
            )
            positions = (
                torch.sum(padding, dim=-1)[:, None]
                + torch.cumsum(suffix_padding, dim=1)
                - 1
            )
            embeddings, _ = self.model.paligemma_with_expert.forward(
                attention_mask=self.model._prepare_attention_masks_4d(attention),
                position_ids=positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix],
                use_cache=False,
                adarms_cond=[None, condition],
            )
            suffix = embeddings[1][:, -self.model.config.action_horizon :].to(
                torch.float32
            )
            velocity = self.model.action_out_proj(suffix)
            dt = torch.tensor(
                -1.0 / config.num_steps, dtype=torch.float32, device=sample.device
            )
            return sample + dt * velocity, time + dt

    class InitializeTime(torch.nn.Module):
        def forward(self, noise):
            return torch.ones((), dtype=torch.float32, device=noise.device)

    class Finite(torch.nn.Module):
        def forward(self, sample):
            return torch.isfinite(sample).all().reshape(1)

    prefix, step, initialize, finite = (
        Prefix().eval(),
        Step().eval(),
        InitializeTime(),
        Finite(),
    )
    prefix_args = tuple(values[name] for name in prefix_names)
    with torch.inference_mode():
        context = prefix(*prefix_args)
        frozen_context = tuple(value.clone() for value in context)
        time = initialize(values["noise"])
        step_args = (values["state"], values["noise"], time, *context)
        step_output = step(*step_args)
        sample = values["noise"]
        for _ in range(config.num_steps):
            sample, time = step(values["state"], sample, time, *context)
        for original, after in zip(frozen_context, context, strict=True):
            torch.testing.assert_close(original, after, rtol=0, atol=0)
        reference = model.sample_actions(
            config.device,
            prepared.observation,
            noise=values["noise"].clone(),
            num_steps=config.num_steps,
        )
        torch.testing.assert_close(
            sample, reference, atol=absolute_tolerance, rtol=relative_tolerance
        )

    def declare(name, implementation, input_names, args, results):
        return tensor_region(
            name,
            inputs=tuple(
                Value(name, _tensor_type(arg))
                for name, arg in zip(input_names, args, strict=True)
            ),
            outputs=tuple(_tensor_type(result) for result in results),
        )(implementation)

    prefix = declare("openpi_prefix", prefix, prefix_names, prefix_args, context)
    step_names = (
        "state",
        "sample",
        "time",
        "prefix_padding",
        *(f"cache_{index}" for index in range(len(context) - 1)),
    )
    step = declare("openpi_step", step, step_names, step_args, step_output)
    initialize = declare(
        "openpi_time", initialize, ("noise",), (values["noise"],), (step_args[2],)
    )
    finite = declare("openpi_finite", finite, ("sample",), (sample,), (finite(sample),))
    program = compose_openpi_invocation(
        ports=tuple(
            InputPort(name, _tensor_type(value), device=str(value.device))
            for name, value in values.items()
        ),
        prefix=prefix,
        step=step,
        initialize=initialize,
        finite=finite,
        num_steps=config.num_steps,
        state_in_prefix=bool(model.config.discrete_state_input),
    )
    arguments = {
        next(
            name
            for name in program.regions
            if name.startswith("vf_cached_openpi_prefix")
        ): prefix_args,
        "openpi_step": step_args,
        "openpi_time": (values["noise"],),
        "openpi_finite": (sample,),
    }
    return OpenPIFrontend(program, arguments, reference, sample)


def capture_openpi_frontend(
    frontend: OpenPIFrontend,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
):
    """Return explicit per-region capture results, never an implicit eager fallback."""
    return tuple(
        capture_region(
            region,
            frontend.program.regions[region.name],
            frontend.example_args[region.name],
            strict=False,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for region in frontend.program.module.regions
    )
