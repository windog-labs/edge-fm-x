"""Strict real-weight RDT reference primitives using the official three models.

This module is not an IR capture or no-Python deployment claim. It keeps the
published diffusion scheduler and raw frontend available as reference oracles.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.adapters.rdt.rdt_assets import verify_assets, verify_source

OFFICIAL_PACKAGES = {
    "torch": "2.1.0+cu121", "torchvision": "0.16.0+cu121",
    "transformers": "4.41.0", "diffusers": "0.27.2", "timm": "1.0.3",
    "accelerate": "0.30.1", "sentencepiece": "0.2.0",
    "numpy": "1.26.4", "huggingface-hub": "0.23.4", "protobuf": "4.25.3",
}


@dataclass(frozen=True)
class RDTConfig:
    source_root: Path
    assets_root: Path
    device: str = "cpu"
    numerical_profile: str = "official-cu121"


@dataclass(frozen=True)
class LoadedRDT:
    config: RDTConfig
    policy: Any
    text_embedder: Any
    vision_tower: Any
    wrapper_class: Any
    source_config: Mapping[str, Any]
    provenance: Mapping[str, Any]


def check_environment(profile: str) -> dict:
    expected = dict(OFFICIAL_PACKAGES)
    if profile == "torch210-cu128":
        expected.update(torch="2.10.0+cu128", torchvision="0.25.0+cu128")
    elif profile != "official-cu121":
        raise ValueError("select an explicit official or Torch 2.10 numerical profile")
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"RDT numerical environment differs from {profile}: {actual}")
    return {"profile": profile, "packages": actual,
            "official_torch_baseline": profile == "official-cu121"}


def strict_parameter_subset(model, state: Mapping[str, Any], *, unused_prefixes=(), unused_keys=()) -> dict:
    """Select a documented subnetwork, never silently accept missing model keys."""
    import torch

    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("checkpoint must be a nonempty tensor-only state dictionary")
    expected = set(model.state_dict())
    missing = expected - set(state)
    unused = set(state) - expected
    forbidden = {
        key for key in unused
        if key not in unused_keys and not key.startswith(tuple(unused_prefixes))
    }
    if missing or forbidden:
        raise ValueError(f"incomplete/unknown checkpoint: missing={sorted(missing)}, unexpected={sorted(forbidden)}")
    selected = {key: state[key] for key in expected}
    result = model.load_state_dict(selected, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("strict pretrained parameter loading failed")
    return {
        "strict": True, "missing": [], "unexpected": [],
        "active_state_tensor_count": len(expected),
        "active_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_state_tensor_count": len(state),
        "checkpoint_state_elements_including_aliases": sum(value.numel() for value in state.values()),
        "explicitly_unused_keys": sorted(unused),
    }


def _import_source(root: Path):
    root = root.resolve()
    for name in ("models", "configs", "scripts"):
        imported = sys.modules.get(name)
        if imported is not None:
            locations = [Path(path).resolve() for path in getattr(imported, "__path__", ())]
            if not locations or any(not path.is_relative_to(root) for path in locations):
                raise ValueError(f"upstream import namespace collision: {name}")
    sys.path.insert(0, str(root))
    modules = {
        "runner": importlib.import_module("models.rdt_runner"),
        "text": importlib.import_module("models.multimodal_encoder.t5_encoder"),
        "vision": importlib.import_module("models.multimodal_encoder.siglip_encoder"),
        "wrapper": importlib.import_module("scripts.agilex_model"),
    }
    if any(not Path(module.__file__).resolve().is_relative_to(root) for module in modules.values()):
        raise ValueError("RDT imports resolved outside the verified source snapshot")
    return modules


def load_rdt(config: RDTConfig, *, on_stage=None) -> LoadedRDT:
    """Verify complete assets before loading any real encoder or action-head weight."""
    provenance = {
        "environment": check_environment(config.numerical_profile),
        "source": verify_source(config.source_root),
        "assets": verify_assets(config.assets_root),
        "pickle_loading": "torch.load(weights_only=True, map_location='cpu')",
        "physical_action_units_verified": False,
    }
    def emit(phase):
        if on_stage is not None:
            on_stage(phase, provenance)
    emit("complete_assets_verified")
    import torch
    import yaml
    from safetensors.torch import load_file
    from transformers import (
        AutoTokenizer,
        SiglipVisionConfig,
        SiglipVisionModel,
        T5Config,
        T5EncoderModel,
    )

    modules = _import_source(config.source_root)
    root = config.assets_root
    source_config = yaml.safe_load((config.source_root / "configs/base.yaml").read_text())
    parameters = json.loads((root / "policy/config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        root / "text", model_max_length=parameters["max_lang_cond_len"], local_files_only=True,
    )
    from transformers import SiglipImageProcessor
    image_processor = SiglipImageProcessor.from_pretrained(root / "vision", local_files_only=True)
    emit("processors_initialized")
    architecture_keys = (
        "action_dim", "pred_horizon", "lang_token_dim", "img_token_dim",
        "state_token_dim", "max_lang_cond_len", "img_cond_len",
        "lang_pos_embed_config", "img_pos_embed_config",
    )
    policy = modules["runner"].RDTRunner(
        config=parameters, dtype=torch.bfloat16,
        **{name: parameters[name] for name in architecture_keys},
    )
    state = torch.load(root / "policy/pytorch_model.bin", map_location="cpu", weights_only=True)
    provenance["policy_load"] = strict_parameter_subset(policy, state)
    del state
    policy = policy.eval().to(device=config.device, dtype=torch.bfloat16)
    emit("policy_parameters_loaded_strictly")

    text_config = T5Config.from_pretrained(root / "text", local_files_only=True)
    with torch.device("meta"):
        text_model = T5EncoderModel(text_config)
    state = torch.load(root / "text/pytorch_model.bin", map_location="cpu", weights_only=True)
    if "encoder.embed_tokens.weight" in state and not torch.equal(state["shared.weight"], state["encoder.embed_tokens.weight"]):
        raise ValueError("T5 shared/encoder embedding alias values differ")
    provenance["text_load"] = strict_parameter_subset(
        text_model, state, unused_prefixes=("decoder.",), unused_keys=("lm_head.weight",),
    )
    del state
    text_model.tie_weights()
    text_model = text_model.eval().to(device=config.device, dtype=torch.bfloat16)
    text_model.tie_weights()
    provenance["text_load"]["parameter_elements_before_alias_tying"] = provenance["text_load"]["active_parameter_elements"]
    provenance["text_load"]["active_parameter_elements"] = sum(parameter.numel() for parameter in text_model.parameters())
    emit("text_encoder_parameters_loaded_strictly")
    # The upstream wrapper hardcodes a Hub repo-name allowlist. Construct only its
    # tokenizer/model fields from verified local assets; call its unchanged method.
    text_embedder = object.__new__(modules["text"].T5Embedder)
    text_embedder.device = config.device
    text_embedder.model_max_length = parameters["max_lang_cond_len"]
    text_embedder.tokenizer = tokenizer
    text_embedder.model = text_model

    vision_config = SiglipVisionConfig.from_pretrained(root / "vision", local_files_only=True)
    # SigLIP has nonpersistent position_ids absent from its checkpoint. Let the
    # official constructor materialize those buffers; never replace weights or
    # silently leave uninitialized meta buffers after a strict parameter load.
    vision_model = SiglipVisionModel(vision_config)
    state = load_file(str(root / "vision/model.safetensors"), device="cpu")
    provenance["vision_load"] = strict_parameter_subset(
        vision_model, state, unused_prefixes=("text_model.",), unused_keys=("logit_scale", "logit_bias"),
    )
    del state
    vision_model = vision_model.eval().to(device=config.device, dtype=torch.bfloat16)
    emit("vision_encoder_parameters_loaded_strictly")

    tower = object.__new__(modules["vision"].SiglipVisionTower)
    torch.nn.Module.__init__(tower)
    tower.vision_tower = vision_model
    tower.image_processor = image_processor
    tower.vision_tower_name = str(root / "vision")
    tower.select_feature = "patch"
    tower.is_loaded = True
    tower.eval()
    provenance["scheduler"] = {
        "class": type(policy.noise_scheduler_sample).__name__,
        "config": dict(policy.noise_scheduler_sample.config),
        "num_inference_steps": policy.num_inference_timesteps,
    }
    provenance["online_parameter_elements"] = {
        "rdt_policy": sum(parameter.numel() for parameter in policy.parameters()),
        "t5_encoder": sum(parameter.numel() for parameter in text_model.parameters()),
        "siglip_vision_encoder": sum(parameter.numel() for parameter in vision_model.parameters()),
    }
    provenance["attention_implementation"] = {
        "t5": text_model.config._attn_implementation,
        "siglip": vision_model.config._attn_implementation,
        "rdt_self_attention_fused": bool(policy.model.blocks[0].attn.fused_attn),
        "rdt_cross_attention_fused": bool(policy.model.blocks[0].cross_attn.fused_attn),
    }
    return LoadedRDT(config, policy, text_embedder, tower,
                     modules["wrapper"].RoboticDiffusionTransformerModel,
                     source_config, provenance)


def official_agilex_output_transform(loaded: LoadedRDT):
    """Expose the verified wrapper's unchanged robot-output method as Tensor code.

    Source verification belongs to load_rdt; this does not infer another
    embodiment or certify physical calibration. It preserves the upstream
    index ordering, dtype and gripper-scale construction instead of replacing
    them with a sorted active-dimension selection or FP32 scalar arithmetic.
    """
    import torch

    wrapper = object.__new__(loaded.wrapper_class)
    wrapper.args = loaded.source_config

    class RobotOutput(torch.nn.Module):
        def forward(self, unified):
            return wrapper._unformat_action_to_joint(unified)

    return RobotOutput().eval()


def official_agilex_reference(loaded: LoadedRDT, *, instruction: str, images,
                            proprio, control_frequency: float, seed: int) -> dict:
    """Run real T5 + SigLIP + official five-step RDT on explicitly selected AgileX I/O.

    Returns all unified actions, official robot-space actions, region inputs and
    the exact generated noise/RNG states. This API does not infer embodiment from
    model name, and other embodiments must supply their own checked I/O Adapter.
    """
    import math
    import time

    import torch

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("a real nonempty language instruction is required")
    if len(images) != 6 or not any(image is not None for image in images):
        raise ValueError("explicit history-major six camera slots are required")
    if not math.isfinite(control_frequency) or control_frequency <= 0:
        raise ValueError("control frequency must be finite and positive")
    if proprio.shape != (1, 14) or not torch.isfinite(proprio).all():
        raise ValueError("AgileX proprio must be finite [1,14], with native gripper scale")
    if type(seed) is not int or seed < 0:
        raise ValueError("an explicit nonnegative RNG seed is required")
    traces = {"denoiser_inputs": [], "model_outputs": [], "state_adaptor_inputs": []}
    class CapturedWrapper(loaded.wrapper_class):
        def _unformat_action_to_joint(self, action):
            traces["unified_actions"] = action.detach().clone()
            return super()._unformat_action_to_joint(action)
    wrapper = object.__new__(CapturedWrapper)
    wrapper.args = loaded.source_config
    wrapper.dtype = torch.bfloat16
    wrapper.image_size = None
    wrapper.device = loaded.config.device
    wrapper.control_frequency = control_frequency
    wrapper.image_processor = loaded.vision_tower.image_processor
    wrapper.vision_model = loaded.vision_tower
    wrapper.policy = loaded.policy

    def capture_vision(module, args, output):
        traces["pixel_values"] = args[0].detach().clone()
        traces["image_tokens"] = output.detach().clone()
    def capture_model(module, args, kwargs, output):
        traces["denoiser_inputs"].append(tuple(arg.detach().clone() for arg in args))
        traces["model_outputs"].append(output.detach().clone())
    def capture_state(module, args, output):
        traces["state_adaptor_inputs"].append(args[0].detach().clone())
    def capture_text_inputs(module, args, kwargs):
        traces["token_ids"] = kwargs["input_ids"].detach().clone()
        traces["text_attention_mask"] = kwargs["attention_mask"].detach().clone()
    handles = [
        loaded.vision_tower.register_forward_hook(capture_vision),
        loaded.policy.model.register_forward_hook(capture_model, with_kwargs=True),
        loaded.policy.state_adaptor.register_forward_hook(capture_state),
        loaded.text_embedder.model.register_forward_pre_hook(capture_text_inputs, with_kwargs=True),
    ]
    device = torch.device(loaded.config.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            if device.type == "cuda":
                before = torch.Generator(device=device).manual_seed(seed).get_state()
                torch.cuda.set_rng_state(before, device)
            else:
                torch.random.default_generator.manual_seed(seed)
                before = torch.get_rng_state()
            noise = torch.randn((1, loaded.policy.pred_horizon, loaded.policy.action_dim), dtype=torch.bfloat16, device=device)
            expected_after = torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()
            if devices:
                torch.cuda.set_rng_state(before, device)
                torch.cuda.synchronize(device)
            else:
                torch.set_rng_state(before)
            start = time.perf_counter()
            lang, lang_mask = loaded.text_embedder.get_text_embeddings([instruction])
            actions = wrapper.step(proprio, images, lang)
            if devices:
                torch.cuda.synchronize(device)
            traces["reference_wall_seconds"] = time.perf_counter() - start
            after = torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()
            if not torch.equal(after, expected_after):
                raise ValueError("official pipeline consumed RNG beyond its saved initial noise")
            traces.update(language_tokens=lang.detach().clone(), language_mask=lang_mask.detach().clone(),
                          noise=noise, rng_before=before, rng_after=after, robot_actions=actions)
    finally:
        for handle in handles:
            handle.remove()
    if traces["unified_actions"].shape != (1, loaded.policy.pred_horizon, loaded.policy.action_dim):
        raise ValueError("official output is not the complete unified action chunk")
    if not torch.isfinite(traces["unified_actions"]).all() or not torch.isfinite(actions).all():
        raise ValueError("official complete actions contain non-finite values")
    if len(traces["model_outputs"]) != loaded.policy.num_inference_timesteps:
        raise ValueError("official scheduler did not execute every denoising step")
    actual_noise = traces["state_adaptor_inputs"][1][..., :loaded.policy.action_dim]
    if not torch.equal(actual_noise, noise):
        raise ValueError("captured official initial sample differs from saved noise")
    traces["scheduler_timesteps"] = loaded.policy.noise_scheduler_sample.timesteps.clone()
    traces["scheduler_sigmas"] = loaded.policy.noise_scheduler_sample.sigmas.clone()
    return traces


def partition_reference(loaded: LoadedRDT, reference: Mapping[str, Any]) -> dict:
    """Independently replay exact prefix/denoiser/scheduler stages from saved noise.

    This intentionally uses the official scheduler engine, including its history
    and dtype conversions. Python execution here is an oracle for future tensor
    carry lowering, not evidence that scheduler state is already deployed.
    """
    import torch

    policy = loaded.policy
    noise = reference["noise"]
    if noise.shape != (1, policy.pred_horizon, policy.action_dim) or not torch.isfinite(noise).all():
        raise ValueError("saved complete initial noise must be finite with the official shape")
    state_with_mask = reference["state_adaptor_inputs"][0]
    mask = state_with_mask[..., policy.action_dim:]
    language = reference["language_tokens"]
    language_mask = reference["language_mask"].to(torch.bool)
    if not language_mask.all():
        raise ValueError("the single-instruction official AgileX wrapper uses an all-valid language mask")
    image_tokens = reference["image_tokens"].reshape(1, -1, reference["image_tokens"].shape[-1])
    scheduler = type(policy.noise_scheduler_sample).from_config(dict(policy.noise_scheduler_sample.config))
    scheduler.set_timesteps(policy.num_inference_timesteps)
    if not torch.equal(scheduler.timesteps, reference["scheduler_timesteps"]):
        raise ValueError("fresh scheduler timestep schedule differs from official reference")
    if not torch.equal(scheduler.sigmas, reference["scheduler_sigmas"]):
        raise ValueError("fresh scheduler sigma schedule differs from official reference")
    model_outputs = []
    samples = []
    history = []
    with torch.inference_mode():
        lang, image, state = policy.adapt_conditions(language, image_tokens, state_with_mask)
        sample = noise.clone()
        action_mask = mask.expand(-1, policy.pred_horizon, -1)
        frequency = reference["denoiser_inputs"][0][1]
        for index, timestep in enumerate(scheduler.timesteps):
            adapted_actions = policy.state_adaptor(torch.cat((sample, action_mask), dim=2))
            state_actions = torch.cat((state, adapted_actions), dim=1)
            model_output = policy.model(state_actions, frequency, timestep.unsqueeze(-1).to(sample.device),
                                        lang, image, lang_mask=language_mask)
            model_outputs.append(model_output.detach().clone())
            sample = scheduler.step(model_output, timestep, sample).prev_sample.to(state.dtype)
            samples.append(sample.detach().clone())
            history.append({
                "step_index": scheduler.step_index, "lower_order_nums": scheduler.lower_order_nums,
                "valid_model_outputs": [value is not None for value in scheduler.model_outputs],
                "timestep": int(timestep),
                "model_output_exact": torch.equal(model_output, reference["model_outputs"][index]),
            })
        actions = sample * action_mask
    if not torch.isfinite(actions).all():
        raise ValueError("partitioned complete output contains non-finite values")
    return {
        "actions": actions, "samples": tuple(samples), "model_outputs": tuple(model_outputs),
        "scheduler_history": history, "complete_chunk_exact": torch.equal(actions, reference["unified_actions"]),
        "scheduler_state_lowered_to_cpp": False,
    }
