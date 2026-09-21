"""CogACT tensor partition with an explicit externally produced CUDA RNG tape.

The producer owns the 11 real upstream global-PRNG draws. The Invocation tracks
tape consumption and emits its final state receipt; it is not an autonomous
C++ Torch/Philox implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.frontend import (
    InvocationBuilder,
    capture_region,
    save_exported_region,
    tensor_region,
    tensor_type_from_torch,
)
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


@dataclass
class CogACTPartition:
    program: Any
    inputs: dict[str, Any]
    examples: dict[str, tuple[Any, ...]]
    implementations: dict[str, Any]

    def bind_inputs(self, revision: int) -> dict[str, InputBinding]:
        return {port.name: InputBinding(
            TensorView(self.inputs[port.name], port.payload.shape, port.payload.dtype, device=port.device),
            InputStamp(revision=revision),
        ) for port in self.program.module.inputs}


def tensor_type(value: Any) -> TensorType:
    return tensor_type_from_torch(value)


def make_model_inputs(model: Any, image: Any, instruction: str) -> dict[str, Any]:
    return prepare_cogact_observation(model.vlm.llm_backbone.tokenizer,
        model.vlm.get_prompt_builder(), model.vlm.vision_backbone.image_transform,
        image, instruction, device=model.vlm.device)


def prepare_cogact_observation(tokenizer, prompt, image_transform, image, instruction, *, device):
    """Apply the checked upstream processors without requiring model weights."""
    import torch
    from transformers import LlamaTokenizerFast

    if not isinstance(tokenizer, LlamaTokenizerFast):
        raise TypeError("the released CogACT profile requires the actual Llama fast tokenizer")
    prompt.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    tokens = tokenizer(prompt.get_prompt(), truncation=True, return_tensors="pt").input_ids.to(device)
    tokens = torch.cat([tokens, torch.tensor([[29871, 2]], dtype=torch.int64, device=tokens.device)], dim=1)
    if tokens.shape[0] != 1 or bool((tokens == tokenizer.pad_token_id).any()):
        raise ValueError("this fixed all-multimodal profile excludes padding and batches other than one")
    pixels = image_transform(image)
    if not isinstance(pixels, dict) or set(pixels) != {"dino", "siglip"}:
        raise ValueError("the released CogACT profile requires both actual Dino and SigLIP inputs")
    return {"tokens": tokens, **{key: value.unsqueeze(0).to(device) for key, value in pixels.items()}}


def produce_rng_tape(model: Any) -> dict[str, Any]:
    """Perform the actual ordered upstream draws, preserving eta-zero consumption."""
    import torch

    device = model.vlm.device
    if (device.type != "cuda" or model.action_model.ddim_diffusion.num_timesteps != 10
            or model.future_action_window_size != 15 or model.action_model.in_channels != 7):
        raise ValueError("the bounded real CUDA profile requires the official ten-step scheduler")
    states = [torch.cuda.get_rng_state(device).clone()]
    initial = torch.randn(1, 16, 7, device=device).to(next(model.action_model.net.parameters()).dtype)
    states.append(torch.cuda.get_rng_state(device).clone())
    template = torch.cat([initial, initial], dim=0)
    draws = []
    for _ in range(10):
        draws.append(torch.randn_like(template))
        states.append(torch.cuda.get_rng_state(device).clone())
    return {"initial_noise": initial, "step_noise": torch.stack(draws),
            "rng_states": torch.stack(states).to(device), "rng_before": states[0].to(device)}


def build_cogact_partition(model: Any, supplied: dict[str, Any], *, unnorm_key: str) -> CogACTPartition:
    import numpy as np
    import torch

    expected = {"tokens", "dino", "siglip", "initial_noise", "step_noise", "rng_states", "rng_before"}
    if set(supplied) != expected:
        raise ValueError("complete explicit observation and RNG tape inputs required")
    if tuple(supplied["initial_noise"].shape) != (1, 16, 7) or tuple(supplied["step_noise"].shape) != (10, 2, 16, 7):
        raise ValueError("unexpected full-chunk RNG draw shapes")
    if supplied["rng_states"].shape != (12, supplied["rng_before"].numel()):
        raise ValueError("RNG tape must retain before, after initial draw and all ten step states")
    expected_dtypes = {"tokens": torch.int64, "rng_states": torch.uint8, "rng_before": torch.uint8,
                       "dino": torch.float32, "siglip": torch.float32,
                       "initial_noise": torch.float32, "step_noise": torch.float32}
    if any(supplied[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("the released precision profile requires FP32 observations/noise and byte-exact RNG states")
    if (supplied["tokens"].ndim != 2 or supplied["tokens"].shape[0] != 1
            or tuple(supplied["dino"].shape) != (1, 3, 224, 224)
            or tuple(supplied["siglip"].shape) != (1, 3, 224, 224)
            or supplied["rng_before"].ndim != 1):
        raise ValueError("unsupported batch-one image/token/RNG profile")
    if any(value.device != model.vlm.device or not value.is_contiguous() for value in supplied.values()):
        raise ValueError("partition inputs must be contiguous on the model CUDA device")
    scheduler = model.action_model.ddim_diffusion
    if (scheduler.num_timesteps != 10 or scheduler.original_num_steps != 100
            or scheduler.timestep_map != list(range(0, 100, 10))
            or str(scheduler.model_mean_type) != "ModelMeanType.EPSILON"
            or str(scheduler.model_var_type) != "ModelVarType.FIXED_SMALL"):
        raise ValueError("unsupported scheduler profile; do not approximate another scheduler")
    decoder = model.vlm.llm_backbone.llm.model
    if decoder.config._attn_implementation != "sdpa" or any(hasattr(layer.self_attn, "past_key_value") for layer in decoder.layers):
        raise ValueError("the fixed unpadded prefill profile requires native SDPA without static past state")

    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = model.vlm.vision_backbone
            self.projector = model.vlm.projector
            self.llm = model.vlm.llm_backbone
            self.autocast_enabled = model.vlm.enable_mixed_precision_training
            self.dino_selection = (len(self.vision.dino_featurizer.blocks) - 2,)
            self.siglip_selection = (len(self.vision.siglip_featurizer.blocks) - 2,)

        def forward(self, tokens, dino, siglip):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled):
                indices = torch.arange(tokens.shape[0], dtype=torch.long, device=tokens.device)
                # Same upstream second-last features; tuples avoid Torch 2.10's
                # strict-export assertion when timm copies a captured Python set.
                dino_patches = self.vision.dino_featurizer.get_intermediate_layers(dino[indices], n=self.dino_selection)[0]
                siglip_patches = self.vision.siglip_featurizer.get_intermediate_layers(siglip[indices], n=self.siglip_selection)[0]
                patches = torch.cat([dino_patches, siglip_patches], dim=2)
                projected = self.projector(patches)
                embeddings = self.llm.embed_input_ids(tokens)
                fused = torch.cat([embeddings[indices, :1, :], projected, embeddings[indices, 1:, :]], dim=1)
                # This fixed prefill has no padding or prior tokens. Keep the
                # upstream eager implicit-causal SDPA route under export too.
                positions = torch.arange(fused.shape[1], device=fused.device)
                position_ids = positions.unsqueeze(0)
                hidden = fused
                for layer in self.llm.llm.model.layers:
                    hidden = layer(hidden, attention_mask=None, position_ids=position_ids,
                                   past_key_value=None, output_attentions=False, use_cache=False,
                                   cache_position=positions)[0]
                hidden = self.llm.llm.model.norm(hidden)
            return hidden[:, -1:, :].to(torch.float32)

    class Initialize(torch.nn.Module):
        def forward(self, initial_noise, rng_states, rng_before):
            return (torch.cat([initial_noise, initial_noise], dim=0),
                    torch.zeros_like(initial_noise[:, 0, 0], dtype=torch.int64),
                    rng_states[1].clone(), (rng_states[0] == rng_before).all().reshape(1))

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = model.action_model.net
            for name in ("sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
                         "alphas_cumprod", "alphas_cumprod_prev"):
                self.register_buffer(name, torch.from_numpy(getattr(scheduler, name)).to(model.vlm.device))
            self.register_buffer("timestep_map", torch.tensor(scheduler.timestep_map, dtype=torch.int64, device=model.vlm.device))

        def forward(self, cognition, sample, index, receipt, valid, noise_tape, rng_states):
            small_t = (9 - index).expand(sample.shape[0])
            original_t = self.timestep_map.index_select(0, small_t)
            uncondition = self.net.z_embedder.uncondition.unsqueeze(0).expand(1, 1, -1)
            z = torch.cat([cognition, uncondition], dim=0)
            prediction = self.net.forward_with_cfg(sample, original_t, z, 1.5)
            recip = self.sqrt_recip_alphas_cumprod.index_select(0, small_t).float().reshape(2, 1, 1)
            recipm1 = self.sqrt_recipm1_alphas_cumprod.index_select(0, small_t).float().reshape(2, 1, 1)
            pred_xstart = recip * sample - recipm1 * prediction
            eps = (recip * sample - pred_xstart) / recipm1
            alpha = self.alphas_cumprod.index_select(0, small_t).float().reshape(2, 1, 1)
            alpha_prev = self.alphas_cumprod_prev.index_select(0, small_t).float().reshape(2, 1, 1)
            sigma = 0.0 * torch.sqrt((1 - alpha_prev) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_prev)
            noise = noise_tape.index_select(0, index).squeeze(0)
            mean = pred_xstart * torch.sqrt(alpha_prev) + torch.sqrt(1 - alpha_prev - sigma ** 2) * eps
            nonzero = (small_t != 0).float().reshape(2, 1, 1)
            result = mean + nonzero * sigma * noise
            current = rng_states.index_select(0, index + 1).squeeze(0)
            next_receipt = rng_states.index_select(0, index + 2).squeeze(0)
            return result, index + 1, next_receipt, valid & (receipt == current).all().reshape(1)

    class Finish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pad_token_id = model.vlm.llm_backbone.tokenizer.pad_token_id
            self.register_buffer("required_suffix", torch.tensor([[29871, 2]], dtype=torch.int64, device=model.vlm.device))
            stats = model.get_action_stats(unnorm_key)
            for name, value in {"high": np.array(stats["q99"]), "low": np.array(stats["q01"]),
                                "mask": np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))}.items():
                self.register_buffer(name, torch.from_numpy(value).to(model.vlm.device))

        def forward(self, sample, index, receipt, valid, rng_states, tokens):
            raw = sample[0].contiguous()
            clipped = raw.clamp(-1, 1)
            gripper = torch.where(clipped[:, 6:] < 0.5, 0.0, 1.0)
            normalized = torch.cat([clipped[:, :6], gripper], dim=1)
            scaled = (0.5 * (normalized + 1)).to(torch.float64) * (self.high - self.low) + self.low
            native = torch.where(self.mask, scaled, normalized.to(torch.float64))
            accepted = (valid & (index == 10) & torch.isfinite(raw).all().reshape(1)
                        & torch.isfinite(native).all().reshape(1) & (receipt == rng_states[-1]).all().reshape(1)
                        & (tokens != self.pad_token_id).all().reshape(1)
                        & (tokens[:, -2:] == self.required_suffix).all().reshape(1))
            return raw, normalized, native, receipt.clone(), index + 1, accepted

    implementations = {"prefix": Prefix().eval(), "initialize": Initialize().eval(), "step": Step().eval(), "finish": Finish().eval()}
    inputs = dict(supplied)
    with torch.inference_mode():
        context = implementations["prefix"](inputs["tokens"], inputs["dino"], inputs["siglip"])
        initial = implementations["initialize"](inputs["initial_noise"], inputs["rng_states"], inputs["rng_before"])
        step_args = (context, *initial, inputs["step_noise"], inputs["rng_states"])
        step_outputs = implementations["step"](*step_args)
        finish_args = (*step_outputs, inputs["rng_states"], inputs["tokens"])
        finish_outputs = implementations["finish"](*finish_args)
    declarations = (
        ("prefix", ("tokens", "dino", "siglip"), (inputs["tokens"], inputs["dino"], inputs["siglip"]), (context,)),
        ("initialize", ("initial_noise", "rng_states", "rng_before"), (inputs["initial_noise"], inputs["rng_states"], inputs["rng_before"]), initial),
        ("step", ("cognition", "sample", "index", "receipt", "valid", "noise_tape", "rng_states"), step_args, step_outputs),
        ("finish", ("sample", "index", "receipt", "valid", "rng_states", "tokens"), finish_args, finish_outputs),
    )
    examples_by_implementation = {}
    for name, names, arguments, outputs in declarations:
        region_name = "cogact_" + name
        tensor_region(region_name, inputs=[Value(key, tensor_type(value)) for key, value in zip(names, arguments, strict=True)],
                      outputs=[tensor_type(value) for value in outputs])(implementations[name])
        examples_by_implementation[id(implementations[name])] = arguments
    output_names = ("raw_action_chunk", "normalized_action_chunk", "native_action_chunk", "rng_after", "draws_consumed")
    builder = InvocationBuilder("cogact_explicit_rng_tape", inputs=[
        InputPort(name, tensor_type(value), device=str(value.device)) for name, value in inputs.items()],
        outputs=[OutputPort(name, tensor_type(value), group="manipulation", device=str(value.device))
                 for name, value in zip(output_names, finish_outputs[:-1], strict=True)],
        metadata={"adapter": "CogACT", "reference_identity": "official-code-public-dependency-candidate",
                  "num_steps": 10, "cfg_scale": 1.5, "eta": 0.0,
                  "rng_boundary": "external producer performs 11 actual Torch CUDA draws; IR consumes tape and emits state receipt",
                  "autonomous_cpp_rng": False, "attention_profile": "batch-one unpadded all-multimodal",
                  "prefix_online_compute": "native checkpoint decoder hidden states; unconsumed lm_head omitted",
                  "complete_checkpoint_still_loaded": True,
                  "prefill_state_profile": "q_len=k_len; implicit causal; no prior state or retained transient KV",
                  "native_postprocessing": "FP32 clip/gripper/scale then FP64 checkpoint q01/q99/mask"})
    symbols = {name: builder.input(name) for name in inputs}
    (cognition,) = builder.call(implementations["prefix"], symbols["tokens"], symbols["dino"], symbols["siglip"], cache=True)
    initial_values = builder.call(implementations["initialize"], symbols["initial_noise"], symbols["rng_states"], symbols["rng_before"])

    def step(_host_index, *carried):
        return builder.call(implementations["step"], cognition, *carried, symbols["step_noise"], symbols["rng_states"])

    carried = builder.iterate(initial_values, step, steps=10)
    result = builder.call(implementations["finish"], *carried, symbols["rng_states"], symbols["tokens"])
    program = builder.finish(dict(zip(output_names, result[:-1], strict=True)), accepted=result[-1])
    examples = {name: examples_by_implementation[id(implementation)] for name, implementation in program.regions.items()}
    return CogACTPartition(program, inputs, examples, implementations)


def capture_cogact_partition(partition: CogACTPartition, output: Path) -> tuple[Any, ...]:
    import json
    import traceback
    from types import SimpleNamespace

    import torch

    from vlaforge.frontend.effect_audit import audit_exported_program

    output.mkdir(parents=True, exist_ok=False)
    captures = []
    failures = []
    for region in partition.program.module.regions:
        outcome = capture_region(region, partition.program.regions[region.name], partition.examples[region.name],
                                 strict=True, absolute_tolerance=0.0, relative_tolerance=0.0)
        if not outcome.supported:
            (output / f"{region.name}.unsupported.json").write_text(json.dumps(outcome.report.to_dict(), indent=2) + "\n")
            # Preserve the upstream traceback that the public structured failure
            # deliberately condenses. This diagnostic is never promoted to a capture.
            if outcome.report.stage in {"torch.export", "contract_validation"}:
                phase = "strict-export"
                try:
                    diagnostic = torch.export.export(partition.program.regions[region.name], partition.examples[region.name], strict=True)
                    if outcome.report.stage == "contract_validation":
                        from vlaforge.numerical_context import snapshot

                        torch.export.save(diagnostic, output / f"{region.name}.diagnostic-only.pt2e")
                        torch.save(tuple(value.detach().cpu() for value in partition.examples[region.name]),
                                   output / f"{region.name}.diagnostic-inputs.pt")
                        (output / f"{region.name}.diagnostic-context.json").write_text(json.dumps(
                            {"scope": "failed capture diagnostic only", "numerical_context": snapshot().to_dict()}, indent=2) + "\n")
                        (output / f"{region.name}.diagnostic-code.txt").write_text("\n\n".join(
                            f"MODULE {name}\n{module.code}" for name, module in diagnostic.graph_module.named_modules()
                            if isinstance(module, torch.fx.GraphModule)))
                        with torch.no_grad():
                            phase = "eager-execution"
                            partition.program.regions[region.name](*partition.examples[region.name])
                            phase = "exported-execution"
                            diagnostic.module()(*partition.examples[region.name])
                except (AssertionError, RuntimeError, ValueError, TypeError):
                    (output / f"{region.name}.upstream-traceback.txt").write_text(f"PHASE: {phase}\n" + traceback.format_exc())
            failures.append(outcome)
            continue
        nested_audits = []
        for name, module in outcome.exported_program.graph_module.named_modules():
            if name and isinstance(module, torch.fx.GraphModule):
                audit = audit_exported_program(SimpleNamespace(
                    graph_module=module, graph_signature=SimpleNamespace(input_specs=())))
                nested_audits.append({"scope": name, "effect_audit": audit.to_dict()})
        (output / f"{region.name}.nested-effect-audit.json").write_text(json.dumps(nested_audits, indent=2) + "\n")
        if any(not item["effect_audit"]["passed"] for item in nested_audits):
            raise ValueError(f"nested Tensor graph effect audit failed: {region.name}")
        save_exported_region(outcome, program_path=output / f"{region.name}.pt2e",
                             evidence_path=output / f"{region.name}.capture.json")
        captures.append(outcome)
    if failures:
        failures[0].require_supported()
    return tuple(captures)
