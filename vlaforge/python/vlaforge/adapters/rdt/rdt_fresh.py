"""Online multimodal RDT fresh chunks through the common InvocationBuilder.

Inputs are explicit token IDs, processed camera pixels, unified state/mask,
control frequency and saved noise. Image decoding and embodiment transforms
remain separately audited input/output boundaries, never guessed from a name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vlaforge.adapters.rdt.rdt_scheduler import make_dpm_solver_step
from vlaforge.frontend import InvocationBuilder, InvocationProgram, tensor_region
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


def _type(value):
    names = {"float32": "f32", "bfloat16": "bf16", "int64": "i64", "bool": "bool"}
    name = str(value.dtype).removeprefix("torch.")
    if name not in names:
        raise ValueError(f"RDT input/output dtype is not declared: {name}")
    return TensorType(tuple(value.shape), names[name])


def _finite_tensor(name, value):
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be an explicit strided tensor")
    if not value.is_contiguous() or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be contiguous, nonempty and finite")


@dataclass(frozen=True)
class RDTFreshProgram:
    program: InvocationProgram
    input_tensors: Mapping[str, Any]
    region_examples: Mapping[str, tuple[Any, ...]]
    eager_actions: Any
    eager_samples: tuple[Any, ...]
    eager_model_outputs: tuple[Any, ...]
    eager_language: Any
    eager_images: Any
    eager_unified_actions: Any = None

    def bind_inputs(self, values=None, *, revisions=None):
        values = self.input_tensors if values is None else values
        names = {port.name for port in self.program.module.inputs}
        if set(values) != names or set(revisions or {}) - names:
            raise ValueError("bindings and revisions must use the declared input ports")
        bindings = {}
        for port in self.program.module.inputs:
            value = values[port.name]
            _finite_tensor(port.name, value)
            if _type(value) != port.payload or str(value.device) != port.device:
                raise ValueError(f"{port.name} changed its locked tensor profile")
            bindings[port.name] = InputBinding(
                TensorView(value, port.payload.shape, port.payload.dtype, device=port.device),
                InputStamp(revision=(revisions or {}).get(port.name)),
            )
        return bindings


def inputs_from_official_reference(loaded, reference):
    """Preserve the captured official tensor boundary and saved initial noise."""
    dimension = loaded.policy.action_dim
    state_and_mask = reference["state_adaptor_inputs"][0]
    values = {
        "token_ids": reference["token_ids"],
        "text_mask": reference["text_attention_mask"],
        "pixels": reference["pixel_values"],
        "unified_state": state_and_mask[..., :dimension],
        "action_mask": state_and_mask[..., dimension:],
        "control_frequency": reference["denoiser_inputs"][0][1],
        "noise": reference["noise"],
    }
    return {name: value.detach().clone().contiguous() for name, value in values.items()}


def build_rdt_fresh_program(loaded, supplied: Mapping[str, Any], *, cache_prefix=True,
                            output_transform=None, output_space=None):
    """Build online inference with an optional explicit pure-Tensor output Adapter.

    Without a transform the original unified-only ABI is unchanged. An explicit
    transform publishes the complete transformed chunk plus a unified auxiliary;
    its selected embodiment/units must be named by the caller and independently
    checked against the official output. No model-name dispatch is performed.
    """
    import torch

    if output_transform is None:
        if output_space is not None:
            raise ValueError("output space requires an explicit output transform")
    elif (not isinstance(output_transform, torch.nn.Module)
          or not isinstance(output_space, str) or not output_space.strip()):
        raise ValueError("output transform requires a Tensor module and explicit output space")

    required = {"token_ids", "text_mask", "pixels", "unified_state", "action_mask", "control_frequency", "noise"}
    if set(supplied) != required:
        raise ValueError(f"explicit RDT tensor inputs required: {sorted(required)}")
    for name, value in supplied.items():
        _finite_tensor(name, value)
    inputs = {name: supplied[name].detach().clone() for name in sorted(required)}
    policy = loaded.policy
    noise = inputs["noise"]
    if noise.dtype != torch.bfloat16 or noise.shape != (1, policy.pred_horizon, policy.action_dim):
        raise ValueError("RDT profile requires a complete BF16 [1,horizon,action_dim] noise tensor")
    if any(value.device != noise.device for value in inputs.values()):
        raise ValueError("all explicit input tensors must share the noise device")
    if inputs["token_ids"].dtype != torch.int64 or inputs["token_ids"].ndim != 2 or inputs["token_ids"].shape[0] != 1:
        raise ValueError("text input requires int64 [1,length] token IDs")
    if inputs["text_mask"].shape != inputs["token_ids"].shape or inputs["text_mask"].dtype not in (torch.int64, torch.bool):
        raise ValueError("text attention mask must match token IDs")
    if not torch.all((inputs["text_mask"] == 0) | (inputs["text_mask"] == 1)) or not inputs["text_mask"].any():
        raise ValueError("text mask must be binary and contain a valid token")
    for name in ("unified_state", "action_mask"):
        if inputs[name].shape != (1, 1, policy.action_dim) or inputs[name].dtype != noise.dtype:
            raise ValueError(f"{name} must be BF16 [1,1,action_dim]")
    if not torch.all((inputs["action_mask"] == 0) | (inputs["action_mask"] == 1)) or not inputs["action_mask"].any():
        raise ValueError("action mask must explicitly select valid dimensions")
    if inputs["control_frequency"].shape != (1,) or not torch.all(inputs["control_frequency"] > 0):
        raise ValueError("control frequency must be a positive [1] tensor")
    pixels = inputs["pixels"]
    vision_config = loaded.vision_tower.config
    if loaded.vision_tower.select_feature != "patch":
        raise ValueError("this vision profile requires the official last-hidden patch features")
    if pixels.dtype != noise.dtype or tuple(pixels.shape) != (6, 3, vision_config.image_size, vision_config.image_size):
        raise ValueError("this recorded profile requires six explicit history-major processed views")
    steps = policy.num_inference_timesteps
    solver = make_dpm_solver_step(policy.noise_scheduler_sample, steps, device=noise.device)

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = loaded.text_embedder.model

        def forward(self, token_ids, text_mask):
            return self.encoder(input_ids=token_ids, attention_mask=text_mask)["last_hidden_state"]

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = loaded.vision_tower.vision_tower

        def forward(self, pixels):
            features = self.encoder(pixels)["last_hidden_state"].to(pixels.dtype)
            return features.reshape(1, -1, features.shape[-1])

    class Conditions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = policy.lang_adaptor
            self.image = policy.img_adaptor
            self.state = policy.state_adaptor

        def forward(self, language, image, state, mask):
            return self.language(language), self.image(image), self.state(torch.cat((state, mask), dim=2))

    class Initialize(torch.nn.Module):
        def forward(self, noise):
            return (torch.zeros_like(noise), torch.zeros_like(noise),
                    torch.zeros_like(noise[:, 0, 0], dtype=torch.int64),
                    torch.zeros_like(noise[:, 0, 0], dtype=torch.int64))

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = policy.model
            self.state_adaptor = policy.state_adaptor
            self.horizon = policy.pred_horizon
            self.register_buffer("timesteps", solver.timesteps.clone())

        def forward(self, sample, step_index, state, language, image, mask, frequency, text_mask):
            actions = self.state_adaptor(torch.cat((sample, mask.expand(-1, self.horizon, -1)), dim=2))
            state_actions = torch.cat((state, actions), dim=1)
            timestep = self.timesteps.index_select(0, step_index)
            return self.model(state_actions, frequency, timestep, language, image, lang_mask=text_mask.to(torch.bool))

    class Finish(torch.nn.Module):
        def forward(self, sample, mask):
            actions = sample * mask.expand_as(sample)
            return actions, torch.isfinite(actions).all().reshape(1)

    text, vision, conditions, initialize, denoiser, finish = (
        Text().eval(), Vision().eval(), Conditions().eval(), Initialize().eval(), Denoiser().eval(), Finish().eval(),
    )
    with torch.inference_mode():
        language = text(inputs["token_ids"], inputs["text_mask"])
        images = vision(pixels)
        adapted = conditions(language, images, inputs["unified_state"], inputs["action_mask"])
        older, previous, count, index = initialize(noise)
        initial = (older, previous, count, index)
        sample = noise
        samples, model_outputs = [], []
        denoiser_args = (sample, index, adapted[2], adapted[0], adapted[1], inputs["action_mask"],
                         inputs["control_frequency"], inputs["text_mask"])
        for _ in range(steps):
            current = denoiser(sample, index, adapted[2], adapted[0], adapted[1], inputs["action_mask"],
                               inputs["control_frequency"], inputs["text_mask"])
            if not samples:
                solver_args = (sample, current, older, previous, count, index)
            sample, older, previous, count, index = solver(sample, current, older, previous, count, index)
            samples.append(sample.clone())
            model_outputs.append(current.clone())
        actions, accepted = finish(sample, inputs["action_mask"])
    _finite_tensor("complete action chunk", actions)

    class TransformOutput(torch.nn.Module):
        def __init__(self, transform):
            super().__init__()
            self.transform = transform

        def forward(self, unified, accepted):
            transformed = self.transform(unified)
            return transformed, accepted & torch.isfinite(transformed).all().reshape(1)

    final_actions = actions
    transform = None
    if output_transform is not None:
        transform = TransformOutput(output_transform)
        with torch.inference_mode():
            final_actions, final_accepted = transform(actions, accepted)
        _finite_tensor("complete transformed action chunk", final_actions)
        if (final_actions.ndim != 3 or final_actions.shape[:2] != actions.shape[:2]
                or final_actions.device != actions.device or not final_accepted.all()):
            raise ValueError("output transform must preserve complete batch/horizon and device")

    declarations = (
        (text, "rdt_text", ("token_ids", "text_mask"), (inputs["token_ids"], inputs["text_mask"]), (language,)),
        (vision, "rdt_vision", ("pixels",), (pixels,), (images,)),
        (conditions, "rdt_conditions", ("language", "image", "state", "mask"),
         (language, images, inputs["unified_state"], inputs["action_mask"]), adapted),
        (initialize, "rdt_initialize", ("noise",), (noise,), initial),
        (denoiser, "rdt_denoiser", ("sample", "index", "state", "language", "image", "mask", "frequency", "text_mask"),
         denoiser_args, (model_outputs[0],)),
        (solver, "rdt_solver", ("sample", "current", "older", "previous", "count", "index"),
         solver_args, (noise, *initial)),
        (finish, "rdt_finish", ("sample", "mask"), (samples[-1], inputs["action_mask"]), (actions, accepted)),
    )
    if transform is not None:
        declarations += ((transform, "rdt_output", ("unified", "accepted"),
                          (actions, accepted), (final_actions, final_accepted)),)
    examples = {}
    for module, name, names, args, outputs in declarations:
        tensor_region(name, inputs=(Value(key, _type(value)) for key, value in zip(names, args, strict=True)),
                      outputs=(_type(value) for value in outputs))(module)
        # Intermediate examples must be ordinary detached tensors: export may
        # inspect autograd metadata even though these Regions are inference-only.
        examples[id(module)] = tuple(value.detach().clone() for value in args)
    builder = InvocationBuilder(
        "rdt_fresh_chunk", inputs=(InputPort(name, _type(value), device=str(value.device)) for name, value in inputs.items()),
        outputs=(OutputPort("action_chunk", _type(final_actions), group="manipulation", device=str(noise.device)),)
        + ((OutputPort("unified_action_chunk", _type(actions), group="manipulation", device=str(noise.device), output_id=1),)
           if transform is not None else ()),
        metadata={"adapter": "RDT", "boundary": "token_ids_processed_pixels_unified_state_to_complete_unified_chunk",
                  "online_language_encoder": True, "online_vision_encoder": True, "num_steps": steps,
                  "scheduler": "diffusers_0.27.2_DPMSolverMultistepScheduler",
                  "history": ["older_model_output", "previous_model_output", "valid_count", "device_step_index"],
                  "coefficients": "official_CPU_primitive_graph_constants_selected_by_device_index",
                  "rng": "saved_noise_input_no_internal_draw", "physical_action_units_verified": False,
                  **({"boundary": "token_ids_processed_pixels_unified_state_to_transformed_and_unified_chunks",
                      "output_space": output_space, "unified_auxiliary_preserved": True}
                     if transform is not None else {})},
    )
    symbols = {name: builder.input(name) for name in inputs}
    (lang,) = builder.call(text, symbols["token_ids"], symbols["text_mask"], cache=cache_prefix)
    (image,) = builder.call(vision, symbols["pixels"], cache=cache_prefix)
    lang, image, state = builder.call(conditions, lang, image, symbols["unified_state"], symbols["action_mask"], cache=cache_prefix)
    start = builder.call(initialize, symbols["noise"])

    def step(_host_index, sample, older, previous, count, index):
        (current,) = builder.call(denoiser, sample, index, state, lang, image, symbols["action_mask"],
                                  symbols["control_frequency"], symbols["text_mask"])
        return builder.call(solver, sample, current, older, previous, count, index)

    sample, *_ = builder.iterate((symbols["noise"], *start), step, steps=steps)
    chunk, accepted = builder.call(finish, sample, symbols["action_mask"])
    outputs = {"action_chunk": chunk}
    if transform is not None:
        final_chunk, accepted = builder.call(transform, chunk, accepted)
        outputs = {"action_chunk": final_chunk, "unified_action_chunk": chunk}
    program = builder.finish(outputs, accepted=accepted)
    return RDTFreshProgram(program, MappingProxyType(inputs),
                           MappingProxyType({name: examples[id(module)] for name, module in program.regions.items()}),
                           final_actions, tuple(samples), tuple(model_outputs), language, images, actions)
