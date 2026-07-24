"""Real-checkpoint BF16 torch.export audit for OpenVLA prefill/decode regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.adapters.openvla_real import (
    _checkpoint_shards,
    _deterministic_image,
    _validate_checkpoint,
)
from vlaforge.frontend import (
    ModelFrontendAudit,
    RegionAuditRecord,
    capture_region,
    save_exported_region,
)
from vlaforge.ir.program import TensorRegion, Value
from vlaforge.ir.types import TensorType


@dataclass(frozen=True, slots=True)
class OpenVLAFrontendConfig:
    checkpoint_path: Path
    revision: str
    device: str = "cpu"
    unnorm_key: str = "bridge_orig"
    instruction: str = "pick up the block"
    tolerance: float = 0.0
    cpu_threads: int = 16


def audit_real_openvla_frontend(
    config: OpenVLAFrontendConfig,
    *,
    report_path: str | Path | None = None,
    export_dir: str | Path | None = None,
    torchscript_dir: str | Path | None = None,
) -> ModelFrontendAudit:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    checkpoint = config.checkpoint_path.resolve()
    _validate_checkpoint(checkpoint)
    if config.device != "cpu":
        raise ValueError(
            "OpenVLA frontend audit currently uses BF16 CPU to avoid "
            "bitsandbytes FakeTensor mutation during torch.export"
        )
    torch.set_num_threads(config.cpu_threads)
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cpu"},
    ).eval()
    action_dim = int(model.get_action_dim(config.unnorm_key))
    image = _deterministic_image(np, Image)
    prompt = (
        "In: What action should the robot take to "
        f"{config.instruction.strip().lower()}?\nOut: "
    )
    model_inputs = processor(prompt, image).to(
        config.device, dtype=torch.bfloat16
    )
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor([[29871]], dtype=input_ids.dtype),
            ),
            dim=1,
        )

    class PrefillModule(torch.nn.Module):
        def __init__(self, source_model: Any):
            super().__init__()
            self.model = source_model

        def forward(
            self, pixels: Any, tokens: Any, masks: Any
        ) -> tuple[Any, ...]:
            patch_features = self.model.vision_backbone(pixels)
            projected_patches = self.model.projector(patch_features)
            projected_mask = masks.new_ones(
                (
                    projected_patches.shape[0],
                    projected_patches.shape[1],
                )
            )
            token_embeddings = self.model.get_input_embeddings()(tokens)
            embeddings = torch.cat(
                (
                    token_embeddings[:, :1, :],
                    projected_patches,
                    token_embeddings[:, 1:, :],
                ),
                dim=1,
            )
            multimodal_mask = torch.cat(
                (masks[:, :1], projected_mask, masks[:, 1:]), dim=1
            )
            output = self.model.language_model(
                input_ids=None,
                inputs_embeds=embeddings,
                attention_mask=multimodal_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            return (output[0][:, -1, :],) + tuple(
                value for layer in output[1] for value in layer
            )

    prefill_module = PrefillModule(model).eval()
    prefill_args = (pixel_values, input_ids, attention_mask)
    with torch.inference_mode():
        official_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=None,
            use_cache=True,
            return_dict=False,
        )
        official_prefill_outputs = (
            official_output[0][:, -1, :],
        ) + tuple(value for layer in official_output[1] for value in layer)
        original_prefill_outputs = prefill_module(*prefill_args)
    _assert_tensor_tuples_exact(
        torch, official_prefill_outputs, original_prefill_outputs
    )
    _specialize_exportable_rope(torch, model)
    with torch.inference_mode():
        prefill_outputs = prefill_module(*prefill_args)
    _assert_tensor_tuples_exact(
        torch, original_prefill_outputs, prefill_outputs
    )
    prefill_region = TensorRegion(
        "generate_action_tokens_prefill",
        tuple(
            Value(name, _tensor_type(torch, value))
            for name, value in zip(
                ("pixel_values", "input_ids", "attention_mask"),
                prefill_args,
                strict=True,
            )
        ),
        tuple(_tensor_type(torch, value) for value in prefill_outputs),
    )
    prefill_capture = capture_region(
        prefill_region,
        prefill_module,
        prefill_args,
        strict=False,
        absolute_tolerance=config.tolerance,
        relative_tolerance=0.0,
    )
    records = [
        RegionAuditRecord.from_capture(
            prefill_capture,
            source_location=(
                f"{checkpoint}/modeling_prismatic.py:"
                "OpenVLAForActionPrediction.forward"
            ),
            major_compute=True,
        )
    ]
    _save_capture(prefill_capture, export_dir)
    del prefill_capture

    cache_layers = (len(prefill_outputs) - 1) // 2

    class DecodeModule(torch.nn.Module):
        def __init__(self, source_model: Any, layers: int):
            super().__init__()
            self.model = source_model
            self.layers = layers

        def forward(self, token: Any, *cache_values: Any) -> tuple[Any, ...]:
            cache = tuple(
                (
                    cache_values[index * 2],
                    cache_values[index * 2 + 1],
                )
                for index in range(self.layers)
            )
            output = self.model(
                input_ids=token,
                attention_mask=None,
                pixel_values=None,
                past_key_values=cache,
                use_cache=True,
                return_dict=False,
            )
            return (output[0][:, -1, :],) + tuple(
                value for layer in output[1] for value in layer
            )

    first_token = torch.argmax(prefill_outputs[0], dim=-1, keepdim=True)
    decode_module = DecodeModule(model, cache_layers).eval()
    decode_args = (first_token, *prefill_outputs[1:])
    decode_names = (
        "token",
        *tuple(
            f"cache_{layer}_{kind}"
            for layer in range(cache_layers)
            for kind in ("key", "value")
        ),
    )
    with torch.inference_mode():
        first_decode_outputs = decode_module(*decode_args)
    decode_region = TensorRegion(
        "generate_action_tokens_decode_step",
        tuple(
            Value(name, _tensor_type(torch, value))
            for name, value in zip(decode_names, decode_args, strict=True)
        ),
        tuple(_tensor_type(torch, value) for value in first_decode_outputs),
    )
    decode_capture = capture_region(
        decode_region,
        decode_module,
        decode_args,
        strict=False,
        absolute_tolerance=config.tolerance,
        relative_tolerance=0.0,
    )
    records.append(
        RegionAuditRecord.from_capture(
            decode_capture,
            source_location="transformers:LlamaForCausalLM.forward",
            major_compute=True,
        )
    )
    _save_capture(decode_capture, export_dir)
    del decode_capture

    explicit_tokens = [first_token]
    current_cache = prefill_outputs[1:]
    current_token = first_token
    with torch.inference_mode():
        for _ in range(action_dim - 1):
            decoded = decode_module(current_token, *current_cache)
            current_token = torch.argmax(decoded[0], dim=-1, keepdim=True)
            explicit_tokens.append(current_token)
            current_cache = decoded[1:]
        explicit_action_tokens = torch.cat(explicit_tokens, dim=1)
        official_generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=action_dim,
            do_sample=False,
            use_cache=True,
        )[:, -action_dim:]
    token_ids_equal = bool(torch.equal(explicit_action_tokens, official_generated))

    stats = model.get_action_stats(config.unnorm_key)

    class DetokenizeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "bin_centers",
                torch.as_tensor(model.bin_centers, dtype=torch.float64),
            )
            self.register_buffer(
                "mask",
                torch.as_tensor(
                    stats.get(
                        "mask",
                        np.ones_like(stats["q01"], dtype=bool),
                    ),
                    dtype=torch.bool,
                ),
            )
            self.register_buffer(
                "high", torch.as_tensor(stats["q99"], dtype=torch.float64)
            )
            self.register_buffer(
                "low", torch.as_tensor(stats["q01"], dtype=torch.float64)
            )
            self.vocab_size = int(model.vocab_size)

        def forward(self, tokens: Any) -> Any:
            discretized = torch.clamp(
                self.vocab_size - tokens[0] - 1,
                min=0,
                max=self.bin_centers.shape[0] - 1,
            )
            normalized = self.bin_centers[discretized]
            return torch.where(
                self.mask,
                0.5 * (normalized + 1) * (self.high - self.low) + self.low,
                normalized,
            )

    detokenize_module = DetokenizeModule().eval()
    detokenize_region = TensorRegion(
        "detokenize_action",
        (
            Value(
                "action_tokens",
                _tensor_type(torch, explicit_action_tokens),
            ),
        ),
        (TensorType((action_dim,), "f64"),),
    )
    detokenize_capture = capture_region(
        detokenize_region,
        detokenize_module,
        (explicit_action_tokens,),
        strict=True,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    records.append(
        RegionAuditRecord.from_capture(
            detokenize_capture,
            source_location=(
                f"{checkpoint}/modeling_prismatic.py:"
                "OpenVLAForActionPrediction.predict_action"
            ),
            major_compute=False,
        )
    )
    _save_capture(detokenize_capture, export_dir)
    torchscript_saved = _save_combined_torchscript(
        torch,
        model,
        cache_layers,
        detokenize_module,
        prefill_args,
        decode_args,
        prefill_outputs,
        first_decode_outputs,
        explicit_action_tokens,
        torchscript_dir,
    )

    shard_digests = tuple(
        sorted(
            (str(item["file"]), str(item["sha256"]))
            for item in _checkpoint_shards(checkpoint)
        )
    )
    report = ModelFrontendAudit(
        model="OpenVLA",
        checkpoint_path=str(checkpoint),
        checkpoint_revision=config.revision,
        checkpoint_digests=shard_digests,
        torch_version=torch.__version__,
        device=config.device,
        persistent_states=(),
        persistent_state_evidence_complete=True,
        regions=tuple(records),
        validation_passed=token_ids_equal,
        validation_checks=tuple(
            sorted(
                (
                    ("action_dim", str(action_dim)),
                    (
                        "explicit_token_ids",
                        ",".join(str(int(item)) for item in explicit_action_tokens[0]),
                    ),
                    (
                        "official_token_ids",
                        ",".join(str(int(item)) for item in official_generated[0]),
                    ),
                    ("token_ids_equal", str(token_ids_equal).lower()),
                )
            )
        ),
        notes=(
            "The same checkpoint is loaded in BF16 on CPU for exportability.",
            "The fixed deployment profile replaces only the Transformers "
            "no_grad/autocast RoPE wrapper with a bit-exact pure tensor module "
            "so PyTorch 2.6 can deserialize the saved ExportedProgram.",
            *(
                (
                    "The same pure regions are also frozen as TorchScript "
                    "archives for the no-Python C++ fallback backend after "
                    "AOTInductor exceeded the host memory budget.",
                )
                if torchscript_saved
                else ()
            ),
            "NF4 loading remains a deployment strategy; bitsandbytes Params4bit "
            "cannot currently be fake-tensor exported by PyTorch 2.6.",
            "KV is explicit loop-carried SSA and is not persistent policy state.",
        ),
    )
    if report_path is not None:
        report.write(report_path)
    return report


def _save_capture(capture: Any, export_dir: str | Path | None) -> None:
    if export_dir is None:
        return
    output = Path(export_dir)
    save_exported_region(
        capture,
        program_path=output / f"{capture.region.name}.pt2e",
        evidence_path=output / f"{capture.region.name}.capture.json",
    )


def _save_combined_torchscript(
    torch: Any,
    model: Any,
    cache_layers: int,
    detokenize_module: Any,
    prefill_arguments: tuple[Any, ...],
    decode_arguments: tuple[Any, ...],
    expected_prefill: tuple[Any, ...],
    expected_decode: tuple[Any, ...],
    action_tokens: Any,
    output_dir: str | Path | None,
) -> bool:
    if output_dir is None:
        return False
    _sanitize_torchscript_qualified_names(model)

    class CombinedRegions(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.detokenizer = detokenize_module

        def prefill(
            self, pixels: Any, tokens: Any, masks: Any
        ) -> tuple[Any, ...]:
            patch_features = self.model.vision_backbone(pixels)
            projected_patches = self.model.projector(patch_features)
            projected_mask = masks.new_ones(
                (
                    projected_patches.shape[0],
                    projected_patches.shape[1],
                )
            )
            token_embeddings = self.model.get_input_embeddings()(tokens)
            embeddings = torch.cat(
                (
                    token_embeddings[:, :1, :],
                    projected_patches,
                    token_embeddings[:, 1:, :],
                ),
                dim=1,
            )
            multimodal_mask = torch.cat(
                (masks[:, :1], projected_mask, masks[:, 1:]), dim=1
            )
            output = self.model.language_model(
                input_ids=None,
                inputs_embeds=embeddings,
                attention_mask=multimodal_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            return (output[0][:, -1, :],) + tuple(
                value for layer in output[1] for value in layer
            )

        def decode(self, token: Any, *cache_values: Any) -> tuple[Any, ...]:
            cache = tuple(
                (
                    cache_values[index * 2],
                    cache_values[index * 2 + 1],
                )
                for index in range(cache_layers)
            )
            output = self.model(
                input_ids=token,
                attention_mask=None,
                pixel_values=None,
                past_key_values=cache,
                use_cache=True,
                return_dict=False,
            )
            return (output[0][:, -1, :],) + tuple(
                value for layer in output[1] for value in layer
            )

        def detokenize(self, tokens: Any) -> Any:
            return self.detokenizer(tokens)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace_module(
            CombinedRegions().eval(),
            {
                "prefill": prefill_arguments,
                "decode": decode_arguments,
                "detokenize": (action_tokens,),
            },
            strict=False,
            check_trace=False,
        )
        actual_prefill = tuple(traced.prefill(*prefill_arguments))
        actual_decode = tuple(traced.decode(*decode_arguments))
        actual_action = traced.detokenize(action_tokens)
    _assert_tensor_tuples_exact(torch, expected_prefill, actual_prefill)
    _assert_tensor_tuples_exact(torch, expected_decode, actual_decode)
    _assert_tensor_tuples_exact(
        torch,
        (detokenize_module(action_tokens),),
        (actual_action,),
    )
    torch.jit.save(traced, destination / "openvla_regions.pt")
    return True


def _sanitize_torchscript_qualified_names(model: Any) -> None:
    for module in model.modules():
        module_type = type(module)
        qualified_module = str(module_type.__module__)
        if "-" in qualified_module:
            module_type.__module__ = qualified_module.replace("-", "_")


def _specialize_exportable_rope(torch: Any, model: Any) -> None:
    class ExportableRotaryEmbedding(torch.nn.Module):
        def __init__(self, source: Any):
            super().__init__()
            self.register_buffer(
                "inv_freq", source.inv_freq.detach().clone()
            )

        def forward(self, value: Any, position_ids: Any) -> tuple[Any, Any]:
            inv_freq = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            positions = position_ids[:, None, :].float()
            frequencies = (inv_freq @ positions).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            return (
                embedding.cos().to(dtype=value.dtype),
                embedding.sin().to(dtype=value.dtype),
            )

    layers = model.language_model.model.layers
    for layer in layers:
        layer.self_attn.rotary_emb = ExportableRotaryEmbedding(
            layer.self_attn.rotary_emb
        )


def _assert_tensor_tuples_exact(
    torch: Any,
    expected: tuple[Any, ...],
    actual: tuple[Any, ...],
) -> None:
    if len(expected) != len(actual):
        raise RuntimeError("RoPE specialization changed output arity")
    for index, (left, right) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if not torch.equal(left, right):
            error = float(
                torch.max(
                    torch.abs(left.to(torch.float32) - right.to(torch.float32))
                ).item()
            )
            raise RuntimeError(
                "RoPE specialization is not bit-exact: "
                f"output={index} max_abs={error}"
            )


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
