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
            output = self.model(
                input_ids=tokens,
                attention_mask=masks,
                pixel_values=pixels,
                past_key_values=None,
                use_cache=True,
                return_dict=False,
            )
            return (output[0][:, -1, :],) + tuple(
                value for layer in output[1] for value in layer
            )

    prefill_module = PrefillModule(model).eval()
    prefill_args = (pixel_values, input_ids, attention_mask)
    with torch.inference_mode():
        prefill_outputs = prefill_module(*prefill_args)
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
            "NF4 loading remains a deployment strategy; bitsandbytes Params4bit "
            "cannot currently be fake-tensor exported by PyTorch 2.6.",
            "KV is explicit loop-carried SSA and is not persistent policy state.",
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
