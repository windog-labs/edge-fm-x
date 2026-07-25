"""Memory-bounded real OpenVLA CUDA partition capture.

The logical OpenVLA program remains a multimodal prefill followed by a
bounded autoregressive decode loop. This module only refines those two large
TensorRegions into backend-owned two-layer artifacts so a 7B checkpoint can
be compiled and executed on a 12 GB GPU without changing core IR semantics.
Decode KV uses a fixed maximum buffer plus an explicit cache position; the
buffer is loop-carried derived state, never persistent Session state.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.adapters.openvla_frontend import (
    _specialize_exportable_rope,
)
from vlaforge.adapters.openvla_real import (
    _checkpoint_shards,
    _deterministic_image,
    _validate_checkpoint,
)
from vlaforge.frontend import capture_region, save_exported_region
from vlaforge.ir.program import TensorRegion, Value
from vlaforge.ir.types import TensorType


OPENVLA_UPSTREAM_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
OPENVLA_PARTITION_CAPTURE_SCHEMA = (
    "vlaforge.openvla_partition_capture/1"
)
OPENVLA_ACTION_DIM = 7
OPENVLA_INPUT_TOKEN_LENGTH = 19
OPENVLA_PREFIX_LENGTH = 275
OPENVLA_MAX_CACHE_LENGTH = 281
OPENVLA_HIDDEN_SIZE = 4096
OPENVLA_HEADS = 32
OPENVLA_HEAD_DIM = 128
OPENVLA_VOCAB_SIZE = 32064
OPENVLA_LAYER_COUNT = 32
OPENVLA_CHUNK_SIZE = 2


@dataclass(frozen=True, slots=True)
class OpenVLAPartitionCaptureConfig:
    checkpoint_path: Path
    revision: str = OPENVLA_UPSTREAM_REVISION
    device: str = "cuda:0"
    unnorm_key: str = "bridge_orig"
    instruction: str = "pick up the block"
    reference_frontend_report: Path | None = None


def prefill_chunk_names() -> tuple[str, ...]:
    return tuple(
        f"prefill_layers_{start:02d}_{start + OPENVLA_CHUNK_SIZE - 1:02d}"
        for start in range(0, OPENVLA_LAYER_COUNT, OPENVLA_CHUNK_SIZE)
    )


def decode_chunk_names() -> tuple[str, ...]:
    return tuple(
        f"decode_layers_{start:02d}_{start + OPENVLA_CHUNK_SIZE - 1:02d}"
        for start in range(0, OPENVLA_LAYER_COUNT, OPENVLA_CHUNK_SIZE)
    )


def artifact_region_names() -> tuple[str, ...]:
    return (
        "prepare_multimodal_prefix",
        *prefill_chunk_names(),
        "decode_token_embedding",
        *decode_chunk_names(),
        "token_logits_head",
        "detokenize_action",
    )


def capture_real_openvla_partitioned(
    config: OpenVLAPartitionCaptureConfig,
    *,
    output_root: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("partitioned OpenVLA capture requires CUDA")
    if not config.device.startswith("cuda"):
        raise ValueError("partitioned OpenVLA capture requires a CUDA device")
    if config.revision != OPENVLA_UPSTREAM_REVISION:
        raise ValueError("OpenVLA revision is not pinned")

    checkpoint = config.checkpoint_path.resolve()
    _validate_checkpoint(checkpoint)
    expected_tokens = _reference_tokens(
        config.reference_frontend_report,
        checkpoint=checkpoint,
        revision=config.revision,
    )
    output = Path(output_root)
    export_root = output / "source_exports"
    export_root.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
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
    _specialize_exportable_rope(torch, model)
    if len(model.language_model.model.layers) != OPENVLA_LAYER_COUNT:
        raise ValueError("unexpected OpenVLA decoder layer count")
    if int(model.get_action_dim(config.unnorm_key)) != OPENVLA_ACTION_DIM:
        raise ValueError("unexpected OpenVLA action dimension")

    image = _deterministic_image(np, Image)
    prompt = (
        "In: What action should the robot take to "
        f"{config.instruction.strip().lower()}?\nOut: "
    )
    model_inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
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
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype),
            ),
            dim=1,
        )
    if tuple(input_ids.shape) != (1, OPENVLA_INPUT_TOKEN_LENGTH):
        raise ValueError(
            f"unexpected OpenVLA token profile: {tuple(input_ids.shape)}"
        )

    records: list[dict[str, object]] = []
    PrepareModule = _prepare_module_type(torch)
    prepare = PrepareModule(
        model.vision_backbone,
        model.projector,
        model.get_input_embeddings(),
    ).eval().to(config.device)
    prepare_args = (
        pixel_values.to(config.device),
        input_ids.to(config.device),
        attention_mask.to(config.device),
    )
    prepare_region = TensorRegion(
        "prepare_multimodal_prefix",
        (
            Value(
                "pixel_values",
                TensorType((1, 6, 224, 224), "bf16"),
            ),
            Value(
                "input_ids",
                TensorType((1, OPENVLA_INPUT_TOKEN_LENGTH), "i64"),
            ),
            Value(
                "attention_mask",
                TensorType((1, OPENVLA_INPUT_TOKEN_LENGTH), "i64"),
            ),
        ),
        (
            TensorType((1, OPENVLA_PREFIX_LENGTH, OPENVLA_HIDDEN_SIZE), "bf16"),
            TensorType(
                (
                    1,
                    1,
                    OPENVLA_PREFIX_LENGTH,
                    OPENVLA_PREFIX_LENGTH,
                ),
                "bf16",
            ),
            TensorType((1, OPENVLA_PREFIX_LENGTH), "i64"),
            TensorType((OPENVLA_PREFIX_LENGTH,), "i64"),
        ),
        metadata={"logical_stage": "prefill", "memoize": True},
    )
    hidden, causal_mask, position_ids, cache_position = _capture(
        torch,
        prepare_region,
        prepare,
        prepare_args,
        export_root,
        records,
    )
    prepare.to("cpu")
    del prepare
    torch.cuda.empty_cache()

    fixed_cache: list[Any] = []
    for chunk_index, start in enumerate(
        range(0, OPENVLA_LAYER_COUNT, OPENVLA_CHUNK_SIZE)
    ):
        layers = model.language_model.model.layers[
            start : start + OPENVLA_CHUNK_SIZE
        ]
        original_indices = tuple(
            layer.self_attn.layer_idx for layer in layers
        )
        for local_index, layer in enumerate(layers):
            layer.self_attn.layer_idx = local_index
        PrefillChunk = _prefill_chunk_type(torch)
        chunk = PrefillChunk(layers).eval().to(config.device)
        name = prefill_chunk_names()[chunk_index]
        region = _prefill_chunk_region(name)
        outputs = _capture(
            torch,
            region,
            chunk,
            (hidden, causal_mask, position_ids, cache_position),
            export_root,
            records,
        )
        hidden = outputs[0]
        fixed_cache.extend(
            torch.nn.functional.pad(
                value,
                (0, 0, 0, OPENVLA_MAX_CACHE_LENGTH - value.shape[2]),
            )
            for value in outputs[1:]
        )
        chunk.to("cpu")
        for layer, original in zip(
            layers,
            original_indices,
            strict=True,
        ):
            layer.self_attn.layer_idx = original
        del chunk, outputs
        torch.cuda.empty_cache()

    EmbeddingModule = _embedding_module_type(torch)
    embedding = EmbeddingModule(model.get_input_embeddings()).eval().to(
        config.device
    )
    embedding_example = torch.tensor(
        [[expected_tokens[0]]],
        dtype=torch.int64,
        device=config.device,
    )
    embedding_region = TensorRegion(
        "decode_token_embedding",
        (Value("token", TensorType((1, 1), "i64")),),
        (TensorType((1, 1, OPENVLA_HIDDEN_SIZE), "bf16"),),
        metadata={"logical_stage": "decode"},
    )
    _capture(
        torch,
        embedding_region,
        embedding,
        (embedding_example,),
        export_root,
        records,
    )

    HeadModule = _head_module_type(torch)
    head = HeadModule(
        model.language_model.model.norm,
        model.language_model.lm_head,
    ).eval().to(config.device)
    head_region = TensorRegion(
        "token_logits_head",
        (
            Value(
                "last_hidden",
                TensorType((1, 1, OPENVLA_HIDDEN_SIZE), "bf16"),
            ),
        ),
        (
            TensorType((1, OPENVLA_VOCAB_SIZE), "bf16"),
            TensorType((1, 1), "i64"),
        ),
        metadata={"logical_stage": "prefill_and_decode"},
    )
    logits, token = _capture(
        torch,
        head_region,
        head,
        (hidden[:, -1:, :],),
        export_root,
        records,
    )
    del logits
    generated_tokens = [int(token.item())]
    if expected_tokens and generated_tokens[0] != expected_tokens[0]:
        raise RuntimeError(
            "partitioned OpenVLA prefill token differs from L2 reference"
        )

    DecodeChunk = _fixed_decode_chunk_type(torch)
    decode_names = decode_chunk_names()
    for step in range(OPENVLA_ACTION_DIM - 1):
        with torch.inference_mode():
            hidden = embedding(token)
        decode_position = OPENVLA_PREFIX_LENGTH + step
        position_ids = torch.tensor(
            [[decode_position]],
            dtype=torch.int64,
            device=config.device,
        )
        cache_position = position_ids[0]
        for chunk_index, start in enumerate(
            range(0, OPENVLA_LAYER_COUNT, OPENVLA_CHUNK_SIZE)
        ):
            layers = model.language_model.model.layers[
                start : start + OPENVLA_CHUNK_SIZE
            ]
            chunk = DecodeChunk(layers).eval().to(config.device)
            cache_offset = chunk_index * 2 * OPENVLA_CHUNK_SIZE
            chunk_cache = tuple(
                fixed_cache[
                    cache_offset : cache_offset + 2 * OPENVLA_CHUNK_SIZE
                ]
            )
            args = (
                hidden,
                position_ids,
                cache_position,
                *chunk_cache,
            )
            if step == 0:
                outputs = _capture(
                    torch,
                    _decode_chunk_region(decode_names[chunk_index]),
                    chunk,
                    args,
                    export_root,
                    records,
                )
            else:
                with torch.inference_mode():
                    outputs = _as_tuple(chunk(*args))
            hidden = outputs[0]
            fixed_cache[
                cache_offset : cache_offset + 2 * OPENVLA_CHUNK_SIZE
            ] = tuple(outputs[1:])
            chunk.to("cpu")
            del chunk, outputs
            torch.cuda.empty_cache()
        with torch.inference_mode():
            logits, token = head(hidden)
        del logits
        generated_tokens.append(int(token.item()))

    DetokenizeModule = _detokenize_module_type(
        torch,
        model,
        config.unnorm_key,
    )
    detokenizer = DetokenizeModule().eval().to(config.device)
    action_tokens = torch.tensor(
        [generated_tokens],
        dtype=torch.int64,
        device=config.device,
    )
    detokenize_region = TensorRegion(
        "detokenize_action",
        (
            Value(
                "action_tokens",
                TensorType((1, OPENVLA_ACTION_DIM), "i64"),
            ),
        ),
        (TensorType((OPENVLA_ACTION_DIM,), "f64"),),
        metadata={"logical_stage": "postprocess"},
    )
    (action,) = _capture(
        torch,
        detokenize_region,
        detokenizer,
        (action_tokens,),
        export_root,
        records,
    )
    if expected_tokens and tuple(generated_tokens) != expected_tokens:
        raise RuntimeError(
            "partitioned OpenVLA CUDA tokens differ from L2 reference: "
            f"{generated_tokens} != {list(expected_tokens)}"
        )

    torch.cuda.synchronize()
    shard_records = tuple(_checkpoint_shards(checkpoint))
    fixture = {
        "instruction": config.instruction,
        "prompt": prompt,
        "pixel_values": _tensor_identity(prepare_args[0]),
        "input_ids": _tensor_identity(prepare_args[1]),
        "attention_mask": _tensor_identity(prepare_args[2]),
    }
    report: dict[str, object] = {
        "schema": OPENVLA_PARTITION_CAPTURE_SCHEMA,
        "status": "passed",
        "passed": True,
        "evidence_level": "L2-partition-capture",
        "model": "OpenVLA-7B",
        "checkpoint": {
            "path": str(checkpoint),
            "revision": config.revision,
            "shards": shard_records,
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device": config.device,
        },
        "partition": {
            "logical_regions": [
                "generate_action_tokens_prefill",
                "generate_action_tokens_decode_step",
                "detokenize_action",
            ],
            "artifact_regions": list(artifact_region_names()),
            "layer_count": OPENVLA_LAYER_COUNT,
            "layer_chunk_size": OPENVLA_CHUNK_SIZE,
            "prefix_length": OPENVLA_PREFIX_LENGTH,
            "maximum_cache_length": OPENVLA_MAX_CACHE_LENGTH,
            "decode_steps": OPENVLA_ACTION_DIM - 1,
            "kv_semantics": (
                "loop-carried derived cache with fixed max shape and "
                "explicit cache_position"
            ),
            "core_op_delta": 0,
        },
        "fixture": fixture,
        "correctness": {
            "reference_tokens": list(expected_tokens),
            "partitioned_cuda_tokens": generated_tokens,
            "token_ids_equal": tuple(generated_tokens) == expected_tokens,
            "action": [float(value) for value in action.cpu().tolist()],
            "all_export_replays_exact": all(
                float(item["maximum_absolute_error"]) == 0.0
                for item in records
            ),
        },
        "regions": records,
        "timing": {
            "capture_seconds": time.perf_counter() - started,
        },
        "memory": {
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
    }
    destination = (
        Path(report_path)
        if report_path is not None
        else output / "capture.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _capture(
    torch: Any,
    region: TensorRegion,
    module: Any,
    args: tuple[Any, ...],
    export_root: Path,
    records: list[dict[str, object]],
) -> tuple[Any, ...]:
    with torch.inference_mode():
        expected = _as_tuple(module(*args))
    outcome = capture_region(
        region,
        module,
        args,
        strict=False,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    ).require_supported()
    save_exported_region(
        outcome,
        program_path=export_root / f"{region.name}.pt2e",
        evidence_path=export_root / f"{region.name}.capture.json",
    )
    assert outcome.evidence is not None
    records.append(
        {
            "name": region.name,
            "logical_stage": region.metadata.get("logical_stage"),
            "graph_nodes": len(
                tuple(outcome.exported_program.graph_module.graph.nodes)
            ),
            "graph_sha256": outcome.evidence.graph_digest,
            "export_seconds": outcome.evidence.export_seconds,
            "maximum_absolute_error": (
                outcome.evidence.maximum_absolute_error
            ),
            "effect_audit_passed": outcome.evidence.effect_audit.passed,
            "export_path": str(
                export_root / f"{region.name}.pt2e"
            ),
        }
    )
    del outcome
    return expected


def _prepare_module_type(torch: Any) -> type:
    class PrepareMultimodalPrefix(torch.nn.Module):
        def __init__(
            self,
            vision_backbone: Any,
            projector: Any,
            embedding: Any,
        ) -> None:
            super().__init__()
            self.vision_backbone = vision_backbone
            self.projector = projector
            self.embedding = embedding

        def forward(
            self,
            pixels: Any,
            tokens: Any,
            masks: Any,
        ) -> tuple[Any, Any, Any, Any]:
            patches = self.vision_backbone(pixels)
            projected = self.projector(patches)
            projected_mask = masks.new_ones(
                (projected.shape[0], projected.shape[1])
            )
            token_embeddings = self.embedding(tokens)
            hidden = torch.cat(
                (
                    token_embeddings[:, :1, :],
                    projected,
                    token_embeddings[:, 1:, :],
                ),
                dim=1,
            )
            multimodal_mask = torch.cat(
                (masks[:, :1], projected_mask, masks[:, 1:]),
                dim=1,
            )
            cache_position = torch.arange(
                hidden.shape[1],
                dtype=torch.int64,
                device=hidden.device,
            )
            position_ids = cache_position.unsqueeze(0)
            minimum = torch.finfo(hidden.dtype).min
            sequence_length = hidden.shape[1]
            causal_mask = torch.full(
                (sequence_length, multimodal_mask.shape[-1]),
                minimum,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= (
                torch.arange(
                    multimodal_mask.shape[-1],
                    device=hidden.device,
                )
                > cache_position.reshape(-1, 1)
            )
            causal_mask = causal_mask[
                None, None, :, :
            ].expand(hidden.shape[0], 1, -1, -1)
            causal_mask = causal_mask.clone()
            padding = causal_mask.eq(0.0) * multimodal_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask = causal_mask.masked_fill(padding, minimum)
            return hidden, causal_mask, position_ids, cache_position

    return PrepareMultimodalPrefix


def _prefill_chunk_type(torch: Any) -> type:
    from transformers.cache_utils import DynamicCache

    class PrefillLayerChunk(torch.nn.Module):
        def __init__(self, layers: Any) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(list(layers))

        def forward(
            self,
            hidden: Any,
            causal_mask: Any,
            position_ids: Any,
            cache_position: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            cache = DynamicCache()
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )[0]
            values = cache.to_legacy_cache()
            return (
                hidden,
                values[0][0],
                values[0][1],
                values[1][0],
                values[1][1],
            )

    return PrefillLayerChunk


def _fixed_decode_chunk_type(torch: Any) -> type:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    class FixedCacheDecodeChunk(torch.nn.Module):
        def __init__(self, layers: Any) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(list(layers))

        def _layer(
            self,
            layer: Any,
            hidden: Any,
            position_ids: Any,
            cache_position: Any,
            key_cache: Any,
            value_cache: Any,
        ) -> tuple[Any, Any, Any]:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attention = layer.self_attn
            batch, query_length, _ = hidden.size()
            query = attention.q_proj(hidden).view(
                batch,
                query_length,
                attention.num_heads,
                attention.head_dim,
            ).transpose(1, 2)
            key = attention.k_proj(hidden).view(
                batch,
                query_length,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            value = attention.v_proj(hidden).view(
                batch,
                query_length,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            cosine, sine = attention.rotary_emb(value, position_ids)
            query, key = apply_rotary_pos_emb(
                query,
                key,
                cosine,
                sine,
            )
            key_cache = torch.index_copy(
                key_cache,
                2,
                cache_position,
                key,
            )
            value_cache = torch.index_copy(
                value_cache,
                2,
                cache_position,
                value,
            )
            repeated_key = repeat_kv(
                key_cache,
                attention.num_key_value_groups,
            )
            repeated_value = repeat_kv(
                value_cache,
                attention.num_key_value_groups,
            )
            weights = torch.matmul(
                query,
                repeated_key.transpose(2, 3),
            ) / math.sqrt(attention.head_dim)
            future = (
                torch.arange(
                    key_cache.shape[2],
                    device=hidden.device,
                )
                > cache_position.reshape(-1, 1)
            )
            future_mask = (
                future.to(hidden.dtype) * torch.finfo(hidden.dtype).min
            )
            weights = weights + future_mask[:, None, None, :]
            weights = torch.nn.functional.softmax(
                weights,
                dim=-1,
                dtype=torch.float32,
            ).to(query.dtype)
            attended = torch.matmul(weights, repeated_value)
            attended = attended.transpose(1, 2).contiguous().reshape(
                batch,
                query_length,
                attention.hidden_size,
            )
            hidden = residual + attention.o_proj(attended)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)
            return hidden, key_cache, value_cache

        def forward(
            self,
            hidden: Any,
            position_ids: Any,
            cache_position: Any,
            key_0: Any,
            value_0: Any,
            key_1: Any,
            value_1: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            hidden, key_0, value_0 = self._layer(
                self.layers[0],
                hidden,
                position_ids,
                cache_position,
                key_0,
                value_0,
            )
            hidden, key_1, value_1 = self._layer(
                self.layers[1],
                hidden,
                position_ids,
                cache_position,
                key_1,
                value_1,
            )
            return hidden, key_0, value_0, key_1, value_1

    return FixedCacheDecodeChunk


def _embedding_module_type(torch: Any) -> type:
    class DecodeTokenEmbedding(torch.nn.Module):
        def __init__(self, embedding: Any) -> None:
            super().__init__()
            self.embedding = embedding

        def forward(self, token: Any) -> Any:
            return self.embedding(token)

    return DecodeTokenEmbedding


def _head_module_type(torch: Any) -> type:
    class TokenLogitsHead(torch.nn.Module):
        def __init__(self, norm: Any, head: Any) -> None:
            super().__init__()
            self.norm = norm
            self.head = head

        def forward(self, hidden: Any) -> tuple[Any, Any]:
            logits = self.head(self.norm(hidden))[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            return logits, token

    return TokenLogitsHead


def _detokenize_module_type(
    torch: Any,
    model: Any,
    unnorm_key: str,
) -> type:
    import numpy as np

    stats = model.get_action_stats(unnorm_key)

    class DetokenizeAction(torch.nn.Module):
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
                "high",
                torch.as_tensor(stats["q99"], dtype=torch.float64),
            )
            self.register_buffer(
                "low",
                torch.as_tensor(stats["q01"], dtype=torch.float64),
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
                0.5
                * (normalized + 1)
                * (self.high - self.low)
                + self.low,
                normalized,
            )

    return DetokenizeAction


def _prefill_chunk_region(name: str) -> TensorRegion:
    hidden = TensorType(
        (1, OPENVLA_PREFIX_LENGTH, OPENVLA_HIDDEN_SIZE),
        "bf16",
    )
    cache = TensorType(
        (1, OPENVLA_HEADS, OPENVLA_PREFIX_LENGTH, OPENVLA_HEAD_DIM),
        "bf16",
    )
    return TensorRegion(
        name,
        (
            Value("hidden", hidden),
            Value(
                "causal_mask",
                TensorType(
                    (
                        1,
                        1,
                        OPENVLA_PREFIX_LENGTH,
                        OPENVLA_PREFIX_LENGTH,
                    ),
                    "bf16",
                ),
            ),
            Value(
                "position_ids",
                TensorType((1, OPENVLA_PREFIX_LENGTH), "i64"),
            ),
            Value(
                "cache_position",
                TensorType((OPENVLA_PREFIX_LENGTH,), "i64"),
            ),
        ),
        (hidden, cache, cache, cache, cache),
        metadata={"logical_stage": "prefill_layer_chunk"},
    )


def _decode_chunk_region(name: str) -> TensorRegion:
    hidden = TensorType((1, 1, OPENVLA_HIDDEN_SIZE), "bf16")
    cache = TensorType(
        (
            1,
            OPENVLA_HEADS,
            OPENVLA_MAX_CACHE_LENGTH,
            OPENVLA_HEAD_DIM,
        ),
        "bf16",
    )
    return TensorRegion(
        name,
        (
            Value("hidden", hidden),
            Value("position_ids", TensorType((1, 1), "i64")),
            Value("cache_position", TensorType((1,), "i64")),
            Value("key_0", cache),
            Value("value_0", cache),
            Value("key_1", cache),
            Value("value_1", cache),
        ),
        (hidden, cache, cache, cache, cache),
        metadata={
            "logical_stage": "decode_layer_chunk",
            "maximum_cache_length": OPENVLA_MAX_CACHE_LENGTH,
        },
    )


def _reference_tokens(
    path: Path | None,
    *,
    checkpoint: Path,
    revision: str,
) -> tuple[int, ...]:
    if path is None:
        raise ValueError(
            "partitioned capture requires the passing real OpenVLA "
            "frontend report"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "vlaforge.frontend_model_audit/1"
        or not payload.get("passed", False)
        or payload.get("checkpoint_revision") != revision
        or Path(str(payload.get("checkpoint_path"))).resolve() != checkpoint
    ):
        raise ValueError("OpenVLA frontend report identity mismatch")
    checks = {
        str(item["name"]): str(item["value"])
        for item in payload.get("validation_checks", ())
    }
    if checks.get("token_ids_equal") != "true":
        raise ValueError("OpenVLA frontend token parity did not pass")
    return tuple(
        int(item)
        for item in checks["official_token_ids"].split(",")
    )


def _tensor_identity(value: Any) -> dict[str, object]:
    contiguous = value.detach().cpu().contiguous()
    raw = contiguous.view(dtype=getattr(__import__("torch"), "uint8"))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        "size_bytes": value.numel() * value.element_size(),
    }


def _as_tuple(value: object) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)

