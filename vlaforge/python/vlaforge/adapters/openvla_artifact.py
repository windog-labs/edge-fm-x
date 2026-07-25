"""Physical OpenVLA artifact schedule for a generated no-Python Session.

The logical model remains prefill + bounded decode + detokenize.  This Adapter
unrolls the fixed six-step decode only in the backend schedule so the existing
two-layer AOTI packages can be weight-paged on a 12 GB GPU.  KV is explicit
invocation-local SSA throughout; the module declares no persistent state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlaforge.adapters.openvla_partitioned import (
    OPENVLA_ACTION_DIM,
    OPENVLA_HEADS,
    OPENVLA_HEAD_DIM,
    OPENVLA_HIDDEN_SIZE,
    OPENVLA_INPUT_TOKEN_LENGTH,
    OPENVLA_MAX_CACHE_LENGTH,
    OPENVLA_PREFIX_LENGTH,
    OPENVLA_VOCAB_SIZE,
    decode_chunk_names,
    prefill_chunk_names,
)
from vlaforge.frontend import capture_region, save_exported_region
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    OutputPort,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, TensorType


OPENVLA_PREPARE_DECODE_STATE = "prepare_decode_state"
OPENVLA_PACK_ACTION_TOKENS = "pack_action_tokens"
OPENVLA_L4_SUPPORT_REGIONS = (
    OPENVLA_PREPARE_DECODE_STATE,
    OPENVLA_PACK_ACTION_TOKENS,
)

PIXELS = TensorType((1, 6, 224, 224), "bf16")
LANGUAGE = TensorType((1, OPENVLA_INPUT_TOKEN_LENGTH), "i64")
PREFIX_HIDDEN = TensorType(
    (1, OPENVLA_PREFIX_LENGTH, OPENVLA_HIDDEN_SIZE),
    "bf16",
)
DECODE_HIDDEN = TensorType((1, 1, OPENVLA_HIDDEN_SIZE), "bf16")
PREFIX_MASK = TensorType(
    (1, 1, OPENVLA_PREFIX_LENGTH, OPENVLA_PREFIX_LENGTH),
    "bf16",
)
PREFIX_POSITIONS = TensorType((1, OPENVLA_PREFIX_LENGTH), "i64")
PREFIX_CACHE_POSITION = TensorType((OPENVLA_PREFIX_LENGTH,), "i64")
PREFIX_CACHE = TensorType(
    (1, OPENVLA_HEADS, OPENVLA_PREFIX_LENGTH, OPENVLA_HEAD_DIM),
    "bf16",
)
FIXED_CACHE = TensorType(
    (1, OPENVLA_HEADS, OPENVLA_MAX_CACHE_LENGTH, OPENVLA_HEAD_DIM),
    "bf16",
)
TOKEN = TensorType((1, 1), "i64")
ACTION_TOKENS = TensorType((1, OPENVLA_ACTION_DIM), "i64")
DECODE_POSITION = TensorType((1, 1), "i64")
DECODE_CACHE_POSITION = TensorType((1,), "i64")
LOGITS = TensorType((1, OPENVLA_VOCAB_SIZE), "bf16")
ACTION = TensorType((OPENVLA_ACTION_DIM,), "f64")


def build_compiled_openvla_program(
    *,
    input_device: str = "cuda:0",
    output_device: str = "cuda:0",
) -> Any:
    """Build a static artifact schedule without adding a core opcode."""

    builder = ModuleBuilder("openvla_real_artifact_session")
    builder.add_input(
        InputPort("image", PIXELS, device=input_device, alignment=2)
    )
    builder.add_input(
        InputPort(
            "instruction_tokens",
            LANGUAGE,
            device=input_device,
            alignment=8,
        )
    )
    builder.add_input(
        InputPort(
            "instruction_mask",
            LANGUAGE,
            device=input_device,
            alignment=8,
        )
    )
    builder.add_output(
        OutputPort(
            "action",
            ACTION,
            group="manipulation",
            device=output_device,
            alignment=8,
        )
    )

    builder.add_region(
        TensorRegion(
            "prepare_multimodal_prefix",
            (
                Value("pixel_values", PIXELS),
                Value("input_ids", LANGUAGE),
                Value("attention_mask", LANGUAGE),
            ),
            (
                PREFIX_HIDDEN,
                PREFIX_MASK,
                PREFIX_POSITIONS,
                PREFIX_CACHE_POSITION,
            ),
            metadata={"memoize": True, "loop_invariant": True},
        )
    )
    for name in prefill_chunk_names():
        builder.add_region(
            TensorRegion(
                name,
                (
                    Value("hidden", PREFIX_HIDDEN),
                    Value("causal_mask", PREFIX_MASK),
                    Value("position_ids", PREFIX_POSITIONS),
                    Value("cache_position", PREFIX_CACHE_POSITION),
                ),
                (
                    PREFIX_HIDDEN,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                ),
                metadata={"memoize": True, "loop_invariant": True},
            )
        )
    builder.add_region(
        TensorRegion(
            OPENVLA_PREPARE_DECODE_STATE,
            (
                Value("prefill_hidden", PREFIX_HIDDEN),
                *(
                    Value(f"prefill_cache_{index:02d}", PREFIX_CACHE)
                    for index in range(64)
                ),
            ),
            (
                DECODE_HIDDEN,
                *(FIXED_CACHE for _ in range(64)),
                *(
                    value
                    for _ in range(OPENVLA_ACTION_DIM - 1)
                    for value in (DECODE_POSITION, DECODE_CACHE_POSITION)
                ),
            ),
            metadata={"loop_invariant": True},
        )
    )
    builder.add_region(
        TensorRegion(
            "decode_token_embedding",
            (Value("token", TOKEN),),
            (DECODE_HIDDEN,),
        )
    )
    for name in decode_chunk_names():
        builder.add_region(
            TensorRegion(
                name,
                (
                    Value("hidden", DECODE_HIDDEN),
                    Value("position_ids", DECODE_POSITION),
                    Value("cache_position", DECODE_CACHE_POSITION),
                    Value("key_0", FIXED_CACHE),
                    Value("value_0", FIXED_CACHE),
                    Value("key_1", FIXED_CACHE),
                    Value("value_1", FIXED_CACHE),
                ),
                (
                    DECODE_HIDDEN,
                    FIXED_CACHE,
                    FIXED_CACHE,
                    FIXED_CACHE,
                    FIXED_CACHE,
                ),
            )
        )
    builder.add_region(
        TensorRegion(
            "token_logits_head",
            (Value("last_hidden", DECODE_HIDDEN),),
            (LOGITS, TOKEN),
        )
    )
    builder.add_region(
        TensorRegion(
            OPENVLA_PACK_ACTION_TOKENS,
            tuple(
                Value(f"token_{index}", TOKEN)
                for index in range(OPENVLA_ACTION_DIM)
            ),
            (ACTION_TOKENS,),
        )
    )
    builder.add_region(
        TensorRegion(
            "detokenize_action",
            (Value("action_tokens", ACTION_TOKENS),),
            (ACTION,),
        )
    )

    operations = [
        ops.input_read("image_value", "image_revision", PIXELS, "image"),
        ops.input_read(
            "instruction_value",
            "instruction_revision",
            LANGUAGE,
            "instruction_tokens",
        ),
        ops.input_read(
            "instruction_mask_value",
            "instruction_mask_revision",
            LANGUAGE,
            "instruction_mask",
        ),
        ops.transaction_begin("txn"),
        ops.invoke(
            (
                "prefill_hidden_00",
                "prefill_mask",
                "prefill_positions",
                "prefill_cache_position",
            ),
            (
                PREFIX_HIDDEN,
                PREFIX_MASK,
                PREFIX_POSITIONS,
                PREFIX_CACHE_POSITION,
            ),
            "prepare_multimodal_prefix",
            (
                "image_value",
                "instruction_value",
                "instruction_mask_value",
            ),
        ),
    ]
    prefill_caches: list[str] = []
    hidden = "prefill_hidden_00"
    for chunk_index, name in enumerate(prefill_chunk_names()):
        outputs = (
            f"prefill_hidden_{chunk_index + 1:02d}",
            *(f"prefill_cache_{chunk_index * 4 + item:02d}" for item in range(4)),
        )
        operations.append(
            ops.invoke(
                outputs,
                (
                    PREFIX_HIDDEN,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                    PREFIX_CACHE,
                ),
                name,
                (
                    hidden,
                    "prefill_mask",
                    "prefill_positions",
                    "prefill_cache_position",
                ),
            )
        )
        hidden = outputs[0]
        prefill_caches.extend(outputs[1:])

    fixed_caches = [f"fixed_cache_00_{index:02d}" for index in range(64)]
    decode_positions = [
        item
        for step in range(OPENVLA_ACTION_DIM - 1)
        for item in (
            f"decode_position_{step}",
            f"decode_cache_position_{step}",
        )
    ]
    operations.append(
        ops.invoke(
            ("prefill_last_hidden", *fixed_caches, *decode_positions),
            (
                DECODE_HIDDEN,
                *(FIXED_CACHE for _ in fixed_caches),
                *(
                    value
                    for _ in range(OPENVLA_ACTION_DIM - 1)
                    for value in (DECODE_POSITION, DECODE_CACHE_POSITION)
                ),
            ),
            OPENVLA_PREPARE_DECODE_STATE,
            (hidden, *prefill_caches),
        )
    )
    operations.append(
        ops.invoke(
            ("logits_0", "token_0"),
            (LOGITS, TOKEN),
            "token_logits_head",
            ("prefill_last_hidden",),
        )
    )

    tokens = ["token_0"]
    for step in range(OPENVLA_ACTION_DIM - 1):
        operations.append(
            ops.invoke(
                (f"decode_hidden_{step}_00",),
                (DECODE_HIDDEN,),
                "decode_token_embedding",
                (tokens[-1],),
            )
        )
        hidden = f"decode_hidden_{step}_00"
        next_caches = list(fixed_caches)
        for chunk_index, name in enumerate(decode_chunk_names()):
            offset = chunk_index * 4
            outputs = (
                f"decode_hidden_{step}_{chunk_index + 1:02d}",
                *(
                    f"fixed_cache_{step + 1:02d}_{offset + item:02d}"
                    for item in range(4)
                ),
            )
            operations.append(
                ops.invoke(
                    outputs,
                    (
                        DECODE_HIDDEN,
                        FIXED_CACHE,
                        FIXED_CACHE,
                        FIXED_CACHE,
                        FIXED_CACHE,
                    ),
                    name,
                    (
                        hidden,
                        f"decode_position_{step}",
                        f"decode_cache_position_{step}",
                        *fixed_caches[offset : offset + 4],
                    ),
                )
            )
            hidden = outputs[0]
            next_caches[offset : offset + 4] = outputs[1:]
        fixed_caches = next_caches
        token = f"token_{step + 1}"
        operations.append(
            ops.invoke(
                (f"logits_{step + 1}", token),
                (LOGITS, TOKEN),
                "token_logits_head",
                (hidden,),
            )
        )
        tokens.append(token)

    operations.extend(
        (
            ops.invoke(
                ("action_tokens",),
                (ACTION_TOKENS,),
                OPENVLA_PACK_ACTION_TOKENS,
                tokens,
            ),
            ops.invoke(
                ("decoded_action",),
                (ACTION,),
                "detokenize_action",
                ("action_tokens",),
            ),
            ops.validate(
                "action_valid",
                "decoded_action",
                "finite_action",
            ),
            ops.output_create(
                "pending_action",
                "decoded_action",
                ACTION,
                "action",
            ),
            ops.output_group(
                "pending_outputs",
                "manipulation",
                (
                    (
                        "pending_action",
                        PendingOutputType("action", ACTION),
                    ),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (PendingOutputType("action", ACTION),),
                "manipulation",
                "txn",
                "pending_outputs",
                "action_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            Block.of(operations),
            metadata={
                "logical_control": "bounded_decode_6",
                "physical_schedule": "unrolled_weight_paged",
                "persistent_state": "none",
                "derived_cache": "fixed_kv",
                "core_op_delta": 0,
            },
        )
    )
    return builder.build()


def capture_openvla_l4_support_regions(
    output_root: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, dict[str, object]]:
    """Capture the two pure glue Regions required by the physical schedule."""

    import torch

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("OpenVLA L4 support capture requires CUDA")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}

    PrepareDecodeState = _prepare_decode_state_type(torch)
    prepare = PrepareDecodeState().eval().to(device)
    hidden_example = torch.zeros(
        PREFIX_HIDDEN.shape,
        dtype=torch.bfloat16,
        device=device,
    )
    cache_examples = tuple(
        torch.zeros(
            PREFIX_CACHE.shape,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(64)
    )
    prepare_region = next(
        region
        for region in build_compiled_openvla_program().regions
        if region.name == OPENVLA_PREPARE_DECODE_STATE
    )
    records[prepare_region.name] = _capture_support(
        prepare_region,
        prepare,
        (hidden_example, *cache_examples),
        root,
    )
    del prepare, hidden_example, cache_examples
    torch.cuda.empty_cache()

    PackActionTokens = _pack_action_tokens_type(torch)
    pack = PackActionTokens().eval().to(device)
    token_examples = tuple(
        torch.tensor([[31857 + index]], dtype=torch.int64, device=device)
        for index in range(OPENVLA_ACTION_DIM)
    )
    pack_region = next(
        region
        for region in build_compiled_openvla_program().regions
        if region.name == OPENVLA_PACK_ACTION_TOKENS
    )
    records[pack_region.name] = _capture_support(
        pack_region,
        pack,
        token_examples,
        root,
    )
    return records


def _capture_support(
    region: TensorRegion,
    module: Any,
    arguments: tuple[Any, ...],
    root: Path,
) -> dict[str, object]:
    outcome = capture_region(
        region,
        module,
        arguments,
        strict=True,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    ).require_supported()
    program = root / f"{region.name}.pt2e"
    evidence = root / f"{region.name}.capture.json"
    save_exported_region(
        outcome,
        program_path=program,
        evidence_path=evidence,
    )
    assert outcome.evidence is not None
    return {
        "region": region.name,
        "program": str(program),
        "evidence": str(evidence),
        "graph_sha256": outcome.evidence.graph_digest,
        "maximum_absolute_error": (
            outcome.evidence.maximum_absolute_error
        ),
    }


def _prepare_decode_state_type(torch: Any) -> type:
    class PrepareDecodeState(torch.nn.Module):
        def forward(self, hidden: Any, *caches: Any) -> tuple[Any, ...]:
            padded = tuple(
                torch.nn.functional.pad(
                    cache,
                    (
                        0,
                        0,
                        0,
                        OPENVLA_MAX_CACHE_LENGTH
                        - OPENVLA_PREFIX_LENGTH,
                    ),
                )
                for cache in caches
            )
            positions = tuple(
                value
                for step in range(OPENVLA_ACTION_DIM - 1)
                for value in (
                    torch.tensor(
                        [[OPENVLA_PREFIX_LENGTH + step]],
                        dtype=torch.int64,
                        device=caches[0].device,
                    ),
                    torch.tensor(
                        [OPENVLA_PREFIX_LENGTH + step],
                        dtype=torch.int64,
                        device=caches[0].device,
                    ),
                )
            )
            return (hidden[:, -1:, :], *padded, *positions)

    return PrepareDecodeState


def _pack_action_tokens_type(torch: Any) -> type:
    class PackActionTokens(torch.nn.Module):
        def forward(self, *tokens: Any) -> Any:
            return torch.cat(tokens, dim=1)

    return PackActionTokens
