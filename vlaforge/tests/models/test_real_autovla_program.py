from __future__ import annotations

from dataclasses import dataclass

import pytest

from vlaforge.adapters.autovla_real import (
    AUTOVLA_ACTION_START_ID,
    AUTOVLA_ACTION_VOCAB_SIZE,
    AUTOVLA_HIDDEN_SIZE,
    AUTOVLA_INTERMEDIATE_SIZE,
    build_autovla_region_modules,
    build_real_autovla_program,
    resolve_autovla_weight_keys,
)
from vlaforge.analysis import verify
from vlaforge.compiler import compile_module


@dataclass
class _Shaped:
    shape: tuple[int, ...]


def test_real_autovla_program_uses_frozen_generic_core() -> None:
    module = build_real_autovla_program()
    assert verify(module, raise_on_error=False) == ()
    assert module.states == ()
    assert tuple(port.name for port in module.inputs) == (
        "post_attention_hidden",
    )
    assert tuple(port.name for port in module.outputs) == (
        "trajectory",
        "action_tokens",
    )
    assert module.invocations[0].metadata["core_op_delta"] == 0

    compilation = compile_module(
        module,
        default_device="cuda:0",
        state_device="cuda:0",
    )
    assert len(compilation.certificate.caches) == 1
    cache = compilation.certificate.caches[0]
    assert cache.region == "autovla_decoder_mlp"
    assert cache.input_ids == (0,)
    assert cache.state_ids == ()


def test_weight_key_resolution_accepts_lightning_prefix_and_tied_head() -> None:
    prefix = "autovla.vlm.model.language_model"
    layer = f"{prefix}.layers.35"
    state = {
        f"{layer}.post_attention_layernorm.weight": _Shaped(
            (AUTOVLA_HIDDEN_SIZE,)
        ),
        f"{layer}.mlp.gate_proj.weight": _Shaped(
            (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE)
        ),
        f"{layer}.mlp.up_proj.weight": _Shaped(
            (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE)
        ),
        f"{layer}.mlp.down_proj.weight": _Shaped(
            (AUTOVLA_HIDDEN_SIZE, AUTOVLA_INTERMEDIATE_SIZE)
        ),
        f"{prefix}.norm.weight": _Shaped((AUTOVLA_HIDDEN_SIZE,)),
        f"{prefix}.embed_tokens.weight": _Shaped(
            (
                AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE,
                AUTOVLA_HIDDEN_SIZE,
            )
        ),
    }
    keys, layer_index = resolve_autovla_weight_keys(state)
    assert layer_index == 35
    assert keys["projection"] == f"{prefix}.embed_tokens.weight"
    assert keys["gate_proj"] == f"{layer}.mlp.gate_proj.weight"


def test_weight_key_resolution_rejects_undersized_vocabulary() -> None:
    prefix = "autovla.vlm.model"
    layer = f"{prefix}.layers.35"
    state = {
        f"{layer}.post_attention_layernorm.weight": _Shaped(
            (AUTOVLA_HIDDEN_SIZE,)
        ),
        f"{layer}.mlp.gate_proj.weight": _Shaped(
            (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE)
        ),
        f"{layer}.mlp.up_proj.weight": _Shaped(
            (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE)
        ),
        f"{layer}.mlp.down_proj.weight": _Shaped(
            (AUTOVLA_HIDDEN_SIZE, AUTOVLA_INTERMEDIATE_SIZE)
        ),
        f"{prefix}.norm.weight": _Shaped((AUTOVLA_HIDDEN_SIZE,)),
        f"{prefix}.embed_tokens.weight": _Shaped(
            (AUTOVLA_ACTION_START_ID, AUTOVLA_HIDDEN_SIZE)
        ),
    }
    with pytest.raises(ValueError, match="projection"):
        resolve_autovla_weight_keys(state)


def test_synthetic_modules_match_source_rollout_equations() -> None:
    torch = pytest.importorskip("torch")
    hidden_size = 4
    intermediate_size = 7
    vocabulary_size = 8
    generator = torch.Generator().manual_seed(20260726)
    weights = {
        "post_attention_norm": torch.randn(hidden_size, generator=generator),
        "gate_proj": torch.randn(
            intermediate_size, hidden_size, generator=generator
        ),
        "up_proj": torch.randn(
            intermediate_size, hidden_size, generator=generator
        ),
        "down_proj": torch.randn(
            hidden_size, intermediate_size, generator=generator
        ),
        "final_norm": torch.randn(hidden_size, generator=generator),
        "action_projection": torch.randn(
            vocabulary_size, hidden_size, generator=generator
        ),
    }
    codebook = torch.randn(
        vocabulary_size, 6, 4, 2, generator=generator
    )
    decoder, projection, detokenize = build_autovla_region_modules(
        weights,
        codebook,
        rms_norm_eps=1e-6,
        action_start_id=100,
    )
    hidden = torch.randn(1, 10, hidden_size, generator=generator)
    decoded = decoder(hidden)
    logits = projection(decoded)
    trajectory, token_ids = detokenize(logits)

    action_indices = logits.argmax(dim=-1).reshape(-1)
    action_tokens = codebook[action_indices]
    positions = torch.zeros((1, 1, 2))
    headings = torch.zeros((1, 1))
    for step in range(10):
        local = action_tokens[None, step]
        cosine = headings[:, step].cos()
        sine = headings[:, step].sin()
        rotation = torch.stack(
            (
                torch.stack((cosine, sine), dim=-1),
                torch.stack((-sine, cosine), dim=-1),
            ),
            dim=1,
        )
        transformed = torch.bmm(local.flatten(1, 2), rotation)
        transformed = (
            transformed + positions[:, step].unsqueeze(1)
        ).view_as(local)
        next_position = transformed[:, -1].mean(dim=1)
        difference = transformed[:, -1, 0] - transformed[:, -1, 3]
        next_heading = torch.atan2(difference[:, 1], difference[:, 0])
        positions = torch.cat(
            (positions, next_position.unsqueeze(1)), dim=1
        )
        headings = torch.cat(
            (headings, next_heading.unsqueeze(1)), dim=1
        )
    reference = torch.cat(
        (positions, headings.unsqueeze(-1)), dim=-1
    )[0, 1:]
    torch.testing.assert_close(trajectory, reference)
    assert torch.equal(token_ids, action_indices + 100)
