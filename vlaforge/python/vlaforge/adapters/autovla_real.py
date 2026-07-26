"""Real-checkpoint AutoVLA frontend partition.

This adapter intentionally stops at a declared model boundary: the caller
provides the ten post-attention hidden vectors produced by AutoVLA's bounded
autoregressive decode.  VLAForge owns the final Qwen MLP sub-block, action
vocabulary projection, source-faithful codebook rollout, exact reuse contract,
and transactional named outputs.  Camera loading, prompt construction, and
token generation remain outside this partition and are not claimed as captured
evidence.

PyTorch stays a lazy dependency so importing :mod:`vlaforge.adapters` does not
load the model stack.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


AUTOVLA_UPSTREAM_REVISION = "ba34eed74ce6729e7986592d0e66cbaca397b4fa"
AUTOVLA_CHECKPOINT_REPOSITORY = "Zewei-Zhou/AutoVLA"
AUTOVLA_CHECKPOINT_REVISION = "a7d7ba3ed7529b248d2694c2defa31b35208340f"
AUTOVLA_CHECKPOINT_FILENAME = "AutoVLA_PDMS_89.ckpt"
AUTOVLA_CHECKPOINT_SIZE = 16_292_664_780
AUTOVLA_CHECKPOINT_SHA256 = (
    "58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473"
)
AUTOVLA_QWEN_REPOSITORY = "Qwen/Qwen2.5-VL-3B-Instruct"
AUTOVLA_QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
AUTOVLA_QWEN_CONFIG_SHA256 = (
    "7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5"
)
AUTOVLA_CODEBOOK_SHA256 = (
    "e6bf8eff32497eaefca66f488dda1edefe97ed2696d1d23c01839dc27422408e"
)
AUTOVLA_SOURCE_SHA256 = {
    "models/autovla.py": (
        "e34becaf49a06fd89c17579a58753a547efa13f5f9d5711738a4e904eeaa4424"
    ),
    "models/action_tokenizer.py": (
        "eb668739af6a9b245a3f32bbf7701f69f8c6662d35c652874e85dc63b836dcf2"
    ),
    "config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml": (
        "32a7bac3ad99723fcc754b58d29af56e56028c056eda3901b5cf571c189dd3ef"
    ),
}

AUTOVLA_ACTION_START_ID = 151_665
AUTOVLA_ACTION_VOCAB_SIZE = 2_048
AUTOVLA_HIDDEN_SIZE = 2_048
AUTOVLA_INTERMEDIATE_SIZE = 11_008
AUTOVLA_DECODE_STEPS = 10
AUTOVLA_RMS_NORM_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class RealAutoVLAConfig:
    source_root: Path
    checkpoint: Path
    codebook: Path
    qwen_config: Path
    device: str = "cuda:0"
    upstream_revision: str = AUTOVLA_UPSTREAM_REVISION
    checkpoint_revision: str = AUTOVLA_CHECKPOINT_REVISION
    qwen_revision: str = AUTOVLA_QWEN_REVISION


@dataclass(frozen=True, slots=True)
class RealAutoVLARegions:
    decoder_mlp: Any
    action_projection: Any
    trajectory_decode: Any
    example_hidden: Any
    checkpoint_sha256: str
    codebook_sha256: str
    qwen_config_sha256: str
    resolved_keys: Mapping[str, str]
    tensor_shapes: Mapping[str, tuple[int, ...]]
    layer_index: int


def build_real_autovla_program(*, device: str = "cuda:0") -> Any:
    """Build the frozen-core program for the real AutoVLA decoder partition."""

    hidden = TensorType(
        (1, AUTOVLA_DECODE_STEPS, AUTOVLA_HIDDEN_SIZE),
        "bf16",
    )
    logits = TensorType(
        (1, AUTOVLA_DECODE_STEPS, AUTOVLA_ACTION_VOCAB_SIZE),
        "f32",
    )
    trajectory = TensorType((AUTOVLA_DECODE_STEPS, 3), "f32")
    action_tokens = TensorType((AUTOVLA_DECODE_STEPS,), "i64")

    builder = ModuleBuilder("autovla_real_decoder_frontend")
    builder.add_input(
        InputPort(
            "post_attention_hidden",
            hidden,
            device=device,
            alignment=64,
        )
    )
    builder.add_output(
        OutputPort(
            "trajectory",
            trajectory,
            group="planning",
            device=device,
            alignment=64,
        )
    )
    builder.add_output(
        OutputPort(
            "action_tokens",
            action_tokens,
            group="planning",
            device=device,
            alignment=64,
        )
    )
    builder.add_region(
        TensorRegion(
            "autovla_decoder_mlp",
            (Value("hidden", hidden),),
            (hidden,),
            metadata={
                "memoize": True,
                "cache_input_ports": ["post_attention_hidden"],
                "cache_state_slots": [],
                "template": "AutoregressiveTrajectory",
                "partition": "last_qwen_post_attention_mlp",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "autovla_action_projection",
            (Value("decoded_hidden", hidden),),
            (logits,),
            metadata={
                "template": "AutoregressiveTrajectory",
                "partition": "final_norm_and_action_vocab_projection",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "autovla_trajectory_decode",
            (Value("action_logits", logits),),
            (trajectory, action_tokens),
            metadata={
                "template": "AutoregressiveTrajectory",
                "bounded_decode_steps": AUTOVLA_DECODE_STEPS,
                "partition": "action_codebook_rollout",
            },
        )
    )

    pending_trajectory = PendingOutputType("trajectory", trajectory)
    pending_tokens = PendingOutputType("action_tokens", action_tokens)
    body = Block.of(
        (
            ops.input_read(
                "hidden_value",
                "hidden_revision",
                hidden,
                "post_attention_hidden",
            ),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("decoded_hidden",),
                (hidden,),
                "autovla_decoder_mlp",
                ("hidden_value",),
            ),
            ops.invoke(
                ("action_logits",),
                (logits,),
                "autovla_action_projection",
                ("decoded_hidden",),
            ),
            ops.invoke(
                ("trajectory", "action_tokens"),
                (trajectory, action_tokens),
                "autovla_trajectory_decode",
                ("action_logits",),
            ),
            ops.validate(
                "trajectory_valid",
                "trajectory",
                "finite_trajectory",
            ),
            ops.output_create(
                "pending_trajectory",
                "trajectory",
                trajectory,
                "trajectory",
            ),
            ops.output_create(
                "pending_action_tokens",
                "action_tokens",
                action_tokens,
                "action_tokens",
            ),
            ops.output_group(
                "pending_outputs",
                "planning",
                (
                    ("pending_trajectory", pending_trajectory),
                    ("pending_action_tokens", pending_tokens),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (pending_trajectory, pending_tokens),
                "planning",
                "txn",
                "pending_outputs",
                "trajectory_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "plan",
            body,
            metadata={
                "adapter_template": "AutoregressiveTrajectory",
                "source_model": "ucla-mobility/AutoVLA",
                "source_revision": AUTOVLA_UPSTREAM_REVISION,
                "checkpoint_revision": AUTOVLA_CHECKPOINT_REVISION,
                "partition": "post_attention_action_decoder",
                "bounded_decode_steps": AUTOVLA_DECODE_STEPS,
                "core_op_delta": 0,
            },
        )
    )
    return builder.build()


def load_real_autovla_regions(
    config: RealAutoVLAConfig,
) -> RealAutoVLARegions:
    """Load only the released weights required by the declared partition."""

    import numpy as np
    import torch

    _validate_config(config)
    checkpoint_sha256 = _sha256(config.checkpoint)
    if checkpoint_sha256 != AUTOVLA_CHECKPOINT_SHA256:
        raise ValueError("AutoVLA checkpoint SHA256 mismatch")
    codebook_sha256 = _sha256(config.codebook)
    if codebook_sha256 != AUTOVLA_CODEBOOK_SHA256:
        raise ValueError("AutoVLA action codebook SHA256 mismatch")
    qwen_config_sha256 = _sha256(config.qwen_config)
    if qwen_config_sha256 != AUTOVLA_QWEN_CONFIG_SHA256:
        raise ValueError("Qwen configuration SHA256 mismatch")

    qwen = json.loads(config.qwen_config.read_text(encoding="utf-8"))
    expected_config = {
        "hidden_size": AUTOVLA_HIDDEN_SIZE,
        "intermediate_size": AUTOVLA_INTERMEDIATE_SIZE,
        "num_hidden_layers": 36,
        "rms_norm_eps": AUTOVLA_RMS_NORM_EPS,
    }
    for name, expected in expected_config.items():
        if qwen.get(name) != expected:
            raise ValueError(
                f"Qwen config {name} mismatch: "
                f"expected={expected!r} actual={qwen.get(name)!r}"
            )

    checkpoint = torch.load(
        config.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state = _checkpoint_state_dict(checkpoint)
    keys, layer_index = resolve_autovla_weight_keys(state)
    tensors = {name: state[key] for name, key in keys.items()}
    _validate_weight_shapes(tensors)

    action_end = AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE
    projection_source = tensors["projection"]
    if projection_source.shape[0] < action_end:
        raise ValueError(
            "AutoVLA projection vocabulary is too small for configured "
            f"action tokens: {tuple(projection_source.shape)}"
        )
    selected = {
        "post_attention_norm": tensors["post_attention_norm"],
        "gate_proj": tensors["gate_proj"],
        "up_proj": tensors["up_proj"],
        "down_proj": tensors["down_proj"],
        "final_norm": tensors["final_norm"],
        "action_projection": projection_source[
            AUTOVLA_ACTION_START_ID:action_end
        ],
    }
    device_weights = {
        name: value.detach()
        .to(device=config.device, dtype=torch.bfloat16)
        .clone()
        for name, value in selected.items()
    }

    with config.codebook.open("rb") as handle:
        payload = pickle.load(handle)
    try:
        codebook_array = payload["token_all"]["veh"]
    except (KeyError, TypeError) as error:
        raise ValueError("AutoVLA codebook has no token_all/veh table") from error
    codebook = torch.as_tensor(
        np.asarray(codebook_array),
        dtype=torch.float32,
        device=config.device,
    ).clone()
    if tuple(codebook.shape) != (AUTOVLA_ACTION_VOCAB_SIZE, 6, 4, 2):
        raise ValueError(
            f"unexpected AutoVLA vehicle codebook shape: {tuple(codebook.shape)}"
        )

    decoder_mlp, action_projection, trajectory_decode = (
        build_autovla_region_modules(
            device_weights,
            codebook,
            rms_norm_eps=float(qwen["rms_norm_eps"]),
            action_start_id=AUTOVLA_ACTION_START_ID,
        )
    )
    generator = torch.Generator(device=config.device)
    generator.manual_seed(20260726)
    example_hidden = torch.randn(
        (1, AUTOVLA_DECODE_STEPS, AUTOVLA_HIDDEN_SIZE),
        generator=generator,
        device=config.device,
        dtype=torch.bfloat16,
    )
    tensor_shapes = {
        name: tuple(int(item) for item in value.shape)
        for name, value in selected.items()
    }
    del checkpoint, state, tensors, selected, device_weights
    return RealAutoVLARegions(
        decoder_mlp=decoder_mlp,
        action_projection=action_projection,
        trajectory_decode=trajectory_decode,
        example_hidden=example_hidden,
        checkpoint_sha256=checkpoint_sha256,
        codebook_sha256=codebook_sha256,
        qwen_config_sha256=qwen_config_sha256,
        resolved_keys=dict(keys),
        tensor_shapes=tensor_shapes,
        layer_index=layer_index,
    )


def build_autovla_region_modules(
    weights: Mapping[str, Any],
    codebook: Any,
    *,
    rms_norm_eps: float = AUTOVLA_RMS_NORM_EPS,
    action_start_id: int = AUTOVLA_ACTION_START_ID,
) -> tuple[Any, Any, Any]:
    """Create pure modules from real or synthetic partition tensors."""

    import torch
    import torch.nn.functional as functional

    required = {
        "post_attention_norm",
        "gate_proj",
        "up_proj",
        "down_proj",
        "final_norm",
        "action_projection",
    }
    missing = required - set(weights)
    if missing:
        raise ValueError(f"missing AutoVLA partition weights: {sorted(missing)}")

    def rms_norm(hidden: Any, weight: Any) -> Any:
        input_dtype = hidden.dtype
        normalized = hidden.to(torch.float32)
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + rms_norm_eps)
        return weight * normalized.to(input_dtype)

    class DecoderMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in (
                "post_attention_norm",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                self.register_buffer(
                    name,
                    weights[name].detach().clone(),
                    persistent=True,
                )

        def forward(self, hidden: Any) -> Any:
            normalized = rms_norm(hidden, self.post_attention_norm)
            gated = functional.silu(
                functional.linear(normalized, self.gate_proj)
            )
            up = functional.linear(normalized, self.up_proj)
            return hidden + functional.linear(gated * up, self.down_proj)

    class ActionProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "final_norm",
                weights["final_norm"].detach().clone(),
                persistent=True,
            )
            self.register_buffer(
                "action_projection",
                weights["action_projection"].detach().clone(),
                persistent=True,
            )

        def forward(self, hidden: Any) -> Any:
            normalized = rms_norm(hidden, self.final_norm)
            return functional.linear(
                normalized, self.action_projection
            ).to(torch.float32)

    class TrajectoryDecode(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "codebook",
                codebook.detach().to(torch.float32).clone(),
                persistent=True,
            )
            self.action_start_id = int(action_start_id)

        def forward(self, logits: Any) -> tuple[Any, Any]:
            indices = logits.argmax(dim=-1).reshape(-1)
            action_trajectories = self.codebook[indices]
            position = torch.zeros(
                (1, 2),
                dtype=action_trajectories.dtype,
                device=action_trajectories.device,
            )
            heading = torch.zeros(
                (1,),
                dtype=action_trajectories.dtype,
                device=action_trajectories.device,
            )
            positions = []
            headings = []
            for step in range(AUTOVLA_DECODE_STEPS):
                local = action_trajectories[step].reshape(1, 24, 2)
                cosine = heading.cos().reshape(1, 1)
                sine = heading.sin().reshape(1, 1)
                global_x = (
                    local[..., 0] * cosine
                    - local[..., 1] * sine
                    + position[:, None, 0]
                )
                global_y = (
                    local[..., 0] * sine
                    + local[..., 1] * cosine
                    + position[:, None, 1]
                )
                global_trajectory = torch.stack(
                    (global_x, global_y), dim=-1
                ).reshape(1, 6, 4, 2)
                position = global_trajectory[:, -1].mean(dim=1)
                difference = (
                    global_trajectory[:, -1, 0]
                    - global_trajectory[:, -1, 3]
                )
                heading = torch.atan2(difference[:, 1], difference[:, 0])
                positions.append(position)
                headings.append(heading)
            trajectory = torch.cat(
                (
                    torch.stack(positions, dim=1),
                    torch.stack(headings, dim=1).unsqueeze(-1),
                ),
                dim=-1,
            ).squeeze(0)
            token_ids = indices + self.action_start_id
            return trajectory, token_ids

    return (
        DecoderMLP().eval(),
        ActionProjection().eval(),
        TrajectoryDecode().eval(),
    )


def run_real_autovla_chain(regions: RealAutoVLARegions) -> dict[str, Any]:
    """Run the eager partition with deterministic post-attention hidden input."""

    import torch

    with torch.no_grad():
        decoded = regions.decoder_mlp(regions.example_hidden)
        logits = regions.action_projection(decoded)
        trajectory, tokens = regions.trajectory_decode(logits)
    return {
        "decoded_hidden": decoded,
        "action_logits": logits,
        "trajectory": trajectory,
        "action_tokens": tokens,
    }


def capture_real_autovla_regions(
    regions: RealAutoVLARegions,
    output_dir: str | Path,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> tuple[Any, ...]:
    """Strict-export and persist all three real-checkpoint TensorRegions."""

    import torch
    from vlaforge.frontend import capture_region, save_exported_region

    module = build_real_autovla_program()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        decoded = regions.decoder_mlp(regions.example_hidden)
        logits = regions.action_projection(decoded)
    declarations = (
        (
            "autovla_decoder_mlp",
            regions.decoder_mlp,
            (regions.example_hidden,),
        ),
        (
            "autovla_action_projection",
            regions.action_projection,
            (decoded,),
        ),
        (
            "autovla_trajectory_decode",
            regions.trajectory_decode,
            (logits,),
        ),
    )
    captures = []
    for name, implementation, arguments in declarations:
        capture = capture_region(
            module.region(name),
            implementation,
            arguments,
            strict=True,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        capture.require_supported()
        save_exported_region(
            capture,
            program_path=output / f"{name}.pt2e",
            evidence_path=output / f"{name}.capture.json",
        )
        captures.append(capture)
    return tuple(captures)


def resolve_autovla_weight_keys(
    state: Mapping[str, Any],
) -> tuple[dict[str, str], int]:
    """Resolve the final Qwen decoder layer without assuming Lightning prefix."""

    matches = []
    pattern = re.compile(
        r"^(?P<layer>.+\.layers\.(?P<index>\d+))"
        r"\.mlp\.gate_proj\.weight$"
    )
    for key, value in state.items():
        match = pattern.match(str(key))
        if match is None or not hasattr(value, "shape"):
            continue
        shape = tuple(int(item) for item in value.shape)
        if shape == (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE):
            matches.append((int(match.group("index")), match.group("layer"), str(key)))
    if not matches:
        raise ValueError("AutoVLA checkpoint has no compatible Qwen gate projection")
    highest = max(index for index, _, _ in matches)
    layers = [(prefix, key) for index, prefix, key in matches if index == highest]
    if len(layers) != 1:
        raise ValueError(
            "AutoVLA checkpoint final Qwen layer is ambiguous: "
            f"{[key for _, key in layers]}"
        )
    layer_prefix, gate_key = layers[0]
    language_prefix = layer_prefix.rsplit(".layers.", 1)[0]
    direct = {
        "post_attention_norm": (
            f"{layer_prefix}.post_attention_layernorm.weight"
        ),
        "gate_proj": gate_key,
        "up_proj": f"{layer_prefix}.mlp.up_proj.weight",
        "down_proj": f"{layer_prefix}.mlp.down_proj.weight",
        "final_norm": f"{language_prefix}.norm.weight",
    }
    missing = [key for key in direct.values() if key not in state]
    if missing:
        raise ValueError(
            f"AutoVLA checkpoint is missing final-layer tensors: {missing}"
        )

    projection_candidates = [
        str(key)
        for key, value in state.items()
        if str(key).endswith("lm_head.weight")
        and hasattr(value, "shape")
        and len(value.shape) == 2
        and int(value.shape[1]) == AUTOVLA_HIDDEN_SIZE
        and int(value.shape[0])
        >= AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE
    ]
    embedding_key = f"{language_prefix}.embed_tokens.weight"
    if len(projection_candidates) == 1:
        projection_key = projection_candidates[0]
    elif (
        not projection_candidates
        and embedding_key in state
        and len(state[embedding_key].shape) == 2
        and int(state[embedding_key].shape[1]) == AUTOVLA_HIDDEN_SIZE
        and int(state[embedding_key].shape[0])
        >= AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE
    ):
        projection_key = embedding_key
    else:
        raise ValueError(
            "AutoVLA action projection is missing or ambiguous: "
            f"{projection_candidates}"
        )
    direct["projection"] = projection_key
    return direct, highest


def finite_trajectory(value: Any) -> bool:
    """Validator used by both Semantic IR and Plan execution."""

    import torch

    return bool(torch.isfinite(value).all().item())


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("AutoVLA checkpoint root is not a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("AutoVLA checkpoint state_dict is not a mapping")
    return state


def _validate_weight_shapes(weights: Mapping[str, Any]) -> None:
    expected = {
        "post_attention_norm": (AUTOVLA_HIDDEN_SIZE,),
        "gate_proj": (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE),
        "up_proj": (AUTOVLA_INTERMEDIATE_SIZE, AUTOVLA_HIDDEN_SIZE),
        "down_proj": (AUTOVLA_HIDDEN_SIZE, AUTOVLA_INTERMEDIATE_SIZE),
        "final_norm": (AUTOVLA_HIDDEN_SIZE,),
    }
    for name, shape in expected.items():
        actual = tuple(int(item) for item in weights[name].shape)
        if actual != shape:
            raise ValueError(
                f"AutoVLA weight {name} shape mismatch: "
                f"expected={shape} actual={actual}"
            )
    projection = weights["projection"]
    if len(projection.shape) != 2 or int(projection.shape[1]) != AUTOVLA_HIDDEN_SIZE:
        raise ValueError(
            f"AutoVLA projection shape mismatch: {tuple(projection.shape)}"
        )


def _validate_config(config: RealAutoVLAConfig) -> None:
    if config.upstream_revision != AUTOVLA_UPSTREAM_REVISION:
        raise ValueError("AutoVLA upstream revision is not pinned")
    if config.checkpoint_revision != AUTOVLA_CHECKPOINT_REVISION:
        raise ValueError("AutoVLA checkpoint revision is not pinned")
    if config.qwen_revision != AUTOVLA_QWEN_REVISION:
        raise ValueError("AutoVLA Qwen revision is not pinned")
    if not config.device.startswith("cuda"):
        raise ValueError("real AutoVLA frontend evidence requires CUDA")
    for path in (
        config.checkpoint,
        config.codebook,
        config.qwen_config,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.checkpoint.name != AUTOVLA_CHECKPOINT_FILENAME:
        raise ValueError("AutoVLA checkpoint filename mismatch")
    if config.checkpoint.stat().st_size != AUTOVLA_CHECKPOINT_SIZE:
        raise ValueError("AutoVLA checkpoint size mismatch")
    for relative, expected in AUTOVLA_SOURCE_SHA256.items():
        path = config.source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected:
            raise ValueError(
                f"AutoVLA source differs from pinned revision: {relative}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
