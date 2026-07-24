"""Real-checkpoint SmolVLA action-chunk adapter.

Imports of PyTorch and LeRobot are intentionally lazy so the offline IR package
has no model dependency. This adapter validates a real checkpoint in two ways:

1. eager ``predict_action_chunk`` versus the VLAForge IR interpreter split into
   prefix, bounded solver-step, and action-trim TensorRegions;
2. LeRobot's real action queue versus explicit cross-tick StateSlots and
   transaction commits across repeated IR control ticks.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter import Epoch, InputSample, Interpreter
from vlaforge.ir import ops
from vlaforge.ir.attrs import (
    CheckpointPolicy,
    ConsistencyPolicy,
    EpochExpr,
    FreshnessConstraint,
    Ownership,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.program import (
    Block,
    ClockDomain,
    InputStream,
    Policy,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import EpochType, ScalarType, TensorType


OPAQUE = ScalarType("opaque")


@dataclass(frozen=True, slots=True)
class RealSmolVLAConfig:
    policy_path: Path
    vlm_path: Path
    device: str = "cuda"
    num_steps: int = 10
    tolerance: float = 1e-5
    lerobot_revision: str = "unknown"


@dataclass(frozen=True, slots=True)
class RealSmolVLAEvidence:
    schema: str
    evidence_kind: str
    checkpoint_path: str
    checkpoint_sha256: str
    vlm_path: str
    lerobot_revision: str
    torch_version: str
    transformers_version: str
    device: str
    gpu_name: str | None
    num_steps: int
    action_shape: tuple[int, ...]
    eager_seconds: float
    ir_seconds: float
    queue_seconds: tuple[float, ...]
    peak_memory_mb: float | None
    action_max_abs_error: float
    solver_max_abs_errors: tuple[float, ...]
    action_queue_max_abs_errors: tuple[float, ...]
    trace_events: int
    passed: bool

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(slots=True)
class _PrefixContext:
    prefix_pad_masks: Any
    past_key_values: Any

    def __vlaforge_trace__(self) -> dict[str, Any]:
        cache = self.past_key_values
        try:
            layers = len(cache)
        except TypeError:
            layers = len(getattr(cache, "layers", ()))
        get_seq_length = getattr(cache, "get_seq_length", None)
        sequence = int(get_seq_length()) if callable(get_seq_length) else None
        return {
            "kind": "smolvla_prefix_context",
            "prefix_pad_masks": self.prefix_pad_masks,
            "cache_layers": layers,
            "cache_sequence_length": sequence,
        }


def build_real_smolvla_action_program(
    *,
    chunk_size: int,
    max_action_dim: int,
    output_action_dim: int,
    num_steps: int,
) -> Any:
    solver = TensorType((1, chunk_size, max_action_dim), "f32")
    action_chunk = TensorType((1, chunk_size, output_action_dim), "f32")
    action_step = TensorType((1, output_action_dim), "f32")
    cursor = ScalarType("i32")
    builder = ModuleBuilder("smolvla_real_control")
    builder.add_clock(ClockDomain("observation", period_ns=33_333_333))
    builder.add_clock(ClockDomain("control", period_ns=20_000_000))
    builder.add_input(
        InputStream(
            "batch",
            OPAQUE,
            "observation",
            FreshnessConstraint(max_age_ns=50_000_000),
        )
    )
    builder.add_state(
        StateSlot(
            "action_queue",
            action_chunk,
            StateScope.EPISODE,
            "control",
            retention=5,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EPISODE_START,
            authoritative=True,
            ownership=Ownership.HOST,
            checkpoint=CheckpointPolicy.ON_COMMIT,
        )
    )
    builder.add_state(
        StateSlot(
            "queue_cursor",
            cursor,
            StateScope.EPISODE,
            "control",
            retention=5,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EPISODE_START,
            authoritative=True,
            ownership=Ownership.HOST,
            checkpoint=CheckpointPolicy.ON_COMMIT,
        )
    )
    builder.add_input(
        InputStream(
            "noise",
            solver,
            "observation",
            FreshnessConstraint(max_age_ns=50_000_000),
        )
    )
    builder.add_region(
        TensorRegion("prepare_prefix", (Value("batch_arg", OPAQUE),), (OPAQUE,))
    )
    builder.add_region(
        TensorRegion(
            "solver_step",
            (
                Value("prefix_arg", OPAQUE),
                Value("sample_arg", solver),
                Value("step_arg", ScalarType("index")),
            ),
            (solver,),
        )
    )
    builder.add_region(
        TensorRegion(
            "trim_action_chunk",
            (Value("sample_arg", solver),),
            (action_chunk,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_is_empty",
            (Value("cursor_arg", cursor),),
            (ScalarType("bool"),),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_select",
            (
                Value("queue_arg", action_chunk),
                Value("cursor_arg", cursor),
            ),
            (action_step,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_advance",
            (Value("cursor_arg", cursor),),
            (cursor,),
        )
    )
    builder.add_region(TensorRegion("queue_zero", (), (cursor,)))
    loop = Block.of(
        (
            ops.invoke(
                ("sample_next",),
                (solver,),
                "solver_step",
                ("prefix", "sample_iter", "step"),
            ),
            ops.yield_values("sample_next"),
        )
    )
    refill = Block.of(
        (
            ops.invoke(("prefix",), (OPAQUE,), "prepare_prefix", ("batch_value",)),
            ops.for_loop(
                Value("sample_final", solver),
                "noise_value",
                Value("step", ScalarType("index")),
                Value("sample_iter", solver),
                loop,
                lower=0,
                upper=num_steps,
            ),
            ops.invoke(
                ("refilled_queue",),
                (action_chunk,),
                "trim_action_chunk",
                ("sample_final",),
            ),
            ops.invoke(
                ("refilled_action",),
                (action_step,),
                "queue_select",
                ("refilled_queue", "zero_cursor"),
            ),
            ops.invoke(
                ("cursor_after_refill",),
                (cursor,),
                "queue_advance",
                ("zero_cursor",),
            ),
            ops.yield_values(
                "refilled_action",
                "refilled_queue",
                "cursor_after_refill",
            ),
        )
    )
    reuse = Block.of(
        (
            ops.invoke(
                ("queued_action",),
                (action_step,),
                "queue_select",
                ("queue_value", "cursor_value"),
            ),
            ops.invoke(
                ("cursor_after_reuse",),
                (cursor,),
                "queue_advance",
                ("cursor_value",),
            ),
            ops.yield_values(
                "queued_action",
                "queue_value",
                "cursor_after_reuse",
            ),
        )
    )
    body = Block.of(
        (
            ops.sample_input(
                "batch_value",
                "batch_epoch",
                OPAQUE,
                "batch",
                "observation",
                max_age_ns=50_000_000,
            ),
            ops.sample_input(
                "noise_value",
                "noise_epoch",
                solver,
                "noise",
                "observation",
                max_age_ns=50_000_000,
            ),
            ops.transaction_begin("txn", "tick"),
            ops.state_read(
                "queue_snapshot",
                action_chunk,
                "action_queue",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value(
                "queue_value", action_chunk, "queue_snapshot"
            ),
            ops.state_read(
                "cursor_snapshot",
                cursor,
                "queue_cursor",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value("cursor_value", cursor, "cursor_snapshot"),
            ops.invoke(
                ("queue_empty",),
                (ScalarType("bool"),),
                "queue_is_empty",
                ("cursor_value",),
            ),
            ops.invoke(
                ("zero_cursor",),
                (cursor,),
                "queue_zero",
                (),
            ),
            ops.if_op(
                (
                    Value("selected_action", action_step),
                    Value("queue_next", action_chunk),
                    Value("cursor_next", cursor),
                ),
                "queue_empty",
                refill,
                reuse,
            ),
            ops.stage_write(
                "queue_pending",
                action_chunk,
                "action_queue",
                "txn",
                "queue_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.stage_write(
                "cursor_pending",
                cursor,
                "queue_cursor",
                "txn",
                "cursor_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.validate("action_valid", "selected_action", "finite_action"),
            ops.action_create(
                "pending_action", "selected_action", action_step, "tick"
            ),
            ops.transaction_commit(
                "committed_action",
                action_step,
                "txn",
                "pending_action",
                "action_valid",
            ),
            ops.action_publish("committed_action"),
            ops.return_values("committed_action"),
        )
    )
    builder.add_policy(
        Policy(
            "act",
            "control",
            body,
            inputs=(Value("tick", EpochType("control")),),
            metadata={
                "action_generation": "real_iterative_continuous",
                "persistent_state": "action_queue,queue_cursor",
            },
        )
    )
    return builder.build()


def run_real_smolvla(
    config: RealSmolVLAConfig,
    *,
    trace_path: str | Path | None = None,
) -> RealSmolVLAEvidence:
    import torch
    import transformers
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        make_att_2d_masks,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    checkpoint = config.policy_path / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.vlm_path.is_dir():
        raise FileNotFoundError(config.vlm_path)
    if config.num_steps < 1:
        raise ValueError("num_steps must be positive")

    policy_config = PreTrainedConfig.from_pretrained(
        config.policy_path, local_files_only=True
    )
    policy_config.vlm_model_name = str(config.vlm_path)
    policy_config.device = config.device
    policy_config.num_steps = config.num_steps
    policy = SmolVLAPolicy.from_pretrained(
        config.policy_path,
        config=policy_config,
        local_files_only=True,
        strict=False,
    )
    policy.eval()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(
        "pick up the block\n",
        padding="max_length",
        max_length=policy_config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    image_key = next(iter(policy_config.image_features))
    batch = {
        image_key: torch.linspace(
            0,
            1,
            3 * 256 * 256,
            device=config.device,
            dtype=torch.float32,
        ).reshape(1, 3, 256, 256),
        OBS_STATE: torch.linspace(
            -0.2, 0.3, 6, device=config.device, dtype=torch.float32
        ).reshape(1, 6),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(config.device),
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(
            config.device, dtype=torch.bool
        ),
    }
    noise = torch.linspace(
        -1,
        1,
        policy_config.chunk_size * policy_config.max_action_dim,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, policy_config.chunk_size, policy_config.max_action_dim)

    eager_velocities: list[Any] = []
    original_denoise = policy.model.denoise_step

    def capture_eager(*args: Any, **kwargs: Any) -> Any:
        velocity = original_denoise(*args, **kwargs)
        eager_velocities.append(velocity.detach().clone())
        return velocity

    policy.model.denoise_step = capture_eager
    started = time.perf_counter()
    with torch.inference_mode():
        eager_action = policy.predict_action_chunk(batch, noise=noise)
    _synchronize(torch, config.device)
    eager_seconds = time.perf_counter() - started
    policy.model.denoise_step = original_denoise

    module = build_real_smolvla_action_program(
        chunk_size=policy_config.chunk_size,
        max_action_dim=policy_config.max_action_dim,
        output_action_dim=policy_config.action_feature.shape[0],
        num_steps=config.num_steps,
    )
    ir_velocities: list[Any] = []

    def prepare_prefix(input_batch: dict[str, Any]) -> _PrefixContext:
        images, image_masks = policy.prepare_images(input_batch)
        state = policy.prepare_state(input_batch)
        language_tokens = input_batch[OBS_LANGUAGE_TOKENS]
        language_masks = input_batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state=state,
        )
        attention_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=attention_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy_config.use_cache,
            fill_kv_cache=True,
        )
        return _PrefixContext(prefix_pad_masks, past_key_values)

    def solver_step(prefix: _PrefixContext, x_t: Any, step: int) -> Any:
        dt = -1.0 / config.num_steps
        timestep = torch.tensor(
            1.0 + step * dt,
            dtype=torch.float32,
            device=x_t.device,
        ).expand(x_t.shape[0])
        velocity = original_denoise(
            x_t=x_t,
            prefix_pad_masks=prefix.prefix_pad_masks,
            past_key_values=prefix.past_key_values,
            timestep=timestep,
        )
        ir_velocities.append(velocity.detach().clone())
        return x_t + dt * velocity

    action_dim = policy_config.action_feature.shape[0]

    def trim_action_chunk(sample: Any) -> Any:
        return sample[:, :, :action_dim]

    def queue_is_empty(cursor: int) -> bool:
        return int(cursor) >= policy_config.chunk_size

    def queue_select(action_queue: Any, cursor: int) -> Any:
        return action_queue[:, int(cursor)]

    def queue_advance(cursor: int) -> int:
        return int(cursor) + 1

    def queue_zero() -> int:
        return 0

    runtime = Interpreter(
        module,
        regions={
            "prepare_prefix": prepare_prefix,
            "solver_step": solver_step,
            "trim_action_chunk": trim_action_chunk,
            "queue_is_empty": queue_is_empty,
            "queue_select": queue_select,
            "queue_advance": queue_advance,
            "queue_zero": queue_zero,
        },
        validators={"finite_action": lambda action: bool(torch.isfinite(action).all())},
        initial_state={
            "action_queue": torch.zeros_like(eager_action),
            "queue_cursor": policy_config.chunk_size,
        },
    )
    ir_queue_actions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(3):
            timestamp = index * 20_000_000
            ir_result = runtime.run_tick(
                "act",
                Epoch("control", index, timestamp, 0),
                {
                    "batch": InputSample(
                        batch,
                        Epoch("observation", index, timestamp, 0),
                    ),
                    "noise": InputSample(
                        noise,
                        Epoch("observation", index, timestamp, 0),
                    ),
                },
            )
            ir_queue_actions.append(ir_result.returns[0].value.detach().clone())
    _synchronize(torch, config.device)
    ir_seconds = time.perf_counter() - started
    queue_versions = runtime.state_store.versions("action_queue")
    if len(queue_versions) != 4:
        raise RuntimeError(
            f"expected initial + 3 action queue versions, got {len(queue_versions)}"
        )
    ir_action_chunk = queue_versions[1].value

    action_error = float((eager_action - ir_action_chunk).abs().max().cpu())
    solver_errors = tuple(
        float((left - right).abs().max().cpu())
        for left, right in zip(eager_velocities, ir_velocities, strict=True)
    )

    policy.reset()
    queue_actions = []
    queue_seconds = []
    with torch.inference_mode():
        for _ in range(3):
            started = time.perf_counter()
            queue_actions.append(policy.select_action(batch, noise=noise).detach().clone())
            _synchronize(torch, config.device)
            queue_seconds.append(time.perf_counter() - started)
    queue_errors = tuple(
        float((eager - actual).abs().max().cpu())
        for eager, actual in zip(
            queue_actions, ir_queue_actions, strict=True
        )
    )

    if trace_path is not None:
        runtime.trace.write(trace_path)
    peak_memory = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if config.device.startswith("cuda")
        else None
    )
    all_errors = (action_error,) + solver_errors + queue_errors
    passed = (
        len(eager_velocities) == config.num_steps
        and len(ir_velocities) == config.num_steps
        and all(error <= config.tolerance for error in all_errors)
    )
    return RealSmolVLAEvidence(
        schema="vlaforge.real_model_evidence/0.1",
        evidence_kind="real_checkpoint",
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=_sha256(checkpoint),
        vlm_path=str(config.vlm_path.resolve()),
        lerobot_revision=config.lerobot_revision,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        device=config.device,
        gpu_name=(
            torch.cuda.get_device_name()
            if config.device.startswith("cuda")
            else None
        ),
        num_steps=config.num_steps,
        action_shape=tuple(int(dim) for dim in eager_action.shape),
        eager_seconds=eager_seconds,
        ir_seconds=ir_seconds,
        queue_seconds=tuple(queue_seconds),
        peak_memory_mb=peak_memory,
        action_max_abs_error=action_error,
        solver_max_abs_errors=solver_errors,
        action_queue_max_abs_errors=queue_errors,
        trace_events=len(runtime.trace.events),
        passed=passed,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
