"""Deterministic SmolVLA-shaped fixture using only generic core operations.

This adapter is for offline semantic testing. It is deliberately labelled as a
fixture and is not evidence that the real ``lerobot/smolvla_base`` checkpoint
has run.
"""

from __future__ import annotations

import math

from vlaforge.adapters.common import AdapterFixture, FixtureTick
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter.clocks import Epoch, InputSample
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


VECTOR = TensorType((2,), "f32")
ACTION_CHUNK = TensorType((4, 2), "f32")
RNG = ScalarType("i64")
CURSOR = ScalarType("i32")


def build_smolvla_fixture() -> AdapterFixture:
    builder = ModuleBuilder("flow_policy_fixture")
    builder.add_clock(ClockDomain("observation", period_ns=33_333_333))
    builder.add_clock(
        ClockDomain("control", period_ns=20_000_000, deadline_ns=18_000_000)
    )
    builder.add_input(
        InputStream(
            "image",
            VECTOR,
            "observation",
            FreshnessConstraint(max_age_ns=50_000_000),
        )
    )
    builder.add_state(
        StateSlot(
            "action_queue",
            ACTION_CHUNK,
            StateScope.EPISODE,
            "control",
            retention=3,
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
            CURSOR,
            StateScope.EPISODE,
            "control",
            retention=3,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EPISODE_START,
            authoritative=True,
            checkpoint=CheckpointPolicy.ON_COMMIT,
        )
    )
    builder.add_state(
        StateSlot(
            "rng",
            RNG,
            StateScope.SESSION,
            "control",
            retention=3,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EPISODE_START,
            checkpoint=CheckpointPolicy.ON_COMMIT,
        )
    )

    builder.add_region(
        TensorRegion(
            "encode_observation",
            (Value("image_arg", VECTOR),),
            (VECTOR,),
            metadata={"memoize": True},
        )
    )
    builder.add_region(
        TensorRegion(
            "sample_noise",
            (Value("rng_arg", RNG),),
            (ACTION_CHUNK, RNG),
        )
    )
    builder.add_region(
        TensorRegion(
            "solver_step",
            (
                Value("prefix_arg", VECTOR),
                Value("sample_arg", ACTION_CHUNK),
                Value("step_arg", ScalarType("index")),
            ),
            (ACTION_CHUNK,),
        )
    )
    builder.add_region(
        TensorRegion(
            "decode_action_chunk",
            (Value("sample_arg", ACTION_CHUNK),),
            (ACTION_CHUNK,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_is_empty",
            (Value("cursor_arg", CURSOR),),
            (ScalarType("bool"),),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_select",
            (
                Value("queue_arg", ACTION_CHUNK),
                Value("cursor_arg", CURSOR),
            ),
            (VECTOR,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_advance",
            (Value("cursor_arg", CURSOR),),
            (CURSOR,),
        )
    )
    builder.add_region(TensorRegion("queue_zero", (), (CURSOR,)))

    loop_body = Block.of(
        (
            ops.invoke(
                ("sample_next",),
                (ACTION_CHUNK,),
                "solver_step",
                ("prefix", "sample_iter", "step"),
            ),
            ops.yield_values("sample_next"),
        )
    )
    refill_branch = Block.of(
        (
            ops.invoke(
                ("prefix",), (VECTOR,), "encode_observation", ("image_value",)
            ),
            ops.invoke(
                ("sample_initial", "rng_after_refill"),
                (ACTION_CHUNK, RNG),
                "sample_noise",
                ("rng_value",),
            ),
            ops.for_loop(
                Value("sample_final", ACTION_CHUNK),
                "sample_initial",
                Value("step", ScalarType("index")),
                Value("sample_iter", ACTION_CHUNK),
                loop_body,
                lower=0,
                upper=4,
            ),
            ops.invoke(
                ("refilled_queue",),
                (ACTION_CHUNK,),
                "decode_action_chunk",
                ("sample_final",),
            ),
            ops.invoke(
                ("refilled_action",),
                (VECTOR,),
                "queue_select",
                ("refilled_queue", "zero_cursor"),
            ),
            ops.invoke(
                ("cursor_after_refill",),
                (CURSOR,),
                "queue_advance",
                ("zero_cursor",),
            ),
            ops.yield_values(
                "refilled_action",
                "refilled_queue",
                "cursor_after_refill",
                "rng_after_refill",
            ),
        )
    )
    reuse_branch = Block.of(
        (
            ops.invoke(
                ("queued_action",),
                (VECTOR,),
                "queue_select",
                ("queue_value", "cursor_value"),
            ),
            ops.invoke(
                ("cursor_after_reuse",),
                (CURSOR,),
                "queue_advance",
                ("cursor_value",),
            ),
            ops.yield_values(
                "queued_action",
                "queue_value",
                "cursor_after_reuse",
                "rng_value",
            ),
        )
    )
    body = Block.of(
        (
            ops.sample_input(
                "image_value",
                "observation_epoch",
                VECTOR,
                "image",
                "observation",
                max_age_ns=50_000_000,
            ),
            ops.transaction_begin("txn", "tick"),
            ops.state_read(
                "queue_snapshot",
                ACTION_CHUNK,
                "action_queue",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value("queue_value", ACTION_CHUNK, "queue_snapshot"),
            ops.state_read(
                "cursor_snapshot",
                CURSOR,
                "queue_cursor",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value("cursor_value", CURSOR, "cursor_snapshot"),
            ops.state_read(
                "rng_snapshot",
                RNG,
                "rng",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value("rng_value", RNG, "rng_snapshot"),
            ops.invoke(
                ("queue_empty",),
                (ScalarType("bool"),),
                "queue_is_empty",
                ("cursor_value",),
            ),
            ops.invoke(
                ("zero_cursor",),
                (CURSOR,),
                "queue_zero",
                (),
            ),
            ops.if_op(
                (
                    Value("selected_action", VECTOR),
                    Value("queue_next", ACTION_CHUNK),
                    Value("cursor_next", CURSOR),
                    Value("rng_next", RNG),
                ),
                "queue_empty",
                refill_branch,
                reuse_branch,
            ),
            ops.stage_write(
                "queue_pending",
                ACTION_CHUNK,
                "action_queue",
                "txn",
                "queue_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.stage_write(
                "cursor_pending",
                CURSOR,
                "queue_cursor",
                "txn",
                "cursor_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.stage_write(
                "rng_pending",
                RNG,
                "rng",
                "txn",
                "rng_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.validate("action_valid", "selected_action", "finite_action"),
            ops.action_create(
                "pending_action", "selected_action", VECTOR, "tick"
            ),
            ops.transaction_commit(
                "committed_action",
                VECTOR,
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
            metadata={"action_generation": "iterative_continuous"},
        )
    )

    def encode_observation(image: tuple[float, float]) -> tuple[float, float]:
        return tuple(float(item) * 2.0 for item in image)

    def sample_noise(rng: int) -> tuple[tuple[tuple[float, float], ...], int]:
        chunk = tuple(
            (
                ((rng * 17 + 11 + index * 5) % 101) / 100.0,
                ((rng * 29 + 7 + index * 3) % 103) / 100.0,
            )
            for index in range(4)
        )
        return chunk, rng + 1

    def solver_step(
        prefix: tuple[float, float],
        sample: tuple[tuple[float, float], ...],
        step: int,
    ) -> tuple[tuple[float, float], ...]:
        return tuple(
            tuple(
                float(current) + 0.05 * float(context) + 0.01 * step
                for context, current in zip(prefix, action, strict=True)
            )
            for action in sample
        )

    def decode_action_chunk(
        sample: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        return tuple(
            tuple(max(-1.0, min(1.0, float(item))) for item in action)
            for action in sample
        )

    def queue_is_empty(cursor: int) -> bool:
        return cursor >= 4

    def queue_select(
        queue: tuple[tuple[float, float], ...], cursor: int
    ) -> tuple[float, float]:
        return queue[cursor]

    def queue_advance(cursor: int) -> int:
        return cursor + 1

    def queue_zero() -> int:
        return 0

    ticks = tuple(
        FixtureTick(
            tick=Epoch("control", index, index * 20_000_000, 0),
            inputs={
                "image": InputSample(
                    (0.25 + index * 0.1, -0.5 + index * 0.05),
                    Epoch("observation", index, index * 20_000_000, 0),
                )
            },
        )
        for index in range(3)
    )
    return AdapterFixture(
        module=builder.build(),
        regions={
            "encode_observation": encode_observation,
            "sample_noise": sample_noise,
            "solver_step": solver_step,
            "decode_action_chunk": decode_action_chunk,
            "queue_is_empty": queue_is_empty,
            "queue_select": queue_select,
            "queue_advance": queue_advance,
            "queue_zero": queue_zero,
        },
        validators={
            "finite_action": lambda action: all(math.isfinite(item) for item in action)
        },
        initial_state={
            "action_queue": tuple((0.0, 0.0) for _ in range(4)),
            "queue_cursor": 4,
            "rng": 7,
        },
        ticks=ticks,
    )
