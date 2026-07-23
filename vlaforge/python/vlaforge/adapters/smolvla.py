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
RNG = ScalarType("i64")


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
            "prefix_cache",
            VECTOR,
            StateScope.SESSION,
            "observation",
            retention=3,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EPISODE_START,
            freshness=FreshnessConstraint(max_versions=1),
            ownership=Ownership.DEVICE,
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
            (VECTOR, RNG),
        )
    )
    builder.add_region(
        TensorRegion(
            "solver_step",
            (
                Value("prefix_arg", VECTOR),
                Value("sample_arg", VECTOR),
                Value("step_arg", ScalarType("index")),
            ),
            (VECTOR,),
        )
    )
    builder.add_region(
        TensorRegion("decode_action", (Value("sample_arg", VECTOR),), (VECTOR,))
    )

    loop_body = Block.of(
        (
            ops.invoke(
                ("sample_next",),
                (VECTOR,),
                "solver_step",
                ("prefix", "sample_iter", "step"),
            ),
            ops.yield_values("sample_next"),
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
            ops.invoke(
                ("prefix",), (VECTOR,), "encode_observation", ("image_value",)
            ),
            ops.state_read(
                "rng_snapshot",
                RNG,
                "rng",
                "txn",
                epoch=EpochExpr.current("control"),
            ),
            ops.snapshot_value("rng_value", RNG, "rng_snapshot"),
            ops.invoke(
                ("sample_initial", "rng_next"),
                (VECTOR, RNG),
                "sample_noise",
                ("rng_value",),
            ),
            ops.for_loop(
                Value("sample_final", VECTOR),
                "sample_initial",
                Value("step", ScalarType("index")),
                Value("sample_iter", VECTOR),
                loop_body,
                lower=0,
                upper=4,
            ),
            ops.invoke(
                ("decoded_action",),
                (VECTOR,),
                "decode_action",
                ("sample_final",),
            ),
            ops.stage_write(
                "prefix_pending",
                VECTOR,
                "prefix_cache",
                "txn",
                "prefix",
                epoch=EpochExpr("input", "observation"),
            ),
            ops.stage_write(
                "rng_pending",
                RNG,
                "rng",
                "txn",
                "rng_next",
                epoch=EpochExpr.next("control"),
            ),
            ops.validate("action_valid", "decoded_action", "finite_action"),
            ops.action_create(
                "pending_action", "decoded_action", VECTOR, "tick"
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

    def sample_noise(rng: int) -> tuple[tuple[float, float], int]:
        first = ((rng * 17 + 11) % 101) / 100.0
        second = ((rng * 29 + 7) % 103) / 100.0
        return (first, second), rng + 1

    def solver_step(
        prefix: tuple[float, float],
        sample: tuple[float, float],
        step: int,
    ) -> tuple[float, float]:
        return tuple(
            float(current) + 0.05 * float(context) + 0.01 * step
            for context, current in zip(prefix, sample, strict=True)
        )

    def decode_action(sample: tuple[float, float]) -> tuple[float, float]:
        return tuple(max(-1.0, min(1.0, float(item))) for item in sample)

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
            "decode_action": decode_action,
        },
        validators={
            "finite_action": lambda action: all(math.isfinite(item) for item in action)
        },
        initial_state={"rng": 7},
        ticks=ticks,
    )
