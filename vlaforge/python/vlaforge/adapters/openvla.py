"""Deterministic autoregressive VLA-shaped fixture.

The fixture exercises a different action-generation structure from the flow
policy fixture while using exactly the same core IR. It is not a substitute for
the required real OpenVLA checkpoint gate.
"""

from __future__ import annotations

import math

from vlaforge.adapters.common import AdapterFixture, FixtureTick
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter.clocks import Epoch, InputSample
from vlaforge.ir import ops
from vlaforge.ir.attrs import FreshnessConstraint
from vlaforge.ir.program import (
    Block,
    ClockDomain,
    InputStream,
    Policy,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import EpochType, ScalarType, TensorType


VECTOR = TensorType((2,), "f32")
TOKENS = TensorType((3,), "i64")
TOKEN = ScalarType("i64")


def build_openvla_fixture() -> AdapterFixture:
    builder = ModuleBuilder("autoregressive_policy_fixture")
    builder.add_clock(ClockDomain("observation", period_ns=50_000_000))
    builder.add_clock(ClockDomain("control", period_ns=50_000_000))
    builder.add_input(
        InputStream(
            "image",
            VECTOR,
            "observation",
            FreshnessConstraint(max_age_ns=60_000_000),
        )
    )
    builder.add_input(
        InputStream(
            "instruction",
            TOKENS,
            "observation",
            FreshnessConstraint(max_age_ns=60_000_000),
        )
    )
    builder.add_region(
        TensorRegion(
            "encode_context",
            (
                Value("image_arg", VECTOR),
                Value("instruction_arg", TOKENS),
            ),
            (VECTOR,),
        )
    )
    builder.add_region(
        TensorRegion("initial_action_token", (Value("context_arg", VECTOR),), (TOKEN,))
    )
    builder.add_region(
        TensorRegion(
            "next_action_token",
            (
                Value("context_arg", VECTOR),
                Value("token_arg", TOKEN),
                Value("step_arg", ScalarType("index")),
            ),
            (TOKEN,),
        )
    )
    builder.add_region(
        TensorRegion("detokenize_action", (Value("token_arg", TOKEN),), (VECTOR,))
    )

    token_loop = Block.of(
        (
            ops.invoke(
                ("token_next",),
                (TOKEN,),
                "next_action_token",
                ("context", "token_iter", "token_step"),
            ),
            ops.yield_values("token_next"),
        )
    )
    body = Block.of(
        (
            ops.sample_input(
                "image_value",
                "image_epoch",
                VECTOR,
                "image",
                "observation",
                max_age_ns=60_000_000,
            ),
            ops.sample_input(
                "instruction_value",
                "instruction_epoch",
                TOKENS,
                "instruction",
                "observation",
                max_age_ns=60_000_000,
            ),
            ops.transaction_begin("txn", "tick"),
            ops.invoke(
                ("context",),
                (VECTOR,),
                "encode_context",
                ("image_value", "instruction_value"),
            ),
            ops.invoke(
                ("token_initial",),
                (TOKEN,),
                "initial_action_token",
                ("context",),
            ),
            ops.for_loop(
                Value("token_final", TOKEN),
                "token_initial",
                Value("token_step", ScalarType("index")),
                Value("token_iter", TOKEN),
                token_loop,
                lower=0,
                upper=3,
            ),
            ops.invoke(
                ("decoded_action",),
                (VECTOR,),
                "detokenize_action",
                ("token_final",),
            ),
            ops.validate("action_valid", "decoded_action", "bounded_action"),
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
            metadata={"action_generation": "autoregressive_discrete"},
        )
    )

    def encode_context(
        image: tuple[float, float],
        instruction: tuple[int, int, int],
    ) -> tuple[float, float]:
        language = sum(instruction) / 100.0
        return (
            float(image[0]) + language,
            float(image[1]) - language,
        )

    def initial_action_token(context: tuple[float, float]) -> int:
        return int(round((context[0] - context[1]) * 10.0)) % 64

    def next_action_token(context: tuple[float, float], token: int, step: int) -> int:
        bias = int(round((context[0] + context[1]) * 5.0))
        return (token * 7 + bias + step + 3) % 64

    def detokenize_action(token: int) -> tuple[float, float]:
        first = token / 63.0 * 2.0 - 1.0
        return first, -0.5 * first

    ticks = tuple(
        FixtureTick(
            tick=Epoch("control", index, index * 50_000_000, 0),
            inputs={
                "image": InputSample(
                    (0.1 * index, 0.4 - 0.05 * index),
                    Epoch("observation", index, index * 50_000_000, 0),
                ),
                "instruction": InputSample(
                    (1, 2 + index, 3),
                    Epoch("observation", index, index * 50_000_000, 0),
                ),
            },
        )
        for index in range(3)
    )
    return AdapterFixture(
        module=builder.build(),
        regions={
            "encode_context": encode_context,
            "initial_action_token": initial_action_token,
            "next_action_token": next_action_token,
            "detokenize_action": detokenize_action,
        },
        validators={
            "bounded_action": lambda action: all(
                math.isfinite(item) and -1.0 <= item <= 1.0 for item in action
            )
        },
        initial_state={},
        ticks=ticks,
    )
