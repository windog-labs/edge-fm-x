"""Deterministic autoregressive VLA invocation fixture."""

from __future__ import annotations

import math

from vlaforge.adapters.common import AdapterFixture, FixtureRun
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    OutputPort,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


VECTOR = TensorType((2,), "f32")
TOKENS = TensorType((3,), "i64")
TOKEN = ScalarType("i64")


def build_openvla_fixture() -> AdapterFixture:
    builder = ModuleBuilder("autoregressive_invocation_fixture")
    builder.add_input(InputPort("image", VECTOR))
    builder.add_input(InputPort("instruction", TOKENS))
    builder.add_output(OutputPort("action", VECTOR))
    builder.add_region(
        TensorRegion(
            "encode_context",
            (
                Value("image_arg", VECTOR),
                Value("instruction_arg", TOKENS),
            ),
            (VECTOR,),
            metadata={"memoize": True, "loop_invariant": True},
        )
    )
    builder.add_region(
        TensorRegion(
            "initial_action_token",
            (Value("context_arg", VECTOR),),
            (TOKEN,),
        )
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
        TensorRegion(
            "detokenize_action",
            (Value("token_arg", TOKEN),),
            (VECTOR,),
        )
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
            ops.input_read(
                "image_value",
                "image_revision",
                VECTOR,
                "image",
            ),
            ops.input_read(
                "instruction_value",
                "instruction_revision",
                TOKENS,
                "instruction",
            ),
            ops.transaction_begin("txn"),
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
            ops.validate(
                "output_valid",
                "decoded_action",
                "bounded_action",
            ),
            ops.output_create(
                "pending_output",
                "decoded_action",
                VECTOR,
                "action",
            ),
            ops.output_group(
                "pending_outputs",
                "default",
                (
                    (
                        "pending_output",
                        PendingOutputType("action", VECTOR),
                    ),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (PendingOutputType("action", VECTOR),),
                "default",
                "txn",
                "pending_outputs",
                "output_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
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

    def next_action_token(
        context: tuple[float, float],
        token: int,
        step: int,
    ) -> int:
        bias = int(round((context[0] + context[1]) * 5.0))
        return (token * 7 + bias + step + 3) % 64

    def detokenize_action(token: int) -> tuple[float, float]:
        first = token / 63.0 * 2.0 - 1.0
        return first, -0.5 * first

    runs = tuple(
        FixtureRun(
            inputs={
                "image": InputBinding(
                    TensorView(
                        (0.1 * index, 0.4 - 0.05 * index),
                        (2,),
                        "f32",
                    ),
                    InputStamp(revision=index),
                ),
                "instruction": InputBinding(
                    TensorView((1, 2 + index, 3), (3,), "i64"),
                    InputStamp(revision=index),
                ),
            }
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
                math.isfinite(item) and -1.0 <= item <= 1.0
                for item in action
            )
        },
        initial_state={},
        runs=runs,
    )
