"""Validated transactional fallback using only generic Invocation IR v0.2."""

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
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, TensorType


VECTOR = TensorType((2,), "f32")


def build_transactional_fallback_fixture() -> AdapterFixture:
    builder = ModuleBuilder("transactional_output_fallback_fixture")
    builder.add_input(InputPort("observation", VECTOR))
    builder.add_output(
        OutputPort("action", VECTOR, group="manipulation")
    )
    builder.add_state(
        StateSlot("last_committed_action", VECTOR, retention=3)
    )
    builder.add_region(
        TensorRegion(
            "propose_action",
            (Value("observation_arg", VECTOR),),
            (VECTOR,),
        )
    )
    accept = Block.of((ops.yield_values("candidate_action"),))
    fallback = Block.of((ops.yield_values("last_action_value"),))
    pending_type = PendingOutputType("action", VECTOR)
    body = Block.of(
        (
            ops.input_read(
                "observation_value",
                "observation_revision",
                VECTOR,
                "observation",
            ),
            ops.transaction_begin("txn"),
            ops.state_read_latest(
                "last_action_snapshot",
                VECTOR,
                "last_committed_action",
                "txn",
            ),
            ops.snapshot_value(
                "last_action_value",
                VECTOR,
                "last_action_snapshot",
            ),
            ops.invoke(
                ("candidate_action",),
                (VECTOR,),
                "propose_action",
                ("observation_value",),
            ),
            ops.validate(
                "candidate_valid",
                "candidate_action",
                "bounded_action",
            ),
            ops.if_op(
                (Value("selected_action", VECTOR),),
                "candidate_valid",
                accept,
                fallback,
            ),
            ops.validate(
                "selected_valid",
                "selected_action",
                "bounded_action",
            ),
            ops.stage_write(
                "last_action_pending",
                VECTOR,
                "last_committed_action",
                "txn",
                "selected_action",
            ),
            ops.output_create(
                "pending_action",
                "selected_action",
                VECTOR,
                "action",
            ),
            ops.output_group(
                "pending_outputs",
                "manipulation",
                (("pending_action", pending_type),),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (pending_type,),
                "manipulation",
                "txn",
                "pending_outputs",
                "selected_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "ChunkedAction",
                "fallback": "last_committed_action",
            },
        )
    )

    def propose_action(observation):
        if observation[0] < 0.0:
            return 2.0, float(observation[1])
        return (
            float(observation[0]) * 0.5,
            float(observation[1]) * -0.5,
        )

    values = ((0.4, 0.2), (-1.0, 0.7), (0.8, -0.4))
    return AdapterFixture(
        module=builder.build(),
        regions={"propose_action": propose_action},
        validators={
            "bounded_action": lambda action: all(
                math.isfinite(item) and -1.0 <= item <= 1.0
                for item in action
            )
        },
        initial_state={"last_committed_action": (0.0, 0.0)},
        runs=tuple(
            FixtureRun(
                {
                    "observation": InputBinding(
                        TensorView(value, (2,), "f32"),
                        InputStamp(revision=index),
                    )
                }
            )
            for index, value in enumerate(values)
        ),
        evidence_kind="source_faithful_fixture_l1",
    )
