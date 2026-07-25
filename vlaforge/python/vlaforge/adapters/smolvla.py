"""SmolVLA-shaped ChunkedAction fixture for Invocation IR v0.2.

This is executable model-structure evidence (L1), not proof that the real
``lerobot/smolvla_base`` checkpoint has been captured or compiled.
"""

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
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


VECTOR = TensorType((2,), "f32")
ACTION_CHUNK = TensorType((4, 2), "f32")
RNG = ScalarType("i64")
CURSOR = ScalarType("i32")


def build_smolvla_fixture() -> AdapterFixture:
    builder = ModuleBuilder("smolvla_chunked_action_fixture")
    builder.add_input(InputPort("image", VECTOR))
    builder.add_output(OutputPort("action", VECTOR, group="manipulation"))
    builder.add_state(StateSlot("action_queue", ACTION_CHUNK, retention=3))
    builder.add_state(StateSlot("queue_cursor", CURSOR, retention=3))
    builder.add_state(StateSlot("rng", RNG, retention=3))
    builder.add_region(
        TensorRegion(
            "encode_observation",
            (Value("image_arg", VECTOR),),
            (VECTOR,),
            metadata={"memoize": True, "template": "ChunkedAction"},
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
                ("prefix", "sample_iter", "solver_index"),
            ),
            ops.yield_values("sample_next"),
        )
    )
    refill_branch = Block.of(
        (
            ops.invoke(
                ("prefix",),
                (VECTOR,),
                "encode_observation",
                ("image_value",),
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
                Value("solver_index", ScalarType("index")),
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
    pending_action_type = PendingOutputType("action", VECTOR)
    body = Block.of(
        (
            ops.input_read("image_value", "image_revision", VECTOR, "image"),
            ops.transaction_begin("txn"),
            ops.state_read_latest(
                "queue_snapshot",
                ACTION_CHUNK,
                "action_queue",
                "txn",
            ),
            ops.snapshot_value("queue_value", ACTION_CHUNK, "queue_snapshot"),
            ops.state_read_latest(
                "cursor_snapshot",
                CURSOR,
                "queue_cursor",
                "txn",
            ),
            ops.snapshot_value("cursor_value", CURSOR, "cursor_snapshot"),
            ops.state_read_latest("rng_snapshot", RNG, "rng", "txn"),
            ops.snapshot_value("rng_value", RNG, "rng_snapshot"),
            ops.invoke(
                ("queue_empty",),
                (ScalarType("bool"),),
                "queue_is_empty",
                ("cursor_value",),
            ),
            ops.invoke(("zero_cursor",), (CURSOR,), "queue_zero", ()),
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
            ),
            ops.stage_write(
                "cursor_pending",
                CURSOR,
                "queue_cursor",
                "txn",
                "cursor_next",
            ),
            ops.stage_write(
                "rng_pending",
                RNG,
                "rng",
                "txn",
                "rng_next",
            ),
            ops.validate("action_valid", "selected_action", "finite_action"),
            ops.output_create(
                "pending_action",
                "selected_action",
                VECTOR,
                "action",
            ),
            ops.output_group(
                "pending_outputs",
                "manipulation",
                (("pending_action", pending_action_type),),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (pending_action_type,),
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
            body,
            metadata={
                "action_generation": "iterative_continuous_chunk",
                "adapter_template": "ChunkedAction",
                "persistent_state": "action_queue,queue_cursor,rng",
            },
        )
    )

    def encode_observation(image):
        return tuple(float(item) * 2.0 for item in image)

    def sample_noise(rng):
        chunk = tuple(
            (
                ((rng * 17 + 11 + index * 5) % 101) / 100.0,
                ((rng * 29 + 7 + index * 3) % 103) / 100.0,
            )
            for index in range(4)
        )
        return chunk, rng + 1

    def solver_step(prefix, sample, step):
        return tuple(
            tuple(
                float(current) + 0.05 * float(context) + 0.01 * step
                for context, current in zip(prefix, action, strict=True)
            )
            for action in sample
        )

    def decode_action_chunk(sample):
        return tuple(
            tuple(max(-1.0, min(1.0, float(item))) for item in action)
            for action in sample
        )

    runs = tuple(
        FixtureRun(
            {
                "image": InputBinding(
                    TensorView(
                        (0.25 + (index // 4) * 0.1, -0.5),
                        (2,),
                        "f32",
                    ),
                    InputStamp(revision=100 + index // 4),
                )
            }
        )
        for index in range(6)
    )
    return AdapterFixture(
        module=builder.build(),
        regions={
            "encode_observation": encode_observation,
            "sample_noise": sample_noise,
            "solver_step": solver_step,
            "decode_action_chunk": decode_action_chunk,
            "queue_is_empty": lambda cursor: cursor >= 4,
            "queue_select": lambda queue, cursor: queue[cursor],
            "queue_advance": lambda cursor: cursor + 1,
            "queue_zero": lambda: 0,
        },
        validators={
            "finite_action": lambda action: all(
                math.isfinite(item) for item in action
            )
        },
        initial_state={
            "action_queue": tuple((0.0, 0.0) for _ in range(4)),
            "queue_cursor": 4,
            "rng": 7,
        },
        runs=runs,
        evidence_kind="source_faithful_fixture_l1",
    )
