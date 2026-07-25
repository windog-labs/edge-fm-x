"""π0-shaped flow-matching action-chunk fixture for Invocation IR v0.2.

This is L1 source-contract evidence only. It models a VLM prefix, proprioceptive
conditioning, bounded flow integration, and continuous chunk output without
claiming support for pretrained π0 weights.
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
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


IMAGE = TensorType((2,), "f32")
TOKENS = TensorType((3,), "i64")
PROPRIO = TensorType((2,), "f32")
PREFIX = TensorType((2,), "f32")
ACTION_CHUNK = TensorType((4, 2), "f32")
INDEX = ScalarType("index")


def build_pi0_fixture() -> AdapterFixture:
    builder = ModuleBuilder("pi0_flow_action_chunk_fixture")
    builder.add_input(InputPort("image", IMAGE))
    builder.add_input(InputPort("instruction", TOKENS))
    builder.add_input(InputPort("proprio", PROPRIO))
    builder.add_output(
        OutputPort("action_chunk", ACTION_CHUNK, group="robot_action")
    )
    builder.add_region(
        TensorRegion(
            "pi0_embed_prefix",
            (
                Value("image", IMAGE),
                Value("instruction", TOKENS),
            ),
            (PREFIX,),
            metadata={
                "memoize": True,
                "template": "FlowMatchingAction",
                "source_contract": "LeRobot PI0Policy-like",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "pi0_seed_noise",
            (
                Value("prefix", PREFIX),
                Value("proprio", PROPRIO),
            ),
            (ACTION_CHUNK,),
        )
    )
    builder.add_region(
        TensorRegion(
            "pi0_flow_step",
            (
                Value("prefix", PREFIX),
                Value("proprio", PROPRIO),
                Value("sample", ACTION_CHUNK),
                Value("step", INDEX),
            ),
            (ACTION_CHUNK,),
        )
    )
    builder.add_region(
        TensorRegion(
            "pi0_unpad_actions",
            (Value("sample", ACTION_CHUNK),),
            (ACTION_CHUNK,),
        )
    )
    flow_body = Block.of(
        (
            ops.invoke(
                ("sample_next",),
                (ACTION_CHUNK,),
                "pi0_flow_step",
                ("prefix", "proprio_value", "sample_iter", "flow_step"),
            ),
            ops.yield_values("sample_next"),
        )
    )
    pending = PendingOutputType("action_chunk", ACTION_CHUNK)
    body = Block.of(
        (
            ops.input_read("image_value", "image_rev", IMAGE, "image"),
            ops.input_read(
                "instruction_value",
                "instruction_rev",
                TOKENS,
                "instruction",
            ),
            ops.input_read(
                "proprio_value", "proprio_rev", PROPRIO, "proprio"
            ),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("prefix",),
                (PREFIX,),
                "pi0_embed_prefix",
                ("image_value", "instruction_value"),
            ),
            ops.invoke(
                ("sample_initial",),
                (ACTION_CHUNK,),
                "pi0_seed_noise",
                ("prefix", "proprio_value"),
            ),
            ops.for_loop(
                Value("sample_final", ACTION_CHUNK),
                "sample_initial",
                Value("flow_step", INDEX),
                Value("sample_iter", ACTION_CHUNK),
                flow_body,
                lower=0,
                upper=4,
            ),
            ops.invoke(
                ("action_chunk",),
                (ACTION_CHUNK,),
                "pi0_unpad_actions",
                ("sample_final",),
            ),
            ops.validate("valid", "action_chunk", "finite_chunk"),
            ops.output_create(
                "pending_action_chunk",
                "action_chunk",
                ACTION_CHUNK,
                "action_chunk",
            ),
            ops.output_group(
                "pending_outputs",
                "robot_action",
                (("pending_action_chunk", pending),),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (pending,),
                "robot_action",
                "txn",
                "pending_outputs",
                "valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "FlowMatchingAction",
                "evidence_level": "L1",
                "source_contract": "PI0-like",
            },
        )
    )

    def embed_prefix(image, instruction):
        language = sum(instruction) / 20.0
        return image[0] + language, image[1] - language

    def seed_noise(prefix, proprio):
        return tuple(
            (
                0.1 * (index + 1) + 0.05 * prefix[0],
                -0.08 * (index + 1) + 0.05 * proprio[1],
            )
            for index in range(4)
        )

    def flow_step(prefix, proprio, sample, step):
        dt = -0.25
        return tuple(
            (
                action[0]
                + dt * (0.2 * action[0] - 0.03 * prefix[0] + 0.01 * step),
                action[1]
                + dt
                * (0.2 * action[1] - 0.03 * proprio[1] - 0.01 * step),
            )
            for action in sample
        )

    def unpad(sample):
        return tuple(
            tuple(max(-1.0, min(1.0, float(item))) for item in action)
            for action in sample
        )

    def bindings(revision: int):
        return {
            "image": InputBinding(
                TensorView((0.2, -0.4), (2,), "f32"),
                InputStamp(revision=revision),
            ),
            "instruction": InputBinding(
                TensorView((2, 1, 4), (3,), "i64"),
                InputStamp(revision=revision),
            ),
            "proprio": InputBinding(
                TensorView((-0.1, 0.3), (2,), "f32"),
                InputStamp(revision=revision),
            ),
        }

    return AdapterFixture(
        module=builder.build(),
        regions={
            "pi0_embed_prefix": embed_prefix,
            "pi0_seed_noise": seed_noise,
            "pi0_flow_step": flow_step,
            "pi0_unpad_actions": unpad,
        },
        validators={"finite_chunk": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(bindings(50)),
            FixtureRun(bindings(50)),
            FixtureRun(bindings(51)),
        ),
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, tuple | list):
        return all(_all_finite(item) for item in value)
    return isinstance(value, int | float) and math.isfinite(value)
