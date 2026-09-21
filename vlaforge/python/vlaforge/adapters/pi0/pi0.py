"""π0-shaped flow-matching action-chunk fixture for Invocation IR v0.2.

This is L1 source-contract evidence only. It models a VLM prefix, proprioceptive
conditioning, bounded flow integration, and continuous chunk output without
claiming support for pretrained π0 weights.
"""

from __future__ import annotations

import math

from vlaforge.adapters.shared.fixtures import AdapterFixture, FixtureRun
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.interpreter import InputBinding, InputStamp, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import ScalarType, TensorType


IMAGE = TensorType((2,), "f32")
TOKENS = TensorType((3,), "i64")
PROPRIO = TensorType((2,), "f32")
PREFIX = TensorType((2,), "f32")
ACTION_CHUNK = TensorType((4, 2), "f32")
INDEX = ScalarType("index")


def build_pi0_fixture() -> AdapterFixture:
    @tensor_region(
        "pi0_embed_prefix",
        inputs=(Value("image", IMAGE), Value("instruction", TOKENS)),
        outputs=(PREFIX,),
    )
    def embed_prefix(image, instruction):
        language = sum(instruction) / 20.0
        return image[0] + language, image[1] - language

    @tensor_region(
        "pi0_seed_noise",
        inputs=(Value("prefix", PREFIX), Value("proprio", PROPRIO)),
        outputs=(ACTION_CHUNK,),
    )
    def seed_noise(prefix, proprio):
        return tuple(
            (
                0.1 * (index + 1) + 0.05 * prefix[0],
                -0.08 * (index + 1) + 0.05 * proprio[1],
            )
            for index in range(4)
        )

    @tensor_region(
        "pi0_flow_step",
        inputs=(
            Value("prefix", PREFIX),
            Value("proprio", PROPRIO),
            Value("sample", ACTION_CHUNK),
            Value("step", INDEX),
        ),
        outputs=(ACTION_CHUNK,),
    )
    def flow_step(prefix, proprio, sample, step):
        dt = -0.25
        return tuple(
            (
                action[0] + dt * (0.2 * action[0] - 0.03 * prefix[0] + 0.01 * step),
                action[1] + dt * (0.2 * action[1] - 0.03 * proprio[1] - 0.01 * step),
            )
            for action in sample
        )

    @tensor_region(
        "pi0_unpad_actions",
        inputs=(Value("sample", ACTION_CHUNK),),
        outputs=(ACTION_CHUNK,),
    )
    def unpad(sample):
        return tuple(
            tuple(max(-1.0, min(1.0, float(item))) for item in action)
            for action in sample
        )

    @tensor_region(
        "pi0_finite_chunk",
        inputs=(Value("sample", ACTION_CHUNK),),
        outputs=(ScalarType("bool"),),
    )
    def finite_chunk(sample):
        return _all_finite(sample)

    builder = InvocationBuilder(
        "pi0_flow_action_chunk_fixture",
        inputs=(
            InputPort("image", IMAGE),
            InputPort("instruction", TOKENS),
            InputPort("proprio", PROPRIO),
        ),
        outputs=(OutputPort("action_chunk", ACTION_CHUNK, group="robot_action"),),
    )
    image = builder.input("image")
    instruction = builder.input("instruction")
    proprio = builder.input("proprio")
    (prefix,) = builder.call(embed_prefix, image, instruction, cache=True)
    (sample,) = builder.call(seed_noise, prefix, proprio)
    (sample,) = builder.iterate(
        (sample,),
        lambda step, value: builder.call(flow_step, prefix, proprio, value, step),
        steps=4,
    )
    (action,) = builder.call(unpad, sample)
    (accepted,) = builder.call(finite_chunk, action)
    program = builder.finish(
        {"action_chunk": action},
        accepted=accepted,
        metadata={
            "adapter_template": "FlowMatchingAction",
            "evidence_level": "L1",
            "source_contract": "PI0-like",
        },
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
        module=program.module,
        regions=program.regions,
        validators=program.validators,
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
