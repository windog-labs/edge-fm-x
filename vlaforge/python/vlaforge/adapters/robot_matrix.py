"""Source-faithful L1 fixtures for classic robot policy paradigms.

These programs are deterministic structure tests, not claims that upstream
weights were captured. They intentionally use only the compact Invocation IR
core; model-specific behavior remains in TensorRegions and Adapter state.
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


I32 = ScalarType("i32")
I64 = ScalarType("i64")
BOOL = ScalarType("bool")
INDEX = ScalarType("index")
ACTION = TensorType((2,), "f32")


def _tensor(data: object, type_: TensorType) -> TensorView:
    return TensorView(
        data,
        tuple(int(item) for item in type_.shape),
        type_.dtype,
        type_.layout,
    )


def _finish(
    items: tuple[tuple[str, str, TensorType | ScalarType], ...],
    *,
    group: str,
    transaction: str = "txn",
    condition: str = "valid",
):
    pending = tuple(
        PendingOutputType(output, type_) for _, output, type_ in items
    )
    return (
        *(
            ops.output_create(f"pending_{output}", value, type_, output)
            for value, output, type_ in items
        ),
        ops.output_group(
            "pending_outputs",
            group,
            tuple(
                (f"pending_{output}", type_)
                for (_, output, _), type_ in zip(
                    items, pending, strict=True
                )
            ),
        ),
        ops.transaction_commit(
            "committed_outputs",
            pending,
            group,
            transaction,
            "pending_outputs",
            condition,
        ),
        ops.return_values("committed_outputs"),
    )


def build_rt1_like_fixture() -> AdapterFixture:
    """Short observation history + language -> discrete token + action."""

    history_t = TensorType((3, 2), "f32")
    mask_t = TensorType((3,), "bool")
    language_t = TensorType((4,), "i64")
    builder = ModuleBuilder("rt1_like_discrete_action_fixture")
    builder.add_input(InputPort("observation_history", history_t))
    builder.add_input(InputPort("history_valid_mask", mask_t))
    builder.add_input(InputPort("language_tokens", language_t))
    builder.add_output(OutputPort("action_token", I64, group="robot_action"))
    builder.add_output(OutputPort("action", ACTION, group="robot_action"))
    builder.add_region(
        TensorRegion(
            "rt1_action_token",
            (
                Value("history", history_t),
                Value("mask", mask_t),
                Value("language", language_t),
            ),
            (I64,),
            metadata={"template": "StatelessDiscreteAction"},
        )
    )
    builder.add_region(
        TensorRegion(
            "rt1_detokenize",
            (Value("token", I64),),
            (ACTION,),
        )
    )
    body = Block.of(
        (
            ops.input_read("history", "history_rev", history_t,
                           "observation_history"),
            ops.input_read("mask", "mask_rev", mask_t,
                           "history_valid_mask"),
            ops.input_read("language", "language_rev", language_t,
                           "language_tokens"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("action_token",),
                (I64,),
                "rt1_action_token",
                ("history", "mask", "language"),
            ),
            ops.invoke(
                ("action",),
                (ACTION,),
                "rt1_detokenize",
                ("action_token",),
            ),
            ops.validate("valid", "action", "finite_action"),
            *_finish(
                (
                    ("action_token", "action_token", I64),
                    ("action", "action", ACTION),
                ),
                group="robot_action",
            ),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "StatelessDiscreteAction",
                "evidence_level": "L1",
                "source_contract": "RT-1-like",
            },
        )
    )

    def token(history, mask, language):
        observation = sum(
            sum(history[index]) for index in range(3) if mask[index]
        )
        return int(abs(observation * 10 + sum(language))) % 256

    def detokenize(value):
        normalized = float(value) / 255.0 * 2.0 - 1.0
        return normalized, -0.25 * normalized

    inputs = {
        "observation_history": InputBinding(
            _tensor(((0.1, 0.2), (0.3, -0.1), (0.0, 0.0)), history_t),
            InputStamp(revision=10),
        ),
        "history_valid_mask": InputBinding(
            _tensor((True, True, False), mask_t),
            InputStamp(revision=10),
        ),
        "language_tokens": InputBinding(
            _tensor((1, 2, 3, 4), language_t),
            InputStamp(revision=10),
        ),
    }
    return AdapterFixture(
        module=builder.build(),
        regions={
            "rt1_action_token": token,
            "rt1_detokenize": detokenize,
        },
        validators={"finite_action": _all_finite},
        initial_state={},
        runs=(FixtureRun(inputs),),
    )


def build_act_like_fixture() -> AdapterFixture:
    """ACT-shaped action chunk with queue/cursor owned by the Adapter."""

    observation_t = TensorType((3,), "f32")
    language_t = TensorType((3,), "i64")
    chunk_t = TensorType((3, 2), "f32")
    builder = ModuleBuilder("act_like_chunked_action_fixture")
    builder.add_input(InputPort("observation", observation_t))
    builder.add_input(InputPort("language_tokens", language_t))
    builder.add_output(OutputPort("action", ACTION, group="robot_action"))
    builder.add_state(StateSlot("action_queue", chunk_t, retention=3))
    builder.add_state(StateSlot("queue_cursor", I32, retention=3))
    builder.add_region(
        TensorRegion(
            "act_queue_empty",
            (Value("cursor", I32),),
            (BOOL,),
        )
    )
    builder.add_region(
        TensorRegion(
            "act_predict_chunk",
            (
                Value("observation", observation_t),
                Value("language", language_t),
            ),
            (chunk_t,),
            metadata={"template": "ChunkedAction"},
        )
    )
    builder.add_region(
        TensorRegion(
            "act_select",
            (Value("chunk", chunk_t), Value("cursor", I32)),
            (ACTION,),
        )
    )
    builder.add_region(
        TensorRegion("act_advance", (Value("cursor", I32),), (I32,))
    )
    builder.add_region(TensorRegion("act_zero", (), (I32,)))

    refill = Block.of(
        (
            ops.invoke(
                ("refilled_queue",),
                (chunk_t,),
                "act_predict_chunk",
                ("observation", "language"),
            ),
            ops.invoke(
                ("refilled_action",),
                (ACTION,),
                "act_select",
                ("refilled_queue", "zero"),
            ),
            ops.invoke(
                ("refilled_cursor",),
                (I32,),
                "act_advance",
                ("zero",),
            ),
            ops.yield_values(
                "refilled_action", "refilled_queue", "refilled_cursor"
            ),
        )
    )
    reuse = Block.of(
        (
            ops.invoke(
                ("queued_action",),
                (ACTION,),
                "act_select",
                ("queue", "cursor"),
            ),
            ops.invoke(
                ("next_cursor",),
                (I32,),
                "act_advance",
                ("cursor",),
            ),
            ops.yield_values("queued_action", "queue", "next_cursor"),
        )
    )
    body = Block.of(
        (
            ops.input_read("observation", "observation_rev", observation_t,
                           "observation"),
            ops.input_read("language", "language_rev", language_t,
                           "language_tokens"),
            ops.transaction_begin("txn"),
            ops.state_read_latest("queue_snapshot", chunk_t,
                                  "action_queue", "txn"),
            ops.snapshot_value("queue", chunk_t, "queue_snapshot"),
            ops.state_read_latest("cursor_snapshot", I32,
                                  "queue_cursor", "txn"),
            ops.snapshot_value("cursor", I32, "cursor_snapshot"),
            ops.invoke(("empty",), (BOOL,), "act_queue_empty", ("cursor",)),
            ops.invoke(("zero",), (I32,), "act_zero", ()),
            ops.if_op(
                (
                    Value("action", ACTION),
                    Value("next_queue", chunk_t),
                    Value("next_cursor", I32),
                ),
                "empty",
                refill,
                reuse,
            ),
            ops.stage_write("queue_pending", chunk_t, "action_queue",
                            "txn", "next_queue"),
            ops.stage_write("cursor_pending", I32, "queue_cursor",
                            "txn", "next_cursor"),
            ops.validate("valid", "action", "finite_action"),
            *_finish((("action", "action", ACTION),), group="robot_action"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "ChunkedAction",
                "persistent_state": "action_queue,queue_cursor",
                "evidence_level": "L1",
                "source_contract": "ACT-like",
            },
        )
    )

    def predict(observation, language):
        bias = sum(language) * 0.01
        return tuple(
            (
                observation[0] + bias + index * 0.1,
                observation[1] - bias - index * 0.05,
            )
            for index in range(3)
        )

    regions = {
        "act_queue_empty": lambda cursor: cursor >= 3,
        "act_predict_chunk": predict,
        "act_select": lambda chunk, cursor: chunk[cursor],
        "act_advance": lambda cursor: cursor + 1,
        "act_zero": lambda: 0,
    }
    inputs = {
        "observation": InputBinding(
            _tensor((0.2, -0.3, 0.1), observation_t),
            InputStamp(revision=20),
        ),
        "language_tokens": InputBinding(
            _tensor((2, 4, 1), language_t),
            InputStamp(revision=20),
        ),
    }
    return AdapterFixture(
        module=builder.build(),
        regions=regions,
        validators={"finite_action": _all_finite},
        initial_state={
            "action_queue": ((0.0, 0.0),) * 3,
            "queue_cursor": 3,
        },
        runs=tuple(FixtureRun(inputs) for _ in range(4)),
    )


def build_octo_like_fixture() -> AdapterFixture:
    """Octo/Diffusion-Policy-like optional modalities and bounded denoise."""

    history_t = TensorType((2, 3), "f32")
    language_t = TensorType((4,), "i64")
    goal_t = TensorType((2,), "f32")
    condition_t = TensorType((4,), "f32")
    chunk_t = TensorType((4, 2), "f32")
    builder = ModuleBuilder("octo_like_diffusion_chunk_fixture")
    builder.add_input(InputPort("observation_history", history_t))
    builder.add_input(
        InputPort(
            "language_tokens",
            language_t,
            required=False,
            default=(0, 0, 0, 0),
        )
    )
    builder.add_input(
        InputPort(
            "goal_image",
            goal_t,
            required=False,
            default=(0.0, 0.0),
        )
    )
    builder.add_output(
        OutputPort("action_chunk", chunk_t, group="robot_action")
    )
    builder.add_region(
        TensorRegion(
            "octo_condition",
            (
                Value("history", history_t),
                Value("language", language_t),
                Value("goal", goal_t),
            ),
            (condition_t,),
            metadata={
                "memoize": True,
                "template": "DiffusionPolicy",
                "optional_modalities": "language_tokens,goal_image",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "octo_initialize",
            (Value("condition", condition_t),),
            (chunk_t,),
        )
    )
    builder.add_region(
        TensorRegion(
            "octo_denoise",
            (
                Value("condition", condition_t),
                Value("sample", chunk_t),
                Value("step", INDEX),
            ),
            (chunk_t,),
        )
    )
    loop = Block.of(
        (
            ops.invoke(
                ("next_sample",),
                (chunk_t,),
                "octo_denoise",
                ("condition", "sample_iter", "denoise_step"),
            ),
            ops.yield_values("next_sample"),
        )
    )
    body = Block.of(
        (
            ops.input_read("history", "history_rev", history_t,
                           "observation_history"),
            ops.input_read("language", "language_rev", language_t,
                           "language_tokens"),
            ops.input_read("goal", "goal_rev", goal_t, "goal_image"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("condition",),
                (condition_t,),
                "octo_condition",
                ("history", "language", "goal"),
            ),
            ops.invoke(
                ("initial_sample",),
                (chunk_t,),
                "octo_initialize",
                ("condition",),
            ),
            ops.for_loop(
                Value("action_chunk", chunk_t),
                "initial_sample",
                Value("denoise_step", INDEX),
                Value("sample_iter", chunk_t),
                loop,
                lower=0,
                upper=3,
            ),
            ops.validate("valid", "action_chunk", "finite_chunk"),
            *_finish(
                (("action_chunk", "action_chunk", chunk_t),),
                group="robot_action",
            ),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "DiffusionPolicy",
                "evidence_level": "L1",
                "source_contract": "Octo-like",
            },
        )
    )

    def condition(history, language, goal):
        return (
            sum(history[-1]),
            sum(language) * 0.01,
            goal[0],
            goal[1],
        )

    def initialize(value):
        return tuple(
            (0.1 * (index + 1), -0.05 * (index + 1))
            for index in range(4)
        )

    def denoise(value, sample, step):
        return tuple(
            (
                action[0] + 0.04 * value[0] - 0.01 * step,
                action[1] + 0.03 * (value[1] + value[3]) + 0.005 * step,
            )
            for action in sample
        )

    required = {
        "observation_history": InputBinding(
            _tensor(((0.0, 0.1, 0.0), (0.2, -0.1, 0.3)), history_t),
            InputStamp(revision=30),
        )
    }
    explicit = {
        "observation_history": InputBinding(
            required["observation_history"].value,
            InputStamp(revision=31),
        ),
        "language_tokens": InputBinding(
            _tensor((1, 3, 2, 4), language_t),
            InputStamp(revision=31),
        ),
        "goal_image": InputBinding(
            _tensor((0.4, -0.2), goal_t),
            InputStamp(revision=31),
        ),
    }
    return AdapterFixture(
        module=builder.build(),
        regions={
            "octo_condition": condition,
            "octo_initialize": initialize,
            "octo_denoise": denoise,
        },
        validators={"finite_chunk": _all_finite},
        initial_state={},
        runs=(
            FixtureRun(required),
            FixtureRun(required),
            FixtureRun(explicit),
        ),
    )


def build_groot_n1_like_fixture() -> AdapterFixture:
    """GR00T N1-like VLM prefix + DiT action expert with schema variants."""

    cameras_t = TensorType((2, 3), "f32")
    state_t = TensorType((4,), "f32")
    language_t = TensorType((4,), "i64")
    prefix_t = TensorType((4,), "f32")
    chunk_t = TensorType((4, 2), "f32")
    logits_t = TensorType((4,), "f32")
    builder = ModuleBuilder("groot_n1_like_multimodal_dit_fixture")
    builder.add_input(InputPort("multi_camera", cameras_t))
    builder.add_input(InputPort("robot_state", state_t))
    builder.add_input(
        InputPort(
            "language_tokens",
            language_t,
            required=False,
            default=(0, 0, 0, 0),
        )
    )
    builder.add_input(InputPort("embodiment_id", I32, value_range=(0, 3)))
    builder.add_output(
        OutputPort("action_chunk", chunk_t, group="robot_action")
    )
    builder.add_output(
        OutputPort("embodiment_logits", logits_t, group="robot_action")
    )
    builder.add_region(
        TensorRegion(
            "groot_vlm_prefix",
            (
                Value("cameras", cameras_t),
                Value("language", language_t),
                Value("embodiment", I32),
            ),
            (prefix_t, logits_t),
            metadata={
                "memoize": True,
                "artifact": "vlm_backbone",
                "backend_extension": "TensorRT/custom",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "groot_initialize_action",
            (Value("state", state_t),),
            (chunk_t,),
        )
    )
    builder.add_region(
        TensorRegion(
            "groot_dit_step",
            (
                Value("prefix", prefix_t),
                Value("state", state_t),
                Value("sample", chunk_t),
                Value("step", INDEX),
            ),
            (chunk_t,),
            metadata={
                "artifact": "action_expert",
                "backend_extension": "TensorRT/custom",
            },
        )
    )
    loop = Block.of(
        (
            ops.invoke(
                ("next_sample",),
                (chunk_t,),
                "groot_dit_step",
                ("prefix", "state", "sample_iter", "dit_step"),
            ),
            ops.yield_values("next_sample"),
        )
    )
    body = Block.of(
        (
            ops.input_read("cameras", "cameras_rev", cameras_t,
                           "multi_camera"),
            ops.input_read("state", "state_rev", state_t, "robot_state"),
            ops.input_read("language", "language_rev", language_t,
                           "language_tokens"),
            ops.input_read("embodiment", "embodiment_rev", I32,
                           "embodiment_id"),
            ops.transaction_begin("txn"),
            ops.invoke(
                ("prefix", "embodiment_logits"),
                (prefix_t, logits_t),
                "groot_vlm_prefix",
                ("cameras", "language", "embodiment"),
            ),
            ops.invoke(
                ("initial_sample",),
                (chunk_t,),
                "groot_initialize_action",
                ("state",),
            ),
            ops.for_loop(
                Value("action_chunk", chunk_t),
                "initial_sample",
                Value("dit_step", INDEX),
                Value("sample_iter", chunk_t),
                loop,
                lower=0,
                upper=4,
            ),
            ops.validate("valid", "action_chunk", "finite_chunk"),
            *_finish(
                (
                    ("action_chunk", "action_chunk", chunk_t),
                    (
                        "embodiment_logits",
                        "embodiment_logits",
                        logits_t,
                    ),
                ),
                group="robot_action",
            ),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "adapter_template": "MultiEmbodimentDiT",
                "evidence_level": "L1",
                "source_contract": "GR00T-N1-like",
            },
        )
    )

    def prefix(cameras, language, embodiment):
        camera = sum(sum(item) for item in cameras)
        language_value = sum(language) * 0.01
        value = (
            camera,
            language_value,
            float(embodiment),
            camera + language_value,
        )
        logits = tuple(
            value[3] - abs(index - embodiment) for index in range(4)
        )
        return value, logits

    def initialize(state):
        return tuple(
            (state[0] + index * 0.05, state[1] - index * 0.03)
            for index in range(4)
        )

    def dit(prefix_value, state, sample, step):
        return tuple(
            (
                action[0] + 0.02 * prefix_value[0] - 0.01 * step,
                action[1] + 0.02 * state[2] + 0.005 * step,
            )
            for action in sample
        )

    inputs = {
        "multi_camera": InputBinding(
            _tensor(((0.1, 0.2, 0.3), (0.2, 0.1, 0.0)), cameras_t),
            InputStamp(revision=40),
        ),
        "robot_state": InputBinding(
            _tensor((0.2, -0.1, 0.3, 0.0), state_t),
            InputStamp(revision=40),
        ),
        "embodiment_id": InputBinding(2, InputStamp(revision=40)),
    }
    return AdapterFixture(
        module=builder.build(),
        regions={
            "groot_vlm_prefix": prefix,
            "groot_initialize_action": initialize,
            "groot_dit_step": dit,
        },
        validators={"finite_chunk": _all_finite},
        initial_state={},
        runs=(FixtureRun(inputs), FixtureRun(inputs)),
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, tuple | list):
        return all(_all_finite(item) for item in value)
    return isinstance(value, int | float) and math.isfinite(value)


ROBOT_MATRIX_FIXTURES = (
    build_rt1_like_fixture,
    build_act_like_fixture,
    build_octo_like_fixture,
    build_groot_n1_like_fixture,
)
