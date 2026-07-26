from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_INPUT_TYPES,
    MINDDRIVE_OUTPUT_TYPES,
    MINDDRIVE_STATE_TYPES,
    MINDDRIVE_UPSTREAM_REVISION,
    build_real_minddrive_program,
)
from vlaforge.analysis import verify
from vlaforge.compiler import compile_module
from vlaforge.interpreter import (
    InputBinding,
    InputStamp,
    Interpreter,
    InterpreterError,
    TensorView,
)
from vlaforge.plan import PlanExecutor
from vlaforge.validation import normalize_plan_trace_for_runtime


@dataclass
class _Calls:
    vision: int = 0
    position: int = 0
    map: int = 0
    detection: int = 0
    decision: int = 0
    action: int = 0
    trajectory: int = 0
    decode: int = 0


def _implementations(calls: _Calls) -> dict[str, Any]:
    def vision_encoder(camera: object) -> object:
        calls.vision += 1
        return ("vision", camera)

    def position_encoder(*_arguments: object) -> object:
        calls.position += 1
        return "position"

    def map_encoder(*arguments: object) -> tuple[object, ...]:
        calls.map += 1
        value = int(arguments[-6]) + 1
        return (value, value, value, value, *(value for _ in range(6)))

    def detection_encoder(*arguments: object) -> tuple[object, ...]:
        calls.detection += 1
        value = int(arguments[-10]) + 1
        return (
            value,
            value,
            value,
            value,
            value,
            value,
            *(value for _ in range(10)),
        )

    def decision_expert(
        _prompt: object,
        detection_tokens: object,
        _map_tokens: object,
    ) -> object:
        calls.decision += 1
        return detection_tokens

    def action_expert(
        _prompt: object,
        detection_tokens: object,
        _map_tokens: object,
    ) -> object:
        calls.action += 1
        return detection_tokens

    def trajectory_decoder(
        action_hidden: object,
        *_arguments: object,
    ) -> tuple[object, ...]:
        calls.trajectory += 1
        value = int(action_hidden) * 100
        return value, value + 1, value + 8, value + 9

    def detection_decoder(
        detection_classes: object,
        *_arguments: object,
    ) -> tuple[object, ...]:
        calls.decode += 1
        value = int(detection_classes) * 100
        return tuple(value + index for index in range(2, 8))

    return {
        "vision_encoder": vision_encoder,
        "position_encoder": position_encoder,
        "map_encoder": map_encoder,
        "detection_encoder": detection_encoder,
        "decision_expert": decision_expert,
        "action_expert": action_expert,
        "trajectory_decoder": trajectory_decoder,
        "detection_decoder": detection_decoder,
    }


def _initial_state() -> dict[str, object]:
    return {name: 0 for name, _ in MINDDRIVE_STATE_TYPES}


def _bindings(
    *,
    camera_revision: int | None,
    other_revision: int | None,
) -> dict[str, InputBinding]:
    result = {}
    for index, (name, payload) in enumerate(MINDDRIVE_INPUT_TYPES):
        revision = (
            camera_revision if name == "camera_images" else other_revision
        )
        result[name] = InputBinding(
            TensorView(
                (name, index),
                tuple(int(item) for item in payload.shape),
                payload.dtype,
                payload.layout,
                "cuda:0",
                64,
            ),
            InputStamp(revision=revision),
        )
    return result


def _runtime(
    *,
    valid: bool = True,
) -> tuple[Interpreter, _Calls]:
    module = build_real_minddrive_program()
    calls = _Calls()
    return (
        Interpreter(
            module,
            regions=_implementations(calls),
            validators={
                "minddrive_output_contract": lambda _value: valid
            },
            initial_state=_initial_state(),
        ),
        calls,
    )


def test_real_minddrive_program_is_small_generic_and_stateful() -> None:
    module = build_real_minddrive_program()
    assert verify(module, raise_on_error=False) == ()
    assert module.name == "minddrive_0_5b_real"
    assert tuple(port.name for port in module.inputs) == tuple(
        name for name, _ in MINDDRIVE_INPUT_TYPES
    )
    assert tuple(port.name for port in module.outputs) == tuple(
        name for name, _ in MINDDRIVE_OUTPUT_TYPES
    )
    assert tuple(state.name for state in module.states) == (
        *(name for name, _ in MINDDRIVE_STATE_TYPES),
    )
    assert tuple(region.name for region in module.regions) == (
        "vision_encoder",
        "position_encoder",
        "map_encoder",
        "detection_encoder",
        "decision_expert",
        "action_expert",
        "trajectory_decoder",
        "detection_decoder",
    )
    assert module.invocations[0].metadata["source_revision"] == (
        MINDDRIVE_UPSTREAM_REVISION
    )
    assert module.invocations[0].metadata["core_op_delta"] == 0
    assert tuple(
        value.name
        for value in module.region("trajectory_decoder").inputs
    ) == (
        "action_hidden",
        "decision_logits",
        "ego_route_command",
        "trajectory_noise",
        "path_noise",
    )
    assert module.region("trajectory_decoder").metadata["rng_semantics"] == (
        "explicit_tensor_inputs"
    )

    compilation = compile_module(
        module,
        default_device="cuda:0",
        state_device="cuda:0",
    )
    assert len(compilation.certificate.caches) == 1
    cache = compilation.certificate.caches[0]
    assert cache.region == "vision_encoder"
    assert cache.input_ids == (0,)
    assert cache.state_ids == ()
    assert cache.enabled
    assert compilation.plan.arena is not None
    assert compilation.plan.arena.device == "cuda:0"


def test_revision_cache_and_stateful_branch_follow_declared_identity() -> None:
    runtime, calls = _runtime()
    first = runtime.run(
        "run",
        _bindings(camera_revision=10, other_revision=20),
    )
    second = runtime.run(
        "run",
        _bindings(camera_revision=10, other_revision=21),
    )
    third = runtime.run(
        "run",
        _bindings(camera_revision=11, other_revision=21),
    )

    assert first.committed_outputs.output("trajectory") == 100
    assert second.committed_outputs.output("trajectory") == 200
    assert third.committed_outputs.output("trajectory") == 300
    assert calls == _Calls(
        vision=2,
        position=3,
        map=3,
        detection=3,
        decision=3,
        action=3,
        trajectory=3,
        decode=3,
    )
    assert (runtime.cache.hits, runtime.cache.misses) == (1, 2)
    assert (
        runtime.state_store.versions("detection_memory_embedding")[-1].version
        == 3
    )
    cache_events = [
        event.data
        for event in runtime.trace.events
        if event.kind == "cache"
    ]
    assert [event["input_revisions"] for event in cache_events] == [
        [["camera_images", 10]],
        [["camera_images", 10]],
        [["camera_images", 11]],
    ]
    assert all(event["state_snapshots"] == [] for event in cache_events)


def test_missing_camera_revision_is_safe_miss() -> None:
    runtime, calls = _runtime()
    for _ in range(2):
        runtime.run(
            "run",
            _bindings(camera_revision=None, other_revision=30),
        )
    assert calls.vision == 2
    assert (runtime.cache.hits, runtime.cache.misses) == (0, 2)


def test_validation_abort_preserves_state_and_previous_outputs() -> None:
    runtime, _ = _runtime()
    runtime.run(
        "run",
        _bindings(camera_revision=10, other_revision=20),
    )
    before = {
        name: runtime.state_store.versions(name)[-1].version
        for name, _ in MINDDRIVE_STATE_TYPES
    }
    previous = runtime.read_output("trajectory")
    runtime.validators["minddrive_output_contract"] = lambda _value: False
    with pytest.raises(InterpreterError, match="failed validation"):
        runtime.run(
            "run",
            _bindings(camera_revision=11, other_revision=21),
        )
    assert runtime.read_output("trajectory") == previous
    assert {
        name: runtime.state_store.versions(name)[-1].version
        for name, _ in MINDDRIVE_STATE_TYPES
    } == before


def test_reset_episode_and_plan_executor_match_semantic_runtime() -> None:
    module = build_real_minddrive_program()
    compilation = compile_module(
        module,
        default_device="cuda:0",
        state_device="cuda:0",
    )
    module = compilation.module
    semantic_calls = _Calls()
    plan_calls = _Calls()
    semantic = Interpreter(
        module,
        regions=_implementations(semantic_calls),
        validators={"minddrive_output_contract": lambda _value: True},
        initial_state=_initial_state(),
    )
    plan = PlanExecutor(
        compilation.plan,
        module,
        regions=_implementations(plan_calls),
        validators={"minddrive_output_contract": lambda _value: True},
        initial_state=_initial_state(),
    )
    runs = (
        _bindings(camera_revision=10, other_revision=20),
        _bindings(camera_revision=10, other_revision=21),
        _bindings(camera_revision=11, other_revision=22),
    )
    semantic_outputs = [
        semantic.run("run", inputs).committed_outputs.output("trajectory")
        for inputs in runs
    ]
    plan_outputs = [
        plan.run("run", inputs).committed_outputs.output("trajectory")
        for inputs in runs
    ]
    assert semantic_outputs == plan_outputs == [100, 200, 300]
    assert (semantic.cache.hits, semantic.cache.misses) == (1, 2)
    assert (plan.cache.hits, plan.cache.misses) == (1, 2)
    assert tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            semantic.trace,
            compilation.plan,
            module,
        )
    ) == tuple(
        event.as_tuple()
        for event in normalize_plan_trace_for_runtime(
            plan.trace,
            compilation.plan,
            module,
        )
    )

    semantic.reset_episode(1)
    reset = semantic.run(
        "run",
        _bindings(camera_revision=10, other_revision=20),
    )
    assert reset.committed_outputs.output("trajectory") == 100
    assert semantic_calls.map == 4
    assert semantic_calls.detection == 4
    assert semantic.state_store.episode == 1
    assert (
        semantic.state_store.versions("detection_memory_embedding")[-1].version
        == 1
    )
