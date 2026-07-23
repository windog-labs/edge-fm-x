from dataclasses import replace

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.interpreter import Epoch, InputSample, Interpreter, InterpreterError
from vlaforge.interpreter.state_store import StateStoreError
from vlaforge.ir import ops
from vlaforge.ir.attrs import (
    ConsistencyPolicy,
    FreshnessConstraint,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.program import Block, ClockDomain, Policy, StateSlot, Value
from vlaforge.ir.types import EpochType, ScalarType
from vlaforge.validation import compare_traces


def execute(factory):
    fixture = factory()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    results = [
        runtime.run_tick("act", item.tick, item.inputs) for item in fixture.ticks
    ]
    return fixture, runtime, results


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_multi_tick_execution_commits_and_publishes_once(factory):
    fixture, runtime, results = execute(factory)
    assert all(len(result.published_actions) == 1 for result in results)
    assert len(
        [event for event in runtime.trace.events if event.kind == "transaction_commit"]
    ) == len(fixture.ticks)
    assert len(
        [event for event in runtime.trace.events if event.kind == "action_publish"]
    ) == len(fixture.ticks)


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_execution_trace_is_deterministic(factory):
    _, first, _ = execute(factory)
    _, second, _ = execute(factory)
    report = compare_traces(first.trace, second.trace)
    assert report.equal, report.format()


def test_stale_input_is_rejected_with_context():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    tick = Epoch("control", 4, 200_000_000, 0)
    inputs = {
        "image": InputSample(
            (0.0, 0.0),
            Epoch("observation", 0, 0, 0),
        )
    }
    with pytest.raises(InterpreterError, match="freshness.stale_input"):
        runtime.run_tick("act", tick, inputs)


def test_stale_state_version_is_rejected_with_context():
    fixture = build_smolvla_fixture()
    module = replace(
        fixture.module,
        states=(
            replace(
                fixture.module.states[0],
                freshness=FreshnessConstraint(max_versions=1),
            ),
        )
        + fixture.module.states[1:],
    )
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    item = fixture.ticks[0]
    tick = replace(item.tick, sequence=4, timestamp_ns=80_000_000)
    inputs = {
        name: replace(
            sample,
            epoch=replace(sample.epoch, sequence=4, timestamp_ns=80_000_000),
        )
        for name, sample in item.inputs.items()
    }
    with pytest.raises(InterpreterError, match="freshness.stale_state"):
        runtime.run_tick("act", tick, inputs)


def test_old_episode_state_is_not_read_after_reset():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    runtime.reset_episode(1)
    item = fixture.ticks[0]
    new_inputs = {
        name: replace(
            sample,
            epoch=replace(sample.epoch, episode=1),
        )
        for name, sample in item.inputs.items()
    }
    with pytest.raises(StateStoreError, match="no committed version"):
        runtime.run_tick(
            "act",
            replace(item.tick, episode=1),
            new_inputs,
        )


def test_failed_validation_discards_staged_state_and_action():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators={"finite_action": lambda _: False},
        initial_state=fixture.initial_state,
    )
    before = runtime.state_store.inspect()
    item = fixture.ticks[0]
    with pytest.raises(InterpreterError, match="failed validation"):
        runtime.run_tick("act", item.tick, item.inputs)
    assert runtime.state_store.inspect() == before
    assert not any(event.kind == "action_publish" for event in runtime.trace.events)


def test_smolvla_if_refills_then_reuses_queue():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    runtime.run_tick("act", fixture.ticks[0].tick, fixture.ticks[0].inputs)
    first_regions = [
        event.data["region"]
        for event in runtime.trace.events
        if event.kind == "region"
    ]
    runtime.run_tick("act", fixture.ticks[1].tick, fixture.ticks[1].inputs)
    second_regions = [
        event.data["region"]
        for event in runtime.trace.events
        if event.kind == "region"
    ][len(first_regions) :]

    assert first_regions.count("encode_observation") == 1
    assert first_regions.count("solver_step") == 4
    assert "encode_observation" not in second_regions
    assert "solver_step" not in second_regions
    assert "queue_select" in second_regions


def test_openvla_for_runs_exact_bounded_token_steps():
    _, runtime, _ = execute(build_openvla_fixture)
    token_steps = [
        event
        for event in runtime.trace.events
        if event.kind == "region"
        and event.data["region"] == "next_action_token"
    ]
    assert len(token_steps) == 3 * 3


def test_explicit_reset_and_abort_clear_state_without_commit():
    builder = ModuleBuilder("explicit_reset_policy")
    builder.add_clock(ClockDomain("control", period_ns=20_000_000))
    builder.add_state(
        StateSlot(
            "history",
            ScalarType("i64"),
            StateScope.EPISODE,
            "control",
            retention=1,
            consistency=ConsistencyPolicy.SNAPSHOT,
            reset=ResetPolicy.EXPLICIT,
        )
    )
    builder.add_policy(
        Policy(
            "reset",
            "control",
            Block.of(
                (
                    ops.transaction_begin("txn", "tick"),
                    ops.reset("history"),
                    ops.transaction_abort("txn", reason="episode_boundary"),
                    ops.return_values(),
                )
            ),
            inputs=(Value("tick", EpochType("control")),),
        )
    )
    runtime = Interpreter(
        builder.build(),
        regions={},
        validators={},
        initial_state={"history": 42},
    )

    result = runtime.run_tick("reset", Epoch("control", 0, 0, 0), {})

    assert result.returns == ()
    assert runtime.state_store.episode == 1
    assert result.state["history"] == []
    assert [event.kind for event in runtime.trace.events] == [
        "transaction_begin",
        "reset",
        "transaction_abort",
    ]
