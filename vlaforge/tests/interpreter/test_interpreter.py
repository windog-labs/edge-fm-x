from dataclasses import replace

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.interpreter import Epoch, InputSample, Interpreter, InterpreterError
from vlaforge.interpreter.state_store import StateStoreError
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


def test_old_episode_state_is_not_read_after_reset():
    fixture = build_openvla_fixture()
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

