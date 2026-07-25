import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.interpreter import Interpreter, InterpreterError
from vlaforge.interpreter.trace import normalize_value
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
        runtime.run(inputs=item.inputs) for item in fixture.runs
    ]
    return fixture, runtime, results


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_each_run_returns_one_committed_output_group(factory):
    fixture, runtime, results = execute(factory)
    assert len(results) == len(fixture.runs)
    assert all(result.committed_outputs.outputs for result in results)
    assert len(
        [
            event
            for event in runtime.trace.events
            if event.kind == "transaction_commit"
        ]
    ) == len(fixture.runs)
    assert not any(
        event.kind == "action_publish" for event in runtime.trace.events
    )


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_execution_trace_is_deterministic(factory):
    _, first, _ = execute(factory)
    _, second, _ = execute(factory)
    report = compare_traces(first.trace, second.trace)
    assert report.equal, report.format()


def test_failed_validation_discards_state_and_output():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators={"finite_action": lambda _: False},
        initial_state=fixture.initial_state,
    )
    before = runtime.state_store.inspect()
    with pytest.raises(InterpreterError, match="failed validation"):
        runtime.run(inputs=fixture.runs[0].inputs)
    assert runtime.state_store.inspect() == before
    with pytest.raises(InterpreterError, match="no committed output"):
        runtime.read_output()


def test_smolvla_refills_then_consumes_adapter_owned_queue():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    outputs = []
    region_offsets = [0]
    for item in fixture.runs:
        outputs.append(
            runtime.run(
                inputs=item.inputs
            ).committed_outputs.output("action")
        )
        region_offsets.append(len(runtime.trace.events))
    assert len(set(outputs[:4])) == 4
    assert (
        runtime.state_store.versions("queue_cursor")[-1].value == 2
    )
    refills = [
        event
        for event in runtime.trace.events
        if event.kind == "region"
        and event.data["region"] == "encode_observation"
    ]
    solver_steps = [
        event
        for event in runtime.trace.events
        if event.kind == "region"
        and event.data["region"] == "solver_step"
    ]
    assert len(refills) == 2
    assert len(solver_steps) == 8


def test_openvla_bounded_decode_runs_exact_steps():
    fixture, runtime, _ = execute(build_openvla_fixture)
    token_steps = [
        event
        for event in runtime.trace.events
        if event.kind == "region"
        and event.data["region"] == "next_action_token"
    ]
    assert len(token_steps) == len(fixture.runs) * 3


def test_episode_reset_restores_initial_authoritative_state():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    first = runtime.run(
        inputs=fixture.runs[0].inputs
    ).committed_outputs.output("action")
    runtime.run(inputs=fixture.runs[1].inputs)
    runtime.reset_episode(1)
    reset = runtime.run(
        inputs=fixture.runs[0].inputs
    ).committed_outputs.output("action")
    assert reset == first
    assert runtime.state_store.episode == 1


def test_trace_hashes_tensor_storage_when_numpy_conversion_is_unavailable():
    class StorageOnlyTensor:
        shape = (2, 64)
        dtype = "bf16"

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            raise TypeError("bf16 has no NumPy representation")

        def contiguous(self):
            return self

        def untyped_storage(self):
            return bytearray(range(256))

    normalized = normalize_value(StorageOnlyTensor())
    assert normalized["tensor"] is True
    assert normalized["shape"] == [2, 64]
    assert normalized["dtype"] == "bf16"
    assert len(normalized["sha256"]) == 64
    assert "repr" not in normalized


def test_trace_does_not_copy_large_tensor_activations():
    class LargeTensor:
        shape = (1, 32, 256)
        dtype = "bf16"

        def detach(self):
            raise AssertionError("large trace tensor must not be copied")

    normalized = normalize_value(LargeTensor())
    assert normalized == {
        "tensor": True,
        "shape": [1, 32, 256],
        "dtype": "bf16",
        "content": "omitted_large_tensor",
    }
