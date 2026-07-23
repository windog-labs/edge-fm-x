from vlaforge.adapters import build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.interpreter import Interpreter
from vlaforge.transforms import (
    canonicalize,
    physicalize_state,
    synthesize_epoch_memoization,
)
from vlaforge.validation import compare_traces


def walk(block):
    for operation in block.operations:
        yield operation
        for region in operation.regions:
            yield from walk(region)


def run(fixture, module):
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    for item in fixture.ticks:
        runtime.run_tick("act", item.tick, item.inputs)
    return runtime.trace


def test_epoch_memoization_uses_observation_epoch():
    fixture = build_smolvla_fixture()
    transformed = synthesize_epoch_memoization(fixture.module)
    encode = next(
        operation
        for operation in walk(transformed.policies[0].body)
        if operation.opcode == "vla.invoke"
        and operation.attributes["region"] == "encode_observation"
    )
    assert encode.attributes["memoize_semantics"] == "epoch_state_signature"
    assert encode.attributes["memoize_key"] == ["observation_epoch"]
    assert verify(transformed, raise_on_error=False) == ()


def test_transforms_preserve_reference_trace():
    fixture = build_smolvla_fixture()
    transformed = physicalize_state(
        synthesize_epoch_memoization(canonicalize(fixture.module)),
        max_in_flight=2,
        consumer_lag=1,
    )
    report = compare_traces(
        run(fixture, fixture.module),
        run(fixture, transformed),
    )
    assert report.equal, report.format()
    plan = transformed.metadata["physical_state_plan"]
    assert plan["rng"]["capacity"] >= 4
