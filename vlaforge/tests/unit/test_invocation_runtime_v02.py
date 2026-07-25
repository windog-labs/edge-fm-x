from dataclasses import replace

import pytest

from vlaforge.adapters import (
    build_driving_diffusion_fixture,
    build_hybrid_external_feature_fixture,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.interpreter import (
    InputBinding,
    InputStamp,
    Interpreter,
    InterpreterError,
    TensorView,
)
from vlaforge.ir.serializer import io_schema_digest


def test_same_revision_hits_and_new_revision_misses_exact_cache():
    fixture = build_openvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    runtime.run(inputs=fixture.runs[0].inputs)
    runtime.run(inputs=fixture.runs[0].inputs)
    runtime.run(inputs=fixture.runs[1].inputs)
    assert (runtime.cache.hits, runtime.cache.misses) == (1, 2)


def test_missing_revision_is_changed_on_every_run():
    fixture = build_openvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    unstamped = {
        name: InputBinding(binding.value)
        for name, binding in fixture.runs[0].inputs.items()
    }
    runtime.run(inputs=unstamped)
    runtime.run(inputs=unstamped)
    assert (runtime.cache.hits, runtime.cache.misses) == (0, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", (9,)),
        ("dtype", "f16"),
        ("layout", "nhwc"),
        ("device", "cuda"),
    ],
)
def test_tensor_binding_contract_errors_are_explicit(field, value):
    fixture = build_openvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    good = fixture.runs[0].inputs["image"]
    assert isinstance(good.value, TensorView)
    bad_view = replace(good.value, **{field: value})
    bad = dict(fixture.runs[0].inputs)
    bad["image"] = InputBinding(bad_view, good.stamp)
    with pytest.raises(InterpreterError, match="contract mismatch"):
        runtime.run(inputs=bad)
    assert runtime._bindings == {}


def test_required_optional_default_and_bounded_valid_count():
    fixture = build_hybrid_external_feature_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    runtime.run(inputs=fixture.runs[0].inputs)
    assert runtime.read_output("trajectory")
    with pytest.raises(InterpreterError, match="required input @external_bev"):
        runtime.run(
            inputs={
                "route_command": fixture.runs[0].inputs["route_command"],
            }
        )

    diffusion = build_driving_diffusion_fixture()
    runtime = Interpreter(
        diffusion.module,
        regions=diffusion.regions,
        validators=diffusion.validators,
    )
    bad = dict(diffusion.runs[0].inputs)
    bad["agent_valid_count"] = InputBinding(
        9,
        InputStamp(revision=30),
    )
    with pytest.raises(InterpreterError, match="outside bounded profile"):
        runtime.run(inputs=bad)


def test_schema_digest_mismatch_is_rejected():
    fixture = build_openvla_fixture()
    actual = io_schema_digest(fixture.module)
    with pytest.raises(InterpreterError, match="schema digest mismatch"):
        Interpreter(
            fixture.module,
            regions=fixture.regions,
            validators=fixture.validators,
            expected_schema_digest="0" * 64,
        )
    assert len(actual) == 64


def test_commit_versions_abort_and_episode_reset():
    fixture = build_smolvla_fixture()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    runtime.run(inputs=fixture.runs[0].inputs)
    assert runtime.state_store.versions("queue_cursor")[-1].version == 1

    failed = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators={"finite_action": lambda _: False},
        initial_state=fixture.initial_state,
    )
    with pytest.raises(InterpreterError, match="failed validation"):
        failed.run(inputs=fixture.runs[0].inputs)
    assert failed.state_store.versions("queue_cursor")[-1].version == 0

    runtime.reset_episode(1)
    versions = runtime.state_store.versions("queue_cursor")
    assert [(item.version, item.episode, item.value) for item in versions] == [
        (0, 1, 4)
    ]
    runtime.run(inputs=fixture.runs[0].inputs)
    assert runtime.state_store.versions("queue_cursor")[-1].version == 1
