import pytest

from vlaforge.adapters import (
    build_openvla_fixture,
    build_real_openvla_action_program,
    build_smolvla_fixture,
)
from vlaforge.analysis import verify
from vlaforge.interpreter import Interpreter


@pytest.mark.model
@pytest.mark.parametrize(
    ("factory", "generation"),
    [
        (build_smolvla_fixture, "iterative_continuous"),
        (build_openvla_fixture, "autoregressive_discrete"),
    ],
)
def test_distinct_model_structures_share_core_ir(factory, generation):
    fixture = factory()
    assert fixture.module.policies[0].metadata["action_generation"] == generation
    assert fixture.evidence_kind == "deterministic_fixture"
    assert verify(fixture.module, raise_on_error=False) == ()
    for operation in fixture.module.policies[0].body.operations:
        lowered = operation.opcode.lower()
        assert "smolvla" not in lowered
        assert "openvla" not in lowered
        assert "pi0" not in lowered


@pytest.mark.model
@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_model_fixture_runs_three_ticks(factory):
    fixture = factory()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    actions = []
    for item in fixture.ticks:
        result = runtime.run_tick("act", item.tick, item.inputs)
        actions.append(result.published_actions[0].value)
    assert len(actions) == 3
    assert len(set(actions)) >= 2


def test_real_openvla_program_is_stateless_and_vla_focused():
    module = build_real_openvla_action_program(action_dim=7)
    assert verify(module, raise_on_error=False) == ()
    assert module.states == ()
    assert tuple(stream.name for stream in module.inputs) == (
        "image",
        "instruction_tokens",
        "instruction_mask",
    )
    assert tuple(
        operation.opcode for operation in module.policies[0].body.operations
    ) == (
        "vla.sample_input",
        "vla.sample_input",
        "vla.sample_input",
        "vla.txn.begin",
        "vla.invoke",
        "vla.invoke",
        "vla.validate",
        "vla.action.create",
        "vla.txn.commit",
        "vla.action.publish",
        "vla.return",
    )
