import pytest

from vlaforge.adapters import (
    DRIVING_FIXTURES,
    MODEL_CONTRACTS,
    ROBOT_MATRIX_FIXTURES,
    build_openvla_fixture,
    build_pi0_fixture,
    build_smolvla_fixture,
)
from vlaforge.analysis import verify
from vlaforge.interpreter import Interpreter
from vlaforge.plan import PlanExecutor, lower_to_plan, physicalize_plan


ALL_FIXTURES = (
    build_openvla_fixture,
    build_smolvla_fixture,
    build_pi0_fixture,
    *ROBOT_MATRIX_FIXTURES,
    *DRIVING_FIXTURES,
)


@pytest.mark.model
@pytest.mark.parametrize("factory", ALL_FIXTURES)
def test_model_fixtures_use_only_generic_invocation_ir(factory):
    fixture = factory()
    assert verify(fixture.module, raise_on_error=False) == ()
    assert fixture.module.invocations
    for operation in fixture.module.invocations[0].body.operations:
        lowered = operation.opcode.lower()
        for model_name in (
            "smolvla",
            "openvla",
            "autovla",
            "diffusiondrive",
            "rt1",
            "octo",
            "groot",
            "pi0",
        ):
            assert model_name not in lowered


@pytest.mark.model
@pytest.mark.parametrize("factory", ALL_FIXTURES)
def test_semantic_and_plan_outputs_and_trace_match(factory):
    fixture = factory()
    plan = physicalize_plan(lower_to_plan(fixture.module))
    semantic = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    scheduled = PlanExecutor(
        plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    for item in fixture.runs:
        semantic_result = semantic.run(inputs=item.inputs)
        plan_result = scheduled.run(inputs=item.inputs)
        assert semantic_result.committed_outputs == (
            plan_result.committed_outputs
        )
        assert semantic_result.state == plan_result.state
    assert semantic.trace.to_data() == scheduled.trace.to_data()


def test_smolvla_queue_is_adapter_state_consumed_across_runs():
    fixture = build_smolvla_fixture()
    assert tuple(state.name for state in fixture.module.states) == (
        "action_queue",
        "queue_cursor",
        "rng",
    )
    invocation = fixture.module.invocations[0]
    assert invocation.metadata["adapter_template"] == "ChunkedAction"
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    cursors = []
    actions = []
    for item in fixture.runs:
        result = runtime.run(inputs=item.inputs)
        actions.append(result.committed_outputs.output("action"))
        cursors.append(runtime.state_store.versions("queue_cursor")[-1].value)
    assert cursors == [1, 2, 3, 4, 1, 2]
    assert len(set(actions)) == len(actions)
    refills = [
        event
        for event in runtime.trace.events
        if event.kind == "region"
        and event.data.get("region") == "encode_observation"
    ]
    assert len(refills) == 2


def test_robot_matrix_covers_declared_paradigms_without_core_extensions():
    expected = {
        "RT-1-like",
        "ACT-like",
        "Octo-like",
        "GR00T-N1-like",
    }
    actual = {
        factory().module.invocations[0].metadata["source_contract"]
        for factory in ROBOT_MATRIX_FIXTURES
    }
    assert actual == expected
    for factory in (*ROBOT_MATRIX_FIXTURES, build_pi0_fixture):
        fixture = factory()
        assert fixture.evidence_kind == "deterministic_fixture"
        assert fixture.module.invocations[0].metadata["evidence_level"] == "L1"


def test_upstream_contract_registry_is_pinned_and_does_not_overclaim():
    assert len(MODEL_CONTRACTS) >= 12
    for contract in MODEL_CONTRACTS:
        assert len(contract.revision) == 40
        assert all(
            character in "0123456789abcdef"
            for character in contract.revision
        )
        assert contract.repository.startswith("https://github.com/")
        assert contract.source_entries
        if "fixture-L4" in contract.current_evidence:
            assert "real no-Python C++ parity" in contract.unsupported


@pytest.mark.parametrize(
    "factory",
    DRIVING_FIXTURES,
)
def test_driving_fixtures_cover_named_outputs_without_action_queue(factory):
    fixture = factory()
    assert "action_queue" not in {state.name for state in fixture.module.states}
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
    )
    result = runtime.run(inputs=fixture.runs[0].inputs)
    assert {item.output for item in result.committed_outputs.outputs} == {
        port.name for port in fixture.module.outputs
    }
