from vlaforge.adapters import build_transactional_fallback_fixture
from vlaforge.analysis import verify
from vlaforge.compiler import compile_module
from vlaforge.interpreter import Interpreter
from vlaforge.plan import PlanExecutor
from vlaforge.validation import compare_traces


def _run(runtime, fixture):
    return tuple(
        runtime.run(
            inputs=item.inputs
        ).committed_outputs.output("action")
        for item in fixture.runs
    )


def test_transactional_fallback_returns_last_committed_output() -> None:
    fixture = build_transactional_fallback_fixture()
    assert verify(fixture.module, raise_on_error=False) == ()
    runtime = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    outputs = _run(runtime, fixture)
    assert outputs == ((0.2, -0.1), (0.2, -0.1), (0.4, 0.2))
    assert not any(
        event.kind == "action_publish" for event in runtime.trace.events
    )


def test_transactional_fallback_plan_refines_semantic_trace() -> None:
    fixture = build_transactional_fallback_fixture()
    compiled = compile_module(fixture.module)
    semantic = Interpreter(
        compiled.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    scheduled = PlanExecutor(
        compiled.plan,
        compiled.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    assert _run(semantic, fixture) == _run(scheduled, fixture)
    report = compare_traces(semantic.trace, scheduled.trace)
    assert report.equal, report.format()
    opcodes = {task.opcode for task in compiled.plan.tasks}
    assert {
        "vla.validate",
        "vla.if",
        "vla.state.stage_write",
        "vla.output.group",
        "vla.txn.commit",
    } <= opcodes
