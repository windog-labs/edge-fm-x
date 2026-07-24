from __future__ import annotations

from dataclasses import replace

import pytest

from vlaforge.adapters import (
    build_openvla_fixture,
    build_real_openvla_action_program,
    build_real_smolvla_action_program,
    build_smolvla_fixture,
)
from vlaforge.interpreter import Interpreter
from vlaforge.plan import (
    PlanExecutor,
    PlanModule,
    TaskKind,
    lower_to_plan,
    verify_plan,
)
from vlaforge.validation import compare_traces


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_lowering_is_deterministic_and_round_trips(factory) -> None:
    fixture = factory()
    first = lower_to_plan(fixture.module)
    second = lower_to_plan(fixture.module)
    restored = PlanModule.from_dict(first.to_dict())

    assert verify_plan(first, raise_on_error=False) == ()
    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert restored == first
    assert restored.digest() == first.digest()
    assert [task.id for task in first.tasks] == list(range(len(first.tasks)))
    assert [buffer.id for buffer in first.buffers] == list(
        range(len(first.buffers))
    )
    assert all(task.source_location for task in first.tasks)


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_plan_executor_trace_matches_semantic_interpreter(factory) -> None:
    fixture = factory()
    semantic = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    plan = lower_to_plan(fixture.module)
    scheduled = PlanExecutor(
        plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )

    for item in fixture.ticks:
        semantic_result = semantic.run_tick("act", item.tick, item.inputs)
        plan_result = scheduled.run_tick("act", item.tick, item.inputs)
        assert semantic_result.returns == plan_result.returns
        assert semantic_result.state == plan_result.state

    report = compare_traces(semantic.trace, scheduled.trace)
    assert report.equal, report.format()


def test_real_model_semantic_programs_lower_without_model_specific_tasks() -> None:
    modules = (
        build_real_smolvla_action_program(
            chunk_size=50,
            max_action_dim=32,
            output_action_dim=6,
            num_steps=10,
        ),
        build_real_openvla_action_program(action_dim=7),
    )
    for module in modules:
        plan = lower_to_plan(module)
        assert verify_plan(plan, raise_on_error=False) == ()
        assert all(
            "smolvla" not in task.opcode.lower()
            and "openvla" not in task.opcode.lower()
            for task in plan.tasks
        )


def test_verifier_rejects_dependency_cycle() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    first, second = plan.tasks[:2]
    assert first.id in second.dependencies
    tasks = (
        replace(first, dependencies=(second.id,)),
        *plan.tasks[1:],
    )
    broken = replace(plan, tasks=tasks)
    rules = {item.rule for item in verify_plan(broken, raise_on_error=False)}
    assert "dependency.cycle" in rules


def test_verifier_rejects_read_before_produce() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    region = next(
        task
        for task in plan.tasks
        if task.kind is TaskKind.REGION and task.inputs
    )
    broken_task = replace(region, dependencies=())
    tasks = tuple(
        broken_task if item.id == region.id else item for item in plan.tasks
    )
    rules = {
        item.rule
        for item in verify_plan(
            replace(plan, tasks=tasks), raise_on_error=False
        )
    }
    assert "buffer.read_before_produce" in rules


def test_verifier_rejects_missing_artifact() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    region = next(task for task in plan.tasks if task.kind is TaskKind.REGION)
    broken_task = replace(region, artifact_id=999)
    diagnostics = verify_plan(
        replace(
            plan,
            tasks=tuple(
                broken_task if item.id == region.id else item
                for item in plan.tasks
            ),
        ),
        raise_on_error=False,
    )
    assert "artifact.missing" in {item.rule for item in diagnostics}


def test_verifier_rejects_publish_before_commit() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    publish = next(
        task for task in plan.tasks if task.kind is TaskKind.PUBLISH
    )
    pending = next(
        task for task in plan.tasks if task.opcode == "vla.action.create"
    )
    broken_task = replace(publish, inputs=(pending.outputs[0],))
    diagnostics = verify_plan(
        replace(
            plan,
            tasks=tuple(
                broken_task if item.id == publish.id else item
                for item in plan.tasks
            ),
        ),
        raise_on_error=False,
    )
    assert "publish.before_commit" in {item.rule for item in diagnostics}


def test_verifier_rejects_missing_freshness_guard() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    sample = next(
        task for task in plan.tasks if task.opcode == "vla.sample_input"
    )
    broken_task = replace(sample, freshness_guard=None)
    diagnostics = verify_plan(
        replace(
            plan,
            tasks=tuple(
                broken_task if item.id == sample.id else item
                for item in plan.tasks
            ),
        ),
        raise_on_error=False,
    )
    assert "freshness.guard_missing" in {
        item.rule for item in diagnostics
    }


def test_verifier_rejects_invalid_loop_bound() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    loop = next(task for task in plan.tasks if task.kind is TaskKind.LOOP)
    attributes = dict(loop.attributes)
    attributes["upper"] = attributes["lower"]
    broken_task = replace(loop, attributes=attributes)
    diagnostics = verify_plan(
        replace(
            plan,
            tasks=tuple(
                broken_task if item.id == loop.id else item
                for item in plan.tasks
            ),
        ),
        raise_on_error=False,
    )
    assert "loop.invalid_bound" in {item.rule for item in diagnostics}
