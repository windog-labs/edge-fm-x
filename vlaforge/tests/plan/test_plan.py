from dataclasses import replace

from vlaforge.adapters import DRIVING_FIXTURES, build_openvla_fixture
from vlaforge.plan import (
    PlanModule,
    TaskKind,
    lower_to_plan,
    verify_plan,
)


def test_plan_v2_is_deterministic_and_round_trips():
    fixtures = (build_openvla_fixture, *DRIVING_FIXTURES)
    for factory in fixtures:
        fixture = factory()
        first = lower_to_plan(fixture.module)
        second = lower_to_plan(fixture.module)
        restored = PlanModule.from_dict(first.to_dict())
        assert verify_plan(first, raise_on_error=False) == ()
        assert first.canonical_json() == second.canonical_json()
        assert restored == first
        assert first.io_schema_digest == restored.io_schema_digest


def test_plan_has_no_clock_or_scheduler_contract():
    plan = lower_to_plan(build_openvla_fixture().module)
    data = plan.to_dict()
    text = plan.canonical_json().lower()
    assert "clock" not in text
    assert "deadline" not in text
    assert "period" not in text
    assert "policy" not in data
    assert data["invocations"][0]["name"] == "act"


def test_plan_verifier_rejects_dependency_cycle():
    plan = lower_to_plan(build_openvla_fixture().module)
    first, second = plan.tasks[:2]
    assert first.id in second.dependencies
    broken = replace(
        plan,
        tasks=(
            replace(first, dependencies=(second.id,)),
            *plan.tasks[1:],
        ),
    )
    rules = {item.rule for item in verify_plan(broken, raise_on_error=False)}
    assert "dependency.cycle" in rules


def test_plan_verifier_rejects_missing_artifact():
    plan = lower_to_plan(build_openvla_fixture().module)
    region = next(task for task in plan.tasks if task.kind is TaskKind.REGION)
    broken_task = replace(region, artifact_id=999)
    broken = replace(
        plan,
        tasks=tuple(
            broken_task if item.id == region.id else item
            for item in plan.tasks
        ),
    )
    rules = {item.rule for item in verify_plan(broken, raise_on_error=False)}
    assert "artifact.missing" in rules


def test_plan_verifier_rejects_invalid_loop_bound():
    plan = lower_to_plan(build_openvla_fixture().module)
    loop = next(task for task in plan.tasks if task.kind is TaskKind.LOOP)
    attributes = dict(loop.attributes)
    attributes["upper"] = attributes["lower"]
    broken_task = replace(loop, attributes=attributes)
    broken = replace(
        plan,
        tasks=tuple(
            broken_task if item.id == loop.id else item
            for item in plan.tasks
        ),
    )
    rules = {item.rule for item in verify_plan(broken, raise_on_error=False)}
    assert "loop.invalid_bound" in rules
