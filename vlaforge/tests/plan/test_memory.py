from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from vlaforge.adapters import (
    DRIVING_FIXTURES,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.ir.types import ScalarType, TensorType
from vlaforge.plan import (
    ArtifactVariant,
    BufferClass,
    PlanExecutor,
    PlanModule,
    UnsafeStateCapacityError,
    can_reuse_physical_storage,
    emit_memory_constants,
    lower_to_plan,
    physicalize_plan,
    state_arena_sizes,
    verify_plan,
)


def test_state_ring_capacity_is_declared_and_deterministic() -> None:
    module = build_smolvla_fixture().module
    first = physicalize_plan(lower_to_plan(module))
    second = physicalize_plan(lower_to_plan(module))
    assert tuple(state.required_capacity for state in first.states) == (3, 3, 3)
    assert tuple(state.slot_capacity for state in first.states) == (3, 3, 3)
    assert [first.states[0].slot_for(index) for index in range(5)] == [
        0,
        1,
        2,
        0,
        1,
    ]
    assert first.canonical_json() == second.canonical_json()
    assert first.arena == second.arena
    assert state_arena_sizes(first)["cpu"] > 0


def test_unsafe_state_capacity_is_rejected() -> None:
    plan = lower_to_plan(build_smolvla_fixture().module)
    with pytest.raises(UnsafeStateCapacityError, match="required=3"):
        physicalize_plan(
            plan,
            state_capacities={"action_queue": 2},
        )


@pytest.mark.parametrize(
    "module",
    (
        build_smolvla_fixture().module,
        build_openvla_fixture().module,
        *(builder().module for builder in DRIVING_FIXTURES),
    ),
)
def test_every_internal_storage_buffer_has_one_allocation(module) -> None:
    physical = physicalize_plan(lower_to_plan(module))
    assert verify_plan(physical, raise_on_error=False) == ()
    assert physical.arena is not None
    mapped = [
        logical_id
        for item in physical.arena.physical_buffers
        for logical_id in item.logical_buffers
    ]
    assert len(mapped) == len(set(mapped))
    assert all(
        physical.buffers[buffer_id].buffer_class
        not in {BufferClass.EXTERNAL_INPUT, BufferClass.EXTERNAL_OUTPUT}
        for buffer_id in mapped
    )
    assert PlanModule.from_dict(physical.to_dict()) == physical


def test_region_workspace_is_explicit_and_aligned() -> None:
    module = build_openvla_fixture().module
    plan = lower_to_plan(
        module,
        artifact_variants={
            "encode_context": ArtifactVariant(
                "cpu_fixture",
                "optimized",
                workspace_size_bytes=1024,
                workspace_alignment=256,
            )
        },
    )
    task = next(
        item
        for item in plan.tasks
        if item.attributes.get("region") == "encode_context"
    )
    assert task.workspace_buffer is not None
    physical = physicalize_plan(plan)
    assert physical.arena is not None
    allocation = next(
        item
        for item in physical.arena.physical_buffers
        if task.workspace_buffer in item.logical_buffers
    )
    assert allocation.size_bytes == 1024
    assert allocation.alignment == 256
    assert allocation.offset % 256 == 0


def test_scalar_storage_is_region_abi_aligned() -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    assert plan.arena is not None
    scalar_states = [
        state for state in plan.states
        if isinstance(state.payload, ScalarType)
    ]
    scalar_buffers = [
        allocation
        for allocation in plan.arena.physical_buffers
        if any(
            isinstance(plan.buffers[logical_id].type, ScalarType)
            for logical_id in allocation.logical_buffers
        )
    ]

    assert scalar_states
    assert scalar_buffers
    assert all(state.alignment is not None for state in scalar_states)
    assert all(state.alignment >= 16 for state in scalar_states)
    assert all(allocation.alignment >= 16 for allocation in scalar_buffers)


def test_dynamic_internal_storage_requires_override() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    internal = next(
        buffer
        for buffer in plan.buffers
        if not buffer.external
        and buffer.buffer_class
        not in {BufferClass.EXTERNAL_INPUT, BufferClass.EXTERNAL_OUTPUT}
    )
    broken_buffer = replace(internal, type=TensorType((None, 2), "f32"))
    broken = replace(
        plan,
        buffers=tuple(
            broken_buffer if item.id == internal.id else item
            for item in plan.buffers
        ),
    )
    with pytest.raises(ValueError, match="dynamic tensor storage"):
        physicalize_plan(broken)


def test_verifier_rejects_overlapping_live_arena_allocations() -> None:
    plan = physicalize_plan(lower_to_plan(build_openvla_fixture().module))
    assert plan.arena is not None
    left, right, *rest = plan.arena.physical_buffers
    broken_right = replace(
        right,
        offset=left.offset,
        first_task=left.first_task,
        last_task=max(left.last_task, right.last_task),
    )
    diagnostics = verify_plan(
        replace(
            plan,
            arena=replace(
                plan.arena,
                physical_buffers=(left, broken_right, *rest),
            ),
        ),
        raise_on_error=False,
    )
    assert "arena.live_overlap" in {
        item.rule for item in diagnostics
    }


def test_verifier_rejects_overlapping_state_rings() -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    left, right, *rest = plan.states
    diagnostics = verify_plan(
        replace(plan, states=(left, replace(right, offset=left.offset), *rest)),
        raise_on_error=False,
    )
    assert "state.overlap" in {
        item.rule for item in diagnostics
    }


def test_memory_constant_emission_is_deterministic() -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    first = emit_memory_constants(plan)
    assert first == emit_memory_constants(plan)
    assert "inline constexpr StateRingDesc kStateRings[]" in first
    assert "inline constexpr BufferDesc kBuffers[]" in first
    assert f"kArenaSize = {plan.arena.size_bytes}u" in first


@pytest.mark.parametrize(
    "factory",
    (build_smolvla_fixture, build_openvla_fixture),
)
def test_lifetime_packing_reuses_arena_and_preserves_results(factory) -> None:
    fixture = factory()
    lowered = lower_to_plan(fixture.module)
    baseline_plan = physicalize_plan(lowered)
    optimized_plan = physicalize_plan(lowered, reuse_temporaries=True)
    assert baseline_plan.arena is not None
    assert optimized_plan.arena is not None
    assert optimized_plan.arena.size_bytes < baseline_plan.arena.size_bytes
    shared_pairs = [
        (left, right)
        for index, left in enumerate(optimized_plan.arena.physical_buffers)
        for right in optimized_plan.arena.physical_buffers[index + 1 :]
        if left.offset < right.offset + right.size_bytes
        and right.offset < left.offset + left.size_bytes
    ]
    assert shared_pairs
    assert all(
        can_reuse_physical_storage(left, right)
        for left, right in shared_pairs
    )
    baseline = PlanExecutor(
        baseline_plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    optimized = PlanExecutor(
        optimized_plan,
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    for item in fixture.runs:
        left = baseline.run(inputs=item.inputs)
        right = optimized.run(inputs=item.inputs)
        assert left.returns == right.returns
        assert left.state == right.state
    assert baseline.trace.to_json() == optimized.trace.to_json()


def test_lifetime_packing_forbids_overlapping_live_aliases() -> None:
    plan = physicalize_plan(
        lower_to_plan(build_openvla_fixture().module),
        reuse_temporaries=True,
    )
    assert plan.arena is not None
    for index, left in enumerate(plan.arena.physical_buffers):
        for right in plan.arena.physical_buffers[index + 1 :]:
            memory_overlaps = (
                left.offset < right.offset + right.size_bytes
                and right.offset < left.offset + left.size_bytes
            )
            assert not (
                not can_reuse_physical_storage(left, right)
                and memory_overlaps
            )


def test_memory_constants_are_valid_cxx17(tmp_path: Path) -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    header = tmp_path / "memory_constants.h"
    source = tmp_path / "memory_constants.cpp"
    header.write_text(emit_memory_constants(plan), encoding="utf-8")
    source.write_text(
        '#include "memory_constants.h"\n'
        "static_assert(vlaforge_generated::kArenaSize > 0);\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(tmp_path),
            str(source),
        ],
        check=True,
    )
