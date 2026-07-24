from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from vlaforge.adapters import (
    build_openvla_fixture,
    build_real_openvla_action_program,
    build_smolvla_fixture,
)
from vlaforge.ir.types import TensorType
from vlaforge.plan import (
    ArtifactVariant,
    PlanModule,
    UnsafeStateCapacityError,
    emit_memory_constants,
    lower_to_plan,
    physicalize_plan,
    state_arena_sizes,
    verify_plan,
)


def test_state_ring_capacity_is_proven_and_deterministic() -> None:
    module = build_smolvla_fixture().module
    first = physicalize_plan(
        lower_to_plan(
            module,
            max_in_flight=2,
            consumer_lag=1,
            fallback_snapshots=1,
        )
    )
    second = physicalize_plan(
        lower_to_plan(
            module,
            max_in_flight=2,
            consumer_lag=1,
            fallback_snapshots=1,
        )
    )

    assert all(state.required_capacity == 5 for state in first.states)
    assert all(state.slot_capacity == 5 for state in first.states)
    assert [first.states[0].slot_for(index) for index in range(7)] == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
    ]
    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert first.arena == second.arena
    assert state_arena_sizes(first)["cpu"] > 0


def test_unsafe_state_capacity_is_rejected() -> None:
    plan = lower_to_plan(
        build_smolvla_fixture().module,
        max_in_flight=2,
        consumer_lag=1,
        fallback_snapshots=1,
    )
    with pytest.raises(UnsafeStateCapacityError, match="required=5"):
        physicalize_plan(
            plan,
            state_capacities={"action_queue": 4},
        )


@pytest.mark.parametrize(
    "module",
    (
        build_smolvla_fixture().module,
        build_openvla_fixture().module,
        build_real_openvla_action_program(action_dim=7),
    ),
)
def test_every_internal_buffer_has_one_physical_allocation(module) -> None:
    physical = physicalize_plan(lower_to_plan(module))
    assert verify_plan(physical, raise_on_error=False) == ()
    assert physical.arena is not None
    mapped = [
        logical_id
        for item in physical.arena.physical_buffers
        for logical_id in item.logical_buffers
    ]
    expected = [
        buffer.id
        for buffer in physical.buffers
        if not buffer.external and buffer.buffer_class.value != "external"
    ]
    assert sorted(mapped) == sorted(expected)
    assert len(mapped) == len(set(mapped))
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
    assert allocation.first_task == allocation.last_task == task.id


def test_dynamic_internal_storage_requires_override() -> None:
    plan = lower_to_plan(build_openvla_fixture().module)
    internal = next(
        buffer
        for buffer in plan.buffers
        if not buffer.external and buffer.buffer_class.value != "external"
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
    broken_arena = replace(
        plan.arena,
        physical_buffers=(left, broken_right, *rest),
    )
    diagnostics = verify_plan(
        replace(plan, arena=broken_arena), raise_on_error=False
    )
    assert "arena.overlapping_live_buffers" in {
        item.rule for item in diagnostics
    }


def test_verifier_rejects_overlapping_state_rings() -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    left, right, *rest = plan.states
    broken_right = replace(right, offset=left.offset)
    diagnostics = verify_plan(
        replace(plan, states=(left, broken_right, *rest)),
        raise_on_error=False,
    )
    assert "state.overlapping_rings" in {item.rule for item in diagnostics}


def test_memory_constant_emission_is_deterministic() -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    first = emit_memory_constants(plan)
    second = emit_memory_constants(plan)
    assert first == second
    assert "inline constexpr StateRingDesc kStateRings[]" in first
    assert "inline constexpr BufferDesc kBuffers[]" in first
    assert f"kArenaSize = {plan.arena.size_bytes}u" in first


def test_memory_constants_are_valid_cxx17(tmp_path: Path) -> None:
    plan = physicalize_plan(lower_to_plan(build_smolvla_fixture().module))
    header = tmp_path / "memory_constants.h"
    source = tmp_path / "memory_constants.cpp"
    header.write_text(
        emit_memory_constants(plan),
        encoding="utf-8",
    )
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
