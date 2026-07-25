"""Bounded state-ring and deterministic static-arena planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from vlaforge.ir.types import (
    CommittedOutputGroupType,
    InputRevisionType,
    IRType,
    PendingOutputGroupType,
    PendingOutputType,
    PendingType,
    ScalarType,
    SnapshotType,
    TensorType,
    TransactionType,
)
from vlaforge.plan.model import (
    BufferClass,
    LogicalBuffer,
    PhysicalBuffer,
    PlanModule,
    StateBinding,
    StaticArenaPlan,
)
from vlaforge.plan.verifier import verify_plan


class MemoryPlanningError(ValueError):
    pass


class UnsafeStateCapacityError(MemoryPlanningError):
    pass


@dataclass(frozen=True, slots=True)
class StorageOverride:
    size_bytes: int | None = None
    alignment: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("storage override size must be non-negative")
        if self.alignment is not None:
            if self.alignment < 1 or self.alignment & (self.alignment - 1):
                raise ValueError(
                    "storage override alignment must be a power of two"
                )
        if self.device is not None and not self.device:
            raise ValueError("storage override device must be non-empty")


def physicalize_plan(
    plan: PlanModule,
    *,
    state_capacities: Mapping[str, int] | None = None,
    buffer_overrides: Mapping[int, StorageOverride] | None = None,
    state_overrides: Mapping[str, StorageOverride] | None = None,
    default_device: str = "cpu",
    tensor_alignment: int = 64,
    reuse_temporaries: bool = False,
) -> PlanModule:
    """Attach proven state rings and one target-device static arena.

    ``reuse_temporaries`` enables deterministic interval packing.  It is an
    explicit optimization switch so baseline plans remain byte-for-byte stable.
    The packed arena is safe to reuse on every invocation because every
    allocation is proven dead before another live allocation may occupy the
    same byte range.
    """

    verify_plan(plan)
    if not default_device:
        raise ValueError("default memory device must be non-empty")
    if tensor_alignment < 1 or tensor_alignment & (tensor_alignment - 1):
        raise ValueError("tensor alignment must be a power of two")
    capacity_map = dict(state_capacities or {})
    buffer_specs = dict(buffer_overrides or {})
    state_specs = dict(state_overrides or {})
    unknown_capacities = sorted(
        set(capacity_map) - {state.name for state in plan.states}
    )
    unknown_state_specs = sorted(
        set(state_specs) - {state.name for state in plan.states}
    )
    unknown_buffers = sorted(
        set(buffer_specs) - {buffer.id for buffer in plan.buffers}
    )
    if unknown_capacities:
        raise KeyError(
            f"state capacities reference unknown states: {unknown_capacities}"
        )
    if unknown_state_specs:
        raise KeyError(
            f"state overrides reference unknown states: {unknown_state_specs}"
        )
    if unknown_buffers:
        raise KeyError(
            f"buffer overrides reference unknown buffers: {unknown_buffers}"
        )

    physical_states = _physicalize_states(
        plan.states,
        capacity_map,
        state_specs,
        default_device=default_device,
        tensor_alignment=tensor_alignment,
    )
    arena = _plan_arena(
        plan,
        buffer_specs,
        default_device=default_device,
        tensor_alignment=tensor_alignment,
        reuse_temporaries=reuse_temporaries,
    )
    result = replace(plan, states=physical_states, arena=arena)
    verify_plan(result)
    return result


def state_arena_sizes(plan: PlanModule) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for state in plan.states:
        if (
            state.device is None
            or state.offset is None
            or state.total_size_bytes is None
        ):
            raise MemoryPlanningError(
                f"state {state.name} has not been physicalized"
            )
        sizes[state.device] = max(
            sizes.get(state.device, 0),
            state.offset + state.total_size_bytes,
        )
    return dict(sorted(sizes.items()))


def storage_size_bytes(type: IRType) -> int:
    if isinstance(type, TensorType):
        element_size = _DTYPE_BYTES.get(type.dtype)
        if element_size is None:
            raise MemoryPlanningError(
                f"unsupported tensor dtype for physical storage: {type.dtype}"
            )
        elements = 1
        for dimension in type.shape:
            if dimension is None:
                raise MemoryPlanningError(
                    "dynamic tensor storage requires an explicit size override"
                )
            elements *= dimension
        return elements * element_size
    if isinstance(type, ScalarType):
        if type.name == "string":
            raise MemoryPlanningError(
                "string values require an external binding"
            )
        return {
            "bool": 1,
            "i32": 4,
            "f16": 2,
            "bf16": 2,
            "f32": 4,
            "index": 8,
            "i64": 8,
            "u64": 8,
            "f64": 8,
            "opaque": 8,
        }[type.name]
    if isinstance(type, InputRevisionType):
        return 8
    if isinstance(type, SnapshotType):
        return 64
    if isinstance(type, PendingType):
        return 56
    if isinstance(type, TransactionType):
        return 32
    if isinstance(type, PendingOutputType | PendingOutputGroupType):
        return 48
    if isinstance(type, CommittedOutputGroupType):
        return 64
    raise MemoryPlanningError(
        f"unsupported IR type for physical storage: {type!r}"
    )


def emit_memory_constants(
    plan: PlanModule,
    *,
    namespace: str = "vlaforge_generated",
) -> str:
    """Emit deterministic constexpr tables consumed by later C++ codegen."""

    verify_plan(plan)
    if plan.arena is None:
        raise MemoryPlanningError("plan has no static arena")
    if not namespace.replace("_", "").isalnum() or namespace[0].isdigit():
        raise ValueError("C++ namespace must be an identifier")
    for state in plan.states:
        if state.total_size_bytes is None:
            raise MemoryPlanningError(
                f"state {state.name} has not been physicalized"
            )

    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        f"namespace {namespace} {{",
        "",
        "struct StateRingDesc {",
        "  std::uint32_t state_id;",
        "  std::uint32_t capacity;",
        "  std::size_t slot_size;",
        "  std::size_t alignment;",
        "  std::size_t offset;",
        "};",
        "",
        "struct BufferDesc {",
        "  std::uint32_t physical_id;",
        "  std::uint32_t logical_id;",
        "  std::uint32_t storage_class;",
        "  std::size_t size;",
        "  std::size_t alignment;",
        "  std::size_t offset;",
        "  std::uint32_t first_task;",
        "  std::uint32_t last_task;",
        "};",
        "",
        f"inline constexpr std::size_t kArenaSize = {plan.arena.size_bytes}u;",
        (
            "inline constexpr std::size_t kArenaAlignment = "
            f"{plan.arena.alignment}u;"
        ),
        "",
    ]
    if plan.states:
        lines.append("inline constexpr StateRingDesc kStateRings[] = {")
        for state in plan.states:
            assert state.slot_capacity is not None
            assert state.slot_size_bytes is not None
            assert state.alignment is not None
            assert state.offset is not None
            lines.append(
                "  {"
                f"{state.state_id}u, {state.slot_capacity}u, "
                f"{state.slot_size_bytes}u, {state.alignment}u, "
                f"{state.offset}u"
                "},"
            )
        lines.extend(["};", ""])
    else:
        lines.extend(
            [
                "inline constexpr const StateRingDesc* kStateRings = nullptr;",
                "",
            ]
        )
    lines.append("inline constexpr BufferDesc kBuffers[] = {")
    class_ids = {item: index for index, item in enumerate(BufferClass)}
    for physical in plan.arena.physical_buffers:
        if len(physical.logical_buffers) != 1:
            raise MemoryPlanningError(
                "initial codegen requires one logical buffer per allocation"
            )
        logical_id = physical.logical_buffers[0]
        lines.append(
            "  {"
            f"{physical.id}u, {logical_id}u, "
            f"{class_ids[physical.buffer_class]}u, "
            f"{physical.size_bytes}u, {physical.alignment}u, "
            f"{physical.offset}u, {physical.first_task}u, "
            f"{physical.last_task}u"
            "},"
        )
    lines.extend(["};", "", f"}}  // namespace {namespace}", ""])
    return "\n".join(lines)


def _physicalize_states(
    states: tuple[StateBinding, ...],
    capacities: Mapping[str, int],
    overrides: Mapping[str, StorageOverride],
    *,
    default_device: str,
    tensor_alignment: int,
) -> tuple[StateBinding, ...]:
    offsets: dict[str, int] = {}
    result = []
    for state in states:
        capacity = capacities.get(state.name, state.required_capacity)
        if capacity < state.required_capacity:
            raise UnsafeStateCapacityError(
                f"state={state.name} capacity={capacity} "
                f"required={state.required_capacity} "
                f"retention={state.retention}"
            )
        override = overrides.get(state.name, StorageOverride())
        payload_size = (
            storage_size_bytes(state.payload)
            if override.size_bytes is None
            else override.size_bytes
        )
        alignment = (
            _natural_alignment(state.payload, tensor_alignment)
            if override.alignment is None
            else override.alignment
        )
        device = override.device or default_device
        slot_size = _align_up(payload_size, alignment)
        offset = _align_up(offsets.get(device, 0), alignment)
        offsets[device] = offset + slot_size * capacity
        result.append(
            replace(
                state,
                slot_capacity=capacity,
                slot_size_bytes=slot_size,
                alignment=alignment,
                offset=offset,
                device=device,
            )
        )
    return tuple(result)


def _plan_arena(
    plan: PlanModule,
    overrides: Mapping[int, StorageOverride],
    *,
    default_device: str,
    tensor_alignment: int,
    reuse_temporaries: bool,
) -> StaticArenaPlan:
    consumers: dict[int, list[int]] = {
        buffer.id: [] for buffer in plan.buffers
    }
    for task in plan.tasks:
        for buffer_id in task.inputs:
            consumers[buffer_id].append(task.id)

    intervals = []
    for buffer in plan.buffers:
        if (
            buffer.external
            or buffer.buffer_class
            in {
                BufferClass.EXTERNAL_INPUT,
                BufferClass.EXTERNAL_OUTPUT,
            }
        ):
            continue
        override = overrides.get(buffer.id, StorageOverride())
        workspace = _workspace_binding(plan, buffer)
        size = (
            (
                workspace.workspace_size_bytes
                if workspace is not None
                else storage_size_bytes(buffer.type)
            )
            if override.size_bytes is None
            else override.size_bytes
        )
        alignment = (
            (
                workspace.workspace_alignment
                if workspace is not None
                else _natural_alignment(buffer.type, tensor_alignment)
            )
            if override.alignment is None
            else override.alignment
        )
        device = (
            override.device
            or (
                None
                if workspace is None
                else workspace.workspace_device
            )
            or default_device
        )
        if device != default_device:
            raise MemoryPlanningError(
                f"buffer {buffer.id} requires device {device}, but the initial "
                f"static arena targets {default_device}"
            )
        producer = buffer.producer_task
        if producer is None:
            raise MemoryPlanningError(
                f"internal buffer {buffer.id} has no producer"
            )
        last = max(consumers[buffer.id], default=producer)
        intervals.append(
            PhysicalBuffer(
                id=len(intervals),
                logical_buffers=(buffer.id,),
                buffer_class=buffer.buffer_class,
                device=device,
                size_bytes=size,
                alignment=alignment,
                offset=0,
                first_task=producer,
                last_task=last,
            )
        )
    if reuse_temporaries:
        physical = _pack_reusable_intervals(intervals)
    else:
        physical = _pack_without_reuse(intervals)
    arena_alignment = max(
        (item.alignment for item in physical),
        default=1,
    )
    arena_size = max(
        (item.offset + item.size_bytes for item in physical),
        default=0,
    )
    return StaticArenaPlan(
        device=default_device,
        size_bytes=_align_up(arena_size, arena_alignment),
        alignment=arena_alignment,
        physical_buffers=tuple(physical),
    )


def can_reuse_physical_storage(
    left: PhysicalBuffer,
    right: PhysicalBuffer,
) -> bool:
    """Return the legality predicate for two logical allocation intervals.

    Reuse is intentionally based on semantic task lifetime, not allocation
    order or tensor names.  Different devices can never alias.  Touching task
    intervals are live together because a producer and consumer may execute in
    the same scheduled task.
    """

    if (
        left.buffer_class is BufferClass.DERIVED_CACHE
        or right.buffer_class is BufferClass.DERIVED_CACHE
    ):
        return False
    return (
        left.device == right.device
        and (
            left.last_task < right.first_task
            or right.last_task < left.first_task
        )
    )


def _pack_without_reuse(
    intervals: list[PhysicalBuffer],
) -> list[PhysicalBuffer]:
    offset = 0
    result = []
    for interval in intervals:
        offset = _align_up(offset, interval.alignment)
        result.append(replace(interval, offset=offset))
        offset += interval.size_bytes
    return result


def _pack_reusable_intervals(
    intervals: list[PhysicalBuffer],
) -> list[PhysicalBuffer]:
    """Deterministic first-fit packing with exact liveness interference."""

    placed: list[PhysicalBuffer] = []
    for interval in sorted(
        intervals,
        key=lambda item: (
            item.first_task,
            -item.size_bytes,
            -item.alignment,
            item.id,
        ),
    ):
        offset = 0
        while True:
            offset = _align_up(offset, interval.alignment)
            conflicts = [
                other
                for other in placed
                if not can_reuse_physical_storage(interval, other)
                and _ranges_overlap(
                    offset,
                    interval.size_bytes,
                    other.offset,
                    other.size_bytes,
                )
            ]
            if not conflicts:
                break
            offset = min(
                other.offset + other.size_bytes
                for other in conflicts
                if other.offset + other.size_bytes > offset
            )
        placed.append(replace(interval, offset=offset))
    return sorted(placed, key=lambda item: item.id)


def _ranges_overlap(
    left_offset: int,
    left_size: int,
    right_offset: int,
    right_size: int,
) -> bool:
    return (
        left_offset < right_offset + right_size
        and right_offset < left_offset + left_size
    )


def _workspace_binding(plan: PlanModule, buffer: LogicalBuffer):
    if buffer.buffer_class is not BufferClass.REGION_WORKSPACE:
        return None
    if buffer.producer_task is None:
        raise MemoryPlanningError("workspace buffer has no producer")
    task = plan.task(buffer.producer_task)
    if task.artifact_id is None:
        raise MemoryPlanningError("workspace task has no artifact")
    return plan.artifacts[task.artifact_id]


def _natural_alignment(type: IRType, tensor_alignment: int) -> int:
    if isinstance(type, TensorType):
        return tensor_alignment
    size = storage_size_bytes(type)
    alignment = 1
    while alignment < min(size, 8):
        alignment *= 2
    return alignment


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


_DTYPE_BYTES = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}
