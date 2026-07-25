"""Compact Scheduled Execution Plan for one passive invocation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from vlaforge.ir.types import IRType, type_from_dict


PLAN_SCHEMA = "vlaforge.scheduled_plan/2"


class TaskKind(str, Enum):
    INPUT = "input"
    REGION = "region"
    LOOP = "loop"
    BRANCH = "branch"
    STATE = "state"
    VALIDATION = "validation"
    COMMIT = "commit"
    OUTPUT = "output"
    CONTROL = "control"


class BufferClass(str, Enum):
    EXTERNAL_INPUT = "external_input"
    EXTERNAL_OUTPUT = "external_output"
    SSA = "ssa"
    CONTROL = "control"
    LOOP_CARRIED = "loop_carried"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_PENDING = "state_pending"
    PENDING_OUTPUT = "pending_output"
    COMMITTED_OUTPUT = "committed_output"
    REGION_WORKSPACE = "region_workspace"
    DERIVED_CACHE = "derived_cache"


@dataclass(frozen=True, slots=True)
class LogicalBuffer:
    id: int
    name: str
    type: IRType
    buffer_class: BufferClass
    producer_task: int | None
    external: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        if self.id < 0 or not self.name:
            raise ValueError("logical buffer requires non-negative id and name")
        if self.external != (self.producer_task is None):
            raise ValueError(
                "external logical buffers must have no producer and internal "
                "logical buffers must have one producer"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.to_dict(),
            "buffer_class": self.buffer_class.value,
            "producer_task": self.producer_task,
            "external": self.external,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class StateBinding:
    state_id: int
    name: str
    payload: IRType
    retention: int
    slot_capacity: int | None = None
    slot_size_bytes: int | None = None
    alignment: int | None = None
    offset: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.state_id < 0 or not self.name or self.retention < 1:
            raise ValueError("invalid state binding")
        if self.slot_capacity is not None and self.slot_capacity < self.retention:
            raise ValueError("state slot capacity must preserve retention")
        physical = (
            self.slot_size_bytes,
            self.alignment,
            self.offset,
            self.device,
        )
        if any(item is not None for item in physical) and not all(
            item is not None for item in physical
        ):
            raise ValueError("state physical fields must be set together")
        if self.slot_size_bytes is not None:
            assert self.alignment is not None
            assert self.offset is not None
            assert self.device is not None
            if self.slot_capacity is None:
                raise ValueError("physicalized state requires slot capacity")
            if self.slot_size_bytes < 0 or self.alignment < 1 or self.offset < 0:
                raise ValueError("invalid state physical layout")
            if self.alignment & (self.alignment - 1):
                raise ValueError("state alignment must be a power of two")
            if self.offset % self.alignment:
                raise ValueError("state offset violates alignment")

    @property
    def required_capacity(self) -> int:
        return self.retention

    def slot_for(self, logical_version: int) -> int:
        if self.slot_capacity is None:
            raise ValueError(f"state {self.name} has not been physicalized")
        if logical_version < 0:
            raise ValueError("logical version must be non-negative")
        return logical_version % self.slot_capacity

    @property
    def total_size_bytes(self) -> int | None:
        if self.slot_capacity is None or self.slot_size_bytes is None:
            return None
        return self.slot_capacity * self.slot_size_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "name": self.name,
            "payload": self.payload.to_dict(),
            "retention": self.retention,
            "slot_capacity": self.slot_capacity,
            "slot_size_bytes": self.slot_size_bytes,
            "alignment": self.alignment,
            "offset": self.offset,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    artifact_id: int
    region_name: str
    backend: str
    variant: str
    artifact_path: str | None = None
    workspace_size_bytes: int = 0
    workspace_alignment: int = 1
    workspace_device: str = "cpu"
    plugin_abi: str = "vlaforge.region_executable/2"

    def __post_init__(self) -> None:
        if (
            self.artifact_id < 0
            or not self.region_name
            or not self.backend
            or not self.variant
            or self.workspace_size_bytes < 0
            or self.workspace_alignment < 1
            or not self.workspace_device
            or not self.plugin_abi
        ):
            raise ValueError("invalid artifact binding")
        if self.workspace_alignment & (self.workspace_alignment - 1):
            raise ValueError("artifact workspace alignment must be a power of two")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "region_name": self.region_name,
            "backend": self.backend,
            "variant": self.variant,
            "artifact_path": self.artifact_path,
            "workspace_size_bytes": self.workspace_size_bytes,
            "workspace_alignment": self.workspace_alignment,
            "workspace_device": self.workspace_device,
            "plugin_abi": self.plugin_abi,
        }


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    kind: TaskKind
    opcode: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    dependencies: tuple[int, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    blocks: tuple[int, ...] = ()
    artifact_id: int | None = None
    workspace_buffer: int | None = None
    source_op: str | None = None
    source_location: str | None = None

    def __post_init__(self) -> None:
        if self.id < 0 or not self.opcode:
            raise ValueError("task requires non-negative id and opcode")
        if tuple(sorted(set(self.dependencies))) != self.dependencies:
            raise ValueError("task dependencies must be sorted and unique")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(
                {
                    str(key): value
                    for key, value in sorted(self.attributes.items())
                }
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "opcode": self.opcode,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "dependencies": list(self.dependencies),
            "attributes": _plain(self.attributes),
            "blocks": list(self.blocks),
            "artifact_id": self.artifact_id,
            "workspace_buffer": self.workspace_buffer,
            "source_op": self.source_op,
            "source_location": self.source_location,
        }


@dataclass(frozen=True, slots=True)
class PlanBlock:
    id: int
    arguments: tuple[int, ...]
    tasks: tuple[int, ...]
    source: str

    def __post_init__(self) -> None:
        if self.id < 0 or not self.source:
            raise ValueError("plan block requires non-negative id and source")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "arguments": list(self.arguments),
            "tasks": list(self.tasks),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PlanInvocation:
    id: int
    name: str
    body_block: int

    def __post_init__(self) -> None:
        if self.id < 0 or not self.name or self.body_block < 0:
            raise ValueError("invalid plan invocation")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "body_block": self.body_block,
        }


@dataclass(frozen=True, slots=True)
class PhysicalBuffer:
    id: int
    logical_buffers: tuple[int, ...]
    buffer_class: BufferClass
    device: str
    size_bytes: int
    alignment: int
    offset: int
    first_task: int
    last_task: int

    def __post_init__(self) -> None:
        if (
            self.id < 0
            or not self.logical_buffers
            or not self.device
            or self.size_bytes < 0
            or self.alignment < 1
            or self.offset < 0
            or self.first_task < 0
            or self.last_task < self.first_task
        ):
            raise ValueError("invalid physical buffer")
        if self.alignment & (self.alignment - 1):
            raise ValueError("physical buffer alignment must be a power of two")
        if self.offset % self.alignment:
            raise ValueError("physical buffer offset violates alignment")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_buffers": list(self.logical_buffers),
            "buffer_class": self.buffer_class.value,
            "device": self.device,
            "size_bytes": self.size_bytes,
            "alignment": self.alignment,
            "offset": self.offset,
            "first_task": self.first_task,
            "last_task": self.last_task,
        }


@dataclass(frozen=True, slots=True)
class StaticArenaPlan:
    device: str
    size_bytes: int
    alignment: int
    physical_buffers: tuple[PhysicalBuffer, ...]

    def __post_init__(self) -> None:
        if not self.device or self.size_bytes < 0 or self.alignment < 1:
            raise ValueError("invalid static arena")
        if self.alignment & (self.alignment - 1):
            raise ValueError("static arena alignment must be a power of two")

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "size_bytes": self.size_bytes,
            "alignment": self.alignment,
            "physical_buffers": [
                item.to_dict() for item in self.physical_buffers
            ],
        }


@dataclass(frozen=True, slots=True)
class PlanModule:
    name: str
    semantic_digest: str
    io_schema_digest: str
    invocations: tuple[PlanInvocation, ...]
    tasks: tuple[Task, ...]
    blocks: tuple[PlanBlock, ...]
    buffers: tuple[LogicalBuffer, ...]
    states: tuple[StateBinding, ...]
    artifacts: tuple[ArtifactBinding, ...]
    arena: StaticArenaPlan | None = None
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise ValueError(f"unsupported scheduled plan schema: {self.schema!r}")
        if (
            not self.name
            or len(self.semantic_digest) != 64
            or len(self.io_schema_digest) != 64
        ):
            raise ValueError("plan requires semantic and I/O SHA-256 digests")

    def invocation(self, name: str) -> PlanInvocation:
        for invocation in self.invocations:
            if invocation.name == name:
                return invocation
        raise KeyError(f"unknown plan invocation: {name}")

    def task(self, task_id: int) -> Task:
        if 0 <= task_id < len(self.tasks) and self.tasks[task_id].id == task_id:
            return self.tasks[task_id]
        raise KeyError(f"unknown task id: {task_id}")

    def block(self, block_id: int) -> PlanBlock:
        if 0 <= block_id < len(self.blocks) and self.blocks[block_id].id == block_id:
            return self.blocks[block_id]
        raise KeyError(f"unknown block id: {block_id}")

    def buffer(self, buffer_id: int) -> LogicalBuffer:
        if (
            0 <= buffer_id < len(self.buffers)
            and self.buffers[buffer_id].id == buffer_id
        ):
            return self.buffers[buffer_id]
        raise KeyError(f"unknown buffer id: {buffer_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "name": self.name,
            "semantic_digest": self.semantic_digest,
            "io_schema_digest": self.io_schema_digest,
            "invocations": [item.to_dict() for item in self.invocations],
            "tasks": [item.to_dict() for item in self.tasks],
            "blocks": [item.to_dict() for item in self.blocks],
            "buffers": [item.to_dict() for item in self.buffers],
            "states": [item.to_dict() for item in self.states],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "arena": None if self.arena is None else self.arena.to_dict(),
        }

    def canonical_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            indent=indent,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanModule":
        arena_data = data.get("arena")
        arena = (
            None
            if arena_data is None
            else StaticArenaPlan(
                device=str(arena_data["device"]),
                size_bytes=int(arena_data["size_bytes"]),
                alignment=int(arena_data["alignment"]),
                physical_buffers=tuple(
                    PhysicalBuffer(
                        id=int(item["id"]),
                        logical_buffers=tuple(
                            int(value)
                            for value in item["logical_buffers"]
                        ),
                        buffer_class=BufferClass(item["buffer_class"]),
                        device=str(item["device"]),
                        size_bytes=int(item["size_bytes"]),
                        alignment=int(item["alignment"]),
                        offset=int(item["offset"]),
                        first_task=int(item["first_task"]),
                        last_task=int(item["last_task"]),
                    )
                    for item in arena_data.get("physical_buffers", ())
                ),
            )
        )
        return cls(
            schema=str(data["schema"]),
            name=str(data["name"]),
            semantic_digest=str(data["semantic_digest"]),
            io_schema_digest=str(data["io_schema_digest"]),
            invocations=tuple(
                PlanInvocation(
                    id=int(item["id"]),
                    name=str(item["name"]),
                    body_block=int(item["body_block"]),
                )
                for item in data.get("invocations", ())
            ),
            tasks=tuple(
                Task(
                    id=int(item["id"]),
                    kind=TaskKind(item["kind"]),
                    opcode=str(item["opcode"]),
                    inputs=tuple(
                        int(value) for value in item.get("inputs", ())
                    ),
                    outputs=tuple(
                        int(value) for value in item.get("outputs", ())
                    ),
                    dependencies=tuple(
                        int(value)
                        for value in item.get("dependencies", ())
                    ),
                    attributes=dict(item.get("attributes", {})),
                    blocks=tuple(
                        int(value) for value in item.get("blocks", ())
                    ),
                    artifact_id=(
                        None
                        if item.get("artifact_id") is None
                        else int(item["artifact_id"])
                    ),
                    workspace_buffer=(
                        None
                        if item.get("workspace_buffer") is None
                        else int(item["workspace_buffer"])
                    ),
                    source_op=(
                        None
                        if item.get("source_op") is None
                        else str(item["source_op"])
                    ),
                    source_location=(
                        None
                        if item.get("source_location") is None
                        else str(item["source_location"])
                    ),
                )
                for item in data.get("tasks", ())
            ),
            blocks=tuple(
                PlanBlock(
                    id=int(item["id"]),
                    arguments=tuple(
                        int(value) for value in item.get("arguments", ())
                    ),
                    tasks=tuple(
                        int(value) for value in item.get("tasks", ())
                    ),
                    source=str(item["source"]),
                )
                for item in data.get("blocks", ())
            ),
            buffers=tuple(
                LogicalBuffer(
                    id=int(item["id"]),
                    name=str(item["name"]),
                    type=type_from_dict(item["type"]),
                    buffer_class=BufferClass(item["buffer_class"]),
                    producer_task=(
                        None
                        if item.get("producer_task") is None
                        else int(item["producer_task"])
                    ),
                    external=bool(item.get("external", False)),
                    source=(
                        None
                        if item.get("source") is None
                        else str(item["source"])
                    ),
                )
                for item in data.get("buffers", ())
            ),
            states=tuple(
                StateBinding(
                    state_id=int(item["state_id"]),
                    name=str(item["name"]),
                    payload=type_from_dict(item["payload"]),
                    retention=int(item["retention"]),
                    slot_capacity=(
                        None
                        if item.get("slot_capacity") is None
                        else int(item["slot_capacity"])
                    ),
                    slot_size_bytes=(
                        None
                        if item.get("slot_size_bytes") is None
                        else int(item["slot_size_bytes"])
                    ),
                    alignment=(
                        None
                        if item.get("alignment") is None
                        else int(item["alignment"])
                    ),
                    offset=(
                        None
                        if item.get("offset") is None
                        else int(item["offset"])
                    ),
                    device=(
                        None
                        if item.get("device") is None
                        else str(item["device"])
                    ),
                )
                for item in data.get("states", ())
            ),
            artifacts=tuple(
                ArtifactBinding(
                    artifact_id=int(item["artifact_id"]),
                    region_name=str(item["region_name"]),
                    backend=str(item["backend"]),
                    variant=str(item["variant"]),
                    artifact_path=(
                        None
                        if item.get("artifact_path") is None
                        else str(item["artifact_path"])
                    ),
                    workspace_size_bytes=int(
                        item.get("workspace_size_bytes", 0)
                    ),
                    workspace_alignment=int(
                        item.get("workspace_alignment", 1)
                    ),
                    workspace_device=str(
                        item.get("workspace_device", "cpu")
                    ),
                    plugin_abi=str(
                        item.get(
                            "plugin_abi",
                            "vlaforge.region_executable/2",
                        )
                    ),
                )
                for item in data.get("artifacts", ())
            ),
            arena=arena,
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        f"plan attribute is not JSON serializable: {type(value).__name__}"
    )
