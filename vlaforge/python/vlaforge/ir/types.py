"""Strongly typed values used by the executable VLA IR.

The types are immutable and serializable. Persistent state is never represented
as a mutable Python object in the IR: a read yields ``SnapshotType`` and a
transactional write yields ``PendingType``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping


class TypeDecodeError(ValueError):
    """Raised when serialized IR contains an unknown or malformed type."""


@dataclass(frozen=True, slots=True)
class IRType:
    kind: ClassVar[str] = "type"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class ScalarType(IRType):
    name: str
    kind: ClassVar[str] = "scalar"

    def __post_init__(self) -> None:
        if self.name not in {
            "bool",
            "index",
            "i32",
            "i64",
            "f16",
            "bf16",
            "f32",
            "f64",
            "string",
            "opaque",
        }:
            raise ValueError(f"unsupported scalar type: {self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name}


@dataclass(frozen=True, slots=True)
class TensorType(IRType):
    shape: tuple[int | None, ...]
    dtype: str
    layout: str = "contiguous"
    kind: ClassVar[str] = "tensor"

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("tensor dtype must be non-empty")
        if any(dim is not None and dim < 0 for dim in self.shape):
            raise ValueError(f"tensor dimensions must be non-negative: {self.shape}")
        if not self.layout:
            raise ValueError("tensor layout must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class EpochType(IRType):
    clock: str
    kind: ClassVar[str] = "epoch"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "clock": self.clock}


@dataclass(frozen=True, slots=True)
class SnapshotType(IRType):
    state: str
    payload: IRType
    kind: ClassVar[str] = "snapshot"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state,
            "payload": self.payload.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PendingType(IRType):
    state: str
    payload: IRType
    kind: ClassVar[str] = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state,
            "payload": self.payload.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TransactionType(IRType):
    kind: ClassVar[str] = "transaction"


@dataclass(frozen=True, slots=True)
class ActionType(IRType):
    payload: IRType
    kind: ClassVar[str] = "action"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload.to_dict()}


@dataclass(frozen=True, slots=True)
class CommittedActionType(IRType):
    payload: IRType
    kind: ClassVar[str] = "committed_action"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload.to_dict()}


@dataclass(frozen=True, slots=True)
class EventType(IRType):
    kind: ClassVar[str] = "event"


@dataclass(frozen=True, slots=True)
class FutureType(IRType):
    payload: IRType
    kind: ClassVar[str] = "future"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload.to_dict()}


_LEAF_TYPES: Mapping[str, type[IRType]] = {
    TransactionType.kind: TransactionType,
    EventType.kind: EventType,
}


def type_from_dict(data: Mapping[str, Any]) -> IRType:
    """Decode an IR type from its canonical dictionary representation."""

    kind = data.get("kind")
    if kind == ScalarType.kind:
        return ScalarType(str(data["name"]))
    if kind == TensorType.kind:
        return TensorType(
            tuple(None if dim is None else int(dim) for dim in data["shape"]),
            str(data["dtype"]),
            str(data.get("layout", "contiguous")),
        )
    if kind == EpochType.kind:
        return EpochType(str(data["clock"]))
    if kind == SnapshotType.kind:
        return SnapshotType(str(data["state"]), type_from_dict(data["payload"]))
    if kind == PendingType.kind:
        return PendingType(str(data["state"]), type_from_dict(data["payload"]))
    if kind == ActionType.kind:
        return ActionType(type_from_dict(data["payload"]))
    if kind == CommittedActionType.kind:
        return CommittedActionType(type_from_dict(data["payload"]))
    if kind == FutureType.kind:
        return FutureType(type_from_dict(data["payload"]))
    if kind in _LEAF_TYPES:
        return _LEAF_TYPES[kind]()
    raise TypeDecodeError(f"unknown IR type kind: {kind!r}")

