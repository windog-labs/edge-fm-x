"""Strongly typed values used by Invocation IR v0.2."""

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
            "u64",
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
class InputRevisionType(IRType):
    """Opaque exact identity for one bound logical input."""

    kind: ClassVar[str] = "input_revision"


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
    kind: ClassVar[str] = "pending_state"

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
class PendingOutputType(IRType):
    output: str
    payload: IRType
    kind: ClassVar[str] = "pending_output"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "output": self.output,
            "payload": self.payload.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PendingOutputGroupType(IRType):
    group: str
    outputs: tuple[PendingOutputType, ...]
    kind: ClassVar[str] = "pending_output_group"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "group": self.group,
            "outputs": [output.to_dict() for output in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class CommittedOutputGroupType(IRType):
    group: str
    outputs: tuple[PendingOutputType, ...]
    kind: ClassVar[str] = "committed_output_group"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "group": self.group,
            "outputs": [output.to_dict() for output in self.outputs],
        }


_LEAF_TYPES: Mapping[str, type[IRType]] = {
    InputRevisionType.kind: InputRevisionType,
    TransactionType.kind: TransactionType,
}


def type_from_dict(data: Mapping[str, Any]) -> IRType:
    kind = data.get("kind")
    if kind == ScalarType.kind:
        return ScalarType(str(data["name"]))
    if kind == TensorType.kind:
        return TensorType(
            tuple(None if dim is None else int(dim) for dim in data["shape"]),
            str(data["dtype"]),
            str(data.get("layout", "contiguous")),
        )
    if kind == SnapshotType.kind:
        return SnapshotType(str(data["state"]), type_from_dict(data["payload"]))
    if kind == PendingType.kind:
        return PendingType(str(data["state"]), type_from_dict(data["payload"]))
    if kind == PendingOutputType.kind:
        return PendingOutputType(
            str(data["output"]), type_from_dict(data["payload"])
        )
    if kind == PendingOutputGroupType.kind:
        return PendingOutputGroupType(
            str(data["group"]),
            tuple(
                PendingOutputType(
                    str(item["output"]),
                    type_from_dict(item["payload"]),
                )
                for item in data["outputs"]
            ),
        )
    if kind == CommittedOutputGroupType.kind:
        return CommittedOutputGroupType(
            str(data["group"]),
            tuple(
                PendingOutputType(
                    str(item["output"]),
                    type_from_dict(item["payload"]),
                )
                for item in data["outputs"]
            ),
        )
    if kind in _LEAF_TYPES:
        return _LEAF_TYPES[kind]()
    raise TypeDecodeError(f"unknown IR type kind: {kind!r}")
