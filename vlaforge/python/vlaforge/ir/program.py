"""Immutable in-memory representation of a stateful temporal VLA program."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from vlaforge.ir.attrs import (
    CheckpointPolicy,
    ConsistencyPolicy,
    Effect,
    FreshnessConstraint,
    Ownership,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.types import IRType


SCHEMA_VERSION = "0.1"


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class ClockDomain:
    name: str
    period_ns: int | None = None
    deadline_ns: int | None = None
    jitter_ns: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("clock name must be non-empty")
        if self.period_ns is not None and self.period_ns <= 0:
            raise ValueError("clock period must be positive")
        if self.deadline_ns is not None and self.deadline_ns <= 0:
            raise ValueError("clock deadline must be positive")
        if self.jitter_ns < 0:
            raise ValueError("clock jitter must be non-negative")


@dataclass(frozen=True, slots=True)
class InputStream:
    name: str
    payload: IRType
    clock: str
    freshness: FreshnessConstraint | None = None


@dataclass(frozen=True, slots=True)
class StateSlot:
    name: str
    payload: IRType
    scope: StateScope
    version_clock: str
    retention: int
    consistency: ConsistencyPolicy = ConsistencyPolicy.SNAPSHOT
    initializer: str | None = None
    reset: ResetPolicy = ResetPolicy.EPISODE_START
    authoritative: bool = False
    freshness: FreshnessConstraint | None = None
    ownership: Ownership = Ownership.HOST
    checkpoint: CheckpointPolicy = CheckpointPolicy.ON_COMMIT

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state name must be non-empty")
        if self.retention < 1:
            raise ValueError(f"state {self.name}: retention must be >= 1")


@dataclass(frozen=True, slots=True)
class Value:
    name: str
    type: IRType

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("%"):
            raise ValueError("SSA value names must be non-empty and omit the '%' prefix")


@dataclass(frozen=True, slots=True)
class Block:
    arguments: tuple[Value, ...] = ()
    operations: tuple["Operation", ...] = ()

    @classmethod
    def of(
        cls,
        operations: Iterable["Operation"],
        arguments: Iterable[Value] = (),
    ) -> "Block":
        return cls(tuple(arguments), tuple(operations))


@dataclass(frozen=True, slots=True)
class Operation:
    opcode: str
    results: tuple[Value, ...] = ()
    operands: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    regions: tuple[Block, ...] = ()
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.opcode:
            raise ValueError("operation opcode must be non-empty")
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))

    def with_attributes(self, **updates: Any) -> "Operation":
        attributes = dict(self.attributes)
        attributes.update(updates)
        return replace(self, attributes=attributes)


@dataclass(frozen=True, slots=True)
class TensorRegion:
    name: str
    inputs: tuple[Value, ...]
    outputs: tuple[IRType, ...]
    effects: tuple[Effect, ...] = (Effect.PURE,)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def pure(self) -> bool:
        return self.effects == (Effect.PURE,)


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    clock: str
    body: Block
    inputs: tuple[Value, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    clocks: tuple[ClockDomain, ...]
    inputs: tuple[InputStream, ...]
    states: tuple[StateSlot, ...]
    regions: tuple[TensorRegion, ...]
    policies: tuple[Policy, ...]
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        for category, objects in (
            ("clock", self.clocks),
            ("input", self.inputs),
            ("state", self.states),
            ("region", self.regions),
            ("policy", self.policies),
        ):
            names = [obj.name for obj in objects]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {category} name in module {self.name}")

    def clock(self, name: str) -> ClockDomain:
        return _named(self.clocks, name, "clock")

    def input(self, name: str) -> InputStream:
        return _named(self.inputs, name, "input")

    def state(self, name: str) -> StateSlot:
        return _named(self.states, name, "state")

    def region(self, name: str) -> TensorRegion:
        return _named(self.regions, name, "region")

    def policy(self, name: str) -> Policy:
        return _named(self.policies, name, "policy")


def _named(objects: Iterable[Any], name: str, category: str) -> Any:
    for obj in objects:
        if obj.name == name:
            return obj
    raise KeyError(f"unknown {category}: {name}")

