"""Immutable representation of a stateful VLA invocation program."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from vlaforge.ir.attrs import Effect, Ownership
from vlaforge.ir.types import IRType, ScalarType, TensorType


SCHEMA_VERSION = "0.2"


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _freeze_literal(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze_literal(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_literal(item) for key, item in value.items()}
        )
    return value


@dataclass(frozen=True, slots=True)
class InputPort:
    name: str
    payload: IRType
    input_id: int | None = None
    required: bool = True
    default: object | None = None
    device: str = "cpu"
    ownership: Ownership = Ownership.EXTERNAL
    alignment: int = 1
    extension: bool = False
    value_range: tuple[int | float, int | float] | None = None
    valid_for: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input name must be non-empty")
        if not isinstance(self.payload, TensorType | ScalarType):
            raise ValueError("input payload must be static Tensor or Scalar/POD")
        if isinstance(self.payload, TensorType) and any(
            dimension is None for dimension in self.payload.shape
        ):
            raise ValueError("deployment input tensor shape must be static")
        if self.input_id is not None and self.input_id < 0:
            raise ValueError("input id must be non-negative")
        if not self.required and self.default is None:
            raise ValueError("optional input requires a static default")
        if self.required and self.default is not None:
            raise ValueError("required input cannot declare a default")
        if not self.device:
            raise ValueError("input device must be non-empty")
        if self.ownership is not Ownership.EXTERNAL:
            raise ValueError("bound input ownership must be external")
        if self.alignment < 1 or self.alignment & (self.alignment - 1):
            raise ValueError("input alignment must be a power of two")
        if self.value_range is not None:
            lower, upper = self.value_range
            if lower > upper:
                raise ValueError("input value_range lower bound exceeds upper")
            if not isinstance(self.payload, ScalarType):
                raise ValueError("input value_range only applies to scalars")
        if self.valid_for is not None and not self.valid_for:
            raise ValueError("valid_for input name must be non-empty")
        object.__setattr__(self, "default", _freeze_literal(self.default))


@dataclass(frozen=True, slots=True)
class OutputPort:
    name: str
    payload: IRType
    output_id: int | None = None
    group: str = "default"
    device: str = "cpu"
    alignment: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("output name must be non-empty")
        if not isinstance(self.payload, TensorType | ScalarType):
            raise ValueError("output payload must be static Tensor or Scalar/POD")
        if isinstance(self.payload, TensorType) and any(
            dimension is None for dimension in self.payload.shape
        ):
            raise ValueError("deployment output tensor shape must be static")
        if self.output_id is not None and self.output_id < 0:
            raise ValueError("output id must be non-negative")
        if not self.group:
            raise ValueError("output group must be non-empty")
        if not self.device:
            raise ValueError("output device must be non-empty")
        if self.alignment < 1 or self.alignment & (self.alignment - 1):
            raise ValueError("output alignment must be a power of two")


@dataclass(frozen=True, slots=True)
class StateSlot:
    name: str
    payload: IRType
    retention: int = 2
    reset_on_episode: bool = True
    ownership: Ownership = Ownership.HOST

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state name must be non-empty")
        if self.retention < 1:
            raise ValueError(f"state {self.name}: retention must be >= 1")
        if self.ownership is Ownership.EXTERNAL:
            raise ValueError("authoritative state cannot have external ownership")


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
        if not self.name:
            raise ValueError("region name must be non-empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def pure(self) -> bool:
        return self.effects == (Effect.PURE,)


@dataclass(frozen=True, slots=True)
class Invocation:
    name: str
    body: Block
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("invocation name must be non-empty")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    inputs: tuple[InputPort, ...]
    outputs: tuple[OutputPort, ...]
    states: tuple[StateSlot, ...]
    regions: tuple[TensorRegion, ...]
    invocations: tuple[Invocation, ...]
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"module schema must be {SCHEMA_VERSION}, got {self.schema_version}"
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        for category, objects in (
            ("input", self.inputs),
            ("output", self.outputs),
            ("state", self.states),
            ("region", self.regions),
            ("invocation", self.invocations),
        ):
            names = [obj.name for obj in objects]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {category} name in module {self.name}")
        input_ids = [port.input_id for port in self.inputs]
        if input_ids != list(range(len(self.inputs))):
            raise ValueError("input ids must be stable contiguous declaration ids")
        output_ids = [port.output_id for port in self.outputs]
        if output_ids != list(range(len(self.outputs))):
            raise ValueError("output ids must be stable contiguous declaration ids")
        input_map = {port.name: port for port in self.inputs}
        for port in self.inputs:
            if port.valid_for is None:
                continue
            target = input_map.get(port.valid_for)
            if target is None or not isinstance(target.payload, TensorType):
                raise ValueError(
                    f"input @{port.name} valid_for target "
                    f"@{port.valid_for} must be a tensor input"
                )
            if port.value_range is None:
                raise ValueError(
                    f"valid-count input @{port.name} requires value_range"
                )
            if target.payload.shape and (
                port.value_range[1] > int(target.payload.shape[0])
            ):
                raise ValueError(
                    f"input @{port.name} upper bound exceeds "
                    f"@{port.valid_for} max profile"
                )

    def input(self, name: str) -> InputPort:
        return _named(self.inputs, name, "input")

    def state(self, name: str) -> StateSlot:
        return _named(self.states, name, "state")

    def output(self, name: str) -> OutputPort:
        return _named(self.outputs, name, "output")

    def region(self, name: str) -> TensorRegion:
        return _named(self.regions, name, "region")

    def invocation(self, name: str) -> Invocation:
        return _named(self.invocations, name, "invocation")


def _named(objects: Iterable[Any], name: str, category: str) -> Any:
    for obj in objects:
        if obj.name == name:
            return obj
    raise KeyError(f"unknown {category}: {name}")
