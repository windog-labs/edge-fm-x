"""Annotations for explicit, export-auditable TensorRegion boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import TensorRegion, Value
from vlaforge.ir.types import IRType


@dataclass(frozen=True, slots=True)
class RegionSpec:
    name: str
    inputs: tuple[Value, ...]
    outputs: tuple[IRType, ...]
    effects: tuple[Effect, ...]

    def as_ir(self) -> TensorRegion:
        return TensorRegion(self.name, self.inputs, self.outputs, self.effects)


def tensor_region(
    name: str,
    *,
    inputs: Iterable[Value],
    outputs: Iterable[IRType],
    effects: Iterable[Effect] = (Effect.PURE,),
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    spec = RegionSpec(name, tuple(inputs), tuple(outputs), tuple(effects))

    def decorate(function: Callable[..., object]) -> Callable[..., object]:
        setattr(function, "__vlaforge_region__", spec)
        return function

    return decorate

