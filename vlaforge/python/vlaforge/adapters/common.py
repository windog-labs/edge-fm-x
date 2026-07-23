"""Shared adapter test contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from vlaforge.interpreter.clocks import Epoch, InputSample
from vlaforge.ir.program import Module


@dataclass(frozen=True, slots=True)
class FixtureTick:
    tick: Epoch
    inputs: Mapping[str, InputSample]


@dataclass(frozen=True, slots=True)
class AdapterFixture:
    module: Module
    regions: Mapping[str, Callable[..., object]]
    validators: Mapping[str, Callable[[object], bool]]
    initial_state: Mapping[str, object]
    ticks: tuple[FixtureTick, ...]
    evidence_kind: str = "deterministic_fixture"

