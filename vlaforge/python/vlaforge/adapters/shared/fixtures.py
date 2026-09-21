"""Shared adapter test contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from vlaforge.interpreter.inputs import InputBinding
from vlaforge.ir.program import Module


@dataclass(frozen=True, slots=True)
class FixtureRun:
    inputs: Mapping[str, InputBinding]


@dataclass(frozen=True, slots=True)
class AdapterFixture:
    module: Module
    regions: Mapping[str, Callable[..., object]]
    validators: Mapping[str, Callable[[object], bool]]
    initial_state: Mapping[str, object]
    runs: tuple[FixtureRun, ...]
    evidence_kind: str = "deterministic_fixture"
