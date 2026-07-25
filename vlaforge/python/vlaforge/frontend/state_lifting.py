"""Source-evidence contract for authoritative cross-Run state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vlaforge.ir.attrs import Ownership
from vlaforge.ir.program import StateSlot
from vlaforge.ir.types import IRType


@dataclass(frozen=True, slots=True)
class PersistentStateEvidence:
    name: str
    payload: IRType
    source_location: str
    cross_run_reason: str
    retention: int = 2
    reset_on_episode: bool = True
    ownership: Ownership = Ownership.HOST

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.source_location
            or not self.cross_run_reason
        ):
            raise ValueError(
                "persistent state evidence requires name, source location, "
                "and cross-Run reason"
            )
        if self.retention < 1:
            raise ValueError("persistent state retention must be positive")
        if self.ownership is Ownership.EXTERNAL:
            raise ValueError("authoritative state cannot be externally owned")

    def as_ir(self) -> StateSlot:
        return StateSlot(
            self.name,
            self.payload,
            retention=self.retention,
            reset_on_episode=self.reset_on_episode,
            ownership=self.ownership,
        )


def lift_persistent_states(
    evidence: Iterable[PersistentStateEvidence],
) -> tuple[StateSlot, ...]:
    items = tuple(evidence)
    names = [item.name for item in items]
    if len(names) != len(set(names)):
        raise ValueError("persistent state evidence contains duplicate names")
    return tuple(item.as_ir() for item in items)
