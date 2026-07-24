"""Evidence-gated lifting of source-retained values into StateSlot declarations."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.ir.attrs import (
    CheckpointPolicy,
    ConsistencyPolicy,
    FreshnessConstraint,
    Ownership,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.program import StateSlot
from vlaforge.ir.types import IRType


@dataclass(frozen=True, slots=True)
class PersistentStateEvidence:
    name: str
    payload: IRType
    source_location: str
    cross_tick_reason: str
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
        if not self.name or not self.source_location or not self.cross_tick_reason:
            raise ValueError(
                "state lifting requires name, source location, and cross-tick reason"
            )
        if self.retention < 1:
            raise ValueError("state retention must be positive")

    def lift(self) -> StateSlot:
        return StateSlot(
            name=self.name,
            payload=self.payload,
            scope=self.scope,
            version_clock=self.version_clock,
            retention=self.retention,
            consistency=self.consistency,
            initializer=self.initializer,
            reset=self.reset,
            authoritative=self.authoritative,
            freshness=self.freshness,
            ownership=self.ownership,
            checkpoint=self.checkpoint,
        )


def lift_persistent_states(
    evidence: tuple[PersistentStateEvidence, ...],
) -> tuple[StateSlot, ...]:
    names = [item.name for item in evidence]
    if len(names) != len(set(names)):
        raise ValueError("state lifting contains duplicate state names")
    return tuple(item.lift() for item in evidence)
