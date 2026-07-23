"""Transaction and action runtime values."""

from __future__ import annotations

from dataclasses import dataclass, field

from vlaforge.interpreter.clocks import Epoch


@dataclass(frozen=True, slots=True)
class SnapshotValue:
    state: str
    version: int
    epoch: Epoch
    value: object


@dataclass(frozen=True, slots=True)
class PendingValue:
    state: str
    epoch: Epoch
    value: object


@dataclass(frozen=True, slots=True)
class PendingAction:
    epoch: Epoch
    value: object


@dataclass(frozen=True, slots=True)
class CommittedAction:
    epoch: Epoch
    value: object
    transaction_id: int


@dataclass(slots=True)
class Transaction:
    id: int
    tick: Epoch
    staged: dict[str, PendingValue] = field(default_factory=dict)
    closed: bool = False
    aborted: bool = False

    def stage(self, pending: PendingValue) -> None:
        if self.closed:
            raise RuntimeError(f"transaction {self.id} is already closed")
        if pending.state in self.staged:
            raise RuntimeError(
                f"transaction {self.id} stages state {pending.state} twice"
            )
        self.staged[pending.state] = pending

    def abort(self) -> None:
        if self.closed:
            raise RuntimeError(f"transaction {self.id} is already closed")
        self.closed = True
        self.aborted = True
        self.staged.clear()


@dataclass(frozen=True, slots=True)
class FutureValue:
    value: object
    completed: bool = True


@dataclass(frozen=True, slots=True)
class EventValue:
    completed: bool = True

