"""Transaction, versioned snapshot, and committed output values."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SnapshotValue:
    state: str
    version: int
    episode: int
    value: object


@dataclass(frozen=True, slots=True)
class PendingValue:
    state: str
    value: object


@dataclass(frozen=True, slots=True)
class PendingOutput:
    output: str
    value: object


@dataclass(frozen=True, slots=True)
class PendingOutputGroup:
    group: str
    outputs: tuple[PendingOutput, ...]

    def __post_init__(self) -> None:
        names = [item.output for item in self.outputs]
        if not self.group or not names or len(names) != len(set(names)):
            raise ValueError("output group requires unique named outputs")


@dataclass(frozen=True, slots=True)
class CommittedOutputGroup:
    group: str
    outputs: tuple[PendingOutput, ...]
    transaction_id: int
    episode: int

    def output(self, name: str) -> object:
        for item in self.outputs:
            if item.output == name:
                return item.value
        raise KeyError(f"output @{name} is not in committed group @{self.group}")


@dataclass(slots=True)
class Transaction:
    id: int
    episode: int
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
