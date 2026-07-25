"""Authoritative state versions allocated only by successful commit."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

from vlaforge.interpreter.transaction import (
    PendingValue,
    SnapshotValue,
    Transaction,
)
from vlaforge.ir.program import Module


class StateStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StateVersion:
    state: str
    version: int
    episode: int
    value: object


class StateStore:
    def __init__(
        self,
        module: Module,
        *,
        initial_values: Mapping[str, object] | None = None,
    ):
        self.module = module
        self._slots = {state.name: state for state in module.states}
        self._versions: dict[str, list[StateVersion]] = {
            state.name: [] for state in module.states
        }
        self._next_transaction = 0
        self._episode = 0
        self._initial_values: dict[str, object] = {}
        for state_name, value in dict(initial_values or {}).items():
            self.initialize(state_name, value)

    @property
    def episode(self) -> int:
        return self._episode

    def initialize(self, state: str, value: object) -> None:
        if state not in self._slots:
            raise KeyError(f"initial value references unknown state {state}")
        if self._versions[state]:
            raise StateStoreError(f"state {state} is already initialized")
        self._versions[state].append(
            StateVersion(state, 0, self._episode, copy.deepcopy(value))
        )
        self._initial_values[state] = copy.deepcopy(value)

    def begin(self) -> Transaction:
        transaction = Transaction(self._next_transaction, self._episode)
        self._next_transaction += 1
        return transaction

    def read_latest(self, state: str) -> SnapshotValue:
        if state not in self._versions:
            raise StateStoreError(f"unknown state {state}")
        candidates = [
            version
            for version in self._versions[state]
            if version.episode == self._episode
        ]
        if not candidates:
            raise StateStoreError(
                f"no committed version for state={state}, "
                f"episode={self._episode}"
            )
        selected = max(candidates, key=lambda item: item.version)
        return SnapshotValue(
            selected.state,
            selected.version,
            selected.episode,
            copy.deepcopy(selected.value),
        )

    def stage(
        self,
        transaction: Transaction,
        state: str,
        value: object,
    ) -> PendingValue:
        if state not in self._slots:
            raise StateStoreError(f"unknown state {state}")
        if transaction.episode != self._episode:
            raise StateStoreError("transaction belongs to another episode")
        pending = PendingValue(state, copy.deepcopy(value))
        transaction.stage(pending)
        return pending

    def commit(self, transaction: Transaction) -> tuple[StateVersion, ...]:
        if transaction.closed:
            raise StateStoreError(
                f"transaction {transaction.id} is already closed"
            )
        if transaction.episode != self._episode:
            raise StateStoreError("transaction belongs to another episode")
        committed: list[StateVersion] = []
        for state_name, pending in transaction.staged.items():
            history = self._versions[state_name]
            next_version = 0 if not history else history[-1].version + 1
            version = StateVersion(
                state_name,
                next_version,
                self._episode,
                copy.deepcopy(pending.value),
            )
            history.append(version)
            retention = self._slots[state_name].retention
            if len(history) > retention:
                del history[: len(history) - retention]
            committed.append(version)
        transaction.closed = True
        return tuple(committed)

    def abort(self, transaction: Transaction) -> None:
        transaction.abort()

    def reset(self, new_episode: int) -> None:
        if new_episode <= self._episode:
            raise StateStoreError(
                f"new episode {new_episode} must exceed current {self._episode}"
            )
        previous_episode = self._episode
        for state_name, slot in self._slots.items():
            if slot.reset_on_episode:
                self._versions[state_name].clear()
                if state_name in self._initial_values:
                    self._versions[state_name].append(
                        StateVersion(
                            state_name,
                            0,
                            new_episode,
                            copy.deepcopy(self._initial_values[state_name]),
                        )
                    )
            else:
                previous = [
                    version
                    for version in self._versions[state_name]
                    if version.episode == previous_episode
                ]
                if previous:
                    latest = max(previous, key=lambda item: item.version)
                    self._versions[state_name].append(
                        StateVersion(
                            state_name,
                            latest.version,
                            new_episode,
                            copy.deepcopy(latest.value),
                        )
                    )
        self._episode = new_episode

    def versions(self, state: str) -> tuple[StateVersion, ...]:
        if state not in self._versions:
            raise KeyError(state)
        return tuple(self._versions[state])

    def inspect(self) -> dict[str, list[dict[str, object]]]:
        return {
            state: [
                {
                    "version": item.version,
                    "episode": item.episode,
                }
                for item in versions
            ]
            for state, versions in self._versions.items()
        }
