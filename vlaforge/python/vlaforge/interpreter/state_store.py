"""Immutable logical state versions backed by a reference in-memory store."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

from vlaforge.interpreter.clocks import Epoch
from vlaforge.interpreter.transaction import PendingValue, SnapshotValue, Transaction
from vlaforge.ir.attrs import ResetPolicy
from vlaforge.ir.program import Module


class StateStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StateVersion:
    state: str
    version: int
    epoch: Epoch
    value: object


class StateStore:
    def __init__(
        self,
        module: Module,
        *,
        initial_values: Mapping[str, object] | None = None,
        initial_epoch: Epoch | None = None,
    ):
        self.module = module
        self._slots = {state.name: state for state in module.states}
        self._versions: dict[str, list[StateVersion]] = {
            state.name: [] for state in module.states
        }
        self._next_transaction = 0
        self._episode = 0 if initial_epoch is None else initial_epoch.episode
        epoch = initial_epoch or Epoch(
            module.clocks[0].name if module.clocks else "init",
            0,
            0,
            self._episode,
        )
        for state_name, value in dict(initial_values or {}).items():
            if state_name not in self._slots:
                raise KeyError(f"initial value references unknown state {state_name}")
            state = self._slots[state_name]
            state_epoch = Epoch(
                state.version_clock,
                epoch.sequence,
                epoch.timestamp_ns,
                epoch.episode,
            )
            self._versions[state_name].append(
                StateVersion(state_name, 0, state_epoch, copy.deepcopy(value))
            )

    @property
    def episode(self) -> int:
        return self._episode

    def begin(self, tick: Epoch) -> Transaction:
        if tick.episode != self._episode:
            raise StateStoreError(
                f"tick episode {tick.episode} does not match state episode "
                f"{self._episode}; reset first"
            )
        transaction = Transaction(self._next_transaction, tick)
        self._next_transaction += 1
        return transaction

    def read(
        self,
        state: str,
        *,
        episode: int,
        max_sequence: int | None = None,
        exact_sequence: int | None = None,
    ) -> SnapshotValue:
        if state not in self._versions:
            raise StateStoreError(f"unknown state {state}")
        candidates = [
            version
            for version in self._versions[state]
            if version.epoch.episode == episode
            and (max_sequence is None or version.epoch.sequence <= max_sequence)
            and (exact_sequence is None or version.epoch.sequence == exact_sequence)
        ]
        if not candidates:
            raise StateStoreError(
                f"no committed version for state={state}, episode={episode}, "
                f"max_sequence={max_sequence}, exact_sequence={exact_sequence}"
            )
        selected = max(candidates, key=lambda item: (item.epoch.sequence, item.version))
        return SnapshotValue(
            selected.state,
            selected.version,
            selected.epoch,
            copy.deepcopy(selected.value),
        )

    def stage(
        self,
        transaction: Transaction,
        state: str,
        epoch: Epoch,
        value: object,
    ) -> PendingValue:
        if state not in self._slots:
            raise StateStoreError(f"unknown state {state}")
        if epoch.episode != self._episode:
            raise StateStoreError(
                f"cannot stage old/new episode state: current={self._episode}, "
                f"target={epoch.episode}"
            )
        pending = PendingValue(state, epoch, copy.deepcopy(value))
        transaction.stage(pending)
        return pending

    def commit(self, transaction: Transaction) -> tuple[StateVersion, ...]:
        if transaction.closed:
            raise StateStoreError(f"transaction {transaction.id} is already closed")
        committed: list[StateVersion] = []
        for state_name, pending in transaction.staged.items():
            history = self._versions[state_name]
            next_version = 0 if not history else history[-1].version + 1
            version = StateVersion(
                state_name,
                next_version,
                pending.epoch,
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

    def reset(self, new_episode: int, states: tuple[str, ...] | None = None) -> None:
        if new_episode <= self._episode:
            raise StateStoreError(
                f"new episode {new_episode} must exceed current {self._episode}"
            )
        selected = set(states or self._slots)
        unknown = selected - set(self._slots)
        if unknown:
            raise StateStoreError(f"reset references unknown states: {sorted(unknown)}")
        for state_name in selected:
            policy = self._slots[state_name].reset
            if states is None or policy in {
                ResetPolicy.EPISODE_START,
                ResetPolicy.EXPLICIT,
                ResetPolicy.ERROR,
            }:
                self._versions[state_name].clear()
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
                    "clock": item.epoch.clock,
                    "sequence": item.epoch.sequence,
                    "timestamp_ns": item.epoch.timestamp_ns,
                    "episode": item.epoch.episode,
                }
                for item in versions
            ]
            for state, versions in self._versions.items()
        }

