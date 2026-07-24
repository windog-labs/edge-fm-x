"""Normalize Python Plan traces to the fixed integer C++ trace contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vlaforge.interpreter.trace import Trace, TraceEvent
from vlaforge.ir.program import Module
from vlaforge.plan.model import PlanModule, Task


TaskSelector = Callable[[TraceEvent, tuple[Task, ...]], Task]


@dataclass(frozen=True, slots=True)
class RuntimeTraceEvent:
    kind: int
    task_id: int
    state_id: int
    logical_version: int
    transaction_id: int
    clock_id: int
    sequence: int
    timestamp_ns: int
    episode: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.kind,
            self.task_id,
            self.state_id,
            self.logical_version,
            self.transaction_id,
            self.clock_id,
            self.sequence,
            self.timestamp_ns,
            self.episode,
        )


def normalize_plan_trace_for_runtime(
    trace: Trace,
    plan: PlanModule,
    module: Module,
    *,
    task_selector: TaskSelector | None = None,
) -> tuple[RuntimeTraceEvent, ...]:
    clock_ids = {
        clock.name: index for index, clock in enumerate(module.clocks)
    }
    state_ids = {
        state.name: index for index, state in enumerate(module.states)
    }
    current_transaction = 0
    result: list[RuntimeTraceEvent] = []
    for event in trace.events:
        task = _task_for_event(event, plan, task_selector=task_selector)
        tick = event.tick
        transaction_id = current_transaction
        if event.kind == "transaction_begin":
            transaction_id = int(event.data["transaction_id"])
            current_transaction = transaction_id
            result.append(
                _event(0, task.id, 0, 0, transaction_id, tick, clock_ids)
            )
        elif event.kind == "input":
            result.append(
                _event(10, task.id, 0, 0, 0, tick, clock_ids)
            )
        elif event.kind == "region":
            result.append(
                _event(
                    11, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
        elif event.kind == "validation":
            result.append(
                _event(
                    12, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
        elif event.kind == "action_pending":
            result.append(
                _event(
                    6, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
        elif event.kind == "state_read":
            result.append(
                _event(
                    1,
                    task.id,
                    state_ids[str(event.data["state"])],
                    int(event.data["version"]),
                    transaction_id,
                    event.data["epoch"],
                    clock_ids,
                )
            )
        elif event.kind == "state_stage":
            result.append(
                _event(
                    2,
                    task.id,
                    state_ids[str(event.data["state"])],
                    0,
                    transaction_id,
                    event.data["epoch"],
                    clock_ids,
                )
            )
        elif event.kind == "transaction_commit":
            for state in event.data["states"]:
                result.append(
                    _event(
                        3,
                        task.id,
                        state_ids[str(state["state"])],
                        int(state["version"]),
                        transaction_id,
                        state["epoch"],
                        clock_ids,
                    )
                )
            result.append(
                _event(
                    4, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
            result.append(
                _event(
                    7, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
        elif event.kind == "transaction_abort":
            result.append(
                _event(
                    5, task.id, 0, 0, transaction_id, tick, clock_ids
                )
            )
        elif event.kind == "action_publish":
            result.append(
                _event(
                    8,
                    task.id,
                    0,
                    0,
                    int(event.data["transaction_id"]),
                    tick,
                    clock_ids,
                )
            )
        elif event.kind == "reset":
            result.append(_event(9, task.id, 0, 0, 0, tick, clock_ids))
    return tuple(result)


def _task_for_event(
    event: TraceEvent,
    plan: PlanModule,
    *,
    task_selector: TaskSelector | None = None,
) -> Task:
    candidates = [task for task in plan.tasks if task.opcode == event.op]
    if event.kind == "input":
        candidates = [
            task
            for task in candidates
            if task.attributes.get("stream") == event.data["stream"]
        ]
    elif event.kind == "region":
        candidates = [
            task
            for task in candidates
            if task.attributes.get("region") == event.data["region"]
        ]
    elif event.kind == "validation":
        candidates = [
            task
            for task in candidates
            if task.attributes.get("contract") == event.data["contract"]
        ]
    elif event.kind in {"state_read", "state_stage"}:
        candidates = [
            task
            for task in candidates
            if task.attributes.get("state") == event.data["state"]
        ]
    if len(candidates) != 1 and task_selector is not None:
        selected = task_selector(event, tuple(candidates))
        if selected not in candidates:
            raise ValueError(
                "runtime trace task selector returned a non-candidate task"
            )
        return selected
    if len(candidates) != 1:
        raise ValueError(
            f"runtime trace task mapping is ambiguous for {event.kind}: "
            f"{[task.id for task in candidates]}"
        )
    return candidates[0]


def _event(
    kind: int,
    task_id: int,
    state_id: int,
    logical_version: int,
    transaction_id: int,
    tick: dict[str, object],
    clock_ids: dict[str, int],
) -> RuntimeTraceEvent:
    return RuntimeTraceEvent(
        kind=kind,
        task_id=task_id,
        state_id=state_id,
        logical_version=logical_version,
        transaction_id=transaction_id,
        clock_id=clock_ids[str(tick["clock"])],
        sequence=int(tick["sequence"]),
        timestamp_ns=int(tick["timestamp_ns"]),
        episode=int(tick["episode"]),
    )
