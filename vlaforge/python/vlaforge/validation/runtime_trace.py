"""Normalize Python traces to the fixed Invocation Runtime trace ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vlaforge.interpreter.trace import Trace, TraceEvent
from vlaforge.ir.program import Module
from vlaforge.plan.model import PlanModule, Task


TaskSelector = Callable[[TraceEvent, tuple[Task, ...]], Task]


@dataclass(frozen=True, slots=True)
class RuntimeTraceEvent:
    """Language-neutral form of ``vlaforge::runtime::TraceEvent``."""

    kind: int
    task_id: int
    subject_id: int
    logical_version: int
    transaction_id: int
    episode: int
    run: int
    revision: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.kind,
            self.task_id,
            self.subject_id,
            self.logical_version,
            self.transaction_id,
            self.episode,
            self.run,
            self.revision,
        )


_KIND = {
    "transaction_begin": 0,
    "state_read": 1,
    "state_stage": 2,
    "state_commit": 3,
    "transaction_commit": 4,
    "transaction_abort": 5,
    "input": 6,
    "region": 7,
    "cache_hit": 8,
    "cache_miss": 9,
    "validation": 10,
    "output_pending": 11,
    "output_group_pending": 12,
    "output_group_commit": 13,
    "reset": 14,
}


def normalize_plan_trace_for_runtime(
    trace: Trace,
    plan: PlanModule,
    module: Module,
    *,
    task_selector: TaskSelector | None = None,
) -> tuple[RuntimeTraceEvent, ...]:
    """Map semantic/Plan trace events to generated C++ trace events.

    Value payloads are deliberately excluded. The stable trace ABI compares
    control/state identity: task and subject IDs, committed state versions,
    transaction, episode, invocation number, and input revision.
    """

    state_ids = {
        state.name: index for index, state in enumerate(module.states)
    }
    input_ids = {port.name: int(port.input_id) for port in module.inputs}
    region_ids = {
        region.name: index for index, region in enumerate(module.regions)
    }
    validator_ids = {
        name: index
        for index, name in enumerate(
            sorted(
                {
                    str(task.attributes["contract"])
                    for task in plan.tasks
                    if task.opcode == "vla.validate"
                }
            )
        )
    }
    output_group_ids = {
        name: index
        for index, name in enumerate(
            sorted({port.group for port in module.outputs})
        )
    }
    current_episode = 0
    current_transaction = 0
    result: list[RuntimeTraceEvent] = []
    for event in trace.events:
        data = event.data
        if event.kind == "reset":
            current_episode = int(data["episode"])
            result.append(
                _event(
                    "reset",
                    0,
                    0,
                    episode=current_episode,
                    run=event.run,
                )
            )
            continue
        task = _task_for_event(
            event,
            plan,
            task_selector=task_selector,
        )
        episode = int(data.get("episode", current_episode))
        if event.kind == "input":
            result.append(
                _event(
                    "input",
                    task.id,
                    input_ids[str(data["input"])],
                    episode=episode,
                    run=event.run,
                    revision=int(data["revision"]),
                )
            )
        elif event.kind == "transaction_begin":
            current_transaction = int(data["transaction_id"])
            result.append(
                _event(
                    "transaction_begin",
                    task.id,
                    0,
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "state_read":
            result.append(
                _event(
                    "state_read",
                    task.id,
                    state_ids[str(data["state"])],
                    logical_version=int(data["version"]),
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "state_stage":
            result.append(
                _event(
                    "state_stage",
                    task.id,
                    state_ids[str(data["state"])],
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "cache":
            region_name = str(data["region"])
            result.append(
                _event(
                    "cache_hit" if bool(data["hit"]) else "cache_miss",
                    task.id,
                    region_ids[region_name],
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "region":
            result.append(
                _event(
                    "region",
                    task.id,
                    region_ids[str(data["region"])],
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "validation":
            result.append(
                _event(
                    "validation",
                    task.id,
                    validator_ids[str(data["contract"])],
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "output_pending":
            # Generated code aliases the value and emits the aggregate group
            # event; it performs no externally visible action here.
            continue
        elif event.kind == "output_group_pending":
            result.append(
                _event(
                    "output_group_pending",
                    task.id,
                    output_group_ids[str(data["group"])],
                    transaction_id=current_transaction,
                    episode=episode,
                    run=event.run,
                )
            )
        elif event.kind == "transaction_commit":
            transaction_id = int(data["transaction_id"])
            for state in data["states"]:
                result.append(
                    _event(
                        "state_commit",
                        task.id,
                        state_ids[str(state["state"])],
                        logical_version=int(state["version"]),
                        transaction_id=transaction_id,
                        episode=int(state["episode"]),
                        run=event.run,
                    )
                )
            result.append(
                _event(
                    "transaction_commit",
                    task.id,
                    0,
                    transaction_id=transaction_id,
                    episode=episode,
                    run=event.run,
                )
            )
            output = data["output"]
            result.append(
                _event(
                    "output_group_commit",
                    task.id,
                    output_group_ids[str(output["group"])],
                    transaction_id=transaction_id,
                    episode=int(output["episode"]),
                    run=event.run,
                )
            )
        elif event.kind == "transaction_abort":
            result.append(
                _event(
                    "transaction_abort",
                    task.id,
                    0,
                    transaction_id=int(data["transaction_id"]),
                    episode=episode,
                    run=event.run,
                )
            )
        else:
            raise ValueError(
                f"unsupported Invocation trace event kind: {event.kind}"
            )
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
            if task.attributes.get("input") == event.data["input"]
        ]
    elif event.kind in {"cache", "region"}:
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
    elif event.kind == "output_pending":
        candidates = [
            task
            for task in candidates
            if task.attributes.get("output") == event.data["output"]
        ]
    elif event.kind == "output_group_pending":
        candidates = [
            task
            for task in candidates
            if task.attributes.get("group") == event.data["group"]
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
    kind: str,
    task_id: int,
    subject_id: int,
    *,
    logical_version: int = 0,
    transaction_id: int = 0,
    episode: int = 0,
    run: int = 0,
    revision: int = 0,
) -> RuntimeTraceEvent:
    return RuntimeTraceEvent(
        kind=_KIND[kind],
        task_id=task_id,
        subject_id=subject_id,
        logical_version=logical_version,
        transaction_id=transaction_id,
        episode=episode,
        run=run,
        revision=revision,
    )
