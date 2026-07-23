"""SSA and persistent-state dependency graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vlaforge.ir.program import Block, Module, Operation


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    value_producers: dict[str, str]
    value_consumers: dict[str, tuple[str, ...]]
    state_readers: dict[str, tuple[str, ...]]
    state_writers: dict[str, tuple[str, ...]]


def _walk(block: Block, prefix: str) -> Iterable[tuple[str, Operation]]:
    for index, operation in enumerate(block.operations):
        node = f"{prefix}/{index}:{operation.opcode}"
        yield node, operation
        for region_index, region in enumerate(operation.regions):
            yield from _walk(region, f"{node}/r{region_index}")


def build_dependency_graph(module: Module) -> DependencyGraph:
    producers: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    writers: dict[str, list[str]] = {}

    for policy in module.policies:
        for value in policy.inputs:
            producers[value.name] = f"policy:{policy.name}:arg"
        for node, operation in _walk(policy.body, f"policy:{policy.name}"):
            for result in operation.results:
                producers[result.name] = node
            for operand in operation.operands:
                consumers.setdefault(operand, []).append(node)
            if operation.opcode == "vla.state.read":
                readers.setdefault(str(operation.attributes["state"]), []).append(node)
            elif operation.opcode == "vla.state.stage_write":
                writers.setdefault(str(operation.attributes["state"]), []).append(node)

    return DependencyGraph(
        value_producers=producers,
        value_consumers={key: tuple(value) for key, value in consumers.items()},
        state_readers={key: tuple(value) for key, value in readers.items()},
        state_writers={key: tuple(value) for key, value in writers.items()},
    )

