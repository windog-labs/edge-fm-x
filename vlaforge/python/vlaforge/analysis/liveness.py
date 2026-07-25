"""Conservative SSA live-range analysis for deterministic planning."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.ir.program import Block, Module


@dataclass(frozen=True, slots=True)
class LiveRange:
    value: str
    first_definition: int
    last_use: int


def analyze_liveness(
    module: Module, invocation_name: str
) -> tuple[LiveRange, ...]:
    invocation = module.invocation(invocation_name)
    positions: dict[str, int] = {}
    last_use: dict[str, int] = {}
    counter = 0

    def visit(block: Block) -> None:
        nonlocal counter
        for argument in block.arguments:
            positions.setdefault(argument.name, counter)
            last_use.setdefault(argument.name, counter)
        for operation in block.operations:
            current = counter
            counter += 1
            for operand in operation.operands:
                last_use[operand] = current
            for result in operation.results:
                positions[result.name] = current
                last_use[result.name] = current
            for region in operation.regions:
                visit(region)

    visit(invocation.body)
    return tuple(
        LiveRange(name, start, last_use.get(name, start))
        for name, start in sorted(positions.items())
    )
