"""Analyze bounded-loop invariance without introducing scheduler semantics."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module


@dataclass(frozen=True, slots=True)
class LoopInvariantDecision:
    region: str
    loop: str
    disposition: str
    reason: str


@dataclass(frozen=True, slots=True)
class LoopInvariantAnalysis:
    decisions: tuple[LoopInvariantDecision, ...]


def analyze_structured_loop_invariance(
    module: Module,
) -> LoopInvariantAnalysis:
    """Record whether region work is prehoisted or loop-carried."""

    verify(module)
    decisions: list[LoopInvariantDecision] = []
    for invocation in module.invocations:
        _analyze_block(
            invocation.body,
            module,
            f"invocation:{invocation.name}",
            decisions,
        )
    return LoopInvariantAnalysis(tuple(decisions))


def _analyze_block(
    block: Block,
    module: Module,
    path: str,
    decisions: list[LoopInvariantDecision],
) -> None:
    producers: dict[str, tuple[str, str]] = {}
    for index, operation in enumerate(block.operations):
        operation_path = f"{path}/{index}:{operation.opcode}"
        if operation.opcode == "vla.invoke":
            region = str(operation.attributes["region"])
            for result in operation.results:
                producers[result.name] = (region, operation_path)
        if operation.opcode == "vla.for" and operation.regions:
            body = operation.regions[0]
            external = _external_uses(body)
            for value in sorted(external):
                producer = producers.get(value)
                if producer is None:
                    continue
                region_name, _ = producer
                region = module.region(region_name)
                if bool(region.metadata.get("loop_invariant", False)):
                    decisions.append(
                        LoopInvariantDecision(
                            region_name,
                            operation_path,
                            "prehoisted",
                            "pure region result is produced in the bounded-for "
                            "preheader and reused by loop-carried SSA",
                        )
                    )
            for nested_index, nested in enumerate(body.operations):
                if nested.opcode != "vla.invoke":
                    continue
                region_name = str(nested.attributes["region"])
                decisions.append(
                    LoopInvariantDecision(
                        region_name,
                        operation_path,
                        "loop_carried",
                        "region consumes induction or loop-carried SSA",
                    )
                )
        for region_index, nested in enumerate(operation.regions):
            _analyze_block(
                nested,
                module,
                f"{operation_path}/region:{region_index}",
                decisions,
            )


def _external_uses(block: Block) -> set[str]:
    local = {argument.name for argument in block.arguments}
    local.update(
        result.name
        for operation in block.operations
        for result in operation.results
    )
    used = {
        operand
        for operation in block.operations
        for operand in operation.operands
    }
    return used - local
