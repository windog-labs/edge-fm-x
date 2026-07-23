"""IR fault mutations used to test verifier detection."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.ir.attrs import EpochExpr
from vlaforge.ir.program import Block, Module, Operation


def _rewrite_first(
    block: Block,
    predicate,
    rewrite,
) -> tuple[Block, bool]:
    operations: list[Operation] = []
    changed = False
    for operation in block.operations:
        current = operation
        if not changed and predicate(operation):
            current = rewrite(operation)
            changed = True
        elif not changed and operation.regions:
            regions = []
            for region in operation.regions:
                rewritten, region_changed = _rewrite_first(region, predicate, rewrite)
                regions.append(rewritten)
                changed = changed or region_changed
            current = replace(operation, regions=tuple(regions))
        operations.append(current)
    return replace(block, operations=tuple(operations)), changed


def mutate_first_policy(module: Module, predicate, rewrite) -> Module:
    policy = module.policies[0]
    body, changed = _rewrite_first(policy.body, predicate, rewrite)
    if not changed:
        raise ValueError("mutation target not found")
    return replace(
        module,
        policies=(replace(policy, body=body),) + module.policies[1:],
    )


def wrong_epoch(module: Module) -> Module:
    clocks = [clock.name for clock in module.clocks]
    return mutate_first_policy(
        module,
        lambda operation: operation.opcode == "vla.state.read",
        lambda operation: operation.with_attributes(
            epoch=EpochExpr.current(
                next(
                    clock
                    for clock in clocks
                    if clock != module.state(str(operation.attributes["state"])).version_clock
                )
            ).to_dict()
        ),
    )


def delete_dependency(module: Module) -> Module:
    return mutate_first_policy(
        module,
        lambda operation: operation.opcode == "vla.invoke" and operation.operands,
        lambda operation: replace(operation, operands=operation.operands[:-1]),
    )


def publish_before_commit(module: Module) -> Module:
    policy = module.policies[0]
    action_name = next(
        operation.results[0].name
        for operation in policy.body.operations
        if operation.opcode == "vla.action.create"
    )
    return mutate_first_policy(
        module,
        lambda operation: operation.opcode == "vla.action.publish",
        lambda operation: replace(operation, operands=(action_name,)),
    )


def duplicate_stage_write(module: Module) -> Module:
    policy = module.policies[0]
    operations = list(policy.body.operations)
    index = next(
        index
        for index, operation in enumerate(operations)
        if operation.opcode == "vla.state.stage_write"
    )
    duplicate = replace(
        operations[index],
        results=tuple(
            replace(result, name=f"{result.name}_duplicate")
            for result in operations[index].results
        ),
    )
    operations.insert(index + 1, duplicate)
    return replace(
        module,
        policies=(
            replace(policy, body=replace(policy.body, operations=tuple(operations))),
        )
        + module.policies[1:],
    )

