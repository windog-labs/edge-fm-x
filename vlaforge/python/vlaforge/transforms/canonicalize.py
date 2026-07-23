"""Deterministic structural canonicalization."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.ir.program import Block, Module, Operation, Policy


def _canonical_block(block: Block) -> Block:
    operations = tuple(
        replace(
            operation,
            attributes={
                key: operation.attributes[key] for key in sorted(operation.attributes)
            },
            regions=tuple(_canonical_block(region) for region in operation.regions),
        )
        for operation in block.operations
    )
    return replace(block, operations=operations)


def canonicalize(module: Module) -> Module:
    """Return a semantically identical module with stable declaration ordering."""

    return replace(
        module,
        clocks=tuple(sorted(module.clocks, key=lambda item: item.name)),
        inputs=tuple(sorted(module.inputs, key=lambda item: item.name)),
        states=tuple(sorted(module.states, key=lambda item: item.name)),
        regions=tuple(sorted(module.regions, key=lambda item: item.name)),
        policies=tuple(
            replace(policy, body=_canonical_block(policy.body))
            for policy in sorted(module.policies, key=lambda item: item.name)
        ),
        metadata={key: module.metadata[key] for key in sorted(module.metadata)},
    )

