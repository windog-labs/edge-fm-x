"""Deterministic canonicalization that preserves stable I/O ids."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.ir.program import Block, Module


def _canonical_block(block: Block) -> Block:
    return replace(
        block,
        operations=tuple(
            replace(
                operation,
                attributes={
                    key: operation.attributes[key]
                    for key in sorted(operation.attributes)
                },
                regions=tuple(
                    _canonical_block(region) for region in operation.regions
                ),
            )
            for operation in block.operations
        ),
    )


def canonicalize(module: Module) -> Module:
    """Canonicalize declarations without renumbering external ports."""

    return replace(
        module,
        inputs=tuple(sorted(module.inputs, key=lambda item: item.input_id)),
        outputs=tuple(sorted(module.outputs, key=lambda item: item.output_id)),
        states=tuple(sorted(module.states, key=lambda item: item.name)),
        regions=tuple(sorted(module.regions, key=lambda item: item.name)),
        invocations=tuple(
            replace(invocation, body=_canonical_block(invocation.body))
            for invocation in sorted(
                module.invocations, key=lambda item: item.name
            )
        ),
        metadata={key: module.metadata[key] for key in sorted(module.metadata)},
    )
