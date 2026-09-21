"""Explicit deployment policy for existing semantic bounded loops."""

from dataclasses import replace

from vlaforge.ir.program import Block, Module
from vlaforge.ir.serializer import io_schema_digest

LOOP_EXECUTIONS = ("source", "off", "batch-only", "prefer", "required")


def configure_loop_execution(module: Module, policy: str) -> tuple[Module, int]:
    """Clone only bounded-loop policy attributes, retaining all value semantics."""
    if policy not in LOOP_EXECUTIONS:
        raise ValueError(f"unsupported loop execution policy: {policy}")
    if policy == "source":
        return module, 0
    selected = 0

    def visit(block: Block) -> Block:
        nonlocal selected
        operations = []
        for operation in block.operations:
            attributes = dict(operation.attributes)
            if operation.opcode == "vla.for":
                selected += 1
                if policy == "off":
                    attributes.pop("replay", None)
                else:
                    attributes["replay"] = policy
            operations.append(
                replace(
                    operation,
                    attributes=attributes,
                    regions=tuple(visit(region) for region in operation.regions),
                )
            )
        return replace(block, operations=tuple(operations))

    configured = replace(
        module,
        invocations=tuple(
            replace(invocation, body=visit(invocation.body))
            for invocation in module.invocations
        ),
    )
    if policy in ("batch-only", "required") and selected == 0:
        raise ValueError(f"loop execution {policy} requires a bounded loop")
    if io_schema_digest(configured) != io_schema_digest(module):
        raise ValueError("loop policy unexpectedly changed the I/O schema")
    return configured, selected
