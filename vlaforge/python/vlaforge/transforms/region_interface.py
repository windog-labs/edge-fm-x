"""Adapt pure Region declarations and calls to an explicit artifact port order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module


def _permutation(order: Sequence[int] | None, count: int) -> tuple[int, ...]:
    values = tuple(range(count)) if order is None else tuple(order)
    if (len(values) != count or any(type(index) is not int for index in values)
            or set(values) != set(range(count))):
        raise ValueError("Region port order must be a complete integer permutation")
    return values


def reorder_region_interface(
    module: Module,
    region_name: str,
    *,
    input_order: Sequence[int] | None = None,
    output_order: Sequence[int] | None = None,
) -> Module:
    """Permute one pure Region and every nested call, retaining SSA identities.

    Each new slot gives its old declaration index. The caller must supply the
    corresponding reordered artifact implementation; this pass neither guesses
    binding names nor certifies artifact semantics. External ports, dtypes,
    consumer operands and named cache dependencies are unchanged.
    """
    verify(module)
    selected = next((region for region in module.regions if region.name == region_name), None)
    if selected is None:
        raise ValueError(f"unknown Region: {region_name}")
    if not selected.pure:
        raise ValueError("Region interface reordering requires a pure Region")
    inputs = _permutation(input_order, len(selected.inputs))
    outputs = _permutation(output_order, len(selected.outputs))
    return _remap_region_ports(module, region_name, inputs, outputs)


def project_region_inputs(module: Module, region_name: str, *, keep_indices: Sequence[int]) -> Module:
    """Adapt calls to a caller-supplied pruned artifact input contract.

    Indices give the exact retained original ports, in artifact order. Results,
    external inputs, states and SSA consumers are not removed. The caller must
    prove that its artifact implements this projection; this pass does not infer
    unused arguments from opaque Region bodies or certify numerical equivalence.
    Cached/effectful Regions require a separate contract and are rejected here.
    """
    verify(module)
    selected = next((region for region in module.regions if region.name == region_name), None)
    if selected is None:
        raise ValueError(f"unknown Region: {region_name}")
    if not selected.pure or selected.metadata.get("memoize", False):
        raise ValueError("Region input projection requires a pure uncached Region")
    inputs = tuple(keep_indices)
    if (any(type(index) is not int or not 0 <= index < len(selected.inputs) for index in inputs)
            or len(set(inputs)) != len(inputs)):
        raise ValueError("Region input projection requires unique in-range integer indices")

    def inspect(block):
        for operation in block.operations:
            if operation.attributes.get("region") == region_name and (
                    operation.opcode != "vla.invoke" or set(operation.attributes) != {"region"}):
                raise ValueError("Region input projection requires plain Region calls")
            for child in operation.regions:
                inspect(child)
    for invocation in module.invocations:
        inspect(invocation.body)
    return _remap_region_ports(module, region_name, inputs, tuple(range(len(selected.outputs))))


def _remap_region_ports(module, region_name, inputs, outputs):

    def rewrite(block: Block) -> Block:
        operations = []
        for operation in block.operations:
            updated = replace(operation, regions=tuple(rewrite(child) for child in operation.regions))
            if operation.opcode == "vla.invoke" and operation.attributes["region"] == region_name:
                updated = replace(updated,
                                  operands=tuple(operation.operands[index] for index in inputs),
                                  results=tuple(operation.results[index] for index in outputs))
            operations.append(updated)
        return replace(block, operations=tuple(operations))

    transformed = replace(
        module,
        regions=tuple(replace(region,
                              inputs=tuple(region.inputs[index] for index in inputs),
                              outputs=tuple(region.outputs[index] for index in outputs))
                      if region.name == region_name else region for region in module.regions),
        invocations=tuple(replace(invocation, body=rewrite(invocation.body))
                          for invocation in module.invocations),
    )
    verify(transformed)
    return transformed
