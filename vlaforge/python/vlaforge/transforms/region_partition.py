"""Compose two pure Region interfaces without changing original SSA consumers."""

from dataclasses import replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Operation, Value


def split_region_invocation(module, target, *, first, second, second_inputs):
    """Route second inputs from original inputs or first outputs (source, index).

    Both implementations/artifacts must be supplied separately by the caller.
    This checks typed composition and rewrites one nested invocation, not the
    implementations' numerical equivalence or backend execution support.
    """
    verify(module)
    original = next((region for region in module.regions if region.name == target), None)
    if original is None:
        raise ValueError("unknown partition target")
    if any(not region.pure or region.metadata.get("memoize", False) for region in (original, first, second)):
        raise ValueError("partition requires pure uncached Regions")
    if (first.inputs != original.inputs or second.outputs != original.outputs
            or len(second.inputs) != len(second_inputs)):
        raise ValueError("partition interfaces do not compose the original Region")
    existing = {region.name for region in module.regions}
    if first.name == second.name or {first.name, second.name} & existing:
        raise ValueError("partition Region name collision")
    routes = tuple(second_inputs)
    for port, route in zip(second.inputs, routes, strict=True):
        if len(route) != 2 or route[0] not in ("input", "first") or type(route[1]) is not int:
            raise ValueError("partition route must name an exact input or first-output index")
        source = original.inputs if route[0] == "input" else first.outputs
        if not 0 <= route[1] < len(source):
            raise ValueError("partition route index out of range")
        payload = source[route[1]].type if route[0] == "input" else source[route[1]]
        if payload != port.type:
            raise ValueError("partition route type mismatch")
    matches, occupied = [], set()

    def inspect(block):
        occupied.update(value.name for value in block.arguments)
        for op in block.operations:
            occupied.update(value.name for value in op.results)
            if op.attributes.get("region") == target:
                if op.opcode != "vla.invoke" or set(op.attributes) != {"region"}:
                    raise ValueError("partition target references must be plain Region calls")
                matches.append(op)
            for child in op.regions:
                inspect(child)
    for invocation in module.invocations:
        inspect(invocation.body)
    if len(matches) != 1:
        raise ValueError("partition requires exactly one target call site")
    selected = matches[0]
    intermediate = []
    for index, payload in enumerate(first.outputs):
        name = "vf_partition_intermediate_" + str(index)
        while name in occupied:
            name += "_"
        occupied.add(name)
        intermediate.append(Value(name, payload))
    head = Operation("vla.invoke", tuple(intermediate), selected.operands,
                     {"region": first.name}, location=selected.location)
    tail_operands = tuple(selected.operands[index] if source == "input" else intermediate[index].name
                          for source, index in routes)
    tail = Operation("vla.invoke", selected.results, tail_operands, {"region": second.name}, location=selected.location)

    def rewrite(block):
        operations = []
        for op in block.operations:
            if op is selected:
                operations.extend((head, tail))
            else:
                operations.append(replace(op, regions=tuple(rewrite(child) for child in op.regions)))
        return replace(block, operations=tuple(operations))
    result = replace(module, regions=tuple(region for region in module.regions if region.name != target) + (first, second),
        invocations=tuple(replace(invocation, body=rewrite(invocation.body)) for invocation in module.invocations))
    verify(result)
    return result
