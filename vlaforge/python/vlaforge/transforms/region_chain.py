"""Compose a pure typed Region from explicitly routed, ordered stages."""

from dataclasses import replace
from typing import Sequence

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Module, Operation, TensorRegion, Value


def compose_region_invocation(
    module: Module,
    target: str,
    *,
    stages: Sequence[TensorRegion],
    stage_inputs: Sequence[Sequence[tuple]],
    output_routes: Sequence[tuple[int, int]],
) -> Module:
    """Replace one pure call site by a checked stage chain, including in loops.

    Input routes are ("input", original_index) or ("stage", earlier_stage,
    output_index). Each original result has one distinct (stage, output) route.
    Original result identities and every consumer, loop, predicate and commit
    remain unchanged, even when results originate in different stages.

    This proves typed call composition only. Implementations, numerical source
    equivalence, precision and backend support require independent evidence.
    All stages execute in order; a false data predicate does not skip stages.
    """
    verify(module)
    original = next((r for r in module.regions if r.name == target), None)
    if original is None:
        raise ValueError("unknown chain target Region")
    stages = tuple(stages)
    if not stages or len(stages) != len(stage_inputs):
        raise ValueError("each nonempty chain stage requires exact input routes")
    if any(not r.pure or r.metadata.get("memoize", False) for r in (original, *stages)):
        raise ValueError("chain requires pure uncached Regions")
    names = [r.name for r in stages]
    if len(names) != len(set(names)) or set(names) & {r.name for r in module.regions}:
        raise ValueError("chain Region name collision")
    routes = tuple(tuple(route for route in ports) for ports in stage_inputs)
    for index, (stage, operands) in enumerate(zip(stages, routes, strict=True)):
        if len(stage.inputs) != len(operands):
            raise ValueError("stage input route count differs")
        for port, route in zip(stage.inputs, operands, strict=True):
            if not isinstance(route, tuple) or not route:
                raise ValueError("explicit tuple input route required")
            if len(route) == 2 and route[0] == "input" and type(route[1]) is int:
                source, offset = original.inputs, route[1]
                if not 0 <= offset < len(source):
                    raise ValueError("original input route index out of range")
                payload = source[offset].type
            elif (len(route) == 3 and route[0] == "stage" and type(route[1]) is int
                  and type(route[2]) is int and 0 <= route[1] < index):
                source, offset = stages[route[1]].outputs, route[2]
                if not 0 <= offset < len(source):
                    raise ValueError("earlier-stage output route index out of range")
                payload = source[offset]
            else:
                raise ValueError("input route must name original input or strictly earlier stage output")
            if payload != port.type:
                raise ValueError("chain input route type mismatch")
    outputs = tuple(output_routes)
    if len(outputs) != len(original.outputs):
        raise ValueError("complete original output routes required")
    for payload, route in zip(original.outputs, outputs, strict=True):
        if (not isinstance(route, tuple) or len(route) != 2
                or any(type(value) is not int for value in route)
                or not 0 <= route[0] < len(stages)
                or not 0 <= route[1] < len(stages[route[0]].outputs)):
            raise ValueError("exact stage/output route required")
        if stages[route[0]].outputs[route[1]] != payload:
            raise ValueError("chain output route type mismatch")
    if len(set(outputs)) != len(outputs):
        raise ValueError("original results require distinct stage output routes")
    matches, occupied = [], set()

    def inspect(block):
        occupied.update(value.name for value in block.arguments)
        for op in block.operations:
            occupied.update(value.name for value in op.results)
            if op.attributes.get("region") == target:
                if op.opcode != "vla.invoke" or set(op.attributes) != {"region"} or op.regions:
                    raise ValueError("chain target must be a plain Region call")
                matches.append(op)
            for child in op.regions:
                inspect(child)

    for invocation in module.invocations:
        inspect(invocation.body)
    if len(matches) != 1:
        raise ValueError("chain requires exactly one target call site")
    selected = matches[0]
    public = dict(zip(outputs, selected.results, strict=True))
    values = []
    for stage_index, stage in enumerate(stages):
        row = []
        for output_index, payload in enumerate(stage.outputs):
            result = public.get((stage_index, output_index))
            if result is None:
                name = f"vf_chain_{stage_index}_{output_index}"
                while name in occupied:
                    name += "_"
                occupied.add(name)
                result = Value(name, payload)
            row.append(result)
        values.append(tuple(row))
    replacement = []
    for index, stage in enumerate(stages):
        operands = tuple(selected.operands[r[1]] if r[0] == "input" else values[r[1]][r[2]].name
                         for r in routes[index])
        replacement.append(Operation("vla.invoke", values[index], operands,
            {"region": stage.name}, location=selected.location))

    def rewrite(block):
        operations = []
        for op in block.operations:
            if op is selected:
                operations.extend(replacement)
            else:
                operations.append(replace(op, regions=tuple(rewrite(child) for child in op.regions)))
        return replace(block, operations=tuple(operations))

    result = replace(module,
        regions=tuple(r for r in module.regions if r.name != target) + stages,
        invocations=tuple(replace(i, body=rewrite(i.body)) for i in module.invocations))
    verify(result)
    return result
