"""Artifact-experiment wiring for one explicit, uncached terminal partition.

This helper is not a compiler pass or an automatic optimization selection.
"""

from __future__ import annotations

from dataclasses import replace

from vlaforge.analysis import verify
from vlaforge.frontend.tensor_types import tensor_type_from_torch
from vlaforge.ir.program import Operation, TensorRegion, Value
from vlaforge.ir.serializer import module_digest, module_from_data, module_to_data


def interpreter_tensor_callable(function, output_count):
    """Adapt exported pytree returns to the existing reference Interpreter ABI."""
    if type(output_count) is not int or output_count < 1:
        raise ValueError("positive declared Tensor output count required")

    def invoke(*arguments):
        import torch
        from torch.utils import _pytree

        outputs, _ = _pytree.tree_flatten(function(*arguments))
        if len(outputs) != output_count or any(not isinstance(value, torch.Tensor) for value in outputs):
            raise ValueError("exported return differs from declared Tensor outputs")
        return outputs[0] if output_count == 1 else tuple(outputs)
    return invoke


def wire_terminal_partition(module, *, target, partition):
    """Replace one vla.invoke, preserving every other serialized IR field."""
    from torch.export.graph_signature import InputKind, TensorArgument

    partition.validate()
    verify(module)
    before = module_to_data(module)
    original = module.region(target)
    if (not original.pure or original.metadata.get("memoize", False)
            or set(original.metadata) - {"loop_invariant", "memoize"}):
        raise ValueError("target must be pure and uncached, without prior artifact metadata")
    prefix_name, tail_name = target + "__terminal_prefix", target + "__terminal_tail"
    if {prefix_name, tail_name} & {region.name for region in module.regions}:
        raise ValueError("terminal Region name collision")
    nodes = {node.name: node for node in partition.prefix.graph.nodes}
    user_specs = [spec for spec in partition.prefix.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    if any(not isinstance(spec.arg, TensorArgument) for spec in user_specs):
        raise ValueError("initial artifact wiring only supports Tensor USER_INPUT slots")
    if tuple(tensor_type_from_torch(nodes[spec.arg.name].meta["val"]) for spec in user_specs) != tuple(
        value.type for value in original.inputs
    ):
        raise ValueError("original Region inputs differ from the exported user ABI")
    tail_nodes = {node.name: node for node in partition.tail.module.graph.nodes}
    output_types = tuple(tensor_type_from_torch(tail_nodes[name].meta["val"]) for name in partition.tail.output_names)
    if output_types != original.outputs:
        raise ValueError("original Region outputs differ from terminal outputs")
    prefix_types = tuple(tensor_type_from_torch(nodes[name].meta["val"]) for name in partition.frontier_names)
    metadata = {"loop_invariant": original.metadata.get("loop_invariant", False),
                "terminal_partition_ledger_sha256": partition._ledger_sha256,
                "compiled_artifact_verified": False}
    prefix_region = TensorRegion(prefix_name, original.inputs, prefix_types, metadata=metadata)
    tail_region = TensorRegion(
        tail_name,
        tuple(Value(name, tensor_type_from_torch(tail_nodes[name].meta["val"])) for name in partition.tail.input_names),
        original.outputs, metadata=metadata,
    )
    all_names = set()
    matching = []

    def inspect(block):
        all_names.update(value.name for value in block.arguments)
        for operation in block.operations:
            all_names.update(value.name for value in operation.results)
            if operation.attributes.get("region") == target:
                if operation.opcode != "vla.invoke" or set(operation.attributes) != {"region"}:
                    raise ValueError("target references must be plain uncached vla.invoke")
                matching.append(operation)
            for child in operation.regions:
                inspect(child)
    for invocation in module.invocations:
        inspect(invocation.body)
    if len(matching) != 1:
        raise ValueError("terminal wiring requires exactly one target invocation")
    selected = matching[0]
    prefix_results = tuple(Value("vf_terminal_frontier_" + str(index), payload)
                           for index, payload in enumerate(prefix_types))
    if any(value.name in all_names for value in prefix_results):
        raise ValueError("terminal frontier SSA name collision")
    prefix_call = Operation("vla.invoke", prefix_results, selected.operands,
                            {"region": prefix_name}, location=selected.location)
    tail_arguments = tuple(
        selected.operands[route.index] if route.source == "user_input" else prefix_results[route.index].name
        for route in partition.routes
    )
    tail_call = Operation("vla.invoke", selected.results, tail_arguments,
                          {"region": tail_name}, location=selected.location)

    def transform(block):
        operations = []
        for operation in block.operations:
            if operation is selected:
                operations.extend((prefix_call, tail_call))
            elif operation.regions:
                operations.append(replace(operation, regions=tuple(transform(child) for child in operation.regions)))
            else:
                operations.append(operation)
        return replace(block, operations=tuple(operations))
    result = replace(module,
                     regions=tuple(region for region in module.regions if region.name != target) + (prefix_region, tail_region),
                     invocations=tuple(replace(invocation, body=transform(invocation.body)) for invocation in module.invocations))
    verify(result)
    after = module_to_data(result)
    reverse = module_to_data(result)
    reverse["regions"] = before["regions"]
    reversals = 0

    def undo(block):
        nonlocal reversals
        operations, restored, index = block["operations"], [], 0
        while index < len(operations):
            operation = operations[index]
            if operation["attributes"].get("region") == prefix_name:
                assert operations[index + 1]["attributes"].get("region") == tail_name
                restored.append(original_call)
                reversals += 1
                index += 2
            else:
                for child in operation["regions"]:
                    undo(child)
                restored.append(operation)
                index += 1
        block["operations"] = restored

    def original_operation(block):
        for operation in block["operations"]:
            if operation["attributes"].get("region") == target:
                return operation
            for child in operation["regions"]:
                found = original_operation(child)
                if found is not None:
                    return found
        return None
    original_call = next(value for invocation in before["invocations"]
                         if (value := original_operation(invocation["body"])) is not None)
    for invocation in reverse["invocations"]:
        undo(invocation["body"])
    if reversals != 1 or reverse != before or module_to_data(module) != before:
        raise ValueError("terminal wiring changed an original nonselected IR field")
    if module_to_data(module_from_data(after)) != after:
        raise ValueError("terminal wiring changed during serialization roundtrip")
    return result, {
        "schema": "vlaforge.terminal_partition_ir_experiment/1",
        "source_module_digest": module_digest(module), "new_module_digest": module_digest(result),
        "partition_ledger_sha256": partition._ledger_sha256,
        "target": target, "prefix_region": prefix_name, "tail_region": tail_name,
        "target_replacements": 1,
        "original_nonselected_ir_fields_exact": True,
        "original_result_names_retained": [value.name for value in selected.results],
        "prefix_operands": list(prefix_call.operands), "tail_operands": list(tail_call.operands),
        "old_artifact_contracts_inherited": False,
        "compiled_artifact_verified": False, "selected_for_deployment": False,
    }
