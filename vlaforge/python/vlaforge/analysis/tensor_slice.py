"""Explicit closed tensor dataflow slices for independent operator experiments.

No model computation is executed or replaced. A slice is a profiling workload,
not an equivalent deployment: guards outside its dataflow are not transplanted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from vlaforge.analysis.constant_precompute import graph_sha256
from vlaforge.analysis.operator_inventory import _metadata


def _storage_roots(node, memo):
    """Follow schema aliases so a mutation outside the slice cannot be omitted."""
    if node in memo:
        return memo[node]
    schema = getattr(node.target, "_schema", None)
    if node.op in ("placeholder", "get_attr"):
        roots = {node}
    elif schema is None:
        roots = {root for value in node.all_input_nodes for root in _storage_roots(value, memo)}
    else:
        aliases = [result.alias_info for result in schema.returns if result.alias_info is not None]
        roots = {node} if not aliases else set()
        alias_sets = set().union(*(set(alias.before_set) | set(alias.after_set) for alias in aliases))
        for index, argument in enumerate(schema.arguments):
            alias = argument.alias_info
            if alias is None or not (alias_sets & (set(alias.before_set) | set(alias.after_set))):
                continue
            value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
            from torch.fx.node import map_arg
            map_arg(value, lambda dependency: roots.update(_storage_roots(dependency, memo)))
    memo[node] = roots
    return roots


@dataclass(frozen=True)
class TensorSlice:
    module: object
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    ledger: dict

    def validate_inputs(self, values):
        """Check the recorded static profile, not source identity or tensor bytes.

        The returned FX module has no automatic runtime guard. Call this before
        export or experimental execution; artifact guards are a separate layer.
        """
        import torch

        if not isinstance(values, tuple) or len(values) != len(self.input_names):
            raise ValueError("slice input tuple differs from explicit boundaries")
        for value, expected in zip(values, self.ledger["inputs"], strict=True):
            if not isinstance(value, torch.Tensor) or _metadata(value) != expected["metadata"]:
                raise ValueError("slice input metadata differs: " + expected["node"])


def extract_tensor_slice(program, *, inputs, outputs, source_artifact_sha256):
    """Copy the exact dependency closure between explicit tensor boundaries.

    All state must be passed as an explicit boundary tensor; nothing is guessed
    constant from an example. The caller verifies the archive SHA before loading
    and supplies independently captured, same-device boundary values afterward.
    Only static, non-mutating, non-random ATen tensor expressions are supported.
    Export, output verification and measurement are separate caller operations.
    The returned FX module does not enforce its recorded static profile; use
    result.validate_inputs before executing or exporting actual boundary values.
    """
    import torch

    from vlaforge.frontend.effect_audit import audit_exported_program

    if not isinstance(program, torch.export.ExportedProgram) or program.range_constraints:
        raise ValueError("tensor slicing requires a static ExportedProgram")
    if (not isinstance(source_artifact_sha256, str) or len(source_artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_artifact_sha256)):
        raise ValueError("tensor slicing requires an explicit source archive SHA256")
    if not audit_exported_program(program).passed:
        raise ValueError("source Region effect audit failed")

    def names(value):
        if (not isinstance(value, (tuple, list)) or not value
                or any(not isinstance(name, str) or not name for name in value)
                or len(set(value)) != len(value)):
            raise ValueError("tensor boundaries must be nonempty unique node names")
        return tuple(value)

    inputs, outputs = names(inputs), names(outputs)
    if set(inputs) & set(outputs):
        raise ValueError("slice outputs cannot be passthrough inputs")
    nodes = {node.name: node for node in program.graph.nodes}
    if (set(inputs) | set(outputs)) - nodes.keys():
        raise ValueError("tensor boundary node is absent")

    def tensor_node(node):
        value = node.meta.get("val")
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.is_quantized or value.is_complex()
                or any(type(size) is not int for size in (*value.shape, *value.stride()))
                or type(value.storage_offset()) is not int):
            raise ValueError("slice nodes require static strided tensor metadata")
        return _metadata(value)

    for name in (*inputs, *outputs):
        tensor_node(nodes[name])

    selected, reached = set(), set()
    boundary = {nodes[name] for name in inputs}
    pending = [nodes[name] for name in outputs]
    while pending:
        node = pending.pop()
        if node in boundary:
            reached.add(node)
            continue
        if node in selected:
            continue
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            raise ValueError("slice dependency escapes explicit inputs or is not an ATen overload")
        if (node.target.namespace != "aten" or node.target._schema.is_mutable
                or torch.Tag.nondeterministic_seeded in node.target.tags):
            raise ValueError("slice computation must be read-only and have no implicit RNG")
        if node.target._schema.name in {
            "aten::empty", "aten::empty_like", "aten::empty_strided",
            "aten::new_empty", "aten::new_empty_strided",
        }:
            raise ValueError("slice cannot expose uninitialized storage")
        tensor_node(node)
        selected.add(node)
        pending.extend(node.all_input_nodes)
    if reached != boundary:
        raise ValueError("slice has unused input boundaries")

    from torch.fx.node import map_arg
    memo = {}
    protected = {root for node in selected | boundary for root in _storage_roots(node, memo)}
    for node in program.graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None or not schema.is_mutable:
            continue
        for index, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
            written = set()
            map_arg(value, lambda dependency, written=written: written.update(_storage_roots(dependency, memo)))
            if protected & written:
                raise ValueError("source mutation aliases slice storage; explicit dataflow is insufficient")

    source_graph = graph_sha256(program)
    graph = torch.fx.Graph()
    environment = {}

    def metadata(node):
        result = dict(node.meta)
        if "custom" in result:
            result["custom"] = copy.deepcopy(result["custom"])
        return result

    for name in inputs:
        node = nodes[name]
        copied = graph.placeholder(name)
        copied.meta = metadata(node)
        environment[node] = copied
    ordered = [node for node in program.graph.nodes if node in selected]
    for node in ordered:
        copied = graph.node_copy(node, environment.__getitem__)
        copied.meta = metadata(node)
        environment[node] = copied
    graph.output(tuple(environment[nodes[name]] for name in outputs))
    graph.lint()
    module = torch.fx.GraphModule({}, graph)
    if graph_sha256(program) != source_graph:
        raise ValueError("source graph changed during extraction")
    ledger = {
        "schema": "vlaforge.tensor_slice/1",
        "source_artifact_sha256": source_artifact_sha256,
        "source_graph_sha256": source_graph,
        "inputs": [{"node": name, "metadata": tensor_node(nodes[name])} for name in inputs],
        "outputs": [{"node": name, "metadata": tensor_node(nodes[name])} for name in outputs],
        "nodes": [{"node": node.name, "target": str(node.target)} for node in ordered],
        "state_externalized": True,
        "state_copied": False,
        "source_executed": False,
        "source_mutable_aliases_rejected": True,
        "guard_scope": "selected tensor dataflow only; not a deployment transformation",
        "source_output_verified": False,
        "performance_measured": False,
        "selected_for_deployment": False,
    }
    return TensorSlice(module, inputs, outputs, ledger)
