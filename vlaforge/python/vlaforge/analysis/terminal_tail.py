"""Partition an explicit terminal tensor suffix without pruning its prefix.

This produces Region inputs and routes, not a deployment certificate. An Adapter
can call the prefix and tail with existing InvocationBuilder operations; original
USER_INPUT routes go directly to the tail without a prefix passthrough output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vlaforge.analysis.constant_precompute import _clone, _literal, graph_sha256
from vlaforge.analysis.operator_inventory import _metadata
from vlaforge.analysis.tensor_slice import _storage_roots, extract_tensor_slice
from vlaforge.analysis.unused_state import _signature_record, _tensor_record


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _state_records(program):
    return {
        namespace: {name: _tensor_record(value) for name, value in values.items()}
        for namespace, values in (
            ("state", program.state_dict), ("constants", program.constants),
        )
    }


def _tail_digest(tail):
    return _digest({
        "graph": str(tail.module.graph),
        "metadata": {
            node.name: _metadata(node.meta.get("val"))
            for node in tail.module.graph.nodes if node.op != "output"
        },
    })


def _node_records(nodes):
    return [{
        "name": node.name, "op": node.op, "target": str(node.target),
        "args": _literal(node.args), "kwargs": _literal(node.kwargs),
        "metadata": _metadata(node.meta.get("val")),
    } for node in nodes]


@dataclass(frozen=True)
class TailInputRoute:
    node: str
    source: str
    index: int


@dataclass(frozen=True)
class TerminalTailPartition:
    control: object
    prefix: object
    tail: object
    frontier_names: tuple[str, ...]
    routes: tuple[TailInputRoute, ...]
    ledger: dict
    _ledger_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_ledger_sha256", _digest(self.ledger))

    def validate(self):
        """Reject edited graphs, routes, ledger or shared state before saving."""
        if _digest(self.ledger) != self._ledger_sha256:
            raise ValueError("terminal partition ledger changed")
        if ([asdict(route) for route in self.routes] != self.ledger["tail_input_routes"]
                or list(self.frontier_names) != self.ledger["frontier_names"]):
            raise ValueError("terminal partition routes changed")
        if (self.tail.ledger != self.ledger["tail_slice"]
                or tuple(route.node for route in self.routes) != self.tail.input_names
                or tuple(item["node"] for item in self.tail.ledger["outputs"]) != self.tail.output_names):
            raise ValueError("terminal partition tail contract changed")
        for name, program in (("control", self.control), ("prefix", self.prefix)):
            if graph_sha256(program) != self.ledger[name + "_graph_sha256"]:
                raise ValueError("terminal partition graph changed: " + name)
            if _state_records(program) != self.ledger["source_state"]:
                raise ValueError("terminal partition state changed: " + name)
            if (_signature_record(program.module_call_graph) != self.ledger[name + "_module_call_graph"]
                    or _literal(getattr(program, "_guards_code", None)) != self.ledger["source_guards_code"]):
                raise ValueError("terminal partition call signature or guards changed: " + name)
        for attribute in ("state_dict", "constants"):
            original, candidate = getattr(self.control, attribute), getattr(self.prefix, attribute)
            if original.keys() != candidate.keys() or any(
                original[key] is not candidate[key] for key in original
            ):
                raise ValueError("terminal partition no longer shares original state objects")
        if _tail_digest(self.tail) != self.ledger["tail_graph_sha256"]:
            raise ValueError("terminal partition tail changed")


def partition_terminal_tail(program, *, inputs, outputs, source_artifact_sha256):
    """Split all flat Tensor outputs into a continuous, pure terminal suffix.

    The prefix keeps every original placeholder, lifted state object and retained
    computation in its original order, including unused math and assertions.
    Its return value is a tuple of computed frontier tensors. Tail input indices
    address that tuple or the original flattened USER_INPUT signature (including
    literal slots). The tail returns the original flattened tensor outputs in
    order; the Adapter retains the original external output structure.

    The initial contract rejects HOPs, unlifted state, state boundaries, a prefix
    without computed outputs, and exported frontier/output storage aliases. No
    computation runs during partitioning; state bytes are read for lineage. The
    caller verifies the supplied archive SHA and keeps state immutable through
    serialization. All artifacts require new compilation and full-model checks.
    """
    import torch
    from torch.export.graph_signature import (
        InputKind,
        OutputKind,
        OutputSpec,
        TensorArgument,
    )
    from torch.utils import _pytree

    tail = extract_tensor_slice(
        program, inputs=inputs, outputs=outputs,
        source_artifact_sha256=source_artifact_sha256,
    )
    original_hash = graph_sha256(program)
    source_state = _state_records(program)
    nodes = list(program.graph.nodes)
    by_name = {node.name: node for node in nodes}
    selected = {entry["node"] for entry in tail.ledger["nodes"]}
    for node in nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function" or not (
            isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == "aten"
            or node.target is operator.getitem
        ):
            raise ValueError("terminal partition rejects HOPs, unknown calls and unlifted state")
        # This operator's schema deliberately omits its actual storage alias.
        if node.target is torch.ops.aten._unsafe_view.default:
            raise ValueError("terminal partition rejects schema-hidden storage aliases")
    program.validate()
    operations = [node for node in nodes if node.op not in ("placeholder", "output")]
    first = next(index for index, node in enumerate(operations) if node.name in selected)
    if {node.name for node in operations[first:]} != selected:
        raise ValueError("selected computation is not a continuous terminal suffix")
    output_specs = program.graph_signature.output_specs
    if any(spec.kind != OutputKind.USER_OUTPUT or not isinstance(spec.arg, TensorArgument)
           for spec in output_specs):
        raise ValueError("terminal partition requires only Tensor USER_OUTPUT slots")
    if tuple(spec.arg.name for spec in output_specs) != tail.output_names:
        raise ValueError("tail outputs must cover every original output in order")

    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    users = [spec.arg.name for spec in program.graph_signature.input_specs
             if spec.kind == InputKind.USER_INPUT]
    frontier, routes = [], []
    memo = {}
    external_roots = {node for node in nodes if node.op == "placeholder"}
    frontier_roots = set()
    for name in tail.input_names:
        node = by_name[name]
        if node.op == "placeholder":
            if specs[name].kind != InputKind.USER_INPUT or not isinstance(specs[name].arg, TensorArgument):
                raise ValueError("tail boundaries cannot directly expose lifted state")
            routes.append(TailInputRoute(name, "user_input", users.index(name)))
        else:
            roots = _storage_roots(node, memo)
            if roots & (external_roots | frontier_roots):
                raise ValueError("computed frontier aliases input/state or another frontier")
            value = node.meta["val"]
            if not value.is_contiguous() or value.storage_offset() != 0:
                raise ValueError("computed frontier requires contiguous zero-offset storage")
            frontier_roots.update(roots)
            routes.append(TailInputRoute(name, "prefix_output", len(frontier)))
            frontier.append(name)
    if not frontier:
        raise ValueError("terminal partition requires a computed frontier; zero-output prefix unsupported")
    published_roots = {
        root for name in tail.input_names for root in _storage_roots(by_name[name], memo)
    }
    for name in tail.output_names:
        roots = _storage_roots(by_name[name], memo)
        if roots & published_roots:
            raise ValueError("tail output aliases a boundary or another output")
        published_roots.update(roots)

    signature = copy.deepcopy(program.graph_signature)
    signature.output_specs = [
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name), None) for name in frontier
    ]
    call_graph = copy.deepcopy(program.module_call_graph)
    roots = [entry for entry in call_graph if entry.fqn == ""]
    if len(roots) != 1 or roots[0].signature is None:
        raise ValueError("terminal partition requires an explicit root module-call signature")
    for entry in call_graph:
        if entry.fqn and entry.signature is not None and any(
            getattr(argument, "name", None) in selected
            for argument in (*entry.signature.inputs, *entry.signature.outputs)
        ):
            raise ValueError("terminal suffix is referenced by a nested module-call signature")
    root_signature = roots[0].signature
    root_signature.out_spec = _pytree.tree_flatten(tuple(None for _ in frontier))[1]
    if root_signature.outputs:
        root_signature.outputs = [TensorArgument(name) for name in frontier]

    graph, environment = torch.fx.Graph(), {}
    for node in nodes:
        if node.name in selected or node.op == "output":
            continue
        copied = graph.node_copy(node, environment.__getitem__)
        copied.meta = dict(node.meta)
        if "custom" in copied.meta:
            copied.meta["custom"] = copy.deepcopy(copied.meta["custom"])
        environment[node] = copied
    graph.output(tuple(environment[by_name[name]] for name in frontier))
    graph.lint()
    prefix = torch.export.ExportedProgram(
        root=program.graph_module, graph=graph, graph_signature=signature,
        state_dict=dict(program.state_dict), constants=dict(program.constants),
        range_constraints=copy.deepcopy(program.range_constraints),
        module_call_graph=call_graph, example_inputs=program.example_inputs,
        verifiers=program.verifiers,
    )
    prefix.validate()
    control = _clone(program)
    retained_nodes = [node for node in nodes if node.name not in selected and node.op != "output"]
    retained_records = _node_records(retained_nodes)
    if (prefix.graph_signature.input_specs != program.graph_signature.input_specs
            or prefix.call_spec.in_spec != program.call_spec.in_spec
            or _node_records(list(prefix.graph.nodes)[:-1]) != retained_records
            or any(getattr(candidate, "_guards_code", None) != getattr(program, "_guards_code", None)
                   for candidate in (prefix, control))
            or graph_sha256(control) != original_hash
            or graph_sha256(program) != original_hash):
        raise ValueError("terminal partition changed source, retained computation, guards or input ABI")
    result = TerminalTailPartition(control, prefix, tail, tuple(frontier), tuple(routes), {
        "schema": "vlaforge.terminal_tail_partition/1",
        "source_artifact_sha256": source_artifact_sha256,
        "source_graph_sha256": original_hash,
        "control_graph_sha256": graph_sha256(control),
        "prefix_graph_sha256": graph_sha256(prefix),
        "tail_graph_sha256": _tail_digest(tail),
        "tail_slice": copy.deepcopy(tail.ledger),
        "source_state": source_state,
        "source_guards_code": _literal(getattr(program, "_guards_code", None)),
        "control_module_call_graph": _signature_record(control.module_call_graph),
        "prefix_module_call_graph": _signature_record(prefix.module_call_graph),
        "source_input_signature": _signature_record(program.graph_signature.input_specs),
        "source_output_signature": _signature_record(output_specs),
        "source_output_structure": _signature_record(program.call_spec.out_spec),
        "frontier_names": frontier,
        "tail_input_routes": [asdict(route) for route in routes],
        "retained_nodes": [node.name for node in retained_nodes],
        "retained_node_records_sha256": _digest(retained_records),
        "retained_no_output_nodes": [node.name for node in operations[:first] if not node.users],
        "state_objects_shared": True,
        "source_computation_executed": False,
        "dead_code_elimination_performed": False,
        "external_input_passthrough_copies_added": False,
        "source_archive_sha_verified_by_transform": False,
        "full_model_verified": False,
        "performance_measured": False,
        "selected_for_deployment": False,
    })
    result.validate()
    return result


def save_terminal_tail_partition(partition, directory):
    """Save control/prefix EPs and a hash-bound ledger in a fresh directory.

    The tail remains a TensorSlice for existing capture/export APIs: this helper
    does not execute it, synthesize example inputs, or emit a tail artifact.
    """
    import torch

    partition.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, program in (("control", partition.control), ("prefix", partition.prefix)):
        path = directory / (name + ".pt2")
        torch.export.save(program, path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        artifacts[name] = {"file": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    partition.validate()
    report = {
        "schema": "vlaforge.saved_terminal_tail_partition/1",
        "partition": partition.ledger,
        "partition_sha256": partition._ledger_sha256,
        "artifacts": artifacts,
        "tail_artifact_emitted": False,
    }
    (directory / "partition.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
