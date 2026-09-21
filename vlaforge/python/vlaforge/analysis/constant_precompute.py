"""Explicit immutable-state precomputation without changing Region boundaries.

This is a static-precompute ablation, not a kernel-selection pass. Snapshots are
an explicit caller contract: the named state must remain immutable for the
lifetime of the resulting artifact. No example input is treated as a constant.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

_PURE_TARGETS = frozenset({
    "aten.embedding.default", "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.squeeze.default", "aten.expand.default", "aten.view.default",
    "aten.reshape.default", "aten.permute.default", "aten.transpose.int",
    "aten.detach.default", "aten.clone.default", "aten.add.Tensor",
    "aten.add.Scalar", "aten.sub.Tensor", "aten.sub.Scalar", "aten.mul.Tensor",
    "aten.mul.Scalar", "aten.neg.default", "aten.sin.default", "aten.cos.default",
    "aten.arange.default", "aten.arange.start", "aten.arange.start_step",
    "aten.div.Tensor", "aten.div.Scalar", "aten.exp.default", "aten._to_copy.default",
    "aten.to.dtype", "aten.to.device", "aten.to.dtype_layout",
})


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _literal(value: object) -> object:
    import torch

    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite_float": repr(value)}
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": value.name}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return {"torch_literal": str(value)}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_literal(item) for item in value]}
    if isinstance(value, Mapping):
        return {"mapping": [[_literal(key), _literal(item)] for key, item in value.items()]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__qualname__, "fields": {
            field.name: _literal(getattr(value, field.name)) for field in dataclasses.fields(value)
        }}
    raise ValueError(f"unsupported graph literal: {type(value).__qualname__}")


def _metadata(value: Any) -> dict[str, object]:
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError("precompute requires strided tensor metadata")
    if any(type(size) is not int for size in (*value.shape, *value.stride())):
        raise ValueError("precompute requires static shape and stride")
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "storage_offset": value.storage_offset(), "dtype": str(value.dtype),
            "device": str(value.device), "layout": str(value.layout)}


def tensor_record(value: Any) -> dict[str, object]:
    """Hash logical element bytes and metadata; never cast BF16 through FP32."""
    import torch

    metadata = _metadata(value)
    if value.is_complex() or value.is_quantized:
        raise ValueError("complex and quantized precompute tensors are unsupported")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError("precompute tensor is nonfinite")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    metadata.update(bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
    return {**metadata, "sha256": hashlib.sha256(_json(metadata)).hexdigest()}


def graph_sha256(program: Any) -> str:
    """Canonical graph/topology/signature digest, separate from archive SHA."""
    import torch

    nodes = []
    for node in program.graph.nodes:
        value = node.meta.get("val")
        metadata = _metadata(value) if isinstance(value, torch.Tensor) else None
        nodes.append({"name": node.name, "op": node.op,
                      "target": "operator.getitem" if node.target is operator.getitem else str(node.target),
                      "args": _literal(node.args), "kwargs": _literal(node.kwargs),
                      "tensor_metadata": metadata})
    return hashlib.sha256(_json({"nodes": nodes, "signature": _literal(program.graph_signature),
                                "in_spec": str(program.call_spec.in_spec),
                                "out_spec": str(program.call_spec.out_spec)})).hexdigest()


def _state_specs(program: Any) -> dict[str, Any]:
    from torch.export.graph_signature import InputKind

    return {spec.target: spec for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.PARAMETER or
            (spec.kind == InputKind.BUFFER and spec.persistent is True)}


@dataclasses.dataclass(frozen=True)
class ImmutableSnapshot:
    snapshot_id: str
    source_artifact_sha256: str
    source_graph_sha256: str
    tensor_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        hashes = (self.source_artifact_sha256, self.source_graph_sha256, *self.tensor_sha256.values())
        if not self.snapshot_id or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes
        ):
            raise ValueError("immutable snapshot requires identity and SHA256 digests")
        object.__setattr__(self, "tensor_sha256", MappingProxyType(dict(self.tensor_sha256)))

    def to_dict(self) -> dict[str, object]:
        return {"snapshot_id": self.snapshot_id, "source_artifact_sha256": self.source_artifact_sha256,
                "source_graph_sha256": self.source_graph_sha256, "tensor_sha256": dict(self.tensor_sha256),
                "contract": ("named state is immutable for artifact lifetime" if self.tensor_sha256 else
                             "literal-only; no state or user input is approved as a constant dependency")}


def declare_immutable_snapshot(program: Any, targets: Sequence[str], *,
                               snapshot_id: str, source_artifact_sha256: str,
                               literal_only: bool = False) -> ImmutableSnapshot:
    """Explicitly declare only named parameters/persistent buffers immutable.

    The caller, not an equality test on example inputs, chooses these names.
    Archive digest verification belongs to the file-loading orchestration.
    """
    if type(literal_only) is not bool or (literal_only and targets):
        raise ValueError("literal-only snapshot requires no state targets")
    if (not targets and not literal_only) or len(set(targets)) != len(targets):
        raise ValueError("immutable targets must be nonempty and unique")
    specs = _state_specs(program)
    if any(target not in specs for target in targets):
        raise ValueError("immutable target is not a lifted parameter or persistent buffer")
    return ImmutableSnapshot(snapshot_id, source_artifact_sha256, graph_sha256(program),
                             {target: tensor_record(program.state_dict[target])["sha256"] for target in targets})


def _clone(program: Any, graph: Any = None, signature: Any = None, state: Any = None) -> Any:
    import torch

    if graph is None:
        graph = torch.fx.Graph()
        environment = {}
        for node in program.graph.nodes:
            copied = graph.node_copy(node, lambda item: environment[item])
            copied.meta = dict(node.meta)
            if "custom" in copied.meta:
                copied.meta["custom"] = copy.deepcopy(copied.meta["custom"])
            environment[node] = copied
    return torch.export.ExportedProgram(
        root=program.graph_module, graph=graph,
        graph_signature=copy.deepcopy(program.graph_signature) if signature is None else signature,
        state_dict=dict(program.state_dict) if state is None else state,
        range_constraints=copy.deepcopy(program.range_constraints),
        module_call_graph=copy.deepcopy(program.module_call_graph),
        example_inputs=program.example_inputs, constants=dict(program.constants),
        verifiers=program.verifiers,
    )


def _audit_program(program: Any) -> None:
    import torch

    from vlaforge.frontend.effect_audit import audit_exported_program

    if program.range_constraints:
        raise ValueError("constant precompute currently requires a static exported profile")
    for node in program.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function" or not (
            node.target is operator.getitem or
            (isinstance(node.target, torch._ops.OpOverload) and node.target._schema.name.startswith("aten::"))
        ):
            raise ValueError(f"unknown call or unlifted state at {node.name}")
    audit = audit_exported_program(program)
    if not audit.passed:
        raise ValueError("precompute effect audit failed: " + "; ".join(item.message for item in audit.diagnostics))


def _fresh_conversion(node: Any) -> bool:
    import torch

    if str(node.target) not in {"aten.to.dtype", "aten.to.device", "aten.to.dtype_layout"}:
        return False
    source, result = node.args[0].meta.get("val"), node.meta.get("val")
    if not isinstance(source, torch.Tensor) or not isinstance(result, torch.Tensor):
        return False
    for index, argument in enumerate(node.target._schema.arguments):
        actual = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
        if argument.name == "copy" and actual is True:
            return True
    return source.dtype != result.dtype or source.device != result.device


def _alias_safe(root: Any, *, fresh_result: bool = True) -> None:
    """A fresh result must not become externally mutable persistent storage."""
    import torch
    from torch.utils._pytree import tree_leaves

    if (fresh_result and any(result.alias_info is not None for result in root.target._schema.returns)
            and not _fresh_conversion(root)):
        raise ValueError(f"selected output {root.name} aliases its dependencies")
    queue, seen = [root], set()
    while queue:
        value = queue.pop()
        if value in seen:
            continue
        seen.add(value)
        for user in value.users:
            if user.op == "output":
                raise ValueError(f"selected output {root.name} has an external alias escape")
            if user.target is operator.getitem:
                queue.append(user)
                continue
            if not isinstance(user.target, torch._ops.OpOverload):
                raise TypeError(f"selected output reaches unknown call {user.name}")
            schema = user.target._schema
            if "unsafe" in schema.name:
                raise ValueError(f"alias-unsafe operator downstream of {root.name}: {user.name}")
            for index, argument in enumerate(schema.arguments):
                actual = user.args[index] if index < len(user.args) else user.kwargs.get(argument.name)
                if value not in tree_leaves(actual):
                    continue
                alias = argument.alias_info
                if alias is None:
                    continue
                if alias.is_write:
                    raise ValueError(f"selected output {root.name} is mutated through {user.name}")
                incoming = set(alias.before_set) | set(alias.after_set)
                for result in schema.returns:
                    if "*" in incoming or (result.alias_info is not None and "*" in (
                        set(result.alias_info.before_set) | set(result.alias_info.after_set)
                    )):
                        raise ValueError(f"unknown wildcard alias downstream of {root.name}")
                    if result.alias_info is not None and incoming & (
                        set(result.alias_info.before_set) | set(result.alias_info.after_set)
                    ):
                        queue.append(user)


@dataclasses.dataclass(frozen=True)
class PrecomputeResult:
    control: Any
    folded: Any
    ledger: dict[str, object]


def precompute_constants(program: Any, *, output_nodes: Sequence[str],
                         snapshot: ImmutableSnapshot) -> PrecomputeResult:
    """Fold audited state-only subgraphs in a copy, retaining all original ports."""
    import torch
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
    from torch.utils._pytree import tree_leaves, tree_map

    _audit_program(program)
    original_hash = graph_sha256(program)
    if original_hash != snapshot.source_graph_sha256:
        raise ValueError("source graph differs from immutable snapshot")
    specs = _state_specs(program)
    records = {}
    for target, expected in snapshot.tensor_sha256.items():
        if target not in specs:
            raise ValueError("snapshot contains a non-state target")
        records[target] = tensor_record(program.state_dict[target])
        if records[target]["sha256"] != expected:
            raise ValueError(f"immutable snapshot changed: {target}")
    if not output_nodes or len(set(output_nodes)) != len(output_nodes):
        raise ValueError("selected output nodes must be nonempty and unique")
    nodes = {node.name: node for node in program.graph.nodes}
    placeholders = {spec.arg.name: spec for spec in program.graph_signature.input_specs}
    closures, dependencies = {}, {}

    def visit(node: Any, closure: set[Any], leaves: set[str]) -> None:
        if node in closure:
            return
        closure.add(node)
        if node.op == "placeholder":
            spec = placeholders[node.name]
            if spec.target not in specs or spec.target not in snapshot.tensor_sha256:
                raise ValueError(f"unapproved state or user input leaf: {node.name}")
            leaves.add(spec.target)
            return
        if node.op != "call_function" or str(node.target) not in _PURE_TARGETS:
            raise ValueError(f"operator is not in the pure precompute allowlist: {node.name}")
        if node.target._schema.is_mutable:
            raise ValueError(f"mutable precompute operator: {node.name}")
        if str(node.target).startswith("aten.arange.") and (
            not isinstance(node.kwargs.get("dtype"), torch.dtype)
            or not isinstance(node.kwargs.get("device"), (torch.device, str))
        ):
            raise ValueError("literal factory requires explicit dtype and device")
        for argument in tree_leaves((node.args, node.kwargs)):
            if isinstance(argument, torch.fx.Node):
                visit(argument, closure, leaves)
            else:
                _literal(argument)
                if type(argument) is float and not math.isfinite(argument):
                    raise ValueError("nonfinite precompute literal")

    for name in output_nodes:
        if name not in nodes or nodes[name].op != "call_function":
            raise ValueError(f"selected node is not an operator: {name}")
        closure, leaves = set(), set()
        visit(nodes[name], closure, leaves)
        _alias_safe(nodes[name])
        for node in closure:
            if node.op == "placeholder":
                _alias_safe(node, fresh_result=False)
        closures[name], dependencies[name] = closure, leaves

    # Evaluate only proven constant dependencies, never the original full graph.
    all_nodes = set().union(*closures.values())
    values = {}
    with torch.inference_mode():
        for node in program.graph.nodes:
            if node not in all_nodes:
                continue
            if node.op == "placeholder":
                values[node] = program.state_dict[placeholders[node.name].target]
            else:
                args = tree_map(lambda item: values[item] if isinstance(item, torch.fx.Node) else item, node.args)
                kwargs = tree_map(lambda item: values[item] if isinstance(item, torch.fx.Node) else item, node.kwargs)
                values[node] = node.target(*args, **kwargs)

    control = _clone(program)
    folded = _clone(program)
    graph = folded.graph
    signature = copy.deepcopy(folded.graph_signature)
    state = dict(folded.state_dict)
    copied = {node.name: node for node in graph.nodes}
    existing = set(copied) | set(state)
    ledger = []
    for index, name in enumerate(output_nodes):
        output = values[nodes[name]]
        record = tensor_record(output)
        if _metadata(output) != _metadata(nodes[name].meta.get("val")):
            raise ValueError(f"computed output metadata differs: {name}")
        if not output.is_contiguous() or output.storage_offset() != 0:
            raise ValueError("selected precompute output must be fresh contiguous storage")
        target = f"_vlaforge_precomputed_{index}"
        while target in existing:
            target += "_"
        existing.add(target)
        position = next((i for i, spec in enumerate(signature.input_specs)
                         if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER)), len(signature.input_specs))
        before = next(node for node in graph.nodes if node.op != "placeholder") if position == len(signature.input_specs) else copied[signature.input_specs[position].arg.name]
        with graph.inserting_before(before):
            replacement = graph.placeholder(target)
        replacement.meta = dict(copied[name].meta)
        signature.input_specs.insert(position, InputSpec(InputKind.BUFFER, TensorArgument(replacement.name), target, True))
        copied[replacement.name] = replacement
        state[target] = output.detach().clone()
        copied[name].replace_all_uses_with(replacement)
        ledger.append({"output_node": name, "replacement_buffer": target,
                       "dependencies": {key: records[key] for key in sorted(dependencies[name])},
                       "operators": [{"node": node.name, "target": str(node.target),
                                      "args": _literal(node.args), "kwargs": _literal(node.kwargs)} for node in program.graph.nodes
                                     if node in closures[name] and node.op == "call_function"],
                       "output": record})
    removed = []
    for node in reversed(list(graph.nodes)):
        if node.name in {item.name for item in all_nodes} and node.op == "call_function" and not node.users:
            removed.append(node.name)
            graph.erase_node(node)
    graph.lint()
    folded = _clone(folded, graph, signature, state)
    _audit_program(folded)
    for target, expected in snapshot.tensor_sha256.items():
        if tensor_record(program.state_dict[target])["sha256"] != expected:
            raise ValueError("immutable state changed while precomputing")
    if graph_sha256(program) != original_hash or graph_sha256(control) != original_hash:
        raise AssertionError("precompute changed the original/control graph")
    return PrecomputeResult(control, folded, {
        "schema": "vlaforge.constant_precompute/1", "mechanism": "static_precompute",
        "snapshot": snapshot.to_dict(), "source_graph_sha256": original_hash,
        "control_graph_sha256": graph_sha256(control), "folded_graph_sha256": graph_sha256(folded),
        "folds": ledger, "removed_nodes": removed,
        "new_constant_bytes": sum(item["output"]["bytes"] for item in ledger),
        "original_state_retained": True, "original_selected_state_unchanged": True,
        "external_input_output_signature_unchanged": program.call_spec == folded.call_spec,
        "tensor_device_policy": "preserve", "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "implementation_sha256": file_sha256(Path(__file__)),
        "kernel_candidate_selected": False, "full_model_verified": False,
        "compiled_artifact_verified": False, "end_to_end_benchmark_completed": False,
    })


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_precompute_bundle(result: PrecomputeResult, directory: str | Path) -> dict[str, object]:
    """Write new, independent EP archives; original source files are not touched."""
    import torch

    def verify() -> None:
        for name, program in (("control", result.control), ("folded", result.folded)):
            if graph_sha256(program) != result.ledger[f"{name}_graph_sha256"]:
                raise ValueError(f"{name} graph changed before/during export")
            for target, expected in result.ledger["snapshot"]["tensor_sha256"].items():
                if tensor_record(program.state_dict[target])["sha256"] != expected:
                    raise ValueError(f"immutable state changed before/during export: {target}")
        for item in result.ledger["folds"]:
            actual = tensor_record(result.folded.state_dict[item["replacement_buffer"]])
            if actual != item["output"]:
                raise ValueError("precomputed buffer changed before/during export")

    verify()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ledger = copy.deepcopy(result.ledger)
    ledger["exported_artifacts"] = {}
    for name, program in (("control", result.control), ("folded", result.folded)):
        path = directory / f"{name}.pt2"
        torch.export.save(program, path)
        ledger["exported_artifacts"][name] = {"path": path.name, "bytes": path.stat().st_size,
                                              "sha256": file_sha256(path)}
    verify()
    ledger["pre_and_post_serialization_snapshot_verified"] = True
    (directory / "ledger.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    return ledger
