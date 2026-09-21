"""Remove unused lifted state without pruning computation or user inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def _file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _graphs(program):
    import torch

    return {
        name: module
        for name, module in program.graph_module.named_modules()
        if isinstance(module, torch.fx.GraphModule)
    }


def _graph_hashes(program):
    return {
        name: hashlib.sha256(str(module.graph).encode()).hexdigest()
        for name, module in _graphs(program).items()
    }


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _signature_record(value):
    from torch.utils import _pytree

    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "name": value.name}
    if isinstance(value, _pytree.TreeSpec):
        return {"treespec": json.loads(_pytree.treespec_dumps(value))}
    if is_dataclass(value):
        return {
            "type": type(value).__qualname__,
            "fields": {
                item.name: _signature_record(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if type(value) in (tuple, list):
        return {
            "type": type(value).__name__,
            "items": [_signature_record(x) for x in value],
        }
    raise TypeError(f"unsupported exported signature value: {type(value).__qualname__}")


def _signature_hash(program):
    if program.range_constraints:
        raise ValueError("state pruning currently requires a static profile")
    return _canonical_hash(
        {
            "graph_signature": _signature_record(program.graph_signature),
            "module_call_graph": _signature_record(program.module_call_graph),
            "call_spec": _signature_record(tuple(program.call_spec)),
        }
    )


def _tensor_record(value):
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.is_quantized
        or value.is_complex()
        or any(type(n) is not int for n in (*value.shape, *value.stride()))
    ):
        raise ValueError("state pruning requires ordinary static strided tensors")
    owned = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
    owned.copy_(value.detach())
    raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "logical_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _storage_inventory(program):
    import torch

    groups = {}
    for prefix, values in (
        ("state", program.state_dict),
        ("constant", program.constants),
    ):
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("untyped/custom state constants are unsupported")
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            group = groups.setdefault(key, {"bytes": storage.nbytes(), "names": []})
            group["names"].append(prefix + ":" + name)
    return sorted(groups.values(), key=lambda group: tuple(sorted(group["names"])))


def _reject_unlifted_state(program):
    import torch

    for module in _graphs(program).values():
        for node in module.graph.nodes:
            if node.op == "call_module":
                raise ValueError(
                    "unlifted module calls require an explicit state contract"
                )
            if node.op != "get_attr":
                continue
            value = module
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.fx.GraphModule):
                raise TypeError(
                    "unlifted graph attributes require an explicit state contract"
                )


@dataclass(frozen=True)
class PrunedState:
    program: Any
    ledger: dict
    _ledger_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_ledger_sha256", _canonical_hash(self.ledger))


def prune_unused_state(program, *, source_artifact_sha256):
    """Return a new EP, preserving every computation and user-visible port.

    Unused parameter/persistent-buffer placeholders and orphan state-table entries
    are removable. Unlifted graph attributes are rejected. A state tensor remains
    if any retained signature entry names it. The caller must
    keep state immutable during transformation/serialization and verify the
    source archive digest before supplying a loaded program. The source EP is
    not retained by the result; release it separately when measuring memory.
    """
    import torch
    from torch.export.graph_signature import InputKind

    from vlaforge.frontend.effect_audit import audit_exported_program

    if (
        not isinstance(source_artifact_sha256, str)
        or len(source_artifact_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_artifact_sha256)
    ):
        raise ValueError("source artifact SHA256 is required")
    if program.range_constraints:
        raise ValueError("state pruning currently requires a static profile")
    audit = audit_exported_program(program)
    if not audit.passed:
        raise ValueError("state pruning requires a pure exported program")
    _reject_unlifted_state(program)
    source_graphs = _graph_hashes(program)
    original_records = {
        name: _tensor_record(value) for name, value in program.state_dict.items()
    }
    constant_records = {
        name: _tensor_record(value) for name, value in program.constants.items()
    }
    original_storage = _storage_inventory(program)
    placeholders = {
        node.name: node for node in program.graph.nodes if node.op == "placeholder"
    }
    removable = []
    for spec in program.graph_signature.input_specs:
        state = spec.kind == InputKind.PARAMETER or (
            spec.kind == InputKind.BUFFER and spec.persistent is True
        )
        if not state:
            continue
        if spec.arg.name not in placeholders or spec.target not in program.state_dict:
            raise ValueError("lifted state does not match its signature/storage")
        if not placeholders[spec.arg.name].users:
            removable.append(spec)
    removed_nodes = {spec.arg.name for spec in removable}
    orphan_targets = set(program.state_dict) - {
        spec.target
        for spec in program.graph_signature.input_specs
        if spec.target is not None
    }
    for entry in program.module_call_graph:
        if entry.signature is not None:
            arguments = (*entry.signature.inputs, *entry.signature.outputs)
            if any(
                getattr(argument, "name", None) in removed_nodes | orphan_targets
                for argument in arguments
            ):
                raise ValueError(
                    "removable state is referenced by a module-call signature"
                )
    signature = copy.deepcopy(program.graph_signature)
    signature.input_specs = [
        spec for spec in signature.input_specs if spec.arg.name not in removed_nodes
    ]
    retained_targets = {
        spec.target for spec in signature.input_specs if spec.target is not None
    }
    removed_targets = (
        {spec.target for spec in removable} - retained_targets
    ) | orphan_targets
    state = {
        name: value
        for name, value in program.state_dict.items()
        if name not in removed_targets
    }
    graph = torch.fx.Graph()
    environment = {}
    for node in program.graph.nodes:
        if node.name in removed_nodes:
            continue
        copied = graph.node_copy(node, lambda item: environment[item])
        copied.meta = dict(node.meta)
        if "custom" in copied.meta:
            copied.meta["custom"] = copy.deepcopy(copied.meta["custom"])
        environment[node] = copied
    graph.lint()
    candidate = torch.export.ExportedProgram(
        root=program.graph_module,
        graph=graph,
        graph_signature=signature,
        state_dict=state,
        range_constraints=copy.deepcopy(program.range_constraints),
        module_call_graph=copy.deepcopy(program.module_call_graph),
        example_inputs=program.example_inputs,
        constants=dict(program.constants),
        verifiers=program.verifiers,
    )
    candidate.validate()
    if (
        program.call_spec != candidate.call_spec
        or program.graph_signature.output_specs
        != candidate.graph_signature.output_specs
    ):
        raise ValueError("state pruning changed user-visible input/output structure")
    if _graph_hashes(program) != source_graphs:
        raise ValueError("source graphs changed while pruning state")
    if _graph_hashes(candidate).keys() != source_graphs.keys() or any(
        value != _graph_hashes(candidate)[name]
        for name, value in source_graphs.items()
        if name
    ):
        raise ValueError("nested computation changed while pruning state")
    kept_records = {
        name: _tensor_record(value) for name, value in candidate.state_dict.items()
    }
    if kept_records != {
        name: value
        for name, value in original_records.items()
        if name not in removed_targets
    }:
        raise ValueError("retained state changed while pruning")
    after_storage = _storage_inventory(candidate)
    ledger = {
        "schema": "vlaforge.unused_lifted_state_pruning/2",
        "source_artifact_sha256": source_artifact_sha256,
        "source_graph_text_sha256": source_graphs,
        "result_graph_text_sha256": _graph_hashes(candidate),
        "result_signature_sha256": _signature_hash(candidate),
        "removed_placeholders": [
            {"name": spec.arg.name, "target": spec.target, "kind": spec.kind.name}
            for spec in removable
        ],
        "removed_orphan_state_targets": sorted(orphan_targets),
        "removed_state": {
            name: original_records[name] for name in sorted(removed_targets)
        },
        "retained_state": kept_records,
        "retained_constants": constant_records,
        "source_storage_groups": original_storage,
        "result_storage_groups": after_storage,
        "source_unique_storage_bytes": sum(
            group["bytes"] for group in original_storage
        ),
        "result_unique_storage_bytes": sum(group["bytes"] for group in after_storage),
        "computation_nodes_removed": 0,
        "user_ports_unchanged": True,
        "retained_tensors_shared_with_source": True,
        "source_archive_modified": False,
        "runtime_peak_reduction_verified": False,
        "full_model_output_verified": False,
    }
    return PrunedState(candidate, ledger)


def save_pruned_state(result, directory):
    """Save into a new directory and bind the resulting archive to its ledger."""
    import torch

    def verify():
        if _canonical_hash(result.ledger) != result._ledger_sha256:
            raise ValueError("pruning ledger changed before/during serialization")
        if _signature_hash(result.program) != result.ledger["result_signature_sha256"]:
            raise ValueError("pruned signature changed before/during serialization")
        if _graph_hashes(result.program) != result.ledger["result_graph_text_sha256"]:
            raise ValueError("pruned graph changed before/during serialization")
        records = {
            name: _tensor_record(value)
            for name, value in result.program.state_dict.items()
        }
        constants = {
            name: _tensor_record(value)
            for name, value in result.program.constants.items()
        }
        if (
            records != result.ledger["retained_state"]
            or constants != result.ledger["retained_constants"]
            or _storage_inventory(result.program)
            != result.ledger["result_storage_groups"]
        ):
            raise ValueError("retained state changed before/during serialization")

    verify()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "program.pt2"
    torch.export.save(result.program, target)
    verify()
    ledger = copy.deepcopy(result.ledger)
    ledger["exported_artifact"] = {
        "path": target.name,
        "sha256": _file_sha(target),
        "bytes": target.stat().st_size,
    }
    ledger["pre_and_post_serialization_state_verified"] = True
    (directory / "ledger.json").write_text(
        json.dumps(ledger, indent=2, allow_nan=False) + "\n"
    )
    return ledger
