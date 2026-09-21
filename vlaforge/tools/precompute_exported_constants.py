"""Inspect state-only candidates without loading tensors, or write new EPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

from vlaforge.analysis.constant_precompute import _PURE_TARGETS, file_sha256


def inspect_archive(path, output_nodes):
    """Read Torch serde v8 structure only; this is not a tensor/effect audit."""
    if not output_nodes or len(set(output_nodes)) != len(output_nodes):
        raise ValueError("selected nodes must be nonempty and unique")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if
                   item.filename.endswith("/models/model.json") or
                   item.filename == "serialized_exported_program.json"]
        if len(members) != 1 or members[0].file_size > 128 * 1024 * 1024:
            raise ValueError("expected one bounded exported-program JSON record")
        raw = archive.read(members[0])
    payload = json.loads(raw)
    if payload["schema_version"]["major"] != 8:
        raise ValueError("offline inspection supports only Torch serde major 8")
    if payload["range_constraints"]:
        raise ValueError("offline candidate requires a static exported profile")
    module = payload["graph_module"]
    producers = {}
    for node in module["graph"]["nodes"]:
        for output in node["outputs"]:
            if "as_tensor" in output:
                name = output["as_tensor"]["name"]
                if name in producers:
                    raise ValueError("duplicate serialized tensor producer")
                producers[name] = node
    leaves = {}
    for spec in module["signature"]["input_specs"]:
        for kind, value in spec.items():
            argument = value.get("arg", {})
            if "name" in argument:
                leaves[argument["name"]] = (kind, value)
            elif "as_tensor" in argument:
                leaves[argument["as_tensor"]["name"]] = (kind, value)
    records = []
    for root in output_nodes:
        queue, visited, selected, dependencies = [root], set(), [], {}
        while queue:
            name = queue.pop()
            if name in visited:
                continue
            visited.add(name)
            if name in leaves:
                kind, value = leaves[name]
                if kind != "parameter" and not (kind == "buffer" and value.get("persistent") is True):
                    raise ValueError(f"user input or nonpersistent dependency: {name}")
                target = value["parameter_name" if kind == "parameter" else "buffer_name"]
                dependencies[target] = {"kind": kind, "placeholder": name}
                continue
            if name not in producers:
                raise ValueError(f"unknown serialized dependency: {name}")
            node = producers[name]
            if node["target"].removeprefix("torch.ops.") not in _PURE_TARGETS:
                raise ValueError(f"serialized operator is not pure-allowlisted: {node['target']}")
            selected.append({"output": name, "target": node["target"], "inputs": node["inputs"]})
            for argument in node["inputs"]:
                value = argument["arg"]
                if set(value) == {"as_tensor"}:
                    queue.append(value["as_tensor"]["name"])
                elif set(value) - {"as_int", "as_ints", "as_float", "as_floats", "as_bool", "as_bools",
                                  "as_none", "as_string", "as_strings", "as_scalar_type", "as_device",
                                  "as_layout", "as_memory_format"}:
                    raise ValueError(f"unknown/symbolic serialized argument: {value}")
        if root not in producers:
            raise ValueError("selected root must be a tensor operator output")
        records.append({"output_node": root, "dependencies": dependencies, "operators": selected})
    return {"schema": "vlaforge.constant_precompute_inspection/1", "status": "structural_candidate_only",
            "source_artifact_sha256": file_sha256(path), "source_artifact": str(Path(path).resolve()),
            "serialized_graph_record_sha256": hashlib.sha256(raw).hexdigest(),
            "serde_version": payload["schema_version"], "torch_version": payload["torch_version"],
            "candidates": records, "tensor_storage_loaded": False,
            "dependency_tensor_hashes_verified": False, "effect_and_alias_audit_completed": False,
            "constant_precompute_executed": False, "full_model_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "precompute"))
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--snapshot-id")
    declaration = parser.add_mutually_exclusive_group()
    declaration.add_argument("--immutable-state-target", action="append")
    declaration.add_argument("--literal-only", action="store_true",
                             help="Approve no state/input dependency; fold only explicit literal expressions")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "started.json").write_text(json.dumps({"pid": os.getpid(), "mode": args.mode,
        "command": list(sys.argv if argv is None else argv), "tool_sha256": file_sha256(Path(__file__))}, indent=2) + "\n")
    try:
        report = inspect_archive(args.program, args.node)
        report["command"] = list(sys.argv if argv is None else argv)
        report["tool_sha256"] = file_sha256(Path(__file__))
        if args.expected_source_sha256 and report["source_artifact_sha256"] != args.expected_source_sha256:
            raise ValueError("source archive SHA256 mismatch")
        (args.output / "inspection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.mode == "precompute":
            if (not args.expected_source_sha256 or not args.snapshot_id
                    or not (args.immutable_state_target or args.literal_only)):
                raise ValueError("precompute requires expected source SHA, snapshot identity and explicit immutable targets")
            import torch
            from vlaforge.analysis.constant_precompute import (
                declare_immutable_snapshot,
                precompute_constants,
                save_precompute_bundle,
            )

            program = torch.export.load(args.program)
            snapshot = declare_immutable_snapshot(program, args.immutable_state_target or (),
                snapshot_id=args.snapshot_id, source_artifact_sha256=args.expected_source_sha256,
                literal_only=args.literal_only)
            result = precompute_constants(program, output_nodes=args.node, snapshot=snapshot)
            ledger = save_precompute_bundle(result, args.output / "exports")
            report = {"status": "new_exported_programs_written", "ledger": ledger,
                      "tool_sha256": report["tool_sha256"], "command": report["command"],
                      "source_artifact": report["source_artifact"]}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return 0
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"error_type": type(error).__qualname__,
            "error": str(error), "traceback": traceback.format_exc(),
            "mode": args.mode, "source_artifact": str(args.program),
            "tool_sha256": file_sha256(Path(__file__)), "full_model_verified": False}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
