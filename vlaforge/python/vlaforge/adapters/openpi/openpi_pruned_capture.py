"""Join independently validated pruning evidence without upgrading its source.

This is an offline, hash-bound evidence derivation, not another model capture.
Large archives are immutable hardlinks; all consumers verify their complete hash.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_pruned import _validated_regions, reference

SCHEMA = "vlaforge.openpi_validated_pruned_capture/1"
PROOF_SCHEMA = "vlaforge.openpi_pruned_capture_derivation/1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(path):
    return {"path": str(Path(path).resolve()), **file_digest(path)}


def _verify_record(record):
    path = Path(record["path"])
    if not path.is_absolute() or path.is_symlink() or _canonical(_record(path)) != _canonical(record):
        raise ValueError("pruned derivation proof file hash/path mismatch")
    return path.resolve()


def _proof(proof):
    import numpy as np

    if set(proof) != {"schema", "original_reference", "regions_root", "whole_ir"} or proof["schema"] != PROOF_SCHEMA:
        raise ValueError("unknown or incomplete pruned derivation schema")
    path = _verify_record(proof["original_reference"])
    path, source, context, _, entries = reference(path, proof["original_reference"]["sha256"])
    root = Path(proof["regions_root"]).resolve()
    checked = _validated_regions(root, path, entries, context)
    whole_path = _verify_record(proof["whole_ir"])
    whole = json.loads(whole_path.read_text())
    if (whole.get("schema") != "vlaforge.openpi_pruned_whole_ir/1" or whole.get("status") != "passed"
            or whole.get("complete_bitwise_equal") is not True
            or _canonical(whole.get("original_reference")) != _canonical(proof["original_reference"])
            or _canonical(whole.get("regions")) != _canonical(checked)
            or _canonical(whole.get("numerical_context")) != _canonical(context.to_dict())
            or whole.get("no_python_deployment") is not False
            or whole.get("physical_units_verified") is not False
            or whole.get("robot_calibration_verified") is not False):
        raise ValueError("fresh complete whole-IR proof does not bind every pruned Region")
    for filename, digest in (("actions.npz", whole["actions"]), ("invocation_trace.json", whole["trace"])):
        if _canonical(file_digest(whole_path.parent / filename)) != _canonical(digest):
            raise ValueError("whole-IR complete outputs/trace hash differs")
    for filename, key in (("actions.npz", "actions"), ("prepared_inputs.npz", "prepared_inputs")):
        if _canonical(file_digest(path.parent / filename)) != _canonical(source[key]):
            raise ValueError("original complete inputs/reference changed")
    with np.load(path.parent / "actions.npz", allow_pickle=False) as original, np.load(whole_path.parent / "actions.npz", allow_pickle=False) as replay:
        if set(replay.files) != {"normalized", "native"}:
            raise ValueError("whole-IR output set differs")
        for expected, actual in (("normalized_reference", "normalized"), ("physical_reference", "native")):
            a, b = original[expected], replay[actual]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes() or not np.isfinite(a).all():
                raise ValueError("whole-IR complete output bytes do not match official reference")
    return path, source, checked, whole


def _join(source, checked, whole, proof, output):
    joined = deepcopy(source)
    joined.update(schema=SCHEMA, status="passed", source_evidence_level=source.get("evidence_level"),
                  evidence_level="derived unused-lifted-state pruning; independently reloaded Regions and fresh whole IR",
                  derivation=deepcopy(proof), no_python_deployment=False)
    capture = joined["capture"]
    capture.update(schema="vlaforge.openpi_pruned_capture/1", status="passed", saved_reload_verified=True,
                   execution="pruned independent Region replay and fresh complete Invocation IR", full_chunk_fidelity=whole["fidelity"][0])
    for entry in capture["regions"]:
        row = checked[entry["region"]]["record"]
        ledger = json.loads(Path(row["ledger"]["path"]).read_text())
        for field in ("source_graph_text_sha256", "result_graph_text_sha256"):
            graphs = ledger.get(field)
            if not isinstance(graphs, dict) or "" not in graphs or any(type(name) is not str for name in graphs):
                raise ValueError("pruning ledger lacks sealed graph/signature identity")
            for value in graphs.values():
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("pruning ledger lacks sealed graph/signature identity")
        signature = ledger.get("result_signature_sha256")
        if not isinstance(signature, str) or len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
            raise ValueError("pruning ledger lacks sealed graph/signature identity")
        graph_identity = hashlib.sha256(_canonical({"graphs": ledger["result_graph_text_sha256"], "signature": signature}).encode()).hexdigest()
        if not isinstance(ledger.get("retained_state"), dict) or not isinstance(ledger.get("retained_constants"), dict):
            raise ValueError("pruning ledger lacks retained-state identities")
        archive = output / "exported_regions" / (entry["region"] + ".pt2")
        example = output / "exported_regions" / (entry["region"] + ".examples.pt")
        entry["archive"] = {**row["pruned_archive"], "path": str(archive)}
        entry["examples"] = {**entry["examples"], "path": str(example)}
        entry["saved_reload_region_parity"] = "passed"
        original_evidence = deepcopy(entry["capture_evidence"])
        entry["source_capture_evidence"] = original_evidence
        evidence = entry["capture_evidence"]
        evidence.update(schema="vlaforge.openpi_pruned_capture_evidence/1",
                        graph_digest=graph_identity,
                        graph_digest_domain="sha256(canonical {graphs: sealed recursive result_graph_text_sha256, signature: result_signature_sha256}); not original frontend digest",
                        evidence_scope="original observed port/effect/context contract, preserved computation and serialized derived graph; no recapture",
                        pruning_ledger=deepcopy(row["ledger"]))
        entry["pruned_validation_report"] = deepcopy(checked[entry["region"]]["report"])
    return joined


def verify_pruned_capture_join(source, path):
    if source.get("schema") != SCHEMA:
        raise ValueError("not a supported derived pruned capture")
    output = Path(path).resolve().parent
    _, original, checked, whole = _proof(source["derivation"])
    expected = _join(original, checked, whole, source["derivation"], output)
    if _canonical(source) != _canonical(expected):
        raise ValueError("derived capture changed sealed provenance, contracts, graph or numerical context")
    for entry in source["capture"]["regions"]:
        for key in ("archive", "examples"):
            candidate = _verify_record(entry[key])
            if not candidate.is_relative_to(output):
                raise ValueError("derived archive escapes its directory")
    for filename, digest in (("prepared_inputs.npz", source["prepared_inputs"]), ("actions.npz", source["actions"]),
                             ("exported_regions/invocation_ir.json", source["capture"]["invocation_ir"])):
        candidate = output / filename
        if candidate.is_symlink() or _canonical(file_digest(candidate)) != _canonical(digest):
            raise ValueError("derived IR/input/reference file differs")


def prepare_pruned_capture(reference_report, reference_sha256, regions_root, whole_report, whole_sha256, output):
    reference_report, whole_report, output = Path(reference_report).resolve(), Path(whole_report).resolve(), Path(output).resolve()
    if file_digest(reference_report)["sha256"] != reference_sha256 or file_digest(whole_report)["sha256"] != whole_sha256:
        raise ValueError("explicit source proof hashes differ")
    proof = {"schema": PROOF_SCHEMA, "original_reference": _record(reference_report),
             "regions_root": str(Path(regions_root).resolve()), "whole_ir": _record(whole_report)}
    path, source, checked, whole = _proof(proof)
    output.mkdir(parents=True, exist_ok=False)
    (output / "exported_regions").mkdir()
    joined = _join(source, checked, whole, proof, output)
    for filename in ("prepared_inputs.npz", "actions.npz", "exported_regions/invocation_ir.json"):
        shutil.copyfile(path.parent / filename, output / filename)
    for entry in joined["capture"]["regions"]:
        row = checked[entry["region"]]["record"]
        os.link(row["pruned_archive"]["path"], entry["archive"]["path"])
        shutil.copyfile(row["examples"]["path"], entry["examples"]["path"])
    target = output / "capture.json"
    verify_pruned_capture_join(joined, target)
    temporary = output / "capture.pending"
    with temporary.open("x") as stream:
        stream.write(json.dumps(joined, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, target)
    temporary.unlink()
    return _record(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--regions-root", type=Path, required=True)
    parser.add_argument("--whole-report", type=Path, required=True)
    parser.add_argument("--whole-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare_pruned_capture(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
