"""Derived OpenPI evidence with one original Region weight owner per process.

State pruning is the shared analysis pass. This orchestration preserves the
original capture and admits a derived chain only after complete real replay.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource

from vlaforge.adapters.openpi.openpi_capture import _run_invocation, _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_phased import (
    _native_output_transform, _require_original_worker_exited, _verified_file,
    tensor_tree_metadata,
)
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot

REGION_SCHEMA = "vlaforge.openpi_pruned_region/1"


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(path, expected):
    path = Path(path).resolve()
    if file_digest(path)["sha256"] != expected:
        raise ValueError("original persisted reference report hash differs")
    source = json.loads(path.read_text())
    capture = source.get("capture", {})
    if (source.get("status") != "persisted" or capture.get("schema") != "vlaforge.openpi_persisted_capture/1"
            or capture.get("status") != "persisted_awaiting_independent_reload"
            or capture.get("saved_reload_verified") is not False):
        raise ValueError("derived validation requires original incomplete persisted capture")
    _require_original_worker_exited(source["process_identity"])
    context = NumericalContext.from_dict(source["numerical_context"])
    if context.partial or context.to_dict() != capture["numerical_context"]:
        raise ValueError("new derivation requires matching actual complete v2 context")
    ir = path.parent / "exported_regions/invocation_ir.json"
    if file_digest(ir) != capture["invocation_ir"]:
        raise ValueError("original invocation IR differs")
    module = parse_canonical_json(ir.read_text())
    entries = {item["region"]: item for item in capture["regions"]}
    if len(entries) != len(capture["regions"]) or set(entries) != {r.name for r in module.regions}:
        raise ValueError("original capture must contain each region exactly once")
    for entry in entries.values():
        if entry["supported"] is not True or entry["saved_reload_region_parity"] != "not-run":
            raise ValueError("unsupported or already-upgraded original capture")
    return path, source, context, module, entries


def _compare(reference_value, candidate):
    import torch
    from torch.utils._pytree import tree_flatten
    from vlaforge.analysis.numerical_probe import tensor_difference

    left, structure = tree_flatten(reference_value)
    right, candidate_structure = tree_flatten(candidate)
    if structure != candidate_structure or not left or not all(isinstance(x, torch.Tensor) for x in (*left, *right)):
        raise ValueError("complete Tensor output structure changed")
    result = []
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0, check_stride=True)
        row = tensor_difference(a, b)
        if not row["bitwise_equal"]:
            raise ValueError("complete output storage bits changed")
        result.append(row)
    return result


def prune_region(args):
    import torch
    from vlaforge.analysis.unused_state import prune_unused_state, save_pruned_state

    path, source, context, _, entries = reference(args.reference_report, args.reference_sha256)
    entry = entries[args.region]
    archive = _verified_file(entry["archive"], path.parent)
    examples = _verified_file(entry["examples"], path.parent)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = _saved_device(source)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report = {"schema": REGION_SCHEMA, "status": "started", "pid": os.getpid(), "region": args.region,
              "original_reference": {"path": str(path), **file_digest(path)}, "archive": entry["archive"],
              "examples": entry["examples"], "numerical_context": context.to_dict(),
              "source_sha256": file_digest(Path(__file__))["sha256"],
              "pass_source_sha256": file_digest(Path(__file__).parents[2] / "analysis/unused_state.py")["sha256"],
              "caller_context_before": snapshot().to_dict(), "full_model_output_verified": False,
              "no_python_deployment": False, "device_remap": False}

    def save():
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device.type == "cuda":
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write(output / "report.json", report)

    save()
    try:
        with offline_restore(context, acknowledge_process_global=True), torch.inference_mode():
            specimen = torch.load(examples, weights_only=True)
            if tensor_tree_metadata(specimen) != entry["example_metadata"]:
                raise ValueError("original Tensor dtype/device/stride/offset changed")
            original = torch.export.load(archive)
            actual = original.module()(*specimen["args"])
            report["original_saved_vs_captured_example"] = _compare(specimen["expected"], actual)
            result = prune_unused_state(original, source_artifact_sha256=entry["archive"]["sha256"])
            candidate = result.program.module()(*specimen["args"])
            report["in_memory_pruned_vs_captured_example"] = _compare(specimen["expected"], candidate)
            ledger = save_pruned_state(result, output / "pruned")
            report["ledger"] = {"path": str(output / "pruned/ledger.json"), **file_digest(output / "pruned/ledger.json")}
            report["pruned_archive"] = {"path": str(output / "pruned/program.pt2"), **file_digest(output / "pruned/program.pt2")}
            report["source_unique_storage_bytes"] = ledger["source_unique_storage_bytes"]
            report["pruned_unique_storage_bytes"] = ledger["result_unique_storage_bytes"]
            if device.type == "cuda":
                report["allocated_before_release"] = torch.cuda.memory_allocated(device)
            del candidate, actual, result, original
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                report["allocated_after_release_before_reload"] = torch.cuda.memory_allocated(device)
            save()
            restored = torch.export.load(output / "pruned/program.pt2")
            reloaded = restored.module()(*specimen["args"])
            report["saved_pruned_vs_captured_example"] = _compare(specimen["expected"], reloaded)
            context.require_current()
            torch.save({"expected": specimen["expected"], "reloaded": reloaded}, output / "outputs.pt")
            report["outputs"] = file_digest(output / "outputs.pt")
        if file_digest(archive) != {k: v for k, v in entry["archive"].items() if k != "path"}:
            raise ValueError("original archive changed")
        report.update(status="passed", caller_context_after=snapshot().to_dict())
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def _validated_regions(root, source_path, entries, context):
    root = Path(root).resolve()
    result = {}
    for name, entry in entries.items():
        record_path = root / name / "report.json"
        row = json.loads(record_path.read_text())
        if (row.get("schema") != REGION_SCHEMA or row.get("status") != "passed" or row.get("region") != name
                or row["original_reference"] != {"path": str(source_path), **file_digest(source_path)}
                or row["archive"] != entry["archive"] or row["examples"] != entry["examples"]
                or row["numerical_context"] != context.to_dict()):
            raise ValueError("pruned region report does not bind original capture/context")
        archive = _verified_file(row["pruned_archive"], root)
        ledger_path = _verified_file(row["ledger"], root)
        ledger = json.loads(ledger_path.read_text())
        if (ledger.get("schema") != "vlaforge.unused_lifted_state_pruning/2"
                or ledger.get("source_artifact_sha256") != entry["archive"]["sha256"]
                or ledger.get("pre_and_post_serialization_state_verified") is not True
                or ledger.get("source_archive_modified") is not False
                or ledger.get("user_ports_unchanged") is not True
                or ledger.get("computation_nodes_removed") != 0
                or ledger.get("exported_artifact") != {"path": archive.name, "sha256": row["pruned_archive"]["sha256"], "bytes": archive.stat().st_size}):
            raise ValueError("pruning transformation ledger is incomplete or mismatched")
        for key in ("original_saved_vs_captured_example", "in_memory_pruned_vs_captured_example", "saved_pruned_vs_captured_example"):
            if not row.get(key) or not all(item.get("bitwise_equal") is True for item in row[key]):
                raise ValueError("pruning lacks complete example validation")
        if file_digest(record_path.parent / "outputs.pt") != row["outputs"]:
            raise ValueError("pruned complete example output archive changed")
        result[name] = {"record": row, "report": {"path": str(record_path), **file_digest(record_path)}}
    return result


def validate_whole(args):
    import numpy as np
    import torch
    from vlaforge.frontend.invocation import InvocationProgram
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    path, source, context, module, entries = reference(args.reference_report, args.reference_sha256)
    checked = _validated_regions(args.regions_root, path, entries, context)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vlaforge.openpi_pruned_whole_ir/1", "status": "started", "pid": os.getpid(),
              "original_reference": {"path": str(path), **file_digest(path)}, "regions": checked,
              "numerical_context": context.to_dict(), "source_sha256": file_digest(Path(__file__))["sha256"],
              "no_python_deployment": False, "physical_units_verified": False, "robot_calibration_verified": False}
    device = _saved_device(source)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save():
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device.type == "cuda":
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write(output / "report.json", report)

    save()
    try:
        for filename, key in (("prepared_inputs.npz", "prepared_inputs"), ("actions.npz", "actions")):
            if file_digest(path.parent / filename) != source[key]:
                raise ValueError("original complete inputs/actions changed")
        with offline_restore(context, acknowledge_process_global=True), torch.inference_mode():
            regions = {name: torch.export.load(row["record"]["pruned_archive"]["path"]).module() for name, row in checked.items()}
            with np.load(path.parent / "prepared_inputs.npz", allow_pickle=False) as bundle:
                tensors = {name: torch.from_numpy(bundle[name]).to(device) for name in bundle.files}
            actions, trace = _run_invocation(module, regions, InvocationProgram(module, regions).validators, tensors)
            context.require_current()
            normalized = actions.detach().cpu().numpy()
            native = _native_output_transform(source)({"state": tensors["state"][0].cpu().numpy(), "actions": normalized[0]})["actions"]
            with np.load(path.parent / "actions.npz", allow_pickle=False) as bundle:
                original = bundle["normalized_reference"]
                original_native = bundle["physical_reference"]
            if normalized.dtype != original.dtype or normalized.shape != original.shape or normalized.tobytes() != original.tobytes():
                raise ValueError("complete normalized action storage changed")
            if native.dtype != original_native.dtype or native.shape != original_native.shape or native.tobytes() != original_native.tobytes():
                raise ValueError("complete native action storage changed")
            contract = NumericContract(**source["tolerances"])
            report["fidelity"] = [compare_action_chunk(a, b, sample_id="recorded-frame", space=space, contract=contract)
                                   for a, b, space in ((original, normalized, "normalized"), (original_native, native, "native-aloha-action-scale"))]
            report["complete_bitwise_equal"] = True
            np.savez(output / "actions.npz", normalized=normalized, native=native)
            report["actions"] = file_digest(output / "actions.npz")
            trace.write(output / "invocation_trace.json")
            report["trace"] = file_digest(output / "invocation_trace.json")
        report.update(status="passed", caller_context_after=snapshot().to_dict())
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("region", "whole"))
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region")
    parser.add_argument("--regions-root", type=Path)
    args = parser.parse_args()
    if args.phase == "region":
        if not args.region:
            parser.error("region phase requires an explicit region")
        prune_region(args)
    else:
        if args.regions_root is None:
            parser.error("whole phase requires independently verified region reports")
        validate_whole(args)


if __name__ == "__main__":
    main()
