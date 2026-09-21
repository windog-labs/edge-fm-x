"""Separate EP and AOTI owners with complete, hash-bound invocation traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource

from vlaforge.adapters.openpi.openpi_aoti import _canonical, _capture_source, _region_result, _tuple, verify_compiled_region
from vlaforge.adapters.openpi.openpi_capture import _run_invocation, _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_phased import _native_output_transform, _require_original_worker_exited, process_identity, tensor_tree_metadata
from vlaforge.adapters.openpi.openpi_pruned import write
from vlaforge.analysis.numerical_probe import tensor_difference
from vlaforge.deployment.aoti_profile import aoti_configs
from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot

TRACE_SCHEMA = "vlaforge.openpi_complete_region_trace/1"


def _cpu_tree(value):
    import torch
    from torch.utils._pytree import tree_map
    return tree_map(lambda x: x.detach().to("cpu", copy=True) if isinstance(x, torch.Tensor) else x, value)


def _compare_tree(expected, actual):
    from torch.utils._pytree import tree_flatten
    left, left_tree = tree_flatten(expected)
    right, right_tree = tree_flatten(actual)
    if left_tree != right_tree or not left:
        raise ValueError("complete region Tensor tree differs")
    return [tensor_difference(a, b) for a, b in zip(left, right, strict=True)]


def _record(path):
    return {"path": str(Path(path).resolve()), **file_digest(Path(path))}


def _verified(record, root=None):
    path = Path(record["path"]).resolve()
    if (root is not None and not path.is_relative_to(root)) or _canonical(_record(path)) != _canonical(record):
        raise ValueError("trace/compile proof file identity differs")
    return path


def _trace_source(path, digest, capture, context):
    path = Path(path).resolve()
    if file_digest(path)["sha256"] != digest:
        raise ValueError("explicit trace report hash differs")
    row = json.loads(path.read_text())
    if (row.get("schema") != TRACE_SCHEMA or row.get("status") != "passed"
            or _canonical(row.get("source_report")) != _canonical(capture)
            or _canonical(row.get("numerical_context")) != _canonical(context.to_dict())
            or row.get("complete_bitwise_equal") is not True
            or row.get("caller_policy_restoration_verified") is not True
            or not row.get("calls")):
        raise ValueError("trace lacks actual complete same-context reference proof")
    _require_original_worker_exited(row["process_identity"])
    for index, call in enumerate(row["calls"]):
        if call.get("index") != index:
            raise ValueError("trace call ordering is incomplete")
        _verified(call["specimen"], path.parent)
    if file_digest(path.parent / "actions.npz") != row["actions"]:
        raise ValueError("complete trace output archive differs")
    return row


def _builds(paths, capture, context, regions, target):
    import torch
    compiled, records = {}, []
    for path in paths:
        path = Path(path).resolve()
        build = json.loads(path.read_text())
        if (build.get("status") != "compiled-unvalidated" or _canonical(build.get("source_report")) != _canonical(capture)
                or build.get("target") != target or build.get("torch") != torch.__version__ or build.get("cuda") != torch.version.cuda
                or _canonical(build.get("numerical_context")) != _canonical(context.to_dict())
                or build.get("caller_policy_restoration_verified") is not True
                or _canonical(build.get("configs")) != _canonical(aoti_configs(build["profile"]))):
            raise ValueError("compile run identity/profile/context differs")
        records.append(_record(path))
        for item in build["regions"]:
            name = item["region"]
            if name in compiled or name not in regions or item.get("status") != "compiled-unvalidated":
                raise ValueError("duplicate, undeclared or incomplete compiled Region")
            if item.get("observed_numerical_context_before") != context.to_dict() or item.get("observed_numerical_context_after") != context.to_dict():
                raise ValueError("actual compile boundaries differ from reference context")
            manifest = _verified(item["manifest"], path.parent)
            compiled[name] = verify_compiled_region(manifest, regions[name], target=target, profile=build["profile"])
    if set(compiled) != set(regions):
        raise ValueError("fresh pure-AOTI execution requires every Region")
    return compiled, records


def run(args):
    import numpy as np
    import torch
    import torch._inductor
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.frontend.invocation import InvocationProgram
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": TRACE_SCHEMA if args.phase == "trace" else "vlaforge.openpi_pruned_aoti_audit/1",
              "status": "started", "pid": os.getpid(), "process_identity": process_identity(), "calls": [],
              "source_implementation": _record(__file__), "no_python_deployment": False,
              "physical_units_verified": False, "robot_calibration_verified": False,
              "single_backend_weight_owner": args.phase, "discarded_warmup_calls": 0}
    device = None
    arrays = {}

    def save():
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device is not None and device.type == "cuda":
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write(output / "report.json", report)

    save()
    try:
        path, source, context, module, entries = _capture_source(args.capture_report, args.capture_sha256)
        device = _saved_device(source)
        if device.type != "cuda":
            raise ValueError("this audit requires the unchanged captured CUDA device")
        torch.cuda.reset_peak_memory_stats(device)
        target = "sm_%d%d" % torch.cuda.get_device_capability(device)
        capture = _record(path)
        report.update(source_report=capture, numerical_context=context.to_dict(), torch=torch.__version__, cuda=torch.version.cuda,
                      target=target, device=str(device), gpu=torch.cuda.get_device_name(device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      caller_context_before=snapshot().to_dict())
        trace_proof = None
        if args.phase == "audit":
            trace_proof = _trace_source(args.reference_trace, args.reference_trace_sha256, capture, context)
            report["reference_trace"] = _record(args.reference_trace)
            compiled, reports = _builds(args.compile_report, capture, context, entries, target)
            report.update(compile_reports=reports, compiled_regions=list(compiled), full_artifact_execution=True,
                          execution="all AOTI Regions in a fresh worker; canonical EP trajectory from a prior exited worker")
        with offline_restore(context, acknowledge_process_global=True), torch.inference_mode():
            if args.phase == "trace":
                backends = {name: torch.export.load(entry["archive"]["path"]).module() for name, entry in entries.items()}
            else:
                backends = {name: torch._inductor.aoti_load_package(item["artifact"]["path"]) for name, item in compiled.items()}
            with np.load(path.parent / "prepared_inputs.npz", allow_pickle=False) as bundle:
                tensors = {name: torch.from_numpy(bundle[name]).to(device) for name in bundle.files}

            def wrap(name):
                def call(*values):
                    index = len(report["calls"])
                    expected = None
                    if trace_proof is not None:
                        if index >= len(trace_proof["calls"]) or trace_proof["calls"][index]["region"] != name:
                            raise ValueError("actual Region invocation order differs from reference trace")
                        expected = torch.load(trace_proof["calls"][index]["specimen"]["path"], map_location="cpu", weights_only=True)
                    actual = _tuple(backends[name](*values))
                    row = {"index": index, "region": name, "inputs_actual_metadata": tensor_tree_metadata(values),
                           "outputs_actual_metadata": tensor_tree_metadata(actual)}
                    specimen_path = output / f"call-{index:03d}.pt"
                    torch.save({"args": _cpu_tree(values), "expected": _cpu_tree(actual)}, specimen_path)
                    row["specimen"] = _record(specimen_path)
                    if expected is not None:
                        row["reference_input_metrics"] = _compare_tree(expected["args"], values)
                        row["canonical_trajectory_output_metrics"] = _compare_tree(expected["expected"], actual)
                        row["reference_inputs_bitwise_equal"] = all(item["bitwise_equal"] for item in row["reference_input_metrics"])
                        row["same_input_output_comparison"] = row["reference_inputs_bitwise_equal"]
                    report["calls"].append(row)
                    save()
                    return _region_result(actual)
                return call

            bound = {name: wrap(name) for name in backends}
            actions, trace = _run_invocation(module, bound, InvocationProgram(module, bound).validators, tensors)
            if trace_proof is not None and len(report["calls"]) != len(trace_proof["calls"]):
                raise ValueError("actual trace lacks reference calls")
            normalized = actions.detach().cpu().numpy()
            native = _native_output_transform(source)({"state": tensors["state"][0].cpu().numpy(), "actions": normalized[0]})["actions"]
            with np.load(path.parent / "actions.npz", allow_pickle=False) as original:
                arrays.update(normalized_reference=original["normalized_reference"], native_reference=original["physical_reference"],
                              normalized_artifact=normalized, native_artifact=native)
            contract = NumericContract(**source["tolerances"])
            report["fidelity"] = [compare_action_chunk(a, b, sample_id="recorded-frame", space=space, contract=contract)
                                   for a, b, space in ((arrays["normalized_reference"], normalized, "normalized"), (arrays["native_reference"], native, "native-aloha-action-scale"))]
            report["complete_bitwise_equal"] = all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                                                    for a, b in ((arrays["normalized_reference"], normalized), (arrays["native_reference"], native)))
            context.require_current()
            report["context_before_guard_exit"] = snapshot().to_dict()
            trace.write(output / "invocation_trace.json")
            report["invocation_trace"] = _record(output / "invocation_trace.json")
        report.update(caller_policy_restoration_verified=True, caller_context_after_restore=snapshot().to_dict())
        np.savez(output / "actions.npz", **arrays)
        report["actions"] = file_digest(output / "actions.npz")
        if args.phase == "audit":
            np.savez(output / "actions_and_region_outputs.npz", **arrays)
            report["outputs"] = file_digest(output / "actions_and_region_outputs.npz")
            report["full_chunk_fidelity"] = report["fidelity"][0]
        passed = report["complete_bitwise_equal"] and all(call.get("reference_inputs_bitwise_equal", True)
                    and all(item["bitwise_equal"] for item in call.get("canonical_trajectory_output_metrics", [])) for call in report["calls"])
        report["status"] = "passed" if passed else "failed"
        if not passed:
            raise ValueError("complete action or canonical input/output trajectory bitwise gate failed")
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("trace", "audit"))
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path)
    parser.add_argument("--reference-trace-sha256")
    parser.add_argument("--compile-report", action="append", type=Path)
    args = parser.parse_args()
    if args.phase == "audit" and not (args.reference_trace and args.reference_trace_sha256 and args.compile_report):
        parser.error("pure AOTI audit requires explicit complete reference trace and compile reports")
    run(args)


if __name__ == "__main__":
    main()
