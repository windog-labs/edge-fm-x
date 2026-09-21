"""Actual OpenPI export persistence and full Invocation IR replay, never a C++ claim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_frontend import capture_openpi_frontend
from vlaforge.interpreter import Interpreter, TensorView
from vlaforge.ir.serializer import canonical_json, parse_canonical_json
from vlaforge.validation.deployment_metrics import compare_action_chunk
from vlaforge.validation.contracts import NumericContract
from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot


def _saved_device(source):
    import torch

    value = source.get("device")
    if not isinstance(value, str) or re.fullmatch(r"cpu|cuda:[0-9]+", value) is None:
        raise ValueError("saved reference requires an explicit CPU or CUDA device")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError("saved CUDA device is unavailable in this process")
        torch.cuda.set_device(device)
    return device


def _run_invocation(module, regions, validators, tensors):
    import torch

    interpreter = Interpreter(module, regions=regions, validators=validators)
    for name, value in tensors.items():
        port = module.input(name)
        interpreter.bind_input(
            name,
            TensorView(
                value,
                tuple(value.shape),
                port.payload.dtype,
                port.payload.layout,
                str(value.device),
                port.alignment,
            ),
        )
    with torch.inference_mode():
        result = interpreter.run()
        actions = interpreter.read_output("normalized_action_chunk")
    return actions, result.trace


def capture_and_reload_openpi(frontend, prepared, output_dir, *, contract, on_progress):
    """Capture, persist and reload every real region, then execute all action steps."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    ir_path = output / "invocation_ir.json"
    ir_path.write_text(canonical_json(frontend.program.module, indent=2) + "\n")
    report = {
        "status": "capturing",
        "regions": [],
        "execution": "Python Invocation IR interpreter with reloaded torch.export regions; not no-Python deployment",
        "invocation_ir": file_digest(ir_path),
        "numerical_context": snapshot().to_dict(),
        "numerical_context_scope": "Python observation and checks; not C++ enforcement",
    }
    on_progress(report)
    outcomes = capture_openpi_frontend(
        frontend,
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    loaded_regions = {}
    for outcome in outcomes:
        region = outcome.region.name
        entry = {
            "region": region,
            "supported": outcome.supported,
            "unsupported_report": outcome.report.to_dict(),
        }
        report["regions"].append(entry)
        on_progress(report)
        if not outcome.supported:
            continue
        entry["capture_evidence"] = outcome.evidence.to_dict()
        archive = output / f"{region}.pt2"
        torch.export.save(outcome.exported_program, archive)
        entry["archive"] = {"path": str(archive), **file_digest(archive)}
        (output / f"{region}.graph.txt").write_text(
            outcome.exported_program.graph_module.code
        )
        module = torch.export.load(archive).module()
        with torch.inference_mode():
            original = frontend.program.regions[region](*frontend.example_args[region])
            restored = module(*frontend.example_args[region])
            torch.testing.assert_close(
                original,
                restored,
                atol=contract.absolute_tolerance,
                rtol=contract.relative_tolerance,
            )
        entry["saved_reload_region_parity"] = "passed"
        loaded_regions[region] = module
        on_progress(report)
    if len(loaded_regions) != len(outcomes):
        report["status"] = "unsupported"
        on_progress(report)
        return report, None
    report["status"] = "reloaded_invocation_running"
    on_progress(report)
    actions, trace = _run_invocation(
        parse_canonical_json(ir_path.read_text()),
        loaded_regions,
        frontend.program.validators,
        prepared.tensors,
    )
    trace.write(output / "reloaded_invocation_trace.json")
    report["full_chunk_fidelity"] = compare_action_chunk(
        frontend.normalized_reference,
        actions,
        sample_id="recorded-frame",
        space="normalized",
        contract=contract,
    )
    report["status"] = (
        "passed"
        if report["full_chunk_fidelity"]["metrics"]["within_tolerance"]
        else "failed"
    )
    on_progress(report)
    return report, actions


def replay_saved_capture(
    capture_report,
    output_dir,
    *,
    ir_sha256,
    numerical_context=None,
    allow_legacy_context=False,
):
    """Restore the recorded arithmetic policy as well as serialized graph data."""
    source = json.loads(Path(capture_report).read_text())
    recorded = source.get("numerical_context")
    supplied = (
        NumericalContext.from_json(Path(numerical_context).read_text())
        if numerical_context is not None
        else None
    )
    if recorded is None:
        if supplied is None or not allow_legacy_context:
            raise ValueError(
                "legacy capture requires an explicit complete numerical context "
                "and allow_legacy_context; missing fields are never defaulted"
            )
        selected = supplied
        legacy_precision = source.get("numerical_execution", {}).get(
            "torch_float32_matmul_precision"
        )
        if (
            legacy_precision is not None
            and legacy_precision != selected.float32_matmul_precision
        ):
            raise ValueError(
                "supplied context differs from the legacy recorded precision"
            )
    else:
        selected = NumericalContext.from_dict(recorded)
        if supplied is not None and supplied != selected:
            raise ValueError(
                "supplied numerical context differs from the saved reference"
            )
    evidence = {
        "context": selected.to_dict(),
        "origin": "recorded-reference"
        if recorded is not None
        else "explicit-complete-context-for-legacy-report",
        "provided_context_file": (
            file_digest(numerical_context) if numerical_context is not None else None
        ),
        "scope": "offline Python process-global and current-thread restoration; not concurrency safe or C++ enforcement",
    }
    with offline_restore(selected, acknowledge_process_global=True):
        report = _replay_saved_capture(
            capture_report,
            output_dir,
            ir_sha256=ir_sha256,
            numerical_execution=evidence,
        )
    report["numerical_execution"]["caller_policy_restoration_verified"] = True
    report["numerical_execution"]["caller_context_after_restore"] = snapshot().to_dict()
    (Path(output_dir) / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _replay_saved_capture(
    capture_report, output_dir, *, ir_sha256, numerical_execution
):
    """Independently replay verified saved regions without constructing OpenPI."""
    import numpy as np
    import torch

    from vlaforge.frontend.invocation import InvocationProgram

    path = Path(capture_report).resolve()
    source = json.loads(path.read_text())
    device = _saved_device(source)
    base = path.parent
    ir_path = base / "exported_regions/invocation_ir.json"
    ir_digest = file_digest(ir_path)
    if ir_digest["sha256"] != ir_sha256:
        raise ValueError("serialized Invocation IR hash mismatch")
    module = parse_canonical_json(ir_path.read_text())
    entries = source["capture"]["regions"]
    if len(entries) != len(module.regions) or {item["region"] for item in entries} != {
        region.name for region in module.regions
    }:
        raise ValueError(
            "saved capture must provide every declared region exactly once"
        )
    regions = {}
    for entry in entries:
        if (
            not entry["supported"]
            or entry.get("saved_reload_region_parity") != "passed"
        ):
            raise ValueError(
                "every saved region must have prior independent capture and reload parity"
            )
        archive = Path(entry["archive"]["path"]).resolve()
        if not archive.is_relative_to(base):
            raise ValueError("saved archive escapes the captured run directory")
        expected = {
            key: value for key, value in entry["archive"].items() if key != "path"
        }
        if file_digest(archive) != expected:
            raise ValueError("saved region archive hash mismatch")
        restored = torch.export.load(archive).module()
        if any(value.device != device for value in restored.parameters()):
            raise ValueError("saved region parameters differ from recorded device")
        regions[entry["region"]] = restored
    for name, key in (
        ("prepared_inputs.npz", "prepared_inputs"),
        ("actions.npz", "actions"),
    ):
        if file_digest(base / name) != source[key]:
            raise ValueError(f"saved reference/input hash mismatch: {name}")
    with np.load(base / "prepared_inputs.npz", allow_pickle=False) as bundle:
        tensors = {
            name: torch.from_numpy(bundle[name]).to(device) for name in bundle.files
        }
    if set(tensors) != {port.name for port in module.inputs}:
        raise ValueError("saved prepared input set mismatch")
    with np.load(base / "actions.npz", allow_pickle=False) as bundle:
        reference = bundle["normalized_reference"]
    contract = NumericContract(**source["tolerances"])
    actions, trace = _run_invocation(
        module, regions, InvocationProgram(module, regions).validators, tensors
    )
    result = compare_action_chunk(
        reference,
        actions,
        sample_id="recorded-frame",
        space="normalized",
        contract=contract,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    trace.write(output / "reloaded_invocation_trace.json")
    np.savez(
        output / "actions.npz",
        normalized_reference=reference,
        normalized_reloaded=actions.detach().cpu().numpy(),
    )
    report = {
        "schema": "vlaforge.openpi_saved_capture_replay/1",
        "pid": os.getpid(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "passed" if result["metrics"]["within_tolerance"] else "failed",
        "source_report": file_digest(path),
        "source_status": source["status"],
        "source_error": source.get("error"),
        "invocation_ir": ir_digest,
        "regions": entries,
        "full_chunk_fidelity": result,
        "actions": file_digest(output / "actions.npz"),
        "execution": "Python Invocation IR interpreter using only hash-verified reloaded torch.export regions",
        "no_python_deployment": "not-run",
        "physical_units_verified": False,
        "robot_calibration_verified": False,
        "openpi_constructed_in_replay": False,
        "numerical_execution": numerical_execution,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ir-sha256", required=True)
    parser.add_argument("--numerical-context", type=Path)
    parser.add_argument("--allow-legacy-context", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    report = replay_saved_capture(**vars(args))
    print(
        json.dumps(
            {
                "status": report["status"],
                "wall_time_seconds": time.monotonic() - start,
                "report": str(args.output_dir / "report.json"),
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
