"""Collect trace-bound activations under an externally supervised device lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
)
from vlaforge.analysis.precision_probe import (
    PrecisionProbe,
    ProbeRegion,
    owned_tensor_snapshot,
    tensor_bundle_digest,
    tensor_identity,
)
from vlaforge.ir.serializer import module_from_data
from vlaforge.numerical_context import NumericalContext

SCHEMA = "vlaforge.precision_probe_protocol/1"
KEYS = {"schema", "module", "invocation", "regions", "sites", "step_keys", "step_groups",
        "profile", "numerical_context", "samples", "evidence", "execution_supervision", "retain_sites"}
SAMPLE_KEYS = {"sample_id", "partition_key", "input_sha256", "noise_sha256", "split", "inputs",
               "noise_names", "expected_outputs", "source_identity"}


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_ref(value, *, tensor=False):
    keys = {"path", "sha256", "format"} if tensor else {"path", "sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("file reference requires exact path/hash/format fields")
    path = Path(value["path"])
    if not path.is_absolute() or file_sha(path) != value["sha256"]:
        raise ValueError("referenced file is not absolute or fails SHA256: " + str(path))
    if tensor and value["format"] not in ("raw", "npy"):
        raise ValueError("tensor format must be raw or npy")
    return path


def load_tensor(reference, port, *, device="cpu"):
    import numpy as np
    import torch

    path = file_ref(reference, tensor=True)
    formats = {"f16": "<f2", "bf16": "<u2", "f32": "<f4", "f64": "<f8",
               "i32": "<i4", "i64": "<i8", "u8": "u1", "u64": "<u8", "bool": "?"}
    dtype = port.payload.dtype
    if dtype not in formats or any(type(size) is not int or size < 1 for size in port.payload.shape):
        raise ValueError("probe tool requires a supported static tensor contract")
    if reference["format"] == "raw":
        raw = path.read_bytes()
        if len(raw) != int(np.prod(port.payload.shape)) * np.dtype(formats[dtype]).itemsize:
            raise ValueError("input/reference raw byte count differs from complete shape")
        if dtype == "bool" and any(value not in (0, 1) for value in raw):
            raise ValueError("noncanonical bool storage")
        array = np.frombuffer(raw, dtype=formats[dtype]).copy().reshape(port.payload.shape)
    else:
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.dtype(formats[dtype]) or tuple(array.shape) != tuple(port.payload.shape):
            raise ValueError("input/reference dtype or complete shape mismatch")
        array = np.array(array, copy=True, order="C")
    value = torch.from_numpy(array)
    if dtype == "bf16":
        value = value.view(torch.bfloat16)
    return value.to(device=device)


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or set(protocol) != KEYS or protocol["schema"] != SCHEMA:
        raise ValueError("unsupported precision probe protocol fields/schema")
    if protocol["execution_supervision"] != "external-exclusive-device-owner-monitor":
        raise ValueError("actual runs require an external exclusive-device owner monitor")
    module = module_from_data(read(file_ref(protocol["module"])))
    if module.states:
        raise ValueError("this diagnostic runner requires an explicit stateless invocation")
    module.invocation(protocol["invocation"])
    if not protocol["regions"] or {item["name"] for item in protocol["regions"]} != {item.name for item in module.regions}:
        raise ValueError("protocol must cover all original regions")
    regions = tuple(ProbeRegion(**item) for item in protocol["regions"])
    if len({item.name for item in regions}) != len(regions):
        raise ValueError("duplicate region declarations")
    for item in regions:
        file_ref({"path": item.path, "sha256": item.artifact_sha256})
    sites = tuple(CalibrationSite(**item) for item in protocol["sites"])
    retain = protocol["retain_sites"]
    if not isinstance(retain, list) or len(set(retain)) != len(retain) or not set(retain) <= {site.name for site in sites}:
        raise ValueError("retained activations must be an explicit unique subset of selected sites")
    context = NumericalContext.from_dict(protocol["numerical_context"])
    if context.partial:
        raise ValueError("old partial numerical context cannot validate calibration")
    for evidence in protocol["evidence"]:
        file_ref(evidence)
    samples, inputs, references = [], [], []
    if not isinstance(protocol["samples"], list) or not protocol["samples"]:
        raise ValueError("explicit calibration and held-out samples are required")
    for item in protocol["samples"]:
        if set(item) != SAMPLE_KEYS or item["split"] not in ("calibration", "held-out"):
            raise ValueError("invalid sample fields/split")
        if set(item["inputs"]) != {port.name for port in module.inputs}:
            raise ValueError("sample must bind every actual input")
        if set(item["expected_outputs"]) != {port.name for port in module.outputs}:
            raise ValueError("reference must cover every complete output")
        values = {port.name: load_tensor(item["inputs"][port.name], port) for port in module.inputs}
        output = tuple(load_tensor(item["expected_outputs"][port.name], port) for port in module.outputs)
        sample = CalibrationSample(**{key: item[key] for key in
                                      ("sample_id", "partition_key", "input_sha256", "noise_sha256")})
        names = item["noise_names"]
        if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(name not in values for name in names):
            raise ValueError("sample requires explicit unique noise inputs")
        if (tensor_bundle_digest({name: value for name, value in values.items() if name not in names}) != sample.input_sha256
                or tensor_bundle_digest({name: values[name] for name in names}) != sample.noise_sha256):
            raise ValueError("sample digest does not match actual observation/noise tensors")
        samples.append(sample)
        inputs.append(values)
        references.append(output)
    return module, regions, sites, context, samples, inputs, references


def run(protocol, output):
    import torch
    from vlaforge.analysis.precision_probe import metadata_digest
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import Interpreter
    from vlaforge.interpreter.inputs import InputStamp, TensorView

    module, regions, sites, context, samples, inputs, references = validate_protocol(protocol)
    collector = PrecisionCalibration(profile_sha256=metadata_digest(protocol["profile"]),
                                      numerical_context_sha256=metadata_digest(context.to_dict()),
                                      step_keys=protocol["step_keys"], sites=sites,
                                      calibration_samples=tuple(sample for sample, item in zip(samples, protocol["samples"], strict=True)
                                                                if item["split"] == "calibration"),
                                      held_out_samples=tuple(sample for sample, item in zip(samples, protocol["samples"], strict=True)
                                                             if item["split"] == "held-out"))
    probe = PrecisionProbe(regions=regions, sites=sites, step_keys=protocol["step_keys"],
                           profile=protocol["profile"], numerical_context=context)
    reports = []
    for index, (sample, values, expected, declaration) in enumerate(zip(samples, inputs, references, protocol["samples"], strict=True)):
        actual = {port.name: values[port.name].to(port.device) for port in module.inputs}
        captured_outputs = []
        def invoke(callables, actual=actual, index=index, captured_outputs=captured_outputs):
            program = InvocationProgram(module, callables)
            session = Interpreter(module, regions=callables, validators=program.validators)
            for port in module.inputs:
                value = actual[port.name]
                if port.payload.layout != "contiguous" or not value.is_contiguous() or value.data_ptr() % port.alignment:
                    raise ValueError("actual tensor does not satisfy supported layout/alignment")
                view = TensorView(value, tuple(value.shape), port.payload.dtype, port.payload.layout,
                                  str(value.device), port.alignment)
                session.bind_input(port.name, view, InputStamp(revision=index + 1))
            session.run(protocol["invocation"])
            outputs = tuple(session.read_output(port.name) for port in module.outputs)
            captured_outputs.extend(owned_tensor_snapshot(value) for value in outputs)
            return outputs
        with torch.no_grad():
            result = probe.observe_sample(sample=sample, inputs=actual, noise_names=declaration["noise_names"],
                                          expected_outputs=expected, invoke=invoke)
        folder = output / f"sample-{index:06d}"
        folder.mkdir()
        for port, value in zip(module.outputs, captured_outputs, strict=True):
            (folder / (port.name + ".bin")).write_bytes(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        if declaration["split"] == "calibration":
            collector = result.publish(collector)
        report = result.report()
        report["split"] = declaration["split"]
        report["source_identity"] = declaration["source_identity"]
        report["invocation_boundary"] = "original IR Interpreter; all declared public outputs read after run"
        report["source_module"] = protocol["module"]
        report["invocation"] = protocol["invocation"]
        retained = []
        for site_index, site in enumerate(sites):
            if site.name not in protocol["retain_sites"]:
                continue
            for step in (range(len(protocol["step_keys"])) if site.stage == "iteration" else (None,)):
                value = result.owned_snapshot(site.name, step)
                name = f"activation-{site_index}-step-{step if step is not None else 'context'}.bin"
                raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                (folder / name).write_bytes(raw)
                retained.append({"site": site.name, "step": step, "split": declaration["split"],
                                 "sample_id": sample.sample_id, "file": name,
                                 "artifact_sha256": site.artifact_sha256, **tensor_identity(value)})
        report["retained_activations"] = retained
        report["complete_raw_outputs"] = {port.name: tensor_identity(value)
                                            for port, value in zip(module.outputs, captured_outputs, strict=True)}
        write(folder / "report.json", report)
        reports.append({"sample_id": sample.sample_id, "split": declaration["split"],
                        "report_sha256": file_sha(folder / "report.json")})
        print("VALIDATED", sample.sample_id, declaration["split"], flush=True)
    write(output / "calibration.json", collector.report())
    plans = {}
    for strategy in ("global", "site", "step", "step-group"):
        plan = collector.fit(strategy, **({"step_groups": protocol["step_groups"]} if strategy == "step-group" else {}))
        write(output / (strategy + ".json"), plan.to_data())
        plans[strategy] = plan.sha256
    write(output / "report.json", {"schema": "vlaforge.precision_probe_campaign/1", "status": "validated",
                                    "samples": reports, "plans": plans, "held_out_used_for_fit": False,
                                    "real_low_precision_kernel_verified": False, "timing_evidence": False,
                                    "protocol_sha256": file_sha(output / "protocol.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("validate", "run"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(args.protocol)
    write(args.output / "protocol.json", protocol)
    try:
        if args.stage == "validate":
            module, _, sites, _, samples, _, _ = validate_protocol(protocol)
            write(args.output / "report.json", {"status": "inputs_validated_no_model_execution",
                                                "samples": [asdict(sample) for sample in samples],
                                                "sites": [asdict(site) for site in sites],
                                                "regions": len(module.regions)})
        else:
            run(protocol, args.output)
    except Exception as error:
        write(args.output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
