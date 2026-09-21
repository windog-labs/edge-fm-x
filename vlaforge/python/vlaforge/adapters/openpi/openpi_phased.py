"""Memory-bounded real OpenPI capture and independent saved-only validation.

The first worker persists graphs, original region examples and expected values.
Only a second worker, after the first exits, may produce validated evidence.
Neither stage is a no-Python deployment or a pretrained-model fixture.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from vlaforge.adapters.openpi.openpi_capture import _run_invocation, _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest, import_openpi
from vlaforge.adapters.openpi.openpi_frontend import capture_openpi_frontend
from vlaforge.ir.serializer import canonical_json, parse_canonical_json
from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot
from vlaforge.validation.contracts import NumericContract
from vlaforge.validation.deployment_metrics import compare_action_chunk


def process_identity(pid=None):
    """Linux PID plus kernel start tick prevents PID reuse from hiding a live owner."""
    pid = os.getpid() if pid is None else pid
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "start_ticks": int(stat[19])}


def _require_original_worker_exited(identity):
    if (
        not isinstance(identity, dict)
        or set(identity) != {"pid", "start_ticks"}
        or any(type(value) is not int or value <= 0 for value in identity.values())
    ):
        raise ValueError("a complete original worker identity is required")
    try:
        current = process_identity(identity["pid"])
    except FileNotFoundError:
        return
    if current == identity:
        raise ValueError("original model worker must exit before saved-only reload")


def tensor_tree_metadata(value):
    """Record actual tensor layouts without coercing dtype, device or strides."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("phased tensor evidence requires strided tensors")
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
        }
    if type(value) in (tuple, list):
        return {
            "kind": type(value).__name__,
            "items": [tensor_tree_metadata(item) for item in value],
        }
    if type(value) is dict and all(type(key) is str for key in value):
        return {
            "kind": "dict",
            "items": {key: tensor_tree_metadata(item) for key, item in value.items()},
        }
    if value is None or type(value) in (str, bool, int, float):
        return {"kind": type(value).__name__, "value": value}
    raise ValueError(f"unsupported saved tensor tree leaf: {type(value).__name__}")


def capture_and_persist_openpi(frontend, output_dir, *, contract, on_progress):
    """Persist without torch.export.load, keeping a single model weight owner."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    ir_path = output / "invocation_ir.json"
    ir_path.write_text(canonical_json(frontend.program.module, indent=2) + "\n")
    context = snapshot()
    report = {
        "schema": "vlaforge.openpi_persisted_capture/1",
        "status": "capturing",
        "regions": [],
        "invocation_ir": file_digest(ir_path),
        "numerical_context": context.to_dict(),
        "saved_reload_verified": False,
        "execution": "actual reference worker capture and serialization only; independent reload pending",
    }
    on_progress(report)
    outcomes = capture_openpi_frontend(
        frontend,
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    for outcome in outcomes:
        region = outcome.region.name
        entry = {
            "region": region,
            "supported": outcome.supported,
            "unsupported_report": outcome.report.to_dict(),
            "saved_reload_region_parity": "not-run",
        }
        report["regions"].append(entry)
        if outcome.supported:
            context.require_current()
            archive = output / f"{region}.pt2"
            torch.export.save(outcome.exported_program, archive)
            entry["archive"] = {"path": str(archive.resolve()), **file_digest(archive)}
            entry["capture_evidence"] = outcome.evidence.to_dict()
            (output / f"{region}.graph.txt").write_text(
                outcome.exported_program.graph_module.code
            )
            arguments = frontend.example_args[region]
            with torch.inference_mode():
                expected = frontend.program.regions[region](*arguments)
            specimen = {"args": arguments, "expected": expected}
            example_path = output / f"{region}.examples.pt"
            torch.save(specimen, example_path)
            entry["examples"] = {
                "path": str(example_path.resolve()),
                **file_digest(example_path),
            }
            entry["example_metadata"] = tensor_tree_metadata(specimen)
            del specimen, expected
        on_progress(report)
    context.require_current()
    report["status"] = (
        "persisted_awaiting_independent_reload"
        if all(item.supported for item in outcomes)
        else "unsupported"
    )
    on_progress(report)
    return report


def _verified_file(record, base):
    path = Path(record["path"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError("phased evidence path escapes original run directory")
    if file_digest(path) != {
        key: value for key, value in record.items() if key != "path"
    }:
        raise ValueError("phased evidence file hash mismatch")
    return path


def _native_processor_context(source):
    """Resolve pinned transform configuration/assets without constructing weights."""
    config = source["processor_config"]
    directory = Path(config["checkpoint_dir"]).resolve()
    provenance = source["checkpoint_provenance"]
    for name, expected in provenance["assets"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_digest(path) != expected:
            raise ValueError("saved native processor asset mismatch")
    configs = import_openpi(Path(config["source_root"]))
    train = configs.get_config(source["config_name"])
    train = replace(train, model=replace(train.model, pytorch_compile_mode=None))
    for name, expected in provenance["model_config"].items():
        if hasattr(train.model, name) and getattr(train.model, name) != expected:
            raise ValueError(f"saved native processor model config mismatch: {name}")
    factory = replace(
        train.data,
        assets=replace(train.data.assets, assets_dir=str(directory / "assets")),
    )
    data = factory.create(directory / "assets", train.model)
    if f"assets/{data.asset_id}/norm_stats.json" not in provenance["assets"]:
        raise ValueError(
            "saved native processor selected an unverified normalization asset"
        )
    normalization = importlib.import_module("openpi.shared.normalize")
    transforms = importlib.import_module("openpi.transforms")
    stats = normalization.load(directory / "assets" / data.asset_id)
    return data, stats, transforms, train.model


def _native_output_transform(source):
    """Construct only pinned official processors, never the model or checkpoint."""
    data, stats, transforms, _ = _native_processor_context(source)
    return transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]
    )


def validate_persisted_openpi(capture_report, output_dir, *, capture_sha256):
    import numpy as np
    import torch

    from vlaforge.frontend.invocation import InvocationProgram

    path = Path(capture_report).resolve()
    digest = file_digest(path)
    if digest["sha256"] != capture_sha256:
        raise ValueError("persisted reference report hash mismatch")
    source = json.loads(path.read_text())
    capture = source.get("capture", {})
    if (
        source.get("status") != "persisted"
        or capture.get("schema") != "vlaforge.openpi_persisted_capture/1"
        or capture.get("status") != "persisted_awaiting_independent_reload"
        or capture.get("saved_reload_verified") is not False
    ):
        raise ValueError(
            "independent reload requires explicit incomplete persistence evidence"
        )
    _require_original_worker_exited(source.get("process_identity"))
    context = NumericalContext.from_dict(source["numerical_context"])
    if capture["numerical_context"] != context.to_dict():
        raise ValueError("persisted reference and capture numerical context mismatch")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_independent_reload/1",
        "status": "started",
        "pid": os.getpid(),
        "source_report": {"path": str(path), **digest},
        "source_worker_exited": True,
        "openpi_model_constructed": False,
        "no_python_deployment": "not-run",
        "regions": [],
        "numerical_context": context.to_dict(),
        "numerical_scope": "explicit offline Python guard, not native runtime enforcement",
        "physical_units_verified": False,
        "robot_calibration_verified": False,
    }
    started = time.monotonic()
    device = _saved_device(source)

    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device.type == "cuda":
            report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            report["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    save()
    try:
        base = path.parent
        ir_path = base / "exported_regions/invocation_ir.json"
        if file_digest(ir_path) != capture["invocation_ir"]:
            raise ValueError("persisted invocation IR hash mismatch")
        module = parse_canonical_json(ir_path.read_text())
        entries = capture["regions"]
        if len(entries) != len(module.regions) or {
            item["region"] for item in entries
        } != {item.name for item in module.regions}:
            raise ValueError(
                "persisted capture must contain every declared region exactly once"
            )
        contract = NumericContract(**source["tolerances"])
        regions = {}
        with offline_restore(context, acknowledge_process_global=True):
            for entry in entries:
                if (
                    entry.get("supported") is not True
                    or entry.get("saved_reload_region_parity") != "not-run"
                ):
                    raise ValueError("invalid persisted region evidence state")
                archive = _verified_file(entry["archive"], base)
                examples = _verified_file(entry["examples"], base)
                restored = torch.export.load(archive).module()
                if any(value.device != device for value in restored.parameters()):
                    raise ValueError(
                        "saved region parameters differ from recorded device"
                    )
                specimen = torch.load(examples, weights_only=True)
                if tensor_tree_metadata(specimen) != entry["example_metadata"]:
                    raise ValueError(
                        "saved region example dtype/device/layout metadata mismatch"
                    )
                with torch.inference_mode():
                    actual = restored(*specimen["args"])
                torch.testing.assert_close(
                    actual,
                    specimen["expected"],
                    atol=contract.absolute_tolerance,
                    rtol=contract.relative_tolerance,
                    check_stride=True,
                )
                report["regions"].append(
                    {
                        "region": entry["region"],
                        "saved_reload_region_parity": "passed",
                        "archive": entry["archive"],
                        "examples": entry["examples"],
                    }
                )
                regions[entry["region"]] = restored
                del specimen, actual
                context.require_current()
                save()
            for filename, key in (
                ("prepared_inputs.npz", "prepared_inputs"),
                ("actions.npz", "actions"),
            ):
                if file_digest(base / filename) != source[key]:
                    raise ValueError(
                        "persisted prepared input or reference hash mismatch"
                    )
            with np.load(base / "prepared_inputs.npz", allow_pickle=False) as bundle:
                tensors = {
                    name: torch.from_numpy(bundle[name]).to(device)
                    for name in bundle.files
                }
            if set(tensors) != {port.name for port in module.inputs}:
                raise ValueError("persisted prepared input set mismatch")
            actions, trace = _run_invocation(
                module, regions, InvocationProgram(module, regions).validators, tensors
            )
            context.require_current()
            with np.load(base / "actions.npz", allow_pickle=False) as bundle:
                reference = bundle["normalized_reference"]
                native_reference = bundle["physical_reference"]
            normalized = actions.detach().cpu().numpy()
            transform = _native_output_transform(source)
            native = transform(
                {
                    "state": tensors["state"][0].detach().cpu().numpy(),
                    "actions": normalized[0],
                }
            )["actions"]
            report["fidelity"] = [
                compare_action_chunk(
                    reference,
                    normalized,
                    sample_id="recorded-frame",
                    space="normalized",
                    contract=contract,
                ),
                compare_action_chunk(
                    native_reference,
                    native,
                    sample_id="recorded-frame",
                    space="native-aloha-action-scale",
                    contract=contract,
                ),
            ]
            if not all(
                item["metrics"]["within_tolerance"] for item in report["fidelity"]
            ):
                raise ValueError("independent full action output validation failed")
            trace.write(output / "reloaded_invocation_trace.json")
            np.savez(
                output / "actions.npz",
                normalized_reference=reference,
                normalized_reloaded=normalized,
                native_action_scale_reference=native_reference,
                native_action_scale_reloaded=native,
            )
            report["actions"] = file_digest(output / "actions.npz")
        report["caller_policy_restoration_verified"] = True
        report["caller_context_after_restore"] = snapshot().to_dict()
        report["status"] = "passed"
        save()
        # This is a new joined report, never an in-place upgrade of the first worker.
        joined = deepcopy(source)
        joined["schema"] = "vlaforge.openpi_validated_phased_capture/1"
        joined["status"] = "passed"
        joined["evidence_level"] = (
            "independent-worker-export-save-reload-full-invocation-parity; source evidence level retained separately"
        )
        joined["source_evidence_level"] = source["evidence_level"]
        joined["capture"]["status"] = "passed"
        joined["capture"]["saved_reload_verified"] = True
        joined["capture"]["full_chunk_fidelity"] = report["fidelity"][0]
        for entry in joined["capture"]["regions"]:
            entry["saved_reload_region_parity"] = "passed"
        joined["independent_reload"] = {
            "path": str(output / "report.json"),
            **file_digest(output / "report.json"),
        }
        joined["persisted_reference_report"] = {"path": str(path), **digest}
        target = base / "validated_capture_report.json"
        with target.open("x") as stream:
            stream.write(json.dumps(joined, indent=2) + "\n")
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


def verify_phased_capture_join(source, path):
    """Bind a passed aggregate back to both immutable worker records and outputs."""
    if source.get("schema") != "vlaforge.openpi_validated_phased_capture/1":
        return
    base = Path(path).resolve().parent
    original_path = _verified_file(source["persisted_reference_report"], base)
    original = json.loads(original_path.read_text())
    replay_record = source["independent_reload"]
    replay_path = Path(replay_record["path"]).resolve()
    if file_digest(replay_path) != {
        key: value for key, value in replay_record.items() if key != "path"
    }:
        raise ValueError("independent reload proof hash mismatch")
    replay = json.loads(replay_path.read_text())
    if (
        original.get("status") != "persisted"
        or replay.get("schema") != "vlaforge.openpi_independent_reload/1"
        or replay.get("status") != "passed"
        or replay.get("source_report") != source["persisted_reference_report"]
        or replay.get("source_worker_exited") is not True
        or replay.get("openpi_model_constructed") is not False
        or replay.get("caller_policy_restoration_verified") is not True
        or replay.get("pid") == original.get("pid")
        or replay.get("numerical_context") != source.get("numerical_context")
        or original.get("numerical_context") != source.get("numerical_context")
        or replay.get("fidelity", [{}])[0]
        != source["capture"].get("full_chunk_fidelity")
        or len(replay.get("fidelity", [])) != 2
        or not all(
            item.get("metrics", {}).get("within_tolerance") is True
            for item in replay.get("fidelity", [])
        )
    ):
        raise ValueError("independent phased capture proof mismatch")
    if file_digest(replay_path.parent / "actions.npz") != replay["actions"]:
        raise ValueError("independent reload full output hash mismatch")
    for key in (
        "prepared_inputs",
        "actions",
        "checkpoint_provenance",
        "processor_config",
        "tolerances",
        "device",
        "config_name",
        "num_steps",
    ):
        if original.get(key) != source.get(key):
            raise ValueError(f"joined capture changed original provenance: {key}")
    actual = source["capture"]["regions"]
    previous = original["capture"]["regions"]
    verified = replay["regions"]
    if len(actual) != len(previous) or len(actual) != len(verified):
        raise ValueError("independent region proof set mismatch")
    for current, before, after in zip(actual, previous, verified, strict=True):
        expected = {**before, "saved_reload_region_parity": "passed"}
        if current != expected or any(
            after.get(key) != current.get(key)
            for key in ("region", "archive", "examples", "saved_reload_region_parity")
        ):
            raise ValueError("independent region proof mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = validate_persisted_openpi(**vars(parser.parse_args()))
    print(json.dumps({"status": report["status"], "pid": report["pid"]}))


if __name__ == "__main__":
    main()
