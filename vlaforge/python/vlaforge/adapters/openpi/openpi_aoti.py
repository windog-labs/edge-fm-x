"""Hash-bound OpenPI artifact orchestration using the generic AOTI compiler CLI.

Both model variants use the saved Invocation IR and declared region interfaces.
No checkpoint construction, model-name dispatch, or C++ runtime claim is made.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import json
import os
from pathlib import Path
import time

from vlaforge.adapters.openpi.openpi_capture import _run_invocation, _saved_device
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.deployment.aoti_export import (
    backend_pass_records,
    backend_program_pass_records,
)
from vlaforge.deployment.aoti_package import package_pass_records, verify_package_audit
from vlaforge.deployment.aoti_profile import PROFILES, aoti_configs
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _capture_source(path, sha256):
    path = Path(path).resolve()
    if file_digest(path)["sha256"] != sha256:
        raise ValueError("capture report SHA256 mismatch")
    source = json.loads(path.read_text())
    from vlaforge.adapters.openpi.openpi_phased import verify_phased_capture_join
    from vlaforge.adapters.openpi.openpi_pruned_capture import SCHEMA, verify_pruned_capture_join

    if source.get("schema") == SCHEMA:
        verify_pruned_capture_join(source, path)
    elif source.get("schema", "").startswith("vlaforge.openpi_validated_pruned_capture/"):
        raise ValueError("unknown derived pruned capture schema")
    else:
        verify_phased_capture_join(source, path)
    capture = source.get("capture")
    if (
        source.get("status") != "passed"
        or not isinstance(capture, dict)
        or capture.get("status") != "passed"
    ):
        raise ValueError(
            "artifact build requires a passed real capture and complete replay"
        )
    context = NumericalContext.from_dict(source["numerical_context"])
    ir_path = path.parent / "exported_regions/invocation_ir.json"
    if file_digest(ir_path) != capture["invocation_ir"]:
        raise ValueError("captured Invocation IR digest mismatch")
    module = parse_canonical_json(ir_path.read_text())
    regions = {item["region"]: item for item in capture["regions"]}
    if len(regions) != len(capture["regions"]) or set(regions) != {
        r.name for r in module.regions
    }:
        raise ValueError("capture must contain each declared region exactly once")
    for name, item in regions.items():
        archive = Path(item["archive"]["path"]).resolve()
        if not archive.is_relative_to(path.parent):
            raise ValueError("captured archive escapes run directory")
        if (
            not item.get("supported")
            or item.get("saved_reload_region_parity") != "passed"
            or file_digest(archive)
            != {k: v for k, v in item["archive"].items() if k != "path"}
        ):
            raise ValueError(f"captured archive evidence mismatch: {name}")
    for filename, key in (
        ("prepared_inputs.npz", "prepared_inputs"),
        ("actions.npz", "actions"),
    ):
        if file_digest(path.parent / filename) != source[key]:
            raise ValueError(f"captured input/reference digest mismatch: {filename}")
    return path, source, context, module, regions


def verify_compiled_region(record_path, region, *, target, profile):
    record_path = Path(record_path).resolve()
    record = json.loads(record_path.read_text())
    configs = aoti_configs(profile)
    artifact = Path(record["artifact"]["path"]).resolve()
    if not artifact.is_relative_to(record_path.parent):
        raise ValueError("compiled artifact escapes its manifest directory")
    digest = file_digest(artifact)
    program_audit = record.get("backend_program_audit")
    if (
        record.get("status") != "passed"
        or record.get("target") != target
        or record.get("inductor_profile") != profile
        or _canonical(record.get("inductor_configs")) != _canonical(configs)
        or record.get("backend_graph_passes", []) != backend_pass_records(configs)
        or not isinstance(program_audit, dict)
        or program_audit.get("passes") != backend_program_pass_records(configs)
        or not isinstance(program_audit.get("rewrites"), list)
        or record.get("backend_package_audit", {}).get("passes", [])
        != package_pass_records(configs)
        or record.get("exported_program", {}).get("sha256")
        != region["archive"]["sha256"]
        or record["artifact"].get("sha256") != digest["sha256"]
        or record["artifact"].get("size_bytes") != digest["size_bytes"]
    ):
        raise ValueError("AOTI artifact source/profile/backend evidence mismatch")
    verify_package_audit(
        artifact,
        configs,
        record.get("backend_package_audit", {}),
        artifact_sha256=digest["sha256"],
    )
    return record


def compile_regions(args):
    import torch
    from vlaforge.cli import main as compiler_main
    import vlaforge.cli
    import vlaforge.deployment.aoti_export
    import vlaforge.deployment.aoti_package
    import vlaforge.deployment.aoti_profile
    import vlaforge.numerical_context

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_aoti_compile/1",
        "status": "started",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "regions": [],
        "no_python_deployment": "not-run",
        "numeric_parity_verified": False,
        "profile": args.profile,
        "orchestrator_source": file_digest(Path(__file__)),
        "context_implementation": file_digest(
            Path(vlaforge.numerical_context.__file__)
        ),
        "configs": aoti_configs(args.profile),
        "source_files": {
            str(Path(module.__file__).name): file_digest(Path(module.__file__))
            for module in (
                vlaforge.cli,
                vlaforge.deployment.aoti_export,
                vlaforge.deployment.aoti_package,
                vlaforge.deployment.aoti_profile,
            )
        },
    }
    started = time.monotonic()

    def save():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        path, source, context, _, regions = _capture_source(
            args.capture_report, args.capture_sha256
        )
        if (
            not args.regions
            or len(set(args.regions)) != len(args.regions)
            or not set(args.regions) <= set(regions)
        ):
            raise ValueError("select unique captured region names")
        device = _saved_device(source)
        if device.type != "cuda":
            raise ValueError(
                "this actual artifact audit requires the recorded CUDA target"
            )
        major, minor = torch.cuda.get_device_capability(device)
        target = f"sm_{major}{minor}"
        report.update(
            source_report={"path": str(path), **file_digest(path)},
            numerical_context=context.to_dict(),
            target=target,
            device=str(device),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numerical_context_scope="offline Python observation/restoration, not native runtime enforcement",
        )
        with offline_restore(context, acknowledge_process_global=True):
            for name in args.regions:
                entry = {
                    "region": name,
                    "status": "compiling",
                    "export": regions[name]["archive"],
                }
                report["regions"].append(entry)
                manifest = output / f"{name}.compile.json"
                command = [
                    "compile-artifact",
                    regions[name]["archive"]["path"],
                    "--output",
                    str(output / f"{name}.pt2"),
                    "--manifest",
                    str(manifest),
                    "--target",
                    target,
                    "--inductor-profile",
                    args.profile,
                ]
                entry["generic_cli_arguments"] = command
                entry["observed_numerical_context_before"] = snapshot().to_dict()
                context.require_current()
                save()
                with (output / f"{name}.compile.log").open("w") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        if compiler_main(command) != 0:
                            raise RuntimeError(f"generic compile CLI failed: {name}")
                entry["observed_numerical_context_after"] = snapshot().to_dict()
                context.require_current()
                verified = verify_compiled_region(
                    manifest, regions[name], target=target, profile=args.profile
                )
                context.require_current()
                entry.update(
                    status="compiled-unvalidated",
                    manifest={"path": str(manifest), **file_digest(manifest)},
                    artifact=verified["artifact"],
                )
                save()
                gc.collect()
                torch.cuda.empty_cache()
        report["caller_policy_restoration_verified"] = True
        report["caller_context_after_restore"] = snapshot().to_dict()
        report["status"] = "compiled-unvalidated"
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def _tuple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _region_result(values):
    return values[0] if len(values) == 1 else values


def _tensor_metrics(expected, actual):
    import torch

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ValueError(
            "artifact must return a Tensor for each captured tensor output"
        )
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(
            "artifact output shape or dtype differs from the captured interface"
        )
    delta = expected.detach().to(torch.float64) - actual.detach().to(torch.float64)
    finite = bool(torch.isfinite(delta).all())
    return {
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "exact": bool(torch.equal(expected, actual)),
        "finite": finite,
        "mse": float(delta.square().mean()) if delta.numel() and finite else None,
        "max_abs": float(delta.abs().max()) if delta.numel() and finite else None,
    }


def audit_artifacts(args):
    import numpy as np
    import torch
    import torch._inductor
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.frontend.invocation import InvocationProgram
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_aoti_audit/1",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "status": "started",
        "region_calls": [],
        "no_python_deployment": "not-run",
        "physical_units_verified": False,
        "robot_calibration_verified": False,
        "discarded_warmup_calls": 0,
    }
    arrays = {}
    started = time.monotonic()

    def save():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        path, source, context, module, regions = _capture_source(
            args.capture_report, args.capture_sha256
        )
        device = _saved_device(source)
        if device.type != "cuda":
            raise ValueError("actual CUDA artifacts require a CUDA capture")
        major, minor = torch.cuda.get_device_capability(device)
        target = f"sm_{major}{minor}"
        compiled = {}
        report.update(
            source_report={"path": str(path), **file_digest(path)},
            device=str(device),
            target=target,
            gpu=torch.cuda.get_device_name(device),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numerical_context=context.to_dict(),
            compile_reports=[],
        )
        for report_path in args.compile_report:
            build = json.loads(report_path.read_text())
            if (
                build.get("status") != "compiled-unvalidated"
                or build.get("source_report", {}).get("sha256") != args.capture_sha256
                or build.get("target") != target
                or NumericalContext.from_dict(build.get("numerical_context")) != context
                or build.get("caller_policy_restoration_verified") is not True
                or build.get("torch") != torch.__version__
                or build.get("cuda") != torch.version.cuda
                or _canonical(build.get("configs"))
                != _canonical(aoti_configs(build["profile"]))
            ):
                raise ValueError(
                    "compile report does not match this capture, context and backend profile"
                )
            report["compile_reports"].append(
                {"path": str(report_path.resolve()), **file_digest(report_path)}
            )
            for item in build["regions"]:
                name = item["region"]
                if (
                    name in compiled
                    or name not in regions
                    or item.get("status") != "compiled-unvalidated"
                ):
                    raise ValueError(
                        "compiled region is duplicated, missing or unfinished"
                    )
                manifest = Path(item["manifest"]["path"])
                if file_digest(manifest) != {
                    k: v for k, v in item["manifest"].items() if k != "path"
                }:
                    raise ValueError("generic compile manifest hash mismatch")
                compiled[name] = verify_compiled_region(
                    manifest, regions[name], target=target, profile=build["profile"]
                )
        if not compiled:
            raise ValueError("artifact audit requires at least one compiled region")
        report["compiled_regions"] = list(compiled)
        report["full_artifact_execution"] = set(compiled) == set(regions)
        report["execution"] = (
            "all AOTI regions through Python Invocation IR"
            if report["full_artifact_execution"]
            else "hybrid AOTI/export through Python Invocation IR; not full artifact execution"
        )
        save()
        with offline_restore(context, acknowledge_process_global=True):
            exports = {
                name: torch.export.load(item["archive"]["path"]).module()
                for name, item in regions.items()
            }
            candidates = {
                name: torch._inductor.aoti_load_package(item["artifact"]["path"])
                for name, item in compiled.items()
            }
            with np.load(
                path.parent / "prepared_inputs.npz", allow_pickle=False
            ) as bundle:
                tensors = {
                    name: torch.from_numpy(bundle[name]).to(device)
                    for name in bundle.files
                }
            with np.load(path.parent / "actions.npz", allow_pickle=False) as bundle:
                reference = bundle["normalized_reference"]
            validators = InvocationProgram(module, exports).validators
            eager, _ = _run_invocation(module, exports, validators, tensors)
            arrays.update(
                normalized_reference=reference,
                normalized_exported=eager.detach().cpu().numpy(),
            )
            calls = []

            def wrap(name):
                def run(*values):
                    expected = _tuple(exports[name](*values))
                    actual = (
                        _tuple(candidates[name](*values))
                        if name in candidates
                        else expected
                    )
                    if len(expected) != len(actual):
                        raise ValueError("artifact output arity mismatch")
                    index = len(calls)
                    metrics = [
                        _tensor_metrics(a, b)
                        for a, b in zip(expected, actual, strict=True)
                    ]
                    calls.append(
                        {
                            "index": index,
                            "region": name,
                            "backend": "aoti" if name in candidates else "torch.export",
                            "same_input_output_metrics": metrics,
                        }
                    )
                    for which, outputs in (
                        ("exported_same_input", expected),
                        ("candidate", actual),
                    ):
                        for item, value in enumerate(outputs):
                            data = value.detach().cpu()
                            if data.dtype == torch.bfloat16:
                                data = data.view(torch.uint16)
                            arrays[f"call_{index:02d}_{which}_{item:02d}"] = (
                                data.numpy().copy()
                            )
                    report["region_calls"] = calls
                    save()
                    return _region_result(actual)

                return run

            bound = {name: wrap(name) for name in regions}
            actual, trace = _run_invocation(module, bound, validators, tensors)
            trace.write(output / "artifact_invocation_trace.json")
            arrays["normalized_artifact"] = actual.detach().cpu().numpy()
            contract = NumericContract(**source["tolerances"])
            report["full_chunk_fidelity"] = compare_action_chunk(
                reference,
                actual,
                sample_id="recorded-frame",
                space="normalized",
                contract=contract,
            )
            report["exported_reference_fidelity"] = compare_action_chunk(
                reference,
                eager,
                sample_id="recorded-frame",
                space="normalized",
                contract=contract,
            )
            context.require_current()
        report["caller_policy_restoration_verified"] = True
        report["caller_context_after_restore"] = snapshot().to_dict()
        np.savez(output / "actions_and_region_outputs.npz", **arrays)
        report["outputs"] = file_digest(output / "actions_and_region_outputs.npz")
        report["bf16_storage_encoding"] = (
            "raw uint16 bit patterns; dtype retained in per-output metrics"
        )
        report["status"] = (
            "passed"
            if report["full_chunk_fidelity"]["metrics"]["within_tolerance"]
            and report["exported_reference_fidelity"]["metrics"]["within_tolerance"]
            and all(
                item["exact"]
                for call in calls
                for item in call["same_input_output_metrics"]
            )
            else "failed"
        )
        save()
        if report["status"] != "passed":
            raise SystemExit(1)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if arrays:
            np.savez(output / "partial_outputs.npz", **arrays)
            report["partial_outputs"] = file_digest(output / "partial_outputs.npz")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--capture-report", type=Path, required=True)
    compile_parser.add_argument("--capture-sha256", required=True)
    compile_parser.add_argument("--output-dir", type=Path, required=True)
    compile_parser.add_argument("--profile", choices=PROFILES, default="eager-numerics")
    compile_parser.add_argument("regions", nargs="+")
    compile_parser.set_defaults(handler=compile_regions)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--capture-report", type=Path, required=True)
    audit_parser.add_argument("--capture-sha256", required=True)
    audit_parser.add_argument(
        "--compile-report", type=Path, action="append", required=True
    )
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.set_defaults(handler=audit_artifacts)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
