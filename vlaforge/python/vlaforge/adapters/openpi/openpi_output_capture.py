"""Capture pinned official output processors without constructing model weights.

This independent stage consumes verified real state and complete normalized
actions. CUDA compilation is optional and uses the public backend interface.
"""

import argparse
import json
import os
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path


def processor_config_for_output(source, checkpoint_dir):
    """Return the pinned processor config, adding legacy capture paths if explicit."""
    existing = source.get("processor_config")
    if isinstance(existing, dict):
        if checkpoint_dir is not None and str(Path(checkpoint_dir).resolve()) != existing.get("checkpoint_dir"):
            raise ValueError("explicit processor checkpoint differs from the captured processor config")
        return dict(existing)
    if checkpoint_dir is None:
        raise ValueError("capture lacks processor_config; pass --processor-checkpoint-dir")
    root = Path(source.get("checkpoint_provenance", {}).get("source", {}).get("root", "")).resolve()
    checkpoint = Path(checkpoint_dir).resolve()
    if not root.is_absolute() or not checkpoint.is_absolute():
        raise ValueError("legacy processor paths must be canonical absolute paths")
    return {"source_root": str(root), "checkpoint_dir": str(checkpoint)}


def output_reference(source, base):
    import numpy as np

    from vlaforge.adapters.openpi.openpi_checkpoint import file_digest

    for key, filename in (("actions", "actions.npz"), ("prepared_inputs", "prepared_inputs.npz")):
        if file_digest(base / filename) != source[key]:
            raise ValueError("actual source tensor archive differs")
    with np.load(base / "actions.npz", allow_pickle=False) as pack:
        normalized = pack["normalized_reference"]
        native = pack["physical_reference"][None]
    with np.load(base / "prepared_inputs.npz", allow_pickle=False) as pack:
        state = pack["state"]
    if (normalized.ndim != 3 or normalized.shape[0] != 1 or native.ndim != 3
            or native.shape[:2] != normalized.shape[:2] or state.ndim != 2
            or state.shape[0] != 1 or not all(np.isfinite(value).all() for value in (state, normalized, native))):
        raise ValueError("complete state/action reference dimensions or finiteness differ")
    if any(value.dtype not in (np.dtype("float32"), np.dtype("float64")) for value in (state, normalized, native)):
        raise TypeError("native processor reference requires explicit floating tensor dtypes")
    return state, normalized, native


def checked_output_rejections(module, inputs):
    """Check the original predicate and the complete, uncropped source tensor."""
    import torch

    rejected = (*inputs[:-1], torch.zeros_like(inputs[-1]))
    if module(*rejected)[1].item():
        raise ValueError("output processor dropped incoming rejection")
    invalid = inputs[1].clone()
    invalid.reshape(-1)[-1] = float("nan")
    if module(inputs[0], invalid, inputs[2])[1].item():
        raise ValueError("output processor accepted nonfinite full normalized output")
    return {"incoming_false_rejected": True, "complete_source_nonfinite_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checked", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--processor-checkpoint-dir", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--package-extraction-root", type=Path)
    parser.add_argument("--materialize-aoti", action="store_true")
    args = parser.parse_args()
    if args.compile and args.device != "cuda:0":
        parser.error("the CUDA compilation stage requires cuda:0")
    if args.package_extraction_root is not None and not args.compile:
        parser.error("explicit package extraction requires the compile stage")
    if args.materialize_aoti and (not args.compile or args.package_extraction_root is not None):
        parser.error("materialized AOTI requires compilation and excludes runtime extraction")
    if args.device == "cuda:0":
        from cogact_gpu_monitor import child_handshake
        child_handshake()
    import numpy as np
    import torch
    import torch._inductor
    import torch._inductor.codecache

    import vlaforge.cli
    import vlaforge.deployment.aoti_export
    import vlaforge.deployment.aoti_package
    import vlaforge.deployment.aoti_profile
    import vlaforge.numerical_context
    from vlaforge.adapters.openpi.openpi_aoti import _capture_source, verify_compiled_region
    from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
    from vlaforge.adapters.openpi.openpi_frontend import _tensor_type
    from vlaforge.adapters.openpi.openpi_output import (
        checked_output_module,
        lower_official_output_transform,
    )
    from vlaforge.adapters.openpi.openpi_phased import (
        _native_output_transform,
        tensor_tree_metadata,
    )
    from vlaforge.cli import main as compiler_main
    from vlaforge.deployment.aoti_profile import aoti_configs
    from vlaforge.frontend import capture_region, tensor_region
    from vlaforge.ir.program import Value
    from vlaforge.numerical_context import offline_restore, snapshot
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "vlaforge.openpi_output_capture/1", "status": "started", "pid": os.getpid(), "scope": "native-action output processor only, not complete model or native Session",
              "model_constructed": False, "no_python_deployment": False,
              "physical_units_verified": False, "robot_calibration_verified": False,
              "tool": file_digest(Path(__file__)), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "device": args.device, "compile_requested": args.compile, "compiled": False,
              "torch": str(torch.__version__), "checks": {}}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def compare(label, actual, golden):
        metric = compare_action_chunk(golden, actual, sample_id="recorded-official-frame0", space="native-aloha-action-scale", contract=NumericContract(absolute_tolerance=0, relative_tolerance=0))
        metric["bitwise_equal"] = actual.dtype == golden.dtype and actual.shape == golden.shape and actual.tobytes() == golden.tobytes()
        report["checks"][label] = metric
        np.save(args.output_dir / (label + ".npy"), actual)
        save()
        if not metric["metrics"]["exact_values"] or not metric["bitwise_equal"]:
            raise ValueError("actual CUDA native-scale processor differs from official: " + label)

    save()
    try:
        _, source, context, _, _ = _capture_source(args.capture_report, args.capture_sha256)
        base = args.capture_report.parent
        source["processor_config"] = processor_config_for_output(source, args.processor_checkpoint_dir)
        report["processor_config"] = source["processor_config"]
        state, normalized, golden = output_reference(source, base)
        module = lower_official_output_transform(_native_output_transform(source))
        module = checked_output_module(module) if args.checked else module
        module = module.to(args.device)
        inputs = (torch.from_numpy(state.copy()).to(args.device), torch.from_numpy(normalized.copy()).to(args.device))
        if args.checked:
            # The shared OpenPI frontend's original final predicate examines every normalized value.
            inputs = (*inputs, torch.isfinite(inputs[1]).all().reshape(1))
            report["incoming_acceptance_source"] = "OpenPI original finite(full normalized action chunk)"
        report.update(source_capture=file_digest(args.capture_report), input_metadata=tensor_tree_metadata(inputs), numerical_context=context.to_dict())
        torch.save(inputs, args.output_dir / "inputs.pt")
        report["inputs"] = file_digest(args.output_dir / "inputs.pt")
        with offline_restore(context, acknowledge_process_global=True), torch.inference_mode():
            actual = module(*inputs)
            results = actual if args.checked else (actual,)
            if args.checked and not results[1].item():
                raise ValueError("native publication predicate rejected actual sample")
            report["checked_publication"] = args.checked
            compare("eager_" + args.device.split(":")[0], results[0].cpu().numpy(), golden)
            if args.checked:
                report["rejection_checks"] = {"eager": checked_output_rejections(module, inputs)}
            input_names = ("state", "normalized_actions", "incoming_accepted") if args.checked else ("state", "normalized_actions")
            declared = tensor_region("openpi_native_output", inputs=tuple(Value(name, _tensor_type(value)) for name, value in zip(input_names, inputs, strict=True)), outputs=tuple(_tensor_type(value) for value in results))(module)
            outcome = capture_region(declared.__vlaforge_region__.as_ir(), declared, inputs, strict=True, absolute_tolerance=0, relative_tolerance=0)
            if not outcome.supported:
                report["capture_failure"] = outcome.report.to_dict()
                raise ValueError("actual native-output Tensor Region capture failed")
            report["capture_evidence"] = outcome.evidence.to_dict()
            export = args.output_dir / "output.exported.pt2"
            torch.export.save(outcome.exported_program, export)
            report["export"] = {"path": str(export), **file_digest(export)}
            reloaded_module = torch.export.load(export).module()
            reloaded = reloaded_module(*inputs)
            if args.checked and not reloaded[1].item():
                raise ValueError("saved publication predicate rejected actual sample")
            compare("saved_reload_" + args.device.split(":")[0], (reloaded[0] if args.checked else reloaded).cpu().numpy(), golden)
            if args.checked:
                report["rejection_checks"]["saved_reload"] = checked_output_rejections(reloaded_module, inputs)
            if not args.compile:
                context.require_current()
                report.update(status="passed", observed_after_capture=snapshot().to_dict())
                return
            target = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
            manifest = args.output_dir / "output.compile.json"
            command = ["compile-artifact", str(export), "--output", str(args.output_dir / "output.aoti.pt2"), "--manifest", str(manifest), "--target", target, "--inductor-profile", "aten-preserving"]
            report["compile_command"] = command
            report["observed_compile_before"] = snapshot().to_dict()
            report["compile_provenance"] = {
                "numerical_context": context.to_dict(), "target": target,
                "profile": "aten-preserving", "configs": aoti_configs("aten-preserving"),
                "context_implementation": file_digest(Path(vlaforge.numerical_context.__file__)),
                "source_files": {Path(item.__file__).name: file_digest(Path(item.__file__)) for item in (
                    vlaforge.cli, vlaforge.deployment.aoti_export,
                    vlaforge.deployment.aoti_package, vlaforge.deployment.aoti_profile)}}
            with (args.output_dir / "compile.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                if compiler_main(command) != 0:
                    raise ValueError("public native-output AOTI compiler failed")
            report["observed_compile_after"] = snapshot().to_dict()
            context.require_current()
            region = {"archive": report["export"], "capture_evidence": report["capture_evidence"]}
            compiled = verify_compiled_region(manifest, region, target=target, profile="aten-preserving")
            with ExitStack() as owners:
                if args.materialize_aoti:
                    from vlaforge.deployment import aoti_materialized
                    artifact = compiled["artifact"]
                    payload = aoti_materialized.materialize_aoti_package(artifact["path"], args.output_dir / "materialized",
                        sha256=artifact["sha256"], size_bytes=artifact["size_bytes"])
                    identity = file_digest(payload)
                    package = owners.enter_context(aoti_materialized.load_materialized_aoti(payload,
                        sha256=identity["sha256"], size_bytes=identity["size_bytes"], device=args.device))
                    report["materialized_package"] = package.extraction
                    report["package_loader_source"] = file_digest(Path(aoti_materialized.__file__))
                elif args.package_extraction_root is None:
                    package = torch._inductor.aoti_load_package(compiled["artifact"]["path"])
                else:
                    from vlaforge.deployment import aoti_load
                    artifact = compiled["artifact"]
                    package = owners.enter_context(aoti_load.load_aoti_package(artifact["path"],
                        extraction_root=args.package_extraction_root, sha256=artifact["sha256"],
                        size_bytes=artifact["size_bytes"], device=args.device))
                    report["package_extraction"] = package.extraction
                    report["package_loader_source"] = file_digest(Path(aoti_load.__file__))
                candidate = package(*inputs)
                if args.checked and not candidate[1].item():
                    raise ValueError("AOTI publication predicate rejected actual sample")
                candidate = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
                compare("aoti_cuda", candidate.cpu().numpy(), golden)
                if args.checked:
                    report["rejection_checks"]["aoti"] = checked_output_rejections(package, inputs)
                context.require_current()
            report.update(compiler_manifest=file_digest(manifest), artifact=compiled["artifact"], compiled=True, status="passed")
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
