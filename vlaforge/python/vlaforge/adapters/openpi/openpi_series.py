"""Pinned multi-observation OpenPI references and complete compiled-IR checks.

Both supported OpenPI configurations use this adapter with their own source
capture and checkpoint. Native measurement remains the generic Session tool.
"""

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.deployment.numerical import strict_json


def load_series_index(path):
    path = Path(path).resolve()
    index = strict_json(path.read_text())
    if index.get("schema") != "vlaforge.openpi_observation_series/1":
        raise ValueError("a versioned observation series is required")
    samples = index.get("samples")
    if not isinstance(samples, list) or not 2 <= len(samples) <= 4096:
        raise ValueError("observation series requires 2..4096 samples")
    names, observations, result = set(), [], []
    for sample in samples:
        name = sample.get("sample_id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("observation series sample IDs must be unique")
        names.add(name)
        source = (path.parent / sample["manifest"]).resolve(strict=True)
        if not source.is_relative_to(path.parent) or file_digest(source)["sha256"] != sample["sha256"]:
            raise ValueError("observation series manifest escaped or changed")
        manifest = strict_json(source.read_text())
        if (manifest.get("schema") != "vlaforge.openpi_input_pack/1"
                or manifest["dataset"] != index["dataset"] or manifest["revision"] != index["revision"]
                or any(manifest[key] != sample[key] for key in ("episode_index", "frame_index"))
                or manifest["noise"]["seed"] != sample["noise_seed"]):
            raise ValueError("observation series provenance differs")
        observation = (sample["episode_index"], sample["frame_index"])
        if any(type(value) is not int or value < 0 for value in observation):
            raise ValueError("observation series indices must be nonnegative integers")
        observations.append(observation)
        result.append((source, sample))
    if observations != sorted(set(observations)):
        raise ValueError("observation series must contain distinct chronological observations")
    return index, tuple(result)


def _checked(path, digest):
    if file_digest(path)["sha256"] != digest:
        raise ValueError("source report identity differs")
    return strict_json(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reference", "direct"), required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--input-series", type=Path, required=True)
    parser.add_argument("--input-series-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--package-extraction-root", type=Path,
        help="Existing canonical owned 0700 directory for direct AOTI archive loads")
    args = parser.parse_args()
    if args.package_extraction_root is not None and args.phase != "direct":
        parser.error("package extraction is only used by direct validation")
    if args.phase == "direct" and any(value is None for value in (
            args.bundle, args.bundle_sha256, args.reference_report, args.reference_sha256)):
        parser.error("direct validation requires a complete bundle and reference identity")
    from cogact_gpu_monitor import child_handshake
    child_handshake()
    import numpy as np
    import torch

    from vlaforge.adapters.openpi.openpi_frontend import (
        OpenPIConfig,
        load_openpi,
        prepare_openpi_inputs,
    )
    from vlaforge.adapters.openpi.openpi_inputs import load_openpi_input_pack
    from vlaforge.deployment import ArtifactKind, load_bundle_manifest
    from vlaforge.deployment.aoti_load import load_aoti_package
    from vlaforge.deployment.aoti_materialized import load_materialized_aoti
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputStamp, Interpreter, TensorView
    from vlaforge.ir.serializer import parse_canonical_json
    from vlaforge.numerical_context import NumericalContext, offline_restore, snapshot
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    capture = _checked(args.capture_report, args.capture_sha256)
    _checked(args.input_series, args.input_series_sha256)
    series, samples = load_series_index(args.input_series)
    context = NumericalContext.from_dict(capture["numerical_context"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vlaforge.openpi_observation_execution/1", "status": "running", "phase": args.phase,
        "pid": os.getpid(), "source_capture_sha256": args.capture_sha256, "input_series_sha256": args.input_series_sha256,
        "samples": [], "files": {}, "numerical_context": context.to_dict(), "model_config": capture["config_name"],
        "num_steps": capture["num_steps"], "dataset": series["dataset"], "revision": series["revision"],
        "tool": file_digest(Path(__file__)), "formal_benchmark": False, "no_python_deployment": False,
        "physical_calibration_verified": False, "board_verified": False, "full_paper_acceptance": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def persist(path, value):
        np.save(path, value, allow_pickle=False)
        report["files"][str(path.relative_to(args.output))] = file_digest(path)

    save()
    try:
        with offline_restore(context, acknowledge_process_global=True), torch.inference_mode(), ExitStack() as owners:
            if args.phase == "reference":
                config = capture["processor_config"]
                loaded = load_openpi(OpenPIConfig(source_root=Path(config["source_root"]),
                    checkpoint_dir=Path(config["checkpoint_dir"]), config_name=capture["config_name"],
                    checkpoint_sha256=capture["checkpoint_provenance"]["checkpoint"]["sha256"],
                    device="cuda:0", num_steps=capture["num_steps"]))
                report["checkpoint_provenance"] = loaded.provenance
            else:
                reference = _checked(args.reference_report, args.reference_sha256)
                if (reference["status"] != "passed" or reference["phase"] != "reference"
                        or reference["input_series_sha256"] != args.input_series_sha256
                        or reference["source_capture_sha256"] != args.capture_sha256
                        or reference["numerical_context"] != context.to_dict()):
                    raise ValueError("complete same-input/model numerical reference required")
                _checked(args.bundle / "bundle.json", args.bundle_sha256)
                manifest = load_bundle_manifest(args.bundle / "bundle.json")
                manifest.verify_files(args.bundle)
                module = parse_canonical_json((args.bundle / manifest.semantic_ir.path).read_text())
                runners = {}
                private = args.package_extraction_root or args.output / "private-packages"
                for item in manifest.region_artifacts:
                    if item.artifact_kind is ArtifactKind.AOTI_MATERIALIZED:
                        owner = load_materialized_aoti(args.bundle / item.artifact_path,
                            sha256=item.artifact_sha256, size_bytes=item.artifact_size_bytes, device="cuda:0")
                    elif item.artifact_kind is ArtifactKind.AOTI_PACKAGE:
                        if args.package_extraction_root is None:
                            private.mkdir(mode=0o700, exist_ok=True)
                        owner = load_aoti_package(args.bundle / item.artifact_path, extraction_root=private,
                            sha256=item.artifact_sha256, size_bytes=item.artifact_size_bytes, device="cuda:0")
                    else:
                        raise ValueError("this validation path requires complete AOTI artifacts")
                    runners[item.region_name] = owners.enter_context(owner)
                    report.setdefault("artifact_loads", {})[item.region_name] = owner.extraction
                interpreter = Interpreter(module, regions=runners, validators=InvocationProgram(module, {}).validators)
                report.update(bundle_sha256=args.bundle_sha256, reference_sha256=args.reference_sha256)
            for ordinal, (path, sample) in enumerate(samples):
                folder = args.output / str(ordinal)
                folder.mkdir()
                entry = {"sample_id": sample["sample_id"], "input_manifest_sha256": file_digest(path)["sha256"],
                    "episode_index": sample["episode_index"], "frame_index": sample["frame_index"]}
                cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                if args.phase == "reference":
                    observation, noise, _ = load_openpi_input_pack(path, source_root=loaded.config.source_root)
                    prepared = prepare_openpi_inputs(loaded, observation, noise=noise)
                    if list(prepared.image_memory_formats) != capture["official_image_memory_formats"]:
                        raise ValueError("official image layout profile changed")
                    normalized = loaded.model.sample_actions("cuda:0", prepared.observation,
                        noise=prepared.tensors["noise"].clone(), num_steps=loaded.config.num_steps)
                    native = loaded.policy._output_transform({"state": prepared.observation.state[0].cpu().numpy(),
                        "actions": normalized[0].cpu().numpy()})["actions"][None]
                    arrays = {name: value.cpu().numpy() for name, value in prepared.tensors.items()}
                    np.savez(folder / "prepared.npz", **arrays)
                    report["files"][str((folder / "prepared.npz").relative_to(args.output))] = file_digest(folder / "prepared.npz")
                    for name, value in (("normalized_action_chunk", normalized.cpu().numpy()), ("native_action_chunk", native)):
                        if not np.isfinite(value).all():
                            raise ValueError("official complete action is nonfinite")
                        persist(folder / (name + ".npy"), value)
                    entry["image_memory_formats"] = list(prepared.image_memory_formats)
                else:
                    if reference["samples"][ordinal]["sample_id"] != sample["sample_id"]:
                        raise ValueError("reference observation ordering differs")
                    base = args.reference_report.parent / str(ordinal)
                    for name in ("prepared.npz", *[item.name + ".npy" for item in module.outputs]):
                        if file_digest(base / name) != reference["files"][str(ordinal) + "/" + name]:
                            raise ValueError("reference complete arrays changed")
                    with np.load(base / "prepared.npz", allow_pickle=False) as pack:
                        for port in module.inputs:
                            value = torch.from_numpy(pack[port.name].copy()).to(port.device)
                            interpreter.bind_input(port.name, TensorView(value, tuple(value.shape), port.payload.dtype,
                                port.payload.layout, port.device, port.alignment), InputStamp(revision=ordinal + 1))
                            binary = folder / (port.name + ".bin")
                            binary.write_bytes(pack[port.name].tobytes())
                            report["files"][str(binary.relative_to(args.output))] = file_digest(binary)
                    execution = interpreter.run()
                    execution.trace.write(folder / "trace.json")
                    report["files"][str((folder / "trace.json").relative_to(args.output))] = file_digest(folder / "trace.json")
                    entry["outputs"] = {}
                    for port in module.outputs:
                        actual = interpreter.read_output(port.name).cpu().numpy()
                        golden = np.load(base / (port.name + ".npy"), allow_pickle=False)
                        persist(folder / (port.name + ".npy"), actual)
                        metric = compare_action_chunk(golden, actual, sample_id=sample["sample_id"], space=port.name,
                            contract=NumericContract(absolute_tolerance=0, relative_tolerance=0))
                        metric["bitwise_equal"] = actual.dtype == golden.dtype and actual.shape == golden.shape and actual.tobytes() == golden.tobytes()
                        entry["outputs"][port.name] = metric
                torch.cuda.synchronize()
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise ValueError("explicit-noise observation execution changed RNG state")
                context.require_current()
                entry["rng_state_unchanged"] = True
                report["samples"].append(entry)
                save()
            report["observed_numerical_context_after"] = snapshot().to_dict()
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "passed"
        if args.phase == "direct" and not all(result["bitwise_equal"] for entry in report["samples"] for result in entry["outputs"].values()):
            report["status"] = "failed_complete_output_parity"
            raise ValueError("compiled full-IR outputs differ on held-out observations")
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
