"""Bounded different-observation RDT validation; not a performance benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from build_real_rdt_fresh import action_fidelity, digest, file_identity, read, verify_capture, write
from build_real_rdt_torchscript import verify_archive

SOURCE = Path(__file__).resolve().parents[1]


def json_provenance(value):
    """Represent nonfinite scheduler configuration values without JSON NaN/Inf."""
    if isinstance(value, dict):
        return {key: json_provenance(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_provenance(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"encoding": "nonfinite-number", "value": repr(value)}
    return value


def history_steps(frame_count, count):
    if type(frame_count) is not int or type(count) is not int or not 8 <= count <= 16 or frame_count <= count:
        raise ValueError("8..16 distinct valid two-frame observations are required")
    return [1 + ((frame_count - 2) * index) // (count - 1) for index in range(count)]


def validate_model_inputs(values, module):
    import torch

    types = {"bf16": torch.bfloat16, "f32": torch.float32, "i64": torch.int64,
             "i32": torch.int32, "bool": torch.bool}
    if set(values) != {port.name for port in module.inputs}:
        raise ValueError("saved model inputs differ from declared ports")
    for port in module.inputs:
        value = values[port.name]
        if (tuple(value.shape) != port.payload.shape or value.dtype != types[port.payload.dtype]
                or not value.is_contiguous() or not torch.isfinite(value).all()):
            raise ValueError(f"saved model input changed static ABI: {port.name}")


def prepare(args):
    import h5py
    from vlaforge.adapters.rdt.rdt_inputs import FIRST_EPISODE_MEMBERS, prepare_recorded_input

    with h5py.File(args.episode_root / FIRST_EPISODE_MEMBERS[0], "r") as episode:
        steps = history_steps(int(episode["observations/qpos"].shape[0]), args.count)
    report = {"status": "preparing", "episode_count": 1, "samples": [], "tool": file_identity(Path(__file__)),
              "full_dataset_verified": False, "formal_performance_measured": False}
    for index, step in enumerate(steps):
        folder = args.output / "observations" / f"sample-{index}"
        manifest = prepare_recorded_input(episode_root=args.episode_root, source_root=args.source_root,
            destination=folder, step=step, seed=args.seed_base + index, image_decoding="upstream-opencv-array")
        report["samples"].append({"index": index, "step": step, "seed": args.seed_base + index,
            "input_manifest": file_identity(folder / "manifest.json"), "inputs": manifest["inputs"]})
        write(args.output / "observations.json", report)
    report["status"] = "distinct_recorded_observations_prepared"
    write(args.output / "observations.json", report)
    return report


def reference(args, module, observations):
    import numpy as np
    import torch
    from PIL import Image
    from vlaforge.adapters.rdt.rdt_fresh import inputs_from_official_reference
    from vlaforge.adapters.rdt.rdt_inputs import load_recorded_input
    from vlaforge.adapters.rdt.rdt_reference import RDTConfig, load_rdt, official_agilex_reference

    report = {"status": "loading_official_models", "samples": [], "tool": file_identity(Path(__file__)),
              "observations_manifest": file_identity(args.output / "observations.json"),
              "timing_is_benchmark": False, "physical_action_units_verified": False}
    write(args.output / "reference.json", report)
    loaded = load_rdt(RDTConfig(args.source_root, args.assets_root, "cuda:0", "torch210-cu128"))
    report["provenance"] = json_provenance(loaded.provenance)
    report["provenance_nonfinite_encoding"] = "tagged metadata only; original in-memory scheduler config unchanged"
    report["status"] = "running_official_reference"
    write(args.output / "reference.json", report)
    for sample in observations["samples"]:
        index = sample["index"]
        pack = args.output / "observations" / f"sample-{index}"
        if file_identity(pack / "manifest.json") != sample["input_manifest"]:
            raise ValueError("recorded observation manifest changed")
        arrays, manifest = load_recorded_input(pack, source_root=args.source_root)
        traces = official_agilex_reference(loaded, instruction=manifest["instruction"],
            images=[Image.fromarray(arrays[f"image_{i}"]) for i in range(6)],
            proprio=torch.from_numpy(arrays["proprio"].copy()),
            control_frequency=manifest["control_frequency"], seed=manifest["seed"])
        inputs = {name: value.cpu() for name, value in inputs_from_official_reference(loaded, traces).items()}
        validate_model_inputs(inputs, module)
        folder = args.output / "references" / f"sample-{index}"
        folder.mkdir(parents=True, exist_ok=False)
        torch.save(inputs, folder / "model-inputs.pt")
        np.save(folder / "official-actions.npy", traces["unified_actions"].float().cpu().numpy(), allow_pickle=False)
        np.save(folder / "official-robot-actions.npy", traces["robot_actions"].float().cpu().numpy(), allow_pickle=False)
        state = {name: traces[name].detach().cpu() for name in (
            "noise", "rng_before", "rng_after", "scheduler_timesteps", "scheduler_sigmas")}
        state["model_outputs"] = tuple(value.detach().cpu() for value in traces["model_outputs"])
        torch.save(state, folder / "official-scheduler.pt")
        native_inputs = args.output / "native-inputs" / f"sample-{index}"
        native_inputs.mkdir(parents=True, exist_ok=False)
        for name, value in inputs.items():
            (native_inputs / (name + ".bin")).write_bytes(value.view(torch.uint8).numpy().tobytes())
        row = {"index": index, "step": sample["step"], "seed": sample["seed"],
               "files": {path.name: file_identity(path) for path in folder.iterdir()},
               "native_inputs": {path.name: file_identity(path) for path in native_inputs.iterdir()},
               "saved_noise_sha256": digest(native_inputs / "noise.bin")}
        report["samples"].append(row)
        write(args.output / "reference.json", report)
        print(json.dumps({"reference_sample": index, "step": sample["step"]}), flush=True)
    if len({row["saved_noise_sha256"] for row in report["samples"]}) != len(report["samples"]):
        raise ValueError("independently seeded samples unexpectedly share saved noise")
    report["status"] = "complete_official_reference_series"
    write(args.output / "reference.json", report)
    return report


def load_reference(args, row, module):
    import numpy as np
    import torch

    folder = args.output / "references" / f"sample-{row['index']}"
    for name, identity in row["files"].items():
        if file_identity(folder / name) != identity:
            raise ValueError("official reference file changed")
    values = torch.load(folder / "model-inputs.pt", map_location="cpu", weights_only=True)
    validate_model_inputs(values, module)
    return values, np.load(folder / "official-actions.npy", allow_pickle=False)


def direct(args, module, captured, references):
    import numpy as np
    import torch
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView

    folder = args.output / "direct"
    folder.mkdir(exist_ok=False)
    report = {"status": "loading_archives", "samples": [], "loaded": [], "calls": {},
              "tool": file_identity(Path(__file__)),
              "capture_sha256": digest(args.capture / "capture.json"),
              "reference_manifest": file_identity(args.output / "reference.json"),
              "same_session_across_samples": True, "no_python_deployment": False,
              "timing_is_benchmark": False, "full_paper_acceptance": False}
    implementations = {}
    for region in module.regions:
        _, artifact = verify_archive(args.compiled_root, captured[region.name])
        loaded = torch.jit.load(str(artifact))
        def call(*inputs, _loaded=loaded, _name=region.name):
            report["calls"][_name] = report["calls"].get(_name, 0) + 1
            return _loaded(*inputs)
        implementations[region.name] = call
        report["loaded"].append({"region": region.name, "artifact": file_identity(artifact)})
        write(folder / "report.json", report)
    executor = Interpreter(module, regions=implementations, validators=InvocationProgram(module, {}).validators)
    for row in references["samples"]:
        values, expected = load_reference(args, row, module)
        bindings = {port.name: InputBinding(TensorView(values[port.name].to(port.device), port.payload.shape,
            port.payload.dtype, device=port.device), InputStamp(revision=row["index"] + 1)) for port in module.inputs}
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            actual = executor.run(inputs=bindings).committed_outputs.output("action_chunk").float().cpu().numpy()
        np.save(folder / f"sample-{row['index']}.npy", actual, allow_pickle=False)
        fidelity = action_fidelity(expected, actual, values["action_mask"].float().numpy(),
            sample_id=f"episode5-step{row['step']}-seed{row['seed']}")
        write(folder / f"sample-{row['index']}.json", fidelity)
        report["samples"].append({"index": row["index"], "step": row["step"], "seed": row["seed"], "gates": fidelity["gates"],
            "active_metrics": fidelity["active"]["metrics"],
            "unified_bits_equal": np.array_equal(expected.view(np.uint32), actual.view(np.uint32)),
            "output": file_identity(folder / f"sample-{row['index']}.npy")})
        write(folder / "report.json", report)
        print(json.dumps({"direct_sample": row["index"], "gates": fidelity["gates"]}), flush=True)
    report["status"] = "complete_direct_series"
    write(folder / "report.json", report)
    return report


def native(args, module, references):
    import numpy as np

    direct_report = read(args.output / "direct/report.json")
    if (direct_report["status"] != "complete_direct_series"
            or direct_report["reference_manifest"] != file_identity(args.output / "reference.json")
            or [row["index"] for row in direct_report["samples"]]
            != [row["index"] for row in references["samples"]]):
        raise ValueError("complete direct series required before native comparison")
    folder = args.output / "native"
    folder.mkdir(exist_ok=False)
    report = {"status": "building_series_runner", "samples": [], "tool": file_identity(Path(__file__)),
              "capture_sha256": digest(args.capture / "capture.json"),
              "reference_manifest": file_identity(args.output / "reference.json"),
              "direct_manifest": file_identity(args.output / "direct/report.json"),
              "same_session_across_samples": True, "timing_is_benchmark": False,
              "no_python_deployment": False, "full_paper_acceptance": False}
    write(folder / "report.json", report)
    build = args.output / "native-build"
    build.mkdir(exist_ok=False)
    # Reuse immutable, verified archives; only the small Session runner is built.
    (build / "artifacts").symlink_to((args.compiled_root / "artifacts").resolve(), target_is_directory=True)
    (build / "direct").symlink_to((args.compiled_root / "direct").resolve(), target_is_directory=True)
    command = [sys.executable, str(SOURCE / "tools/build_real_rdt_torchscript.py"),
        "--mode", "session", "--capture", str(args.capture), "--output", str(build),
        "--repetitions", "1", "--source-revision", "tool-sha256:" + digest(Path(__file__))]
    with (folder / "build-and-original-sample-smoke.log").open("w") as log:
        built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report["build_command"] = command
    report["build_exit_code"] = built.returncode
    write(folder / "report.json", report)
    built.check_returncode()
    bundle = build / "session/bundle"
    runner = bundle / "bin/vlaforge_generated_runner"
    for row in references["samples"]:
        inputs = args.output / "native-inputs" / f"sample-{row['index']}"
        for name, identity in row["native_inputs"].items():
            if file_identity(inputs / name) != identity:
                raise ValueError("native series input changed before execution")
    command = [str(runner), str(bundle), str(args.output / "native-inputs"), str(folder),
               str(len(references["samples"])), "series"]
    report.update(status="running_native_series", command=command, runner=file_identity(runner))
    write(folder / "report.json", report)
    completed = subprocess.run(command, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONHOME": "/no/python/home", "PYTHONPATH": "/no/python/path"})
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    report["exit_code"] = completed.returncode
    for row, candidate in zip(references["samples"], direct_report["samples"], strict=True):
        index = row["index"]
        path = folder / f"run-{index}.bin"
        if not path.is_file():
            continue
        values, expected = load_reference(args, row, module)
        direct_path = args.output / "direct" / f"sample-{index}.npy"
        if file_identity(direct_path) != candidate["output"]:
            raise ValueError("direct output changed before native comparison")
        direct_value = np.load(direct_path, allow_pickle=False)
        actual = (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(expected.shape)
        sample_id = f"episode5-step{row['step']}-seed{row['seed']}"
        fidelity = action_fidelity(expected, actual, values["action_mask"].float().numpy(), sample_id=sample_id)
        same_archive = action_fidelity(direct_value, actual, values["action_mask"].float().numpy(), sample_id=sample_id)
        write(folder / f"sample-{index}-official.json", fidelity)
        write(folder / f"sample-{index}-direct.json", same_archive)
        report["samples"].append({"index": index, "step": row["step"], "seed": row["seed"], "raw": file_identity(path), "gates": fidelity["gates"],
            "active_metrics": fidelity["active"]["metrics"],
            "official_unified_bits_equal": np.array_equal(expected.view(np.uint32), actual.view(np.uint32)),
            "same_archive_bits_equal": np.array_equal(direct_value.view(np.uint32), actual.view(np.uint32))})
    maps = folder / "process-maps.txt"
    report["loaded_python_library"] = any(name in maps.read_text().lower()
        for name in ("libpython", "libtorch_python")) if maps.is_file() else None
    report["process_maps"] = file_identity(maps) if maps.is_file() else None
    report["no_python_deployment"] = completed.returncode == 0 and report["loaded_python_library"] is False
    report["status"] = "complete_native_series" if completed.returncode == 0 and len(report["samples"]) == len(references["samples"]) else "failed_or_incomplete"
    write(folder / "report.json", report)
    completed.check_returncode()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "reference", "direct", "native"), required=True)
    for name in ("output", "source-root", "assets-root", "episode-root", "capture", "compiled-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=2026090600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "prepare":
            result = prepare(args)
        else:
            from vlaforge.adapters.rdt.rdt_reference import check_environment
            check_environment("torch210-cu128")
            _, module, captured = verify_capture(args.capture)
            observations = read(args.output / "observations.json")
            if observations["status"] != "distinct_recorded_observations_prepared":
                raise ValueError("complete verified observation inputs required")
            if args.mode == "reference":
                result = reference(args, module, observations)
            else:
                references = read(args.output / "reference.json")
                if references["status"] != "complete_official_reference_series":
                    raise ValueError("complete official reference series required")
                if (references["observations_manifest"] != file_identity(args.output / "observations.json")
                        or [(row["index"], row["step"], row["seed"]) for row in references["samples"]]
                        != [(row["index"], row["step"], row["seed"]) for row in observations["samples"]]):
                    raise ValueError("official references differ from the complete observation series")
                result = direct(args, module, captured, references) if args.mode == "direct" else native(args, module, references)
    except Exception as error:
        write(args.output / (args.mode + "-failure.json"), {"status": "failed", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "tool": file_identity(Path(__file__))})
        raise
    print(json.dumps({"status": result["status"], "samples": len(result["samples"])}), flush=True)


if __name__ == "__main__":
    main()
