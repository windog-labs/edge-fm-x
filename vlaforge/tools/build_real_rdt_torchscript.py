"""Compile and validate recorded RDT Regions with the public native ATen profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from build_real_rdt_fresh import (
    action_fidelity,
    digest,
    file_identity,
    read,
    runner_source,
    verify_capture,
    write,
)

SOURCE = Path(__file__).resolve().parents[1]


def verify_archive(root, region):
    from vlaforge.deployment import torchscript_export

    artifact = root / "artifacts" / (region["name"] + ".pt")
    manifest = root / "artifacts" / (region["name"] + ".compile.json")
    record = read(manifest)
    audit = record.get("region_validation", {})
    if (
        record.get("status") != "region_cases_passed"
        or record.get("backend") != "torchscript"
        or record.get("exported_program", {}).get("sha256") != region["export"]["sha256"]
        or record.get("artifact", {}).get("sha256") != digest(artifact)
        or record.get("artifact", {}).get("size_bytes") != artifact.stat().st_size
        or audit.get("backend_variant") != "torchscript-aten/1"
        or audit.get("status") != "region_cases_passed"
        or audit.get("jit_optimization") is not False
        or audit.get("archive_device_policy") != "preserve"
        or audit.get("implementation_sha256") != digest(Path(torchscript_export.__file__))
        or audit.get("artifact_sha256") != digest(artifact)
        or not audit.get("validation_cases")
        or any(case.get("bitwise_equal") is not True for case in audit["validation_cases"])
    ):
        raise ValueError("TorchScript candidate differs from its export/profile/implementation evidence")
    return record, artifact


def compile_regions(args, captured):
    folder = args.output / "artifacts"
    folder.mkdir(exist_ok=False)
    report = {"status": "started", "regions": [], "full_model_acceptance": False,
              "capture_sha256": digest(args.capture / "capture.json"),
              "tool": file_identity(Path(__file__)),
              "compiler_cli": file_identity(SOURCE / "python/vlaforge/cli.py"),
              "compiler_helper": file_identity(SOURCE / "python/vlaforge/deployment/torchscript_export.py")}
    write(args.output / "compile.json", report)
    for name, region in captured.items():
        command = [sys.executable, "-m", "vlaforge.cli", "compile-torchscript",
                   str(args.capture / "exports" / (name + ".pt2e")),
                   "--output", str(folder / (name + ".pt")),
                   "--manifest", str(folder / (name + ".compile.json"))]
        log = folder / (name + ".log")
        row = {"name": name, "command": command, "status": "running"}
        report["regions"].append(row)
        write(args.output / "compile.json", report)
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                    env={**os.environ, "PYTHONPATH": str(SOURCE / "python")})
        row.update(exit_code=result.returncode, log=file_identity(log), status="failed")
        if result.returncode == 0:
            manifest, artifact = verify_archive(args.output, region)
            row.update(status="region_cases_passed", artifact=file_identity(artifact),
                       validation=manifest["region_validation"])
        write(args.output / "compile.json", report)
        print(json.dumps({"region": name, "status": row["status"]}), flush=True)
    report["status"] = "all_regions_compiled" if all(
        row["status"] == "region_cases_passed" for row in report["regions"]
    ) else "partial_compile"
    write(args.output / "compile.json", report)
    return report


def direct(args, module, captured):
    import numpy as np
    import torch
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView

    folder = args.output / "direct"
    folder.mkdir(exist_ok=False)
    report = {"status": "started", "samples": [], "loaded": [], "stages": [],
              "capture_sha256": digest(args.capture / "capture.json"),
              "tool": file_identity(Path(__file__)),
              "no_python_deployment": False, "full_paper_acceptance": False,
              "jit_optimization": False, "timing_is_benchmark": False,
              "torch": torch.__version__, "matmul_precision": torch.get_float32_matmul_precision(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    write(folder / "report.json", report)
    implementations = {}
    for region in module.regions:
        _, path = verify_archive(args.output, captured[region.name])
        implementation = torch.jit.load(str(path))

        def observe(*inputs, _implementation=implementation, _name=region.name):
            result = _implementation(*inputs)
            outputs = result if isinstance(result, tuple) else (result,)
            path = folder / f"stage-{len(report['stages']):03d}-{_name}.pt"
            torch.save(tuple(value.detach().cpu().clone() for value in outputs), path)
            report["stages"].append({"region": _name, "output": file_identity(path)})
            write(folder / "report.json", report)
            return result

        implementations[region.name] = observe
        report["loaded"].append({"region": region.name, "artifact": file_identity(path)})
        write(folder / "report.json", report)
        print(json.dumps({"loaded": region.name}), flush=True)
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    with np.load(args.capture / "full_fresh_eager.npz", allow_pickle=False) as packed:
        reference = packed["reference"].copy()
    executor = Interpreter(module, regions=implementations, validators=InvocationProgram(module, {}).validators)
    for repetition in range(args.repetitions):
        bindings = {
            port.name: InputBinding(TensorView(inputs[port.name].to(port.device), port.payload.shape,
                    port.payload.dtype, device=port.device), InputStamp(revision=repetition + 1))
            for port in module.inputs
        }
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            result = executor.run(inputs=bindings).committed_outputs.output("action_chunk")
            torch.cuda.synchronize()
            actual = result.float().cpu().numpy()
        np.save(folder / f"full-actions-{repetition}.npy", actual, allow_pickle=False)
        fidelity = action_fidelity(reference, actual, inputs["action_mask"].float().numpy())
        write(folder / f"fidelity-{repetition}.json", fidelity)
        report["samples"].append({"index": repetition, "gates": fidelity["gates"],
            "active_metrics": fidelity["active"]["metrics"], "primary_acceptance_space": fidelity["primary_acceptance_space"],
            "unified_storage_bits_equal": np.array_equal(actual.view(np.uint32), reference.view(np.uint32)),
            "full_actions": file_identity(folder / f"full-actions-{repetition}.npy")})
        write(folder / "report.json", report)
    report["status"] = "executed_complete_chunks"
    report["all_active_exact"] = all(row["gates"]["exact_values"] for row in report["samples"])
    write(folder / "report.json", report)
    return report


def session(args, module, captured):
    import numpy as np
    import torch
    from vlaforge.deployment import (
        ArtifactIdentity,
        ArtifactKind,
        EffectAudit,
        RegionArtifactContract,
        ValueContract,
        WorkspaceContract,
        build_artifact_compile_bundle,
    )
    from vlaforge.deployment.capabilities import torchscript_backend_capability
    from vlaforge.frontend import InvocationProgram
    from vlaforge.ir.serializer import io_schema_digest

    folder = args.output / "session"
    folder.mkdir(exist_ok=False)
    direct_report = read(args.output / "direct/report.json")
    direct_path = args.output / "direct/full-actions-0.npy"
    capture_sha = digest(args.capture / "capture.json")
    if (direct_report["status"] != "executed_complete_chunks"
            or direct_report["capture_sha256"] != capture_sha
            or direct_report["samples"][0]["full_actions"] != file_identity(direct_path)):
        raise ValueError("complete hash-bound direct output must precede the native Session")
    reference_run = read(args.capture.parent / "report.json")
    contracts, sources = {}, {}
    auxiliary = {"evidence/capture.json": args.capture / "capture.json",
                 "evidence/direct.json": args.output / "direct/report.json"}
    for index, region in enumerate(module.regions):
        name = region.name
        _, artifact = verify_archive(args.output, captured[name])
        evidence = read(args.capture / "exports" / (name + ".capture.json"))
        contracts[name] = RegionArtifactContract(
            region_id=index, region_name=name,
            inputs=tuple(ValueContract.from_dict(item) for item in evidence["inputs"]),
            outputs=tuple(ValueContract.from_dict(item) for item in evidence["outputs"]),
            io_schema_digest=io_schema_digest(module),
            identity=ArtifactIdentity(model_name="RDT-online-fresh",
                upstream_revision=reference_run["provenance"]["source"]["revision"],
                checkpoint_identity="sha256:" + reference_run["provenance"]["assets"]["components"]["policy"]["files"]["pytorch_model.bin"]["sha256"],
                graph_sha256=evidence["graph_digest"]),
            artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE,
            artifact_path=f"artifacts/{name}.pt", artifact_sha256=digest(artifact),
            artifact_size_bytes=artifact.stat().st_size,
            workspace=WorkspaceContract(device="cuda:0"),
            # Select the runtime provider; archive validation alone does not certify replay.
            capability=torchscript_backend_capability(
                "sm_90", ("bf16", "f32", "i32", "i64", "bool"),
                shared_context=True,
            ),
            effect_audit=EffectAudit.from_dict(evidence["effect_audit"]),
            backend_variant="torchscript-aten-context/1",
        )
        sources[name] = artifact
        auxiliary[f"evidence/{name}.compile.json"] = args.output / "artifacts" / (name + ".compile.json")
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    input_folder = folder / "inputs"
    input_folder.mkdir()
    report = {"status": "building", "capture_sha256": capture_sha,
              "backend_variant": "torchscript-aten-context/1",
              "tool": file_identity(Path(__file__)), "samples": [], "inputs": {},
              "no_python_deployment": False, "full_paper_acceptance": False, "timing_is_benchmark": False}
    for port in module.inputs:
        path = input_folder / (port.name + ".bin")
        path.write_bytes(inputs[port.name].contiguous().view(torch.uint8).numpy().tobytes())
        report["inputs"][port.name] = file_identity(path)
    write(folder / "report.json", report)
    bundle = folder / "bundle"
    try:
        build_artifact_compile_bundle(
            module, bundle, region_artifacts=contracts, artifact_sources=sources,
            validators=InvocationProgram(module, {}).cpp_validators(), runner_source=runner_source(module),
            runtime_root=SOURCE, cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={"torchscript": torch.__version__, "cuda": str(torch.version.cuda)},
            profile="verified", loop_execution="source", source_revision=args.source_revision, source_dirty=True,
            default_device="cuda:0", state_device="cuda:0",
            environment={"TORCH_CUDA_ARCH_LIST": "9.0", "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
            auxiliary_files=auxiliary,
        )
    except subprocess.CalledProcessError as error:
        (folder / "build-failure.log").write_text(str(error.stdout or "") + str(error.stderr or ""))
        raise
    runner = bundle / "bin/vlaforge_generated_runner"
    dependencies = subprocess.check_output(["ldd", str(runner)], text=True)
    (folder / "runner.ldd.txt").write_text(dependencies)
    if any(name in dependencies.lower() for name in ("libpython", "libtorch_python")):
        raise ValueError("standalone runner unexpectedly links Python")
    command = [str(runner), str(bundle), str(input_folder), str(folder), str(args.repetitions)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONHOME": "/no/python/home", "PYTHONPATH": "/no/python/path"})
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    report.update(command=command, exit_code=completed.returncode, runner=file_identity(runner))
    with np.load(args.capture / "full_fresh_eager.npz", allow_pickle=False) as packed:
        reference = packed["reference"].copy()
    expected = np.load(direct_path, allow_pickle=False)
    for index in range(args.repetitions):
        path = folder / f"run-{index}.bin"
        if not path.is_file():
            continue
        actual = (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(reference.shape)
        official = action_fidelity(reference, actual, inputs["action_mask"].float().numpy())
        same_archive = action_fidelity(expected, actual, inputs["action_mask"].float().numpy())
        write(folder / f"run-{index}-official.json", official)
        write(folder / f"run-{index}-direct.json", same_archive)
        report["samples"].append({"index": index, "raw": file_identity(path),
            "official": official["gates"], "active_metrics": official["active"]["metrics"],
            "same_archive": same_archive["gates"],
            "same_archive_storage_bits_equal": np.array_equal(expected.view(np.uint32), actual.view(np.uint32))})
    maps = folder / "process-maps.txt"
    report["process_maps"] = file_identity(maps) if maps.is_file() else None
    report["loaded_python_library"] = any(name in maps.read_text().lower()
        for name in ("libpython", "libtorch_python")) if maps.is_file() else None
    report["no_python_deployment"] = completed.returncode == 0 and report["loaded_python_library"] is False
    report["status"] = "executed_complete_chunks" if completed.returncode == 0 and len(report["samples"]) == args.repetitions else "failed_or_incomplete"
    write(folder / "report.json", report)
    completed.check_returncode()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compile", "direct", "session"), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--source-revision", default="frozen-rdt-torchscript-v1")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 16:
        parser.error("bounded numerical validation requires 1..16 repetitions")
    args.capture, args.output = args.capture.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from vlaforge.adapters.rdt.rdt_reference import check_environment
        check_environment("torch210-cu128")
        _, module, captured = verify_capture(args.capture)
        if args.mode == "compile":
            result = compile_regions(args, captured)
        elif args.mode == "direct":
            result = direct(args, module, captured)
        else:
            result = session(args, module, captured)
    except Exception as error:
        write(args.output / (args.mode + "-failure.json"), {"status": "failed", "error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc(), "tool": file_identity(Path(__file__))})
        raise
    print(json.dumps(result, indent=2), flush=True)
    return 3 if result["status"] == "partial_compile" else 0


if __name__ == "__main__":
    raise SystemExit(main())
