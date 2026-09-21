#!/usr/bin/env python3
"""Compile and execute verified full online RDT capture through shared APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile

from vlaforge.adapters.rdt.rdt_assets import file_identity

SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_capture(root):
    from vlaforge.ir.serializer import module_from_data

    record = read(root / "capture.json")
    if record["status"] != "captured_all_regions" or not record["same_torch_parity"]["full_chunk_exact"]:
        raise ValueError("complete capture and same-version full-action reference gate required")
    for filename, key in (("module.json", "module"), ("inputs.pt", "inputs"), ("full_fresh_eager.npz", "full_fresh_eager")):
        if file_identity(root / filename) != record[key]:
            raise ValueError(f"captured file identity changed: {filename}")
    module = module_from_data(read(root / "module.json"))
    records = {item["name"]: item for item in record["regions"]}
    if len(records) != len(record["regions"]) or set(records) != {region.name for region in module.regions}:
        raise ValueError("capture region inventory differs from serialized IR")
    for name, item in records.items():
        for suffix, key in ((".pt2e", "export"), (".capture.json", "evidence"), (".inputs.pt", "examples")):
            if item["status"] != "captured" or file_identity(root / "exports" / (name + suffix)) != item[key]:
                raise ValueError(f"captured region identity changed: {name}{suffix}")
    return record, module, records


def verify_artifact(root, name, captured, profile):
    from vlaforge.deployment.aoti_export import backend_pass_records, backend_program_pass_records
    from vlaforge.deployment.aoti_package import package_pass_records
    from vlaforge.deployment.aoti_profile import aoti_configs

    package = root / (name + ".pt2")
    record = read(root / (name + ".compile.json"))
    configs = aoti_configs(profile)
    program_audit = record.get("backend_program_audit", {})
    expected_program_passes = backend_program_pass_records(configs)
    if (record.get("status") != "passed" or record.get("target") != "sm_90"
        or record.get("inductor_profile") != profile or record.get("inductor_configs") != configs
        or record.get("backend_graph_passes") != backend_pass_records(configs)
        or program_audit.get("passes", []) != expected_program_passes
        or (expected_program_passes and not isinstance(program_audit.get("rewrites"), list))
        or record.get("backend_package_audit", {}).get("passes") != package_pass_records(configs)
        or record.get("exported_program", {}).get("sha256") != captured["export"]["sha256"]
        or record.get("artifact", {}).get("sha256") != digest(package)
        or record["artifact"].get("size_bytes") != package.stat().st_size):
        raise ValueError(f"compiled artifact/profile/source mismatch: {name}")
    return record


def unpack_package(package, destination):
    """Keep the full archive layout and bind every extracted dependency."""
    package_identity = {"sha256": digest(package), "size_bytes": package.stat().st_size}
    manifest_path = destination / "extraction.json"
    if destination.exists():
        manifest = read(manifest_path)
        if destination.is_symlink() or manifest["package"] != package_identity:
            raise ValueError("extraction belongs to a different package")
        with zipfile.ZipFile(package) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if (set(manifest["files"]) != {member.filename for member in members}
                or [member.filename for member in members if member.filename.endswith(".so")] != [manifest["library"]]):
                raise ValueError("extracted dependency inventory changed")
            for member in members:
                relative = member.filename
                path = destination / relative
                if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(destination.resolve())
                    or file_identity(path) != manifest["files"][relative]):
                    raise ValueError(f"extracted dependency changed: {relative}")
                with archive.open(member) as source:
                    source_sha = hashlib.file_digest(source, "sha256").hexdigest()
                if source_sha != manifest["files"][relative]["sha256"]:
                    raise ValueError(f"extracted dependency differs from original package: {relative}")
        return manifest
    destination.mkdir(parents=True)
    manifest = {"package": package_identity, "files": {}, "status": "extracting"}
    with zipfile.ZipFile(package) as archive:
        members = archive.infolist()
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or not path.parts or member.filename in names or path.parts[0] == "extraction.json"
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (mode == stat.S_IFDIR and not member.is_dir())):
                raise ValueError(f"unsafe archive member: {member.filename}")
            names.add(member.filename)
        libraries = [member.filename for member in members if member.filename.endswith(".so")]
        if len(libraries) != 1:
            raise ValueError("raw loader requires one complete precompiled shared library")
        # This profile embeds mmap weights in its .so. Separate constant blobs
        # need the PackageLoader relocation contract, which is not guessed here.
        if any("data/constants" in str(PurePosixPath(name).parent) for name in names):
            raise ValueError("separate constant-blob relocation is not supported by this extraction profile")
        for member in members:
            path = destination / member.filename
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            if path.stat().st_size != member.file_size:
                raise ValueError(f"extracted size mismatch: {member.filename}")
            manifest["files"][member.filename] = file_identity(path)
    manifest.update(status="verified_complete_archive", library=libraries[0],
                    weight_storage="package_shared_library_embedded_mmap")
    write(manifest_path, manifest)
    return manifest


def raw_callable(runner, output_arity):
    def invoke(*values):
        outputs = runner.run(list(values))
        if len(outputs) != output_arity:
            raise ValueError("raw runner output arity differs from captured Region")
        return outputs[0] if output_arity == 1 else tuple(outputs)
    return invoke


def compile_all(args, record, module, captured):
    folder = args.output / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for region in module.regions:
        name = region.name
        item = {"name": name, "status": "started"}
        results.append(item)
        write(args.output / "compile.json", {"regions": results, "status": "running"})
        try:
            package, manifest = folder / (name + ".pt2"), folder / (name + ".compile.json")
            if args.reuse_solver is not None and name == "rdt_solver" and not package.exists() and not manifest.exists():
                verify_artifact(args.reuse_solver, name, captured[name], args.profile)
                shutil.copyfile(args.reuse_solver / package.name, package)
                shutil.copyfile(args.reuse_solver / manifest.name, manifest)
                item["reused_verified_package"] = str(args.reuse_solver)
            if package.exists() or manifest.exists():
                result = verify_artifact(folder, name, captured[name], args.profile)
            else:
                command = [sys.executable, "-m", "vlaforge.cli", "compile-artifact",
                           str(args.capture / "exports" / (name + ".pt2e")),
                           "--output", str(package), "--manifest", str(manifest),
                           "--target", "sm_90", "--inductor-profile", args.profile]
                item["command"] = command
                write(args.output / "compile.json", {"regions": results, "status": "running"})
                with (folder / (name + ".compile.log")).open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                        env={**os.environ, "PYTHONPATH": str(SOURCE / "python"), "TORCHINDUCTOR_COMPILE_THREADS": "1"})
                item["exit_code"] = completed.returncode
                completed.check_returncode()
                result = verify_artifact(folder, name, captured[name], args.profile)
            item.update(status="compiled_candidate", artifact=result["artifact"], compile_seconds=result["compile_seconds"])
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write(args.output / "compile.json", {"regions": results, "status": "running"})
        print(json.dumps({"region": name, "status": item["status"]}), flush=True)
    result = {"status": "compiled_all_candidates" if all(item["status"] == "compiled_candidate" for item in results) else "partial_compile",
              "regions": results, "full_model_numerical_acceptance": False,
              "capture_sha256": digest(args.capture / "capture.json"), "profile": args.profile,
              "tool": file_identity(Path(__file__))}
    write(args.output / "compile.json", result)
    return result


def fidelity(reference, actual, *, sample_id="recorded-episode5-step64"):
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    report = compare_action_chunk(reference, actual, sample_id=sample_id,
                                  space="unified-model-action-space", contract=NumericContract(0, 0))
    metrics = report["metrics"]
    report["gates"] = {
        "exact_values": metrics["exact_values"],
        "technical_tolerance": metrics["maximum_absolute_error"] <= .05 and metrics["mean_absolute_error"] <= .01,
        "paper_numeric_threshold": metrics["mean_squared_error"] <= 1e-5 and metrics["cosine_similarity"] is not None and metrics["cosine_similarity"] >= .9999,
        "full_paper_acceptance": False,
    }
    return report


def action_fidelity(reference, actual, action_mask, *, sample_id="recorded-episode5-step64"):
    import numpy as np

    mask = np.asarray(action_mask)
    if mask.size != reference.shape[-1] or not np.isin(mask, (0, 1)).all():
        raise ValueError("one explicit binary embodiment action mask is required")
    indices = np.flatnonzero(mask.reshape(-1))
    if not len(indices):
        raise ValueError("the embodiment must declare active action dimensions")
    unified = fidelity(reference, actual, sample_id=sample_id)
    active = fidelity(reference[..., indices], actual[..., indices], sample_id=sample_id)
    active["space"] = "active-embodiment-action-space"
    return {"schema": "vlaforge.rdt_action_fidelity/1", "unified": unified, "active": active,
            "active_indices": indices.tolist(), "primary_acceptance_space": "active-embodiment-action-space",
            "gates": {**active["gates"], "unified_paper_numeric_threshold": unified["gates"]["paper_numeric_threshold"]}}


def execute(args, record, module, captured):
    import numpy as np
    import torch
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.frontend import InvocationProgram, load_exported_region
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView

    folder = args.output / args.mode
    folder.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "mode": args.mode, "aoti_loader": args.aoti_loader, "stages": [], "loaded": [], "no_python_deployment": False,
              "capture_sha256": digest(args.capture / "capture.json"), "tool": file_identity(Path(__file__))}
    write(folder / "report.json", report)
    implementations = {}
    for region in module.regions:
        name = region.name
        if args.mode == "exports":
            exported = load_exported_region(args.capture / "exports" / (name + ".pt2e"))
            implementation = exported.module()
        else:
            verify_artifact(args.output / "artifacts", name, captured[name], args.profile)
            package = args.output / "artifacts" / (name + ".pt2")
            if args.aoti_loader == "extracted":
                destination = args.output / "extracted" / name
                extraction = unpack_package(package, destination)
                library = destination / extraction["library"]
                runner = torch._C._aoti.AOTIModelContainerRunnerCuda(str(library), 1, "cuda:0", str(library.parent), True)
                implementation = raw_callable(runner, len(region.outputs))
                report["loaded"].append({"region": name, "extraction": file_identity(destination / "extraction.json")})
            else:
                implementation = torch._inductor.aoti_load_package(str(package))
        def observe(*inputs, _name=name, _implementation=implementation):
            with torch.no_grad():
                result = _implementation(*inputs)
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            sequence = len(report["stages"])
            path = folder / f"stage-{sequence:03d}-{_name}.pt"
            torch.save(tuple(value.detach().cpu().clone() for value in outputs), path)
            report["stages"].append({"sequence": sequence, "region": _name, "output": file_identity(path),
                                     "shapes": [list(value.shape) for value in outputs]})
            write(folder / "report.json", report)
            return result
        implementations[name] = observe
        print(json.dumps({"loaded_region": name, "mode": args.mode}), flush=True)
    inputs = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    bindings = {port.name: InputBinding(TensorView(inputs[port.name].to(port.device), port.payload.shape,
                    port.payload.dtype, device=port.device), InputStamp(revision=1)) for port in module.inputs}
    executor = Interpreter(module, regions=implementations, validators=InvocationProgram(module, {}).validators)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        actual = executor.run(inputs=bindings).committed_outputs.output("action_chunk")
    torch.cuda.synchronize()
    report["instrumented_seconds"] = time.monotonic() - started
    report["timing_is_benchmark"] = False
    actual = actual.float().cpu().numpy()
    np.save(folder / "full_actions.npy", actual, allow_pickle=False)
    with np.load(args.capture / "full_fresh_eager.npz", allow_pickle=False) as reference:
        comparison = action_fidelity(reference["reference"], actual, inputs["action_mask"].float().numpy())
    write(folder / "fidelity.json", comparison)
    report.update(status="executed_complete_chunk", gates=comparison["gates"], metrics=comparison["active"]["metrics"],
                  unified_metrics=comparison["unified"]["metrics"], primary_acceptance_space=comparison["primary_acceptance_space"],
                  full_actions=file_identity(folder / "full_actions.npy"), fidelity=file_identity(folder / "fidelity.json"))
    write(folder / "report.json", report)
    print(json.dumps({"mode": args.mode, "status": report["status"], "gates": report["gates"], "metrics": report["metrics"]}), flush=True)
    return report


def runner_source(module):
    import math

    widths = {"bf16": 2, "f32": 4, "i64": 8, "i32": 4, "bool": 1}
    inputs = []
    for port in module.inputs:
        if port.device != "cuda:0" or port.payload.dtype not in widths or any(size is None for size in port.payload.shape):
            raise ValueError("runner requires fixed contiguous CUDA input profiles")
        inputs.append('{"%s", VLAFORGE_DTYPE_%s, {%s}, %du}' % (
            port.name, port.payload.dtype.upper(), ",".join(map(str, port.payload.shape)),
            math.prod(port.payload.shape) * widths[port.payload.dtype]))
    if len(module.outputs) != 1 or module.outputs[0].payload.dtype != "bf16":
        raise ValueError("this RDT runner requires the full BF16 model action output")
    return Path(__file__).with_name("rdt_fresh_runner.cpp.in").read_text().replace(
        "@INPUTS@", ",\n".join(inputs)).replace("@OUTPUT_COUNT@", str(math.prod(module.outputs[0].payload.shape)))


def session(args, record, module, captured):
    import numpy as np
    import torch
    from vlaforge.deployment import ArtifactIdentity, ArtifactKind, EffectAudit, RegionArtifactContract, ValueContract, WorkspaceContract, build_artifact_compile_bundle
    from vlaforge.deployment.capabilities import aoti_backend_capability
    from vlaforge.frontend import InvocationProgram
    from vlaforge.ir.serializer import io_schema_digest

    folder = args.output / "session"
    folder.mkdir(parents=True, exist_ok=False)
    run = {"status": "started", "capture_sha256": digest(args.capture / "capture.json"),
           "aoti_loader": args.aoti_loader,
           "tool": file_identity(Path(__file__)), "runner_template": file_identity(Path(__file__).with_name("rdt_fresh_runner.cpp.in"))}
    write(folder / "report.json", run)
    reference_run = read(args.capture.parent / "report.json")
    direct_run = read(args.output / "direct/report.json")
    if direct_run["status"] != "executed_complete_chunk" or direct_run["capture_sha256"] != run["capture_sha256"]:
        raise ValueError("a complete direct artifact reference must precede the C++ comparison")
    direct_path = args.output / "direct/full_actions.npy"
    if file_identity(direct_path) != direct_run["full_actions"]:
        raise ValueError("direct output bytes changed")
    contracts, sources = {}, {}
    auxiliary = {"evidence/capture.json": args.capture / "capture.json", "evidence/direct.json": args.output / "direct/report.json"}
    for index, region in enumerate(module.regions):
        name = region.name
        manifest = verify_artifact(args.output / "artifacts", name, captured[name], args.profile)
        artifact = args.output / "artifacts" / (name + ".pt2")
        artifact_path = f"artifacts/{name}.pt2"
        if args.aoti_loader == "extracted":
            destination = args.output / "extracted" / name
            extraction = unpack_package(artifact, destination)
            artifact = destination / extraction["library"]
            artifact_path = f"artifacts/{name}/{extraction['library']}"
            for relative in extraction["files"]:
                if relative != extraction["library"]:
                    auxiliary[f"artifacts/{name}/{relative}"] = destination / relative
            auxiliary[f"evidence/extractions/{name}.json"] = destination / "extraction.json"
        evidence = read(args.capture / "exports" / (name + ".capture.json"))
        contracts[name] = RegionArtifactContract(
            region_id=index, region_name=name, inputs=tuple(ValueContract.from_dict(item) for item in evidence["inputs"]),
            outputs=tuple(ValueContract.from_dict(item) for item in evidence["outputs"]), io_schema_digest=io_schema_digest(module),
            identity=ArtifactIdentity(model_name="RDT-online-fresh", upstream_revision=reference_run["provenance"]["source"]["revision"],
                checkpoint_identity="sha256:" + reference_run["provenance"]["assets"]["components"]["policy"]["files"]["pytorch_model.bin"]["sha256"],
                graph_sha256=evidence["graph_digest"]), artifact_kind=ArtifactKind.AOTI_PACKAGE,
            artifact_path=artifact_path, artifact_sha256=digest(artifact), artifact_size_bytes=artifact.stat().st_size,
            workspace=WorkspaceContract(device="cuda:0"), capability=aoti_backend_capability("sm_90", ("bf16", "f32", "i32", "i64", "bool")),
            effect_audit=EffectAudit.from_dict(evidence["effect_audit"]), backend_variant=f"torch-{torch.__version__}",
        )
        sources[name] = artifact
    values = torch.load(args.capture / "inputs.pt", map_location="cpu", weights_only=True)
    inputs = folder / "inputs"
    inputs.mkdir()
    run["inputs"] = {}
    for port in module.inputs:
        value = values[port.name].contiguous()
        path = inputs / (port.name + ".bin")
        path.write_bytes(value.view(torch.uint8).numpy().tobytes())
        run["inputs"][port.name] = file_identity(path)
    write(folder / "report.json", run)
    bundle = folder / "bundle"
    try:
        build_artifact_compile_bundle(
            module, bundle, region_artifacts=contracts, artifact_sources=sources,
            validators=InvocationProgram(module, {}).cpp_validators(), runner_source=runner_source(module),
            runtime_root=SOURCE, cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={"aoti": torch.__version__, "cuda": str(torch.version.cuda)}, profile="verified",
            loop_execution="source", source_revision=args.source_revision, source_dirty=True,
            default_device="cuda:0", state_device="cuda:0", environment={"TORCH_CUDA_ARCH_LIST": "9.0", "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
            auxiliary_files=auxiliary,
        )
    except subprocess.CalledProcessError as error:
        (folder / "build-failure.log").write_text(str(error.stdout or "") + str(error.stderr or ""))
        raise
    runner = bundle / "bin/vlaforge_generated_runner"
    dependency = subprocess.check_output(["ldd", str(runner)], text=True)
    (folder / "runner.ldd.txt").write_text(dependency)
    if "libpython" in dependency.lower():
        raise ValueError("standalone runner unexpectedly links Python")
    for name, identity in run["inputs"].items():
        if file_identity(inputs / (name + ".bin")) != identity:
            raise ValueError("C++ input changed after assembly")
    command = [str(runner), str(bundle), str(inputs), str(folder), str(args.repetitions)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONHOME": "/no/python/home", "PYTHONPATH": "/no/python/path"})
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    run.update(status="runner_executed", command=command, exit_code=completed.returncode,
               runner=file_identity(runner), ldd=file_identity(folder / "runner.ldd.txt"), samples=[])
    write(folder / "report.json", run)
    reference = np.load(args.capture / "full_fresh_eager.npz", allow_pickle=False)["reference"]
    direct = np.load(direct_path, allow_pickle=False)
    for index in range(args.repetitions):
        path = folder / f"run-{index}.bin"
        if not path.is_file():
            continue
        raw = np.fromfile(path, dtype=np.uint16)
        actual = (raw.astype(np.uint32) << 16).view(np.float32).reshape(reference.shape)
        mask = values["action_mask"].float().numpy()
        official_comparison, direct_comparison = action_fidelity(reference, actual, mask), action_fidelity(direct, actual, mask)
        write(folder / f"run-{index}-official.json", official_comparison)
        write(folder / f"run-{index}-direct.json", direct_comparison)
        run["samples"].append({"index": index, "raw": file_identity(path), "official": official_comparison["gates"],
                               "direct": direct_comparison["gates"], "official_metrics": official_comparison["active"]["metrics"],
                               "direct_storage_bits_equal": np.array_equal(direct.view(np.uint32), actual.view(np.uint32))})
    run["status"] = "executed_complete_chunks" if completed.returncode == 0 and len(run["samples"]) == args.repetitions else "failed_or_incomplete"
    maps_path = folder / "process-maps.txt"
    run["process_maps"] = file_identity(maps_path) if maps_path.is_file() else None
    run["loaded_python_library"] = "libpython" in maps_path.read_text().lower() if maps_path.is_file() else None
    run["no_python_deployment"] = completed.returncode == 0 and run["loaded_python_library"] is False
    run["full_paper_acceptance"] = False
    write(folder / "report.json", run)
    completed.check_returncode()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compile", "exports", "direct", "session"), required=True)
    parser.add_argument("--profile", choices=("eager-numerics", "aten-preserving", "conservative"), default="eager-numerics")
    parser.add_argument("--reuse-solver", type=Path)
    parser.add_argument("--source-revision", default="uncommitted-local-snapshot")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--aoti-loader", choices=("package", "extracted"), default="package")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10000:
        parser.error("repetitions must lie in [1,10000]")
    args.capture, args.output = args.capture.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from vlaforge.adapters.rdt.rdt_reference import check_environment
        check_environment("torch210-cu128")
        record, module, captured = verify_capture(args.capture)
        if args.mode == "compile":
            result = compile_all(args, record, module, captured)
        elif args.mode == "session":
            result = session(args, record, module, captured)
        else:
            result = execute(args, record, module, captured)
    except Exception as error:
        write(args.output / (args.mode + "-failure.json"), {"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        raise
    if result["status"] == "partial_compile":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
