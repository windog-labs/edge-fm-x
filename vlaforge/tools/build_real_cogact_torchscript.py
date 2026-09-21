"""Strict recorded CogACT candidate through public device-preserving TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
OUTPUTS = (("raw_action_chunk", "raw"), ("normalized_action_chunk", "normalized"),
           ("native_action_chunk", "native"), ("rng_after", "rng_after"),
           ("draws_consumed", "draws_consumed"))
SEEDS = (42, 43, 42)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path)
    return {"sha256": digest(path), "size_bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def verify_capture(args):
    from vlaforge.ir.serializer import parse_canonical_json

    if digest(args.capture / "report.json") != args.capture_report_sha256:
        raise ValueError("capture report bytes changed")
    if digest(args.capture / "module.json") != args.module_sha256:
        raise ValueError("captured IR bytes changed")
    report = read(args.capture / "report.json")
    if (report.get("status") != "partition_capture_candidate_verified"
            or report.get("reference_identity") != "official-code-public-dependency-candidate"
            or report.get("original_meta_config_verified") is not False
            or len(report.get("captures", ())) != 4
            or len(report.get("saved_export_full_loop", ())) != 2
            or any(row.get("all_five_outputs_bitwise_exact") is not True
                   or row.get("global_rng_unchanged") is not True for row in report["saved_export_full_loop"])):
        raise ValueError("complete candidate capture is required without provenance promotion")
    module = parse_canonical_json((args.capture / "module.json").read_text())
    regions = {row["region_name"]: row for row in report["captures"]}
    if set(regions) != {region.name for region in module.regions}:
        raise ValueError("capture region coverage changed")
    for name, row in regions.items():
        if (row.get("strict_export") is not True or row.get("maximum_absolute_error") != 0
                or row.get("effect_audit", {}).get("passed") is not True):
            raise ValueError("strict zero-error effect-audited capture required")
        for suffix in ("capture.json", "nested-effect-audit.json"):
            filename = f"{name}.{suffix}"
            expected = report["export_files"][filename]
            if identity(args.capture / "exports" / filename) != {
                    "sha256": expected["sha256"], "size_bytes": expected["size"]}:
                raise ValueError("capture evidence bytes changed")
        if any(scope["effect_audit"]["passed"] is not True for scope in
               read(args.capture / "exports" / f"{name}.nested-effect-audit.json")):
            raise ValueError("nested effect audit failed")
    for run in report["runs"]:
        for filename, expected in run["files"].items():
            if digest(args.capture / f"seed-{run['seed']}" / filename) != expected:
                raise ValueError("official input/output/trace evidence bytes changed")
    return module, report, regions


def verify_archive(root, name, capture_report):
    from vlaforge.deployment import torchscript_export

    artifact = root / "artifacts" / f"{name}.pt"
    record = read(root / "artifacts" / f"{name}.compile.json")
    audit = record.get("validation", {})
    expected = capture_report["export_files"][f"{name}.pt2e"]
    if (record.get("status") != "region_cases_passed"
            or record.get("exported_program") != {"sha256": expected["sha256"], "size_bytes": expected["size"]}
            or record.get("artifact") != identity(artifact)
            or audit.get("status") != "region_cases_passed"
            or audit.get("backend_variant") != "torchscript-aten/1"
            or audit.get("jit_optimization") is not False
            or audit.get("archive_device_policy") != "preserve"
            or audit.get("numerical_provider_enforcement") is not False
            or audit.get("implementation_sha256") != digest(torchscript_export.__file__)
            or audit.get("artifact_sha256") != digest(artifact)
            or not audit.get("validation_cases")
            or any(case.get("bitwise_equal") is not True for case in audit["validation_cases"])
            or not audit.get("effect_audits")
            or any(effect.get("passed") is not True for effect in audit["effect_audits"])):
        raise ValueError("archive/profile/implementation validation identity mismatch")
    return record, artifact


def compile_one(args, captured, name):
    from vlaforge.deployment.torchscript_export import export_torchscript_region
    from vlaforge.frontend import load_exported_region
    from vlaforge.numerical_context import NumericalContext, offline_restore

    folder = args.output / "artifacts"
    folder.mkdir(exist_ok=True)
    path = args.capture / "exports" / f"{name}.pt2e"
    expected = captured["export_files"][path.name]
    if identity(path) != {"sha256": expected["sha256"], "size_bytes": expected["size"]}:
        raise ValueError("real exported program bytes changed")
    evidence = next(row for row in captured["captures"] if row["region_name"] == name)
    report = {"status": "started", "exported_program": identity(path),
              "tool": identity(__file__), "full_model_verified": False,
              "numerical_context_scope": "offline Python only, no runtime provider enforcement"}
    target = folder / f"{name}.compile.json"
    write(target, report)
    try:
        with offline_restore(NumericalContext.from_dict(evidence["observed_numerical_context"]), acknowledge_process_global=True):
            program = load_exported_region(path)
            audit = export_torchscript_region(program, folder / f"{name}.pt")
        report.update(status="region_cases_passed", artifact=identity(folder / f"{name}.pt"), validation=audit)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write(target, report)


def compile_regions(args, module, captured):
    from cogact_gpu_monitor import run_monitored

    folder = args.output / "artifacts"
    folder.mkdir(exist_ok=False)
    rows = []
    for region in module.regions:
        command = [sys.executable, str(Path(__file__)), "--capture", str(args.capture),
                   "--capture-report-sha256", args.capture_report_sha256, "--module-sha256", args.module_sha256,
                   "--output", str(args.output), "--mode", "compile-one", "--region", region.name]
        log = folder / f"{region.name}.log"
        completed = run_monitored(command, folder / f"{region.name}-monitor")
        log.write_text(completed.stdout + completed.stderr)
        row = {"name": region.name, "command": command, "exitcode": completed.returncode, "log": identity(log)}
        rows.append(row)
        write(args.output / "compile.json", {"status": "in_progress", "regions": rows})
        print(json.dumps(row), flush=True)
        if completed.returncode == 0:
            verify_archive(args.output, region.name, captured)
    passed = all(row["exitcode"] == 0 for row in rows)
    write(args.output / "compile.json", {"status": "all_regions_compiled" if passed else "partial_compile", "regions": rows})
    if not passed:
        raise ValueError("one or more real TorchScript regions failed")


def exact_metrics(expected, actual):
    import numpy as np

    left, right = np.asarray(expected), np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"bitwise_equal": False, "shape_dtype_equal": False}
    same = left.tobytes() == right.tobytes()
    a, b = left.astype(np.float64).reshape(-1), right.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return {"bitwise_equal": same, "shape_dtype_equal": True, "shape": list(left.shape),
            "dtype": str(left.dtype), "all_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "mse": float(np.mean((a - b) ** 2)), "max_abs": float(np.max(np.abs(a - b))),
            "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else None}


def direct(args, module, captured):
    import numpy as np
    import torch
    from vlaforge.frontend import InvocationProgram
    from vlaforge.interpreter import InputBinding, InputStamp, Interpreter, TensorView
    from vlaforge.numerical_context import NumericalContext, offline_restore

    folder = args.output / "direct"
    folder.mkdir(exist_ok=False)
    report = {"status": "started", "runs": [], "artifacts": {}, "no_python_deployment": False,
              "capture_report_sha256": args.capture_report_sha256, "module_sha256": args.module_sha256,
              "reference_identity": captured["reference_identity"], "original_meta_config_verified": False,
              "autonomous_cpp_rng": False, "jit_optimization": False, "timing_is_benchmark": False}
    write(folder / "report.json", report)
    context = captured["captures"][0]["observed_numerical_context"]
    if any(row["observed_numerical_context"] != context for row in captured["captures"]):
        raise ValueError("regions have contradictory offline numerical requirements")
    with offline_restore(NumericalContext.from_dict(context), acknowledge_process_global=True), torch.inference_mode(), torch.jit.optimized_execution(False):
        implementations = {}
        active = {"steps": []}
        for region in module.regions:
            _, path = verify_archive(args.output, region.name, captured)
            implementation = torch.jit.load(str(path))
            report["artifacts"][region.name] = identity(path)

            def observe(*inputs, _implementation=implementation, _name=region.name):
                result = _implementation(*inputs)
                if _name == "cogact_step":
                    active["steps"].append(tuple(value.detach().cpu().clone() for value in result))
                return result

            implementations[region.name] = observe
        executor = Interpreter(module, regions=implementations, validators=InvocationProgram(module, {}).validators)
        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        for index, seed in enumerate(SEEDS):
            active["steps"] = []
            reference = args.capture / f"seed-{seed}"
            with np.load(reference / "inputs.npz", allow_pickle=False) as pack:
                inputs = {name: torch.from_numpy(pack[name].copy()).to("cuda:0") for name in pack.files}
            bindings = {port.name: InputBinding(TensorView(inputs[port.name], port.payload.shape,
                        port.payload.dtype, device=port.device), InputStamp(revision=index + 1)) for port in module.inputs}
            result = executor.run(inputs=bindings)
            traces = torch.load(reference / "traces.pt", map_location="cpu", weights_only=False)
            if len(active["steps"]) != 10:
                raise ValueError("full fresh ten-step generation required")
            torch.save(active["steps"], folder / f"run-{index}-steps.pt")
            step_metrics = []
            for actual, expected in zip(active["steps"], traces["partition_steps"], strict=True):
                metrics = [exact_metrics(a.numpy(), b.numpy()) for a, b in zip(expected, actual, strict=True)]
                step_metrics.append(metrics)
            metrics = {}
            for name, filename in OUTPUTS:
                actual = result.committed_outputs.output(name).cpu().numpy()
                np.save(folder / f"run-{index}-{filename}.npy", actual, allow_pickle=False)
                metrics[name] = exact_metrics(np.load(reference / f"{filename}.npy", allow_pickle=False), actual)
            rng_equal = torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state())
            row = {"seed": seed, "index": index, "outputs": metrics, "step_carry_metrics": step_metrics,
                   "global_rng_unchanged": rng_equal}
            report["runs"].append(row)
            write(folder / "report.json", report)
            if (not rng_equal or any(not item["bitwise_equal"] or not item.get("all_finite") for item in metrics.values())
                    or any(not item["bitwise_equal"] or not item.get("all_finite") for step in step_metrics for item in step)):
                raise ValueError("complete TorchScript candidate output or global RNG differs")
        report["status"] = "full_candidate_exact"
        report["files"] = {path.name: identity(path) for path in folder.iterdir() if path.name != "report.json"}
        write(folder / "report.json", report)


def runner_source(module):
    dtypes = {"f32": ("F32", 4), "float32": ("F32", 4), "f64": ("F64", 8), "float64": ("F64", 8),
              "i64": ("I64", 8), "int64": ("I64", 8), "u8": ("U8", 1), "uint8": ("U8", 1), "bool": ("BOOL", 1)}

    def specs(ports):
        rows = []
        for port in ports:
            if port.device != "cuda:0" or port.payload.dtype not in dtypes or any(d is None for d in port.payload.shape):
                raise ValueError("static contiguous CUDA input/output contract required")
            dtype, width = dtypes[port.payload.dtype]
            shape = ",".join(map(str, port.payload.shape))
            rows.append(f'{{"{port.name}",VLAFORGE_DTYPE_{dtype},{{{shape}}},{math.prod(port.payload.shape) * width}u}}')
        return ",\n".join(rows)

    return (Path(__file__).with_name("cogact_fresh_runner.cpp.in").read_text()
            .replace("@INPUTS@", specs(module.inputs)).replace("@OUTPUTS@", specs(module.outputs)))


def normalize_ir_tensor_types(value, *, path="module", ledger=None):
    from vlaforge.frontend.tensor_types import canonical_tensor_dtype
    from vlaforge.ir.types import TensorType

    if ledger is None:
        ledger = []
    if isinstance(value, TensorType):
        dtype = canonical_tensor_dtype(value.dtype)
        if dtype != value.dtype:
            converted = replace(value, dtype=dtype)
            ledger.append({"path": path, "before": value.to_dict(), "after": converted.to_dict(),
                           "operation": "supported dtype alias only; no tensor cast/copy/device/layout change"})
            return converted
        return value
    if is_dataclass(value) and value.__class__.__module__.startswith("vlaforge.ir."):
        return replace(value, **{field.name: normalize_ir_tensor_types(getattr(value, field.name),
            path=f"{path}.{field.name}", ledger=ledger) for field in fields(value)})
    if isinstance(value, Mapping):
        return {key: normalize_ir_tensor_types(item, path=f"{path}[{key!r}]", ledger=ledger) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(normalize_ir_tensor_types(item, path=f"{path}[{index}]", ledger=ledger) for index, item in enumerate(value))
    return value


def artifact_contract(module, index, evidence, artifact, checkpoint_sha256, *, ledger=None):
    from vlaforge.deployment import (
        ArtifactIdentity,
        ArtifactKind,
        EffectAudit,
        RegionArtifactContract,
        ValueContract,
        WorkspaceContract,
    )
    from vlaforge.deployment.capabilities import torchscript_backend_capability
    from vlaforge.ir.serializer import io_schema_digest

    name = evidence["region_name"]

    def values(which):
        contracts = []
        for number, row in enumerate(evidence[which]):
            value = ValueContract.from_dict(row)
            contracts.append(replace(value, type=normalize_ir_tensor_types(value.type,
                path=f"artifact[{name!r}].{which}[{number}].type", ledger=ledger)))
        return tuple(contracts)

    return RegionArtifactContract(region_id=index, region_name=name,
        inputs=values("inputs"), outputs=values("outputs"), io_schema_digest=io_schema_digest(module),
        identity=ArtifactIdentity(model_name="CogACT-public-dependency-candidate",
            upstream_revision="b174a1b86deedfab4d198d935207e7bb0527994e",
            checkpoint_identity="sha256:" + checkpoint_sha256, graph_sha256=evidence["graph_digest"]),
        artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE, artifact_path=f"artifacts/{name}.pt",
        artifact_sha256=digest(artifact), artifact_size_bytes=artifact.stat().st_size,
        workspace=WorkspaceContract(device="cuda:0"),
        capability=torchscript_backend_capability("sm_90", ("f32", "f64", "i64", "u8", "bool")),
        effect_audit=EffectAudit.from_dict(evidence["effect_audit"]), backend_variant="torchscript-aten/1")


def adopt_existing(args, module, captured):
    root = args.reuse_root
    if (root is None or digest(root / "compile.json") != args.reuse_compile_sha256
            or digest(root / "direct/report.json") != args.reuse_direct_sha256):
        raise ValueError("explicit existing compile/direct report hashes required")
    direct_report = read(root / "direct/report.json")
    if (read(root / "compile.json")["status"] != "all_regions_compiled"
            or direct_report["status"] != "full_candidate_exact"
            or direct_report["capture_report_sha256"] != args.capture_report_sha256
            or direct_report["module_sha256"] != args.module_sha256):
        raise ValueError("only same-capture complete validated candidates can be reused")
    for relative, expected in direct_report["files"].items():
        if identity(root / "direct" / relative) != expected:
            raise ValueError("existing direct evidence changed")
    target = args.output / "artifacts"
    target.mkdir(exist_ok=False)
    records = {}
    for region in module.regions:
        _, source = verify_archive(root, region.name, captured)
        if identity(source) != direct_report["artifacts"][region.name]:
            raise ValueError("existing artifact changed after direct verification")
        os.link(source, target / source.name)
        record = root / "artifacts" / f"{region.name}.compile.json"
        shutil.copyfile(record, target / record.name)
        records[region.name] = identity(source)
    shutil.copytree(root / "direct", args.output / "direct")
    shutil.copyfile(root / "compile.json", args.output / "compile.json")
    write(args.output / "reuse.json", {"status": "same_artifacts_verified", "source": str(root),
        "source_compile_sha256": args.reuse_compile_sha256, "source_direct_sha256": args.reuse_direct_sha256,
        "artifacts": records, "recompiled": False})


def session(args, module, captured, regions):
    import numpy as np
    import torch
    from cogact_gpu_monitor import run_monitored
    from vlaforge.deployment import build_artifact_compile_bundle
    from vlaforge.frontend import InvocationProgram
    from vlaforge.ir.serializer import canonical_json, io_schema_digest, module_digest

    direct_report = read(args.output / "direct/report.json")
    if (direct_report.get("status") != "full_candidate_exact" or len(direct_report.get("runs", ())) != 3
            or direct_report.get("capture_report_sha256") != args.capture_report_sha256
            or direct_report.get("module_sha256") != args.module_sha256):
        raise ValueError("three complete candidate-exact direct calls must precede native build")
    for name, expected in direct_report["files"].items():
        if identity(args.output / "direct" / name) != expected:
            raise ValueError("direct evidence bytes changed")
    folder = args.output / "session"
    folder.mkdir(exist_ok=False)
    original_module = module
    ledger = []
    module = normalize_ir_tensor_types(module, ledger=ledger)
    (folder / "original-semantic-ir.json").write_text(canonical_json(original_module, indent=2) + "\n")
    (folder / "canonical-semantic-ir.json").write_text(canonical_json(module, indent=2) + "\n")
    contracts, sources = {}, {}
    auxiliary = {"evidence/capture.json": args.capture / "report.json",
                 "evidence/direct.json": args.output / "direct/report.json"}
    for index, region in enumerate(module.regions):
        name = region.name
        _, artifact = verify_archive(args.output, name, captured)
        if identity(artifact) != direct_report["artifacts"][name]:
            raise ValueError("archive changed after full-loop validation")
        evidence = regions[name]
        contracts[name] = artifact_contract(module, index, evidence, artifact, captured["weights"]["checkpoint_sha256"], ledger=ledger)
        sources[name] = artifact
        auxiliary[f"evidence/{name}.compile.json"] = args.output / "artifacts" / f"{name}.compile.json"
    write(folder / "tensor-type-conversion.json", {
        "schema": "cogact.explicit_tensor_type_alias_conversion/1", "entries": ledger,
        "original_module_sha256": digest(folder / "original-semantic-ir.json"),
        "converted_module_sha256": digest(folder / "canonical-semantic-ir.json"),
        "original_semantic_digest": module_digest(original_module), "converted_semantic_digest": module_digest(module),
        "original_io_digest": io_schema_digest(original_module), "converted_io_digest": io_schema_digest(module),
        "existing_certificate_inherited": False, "tensor_computation_changed": False,
        "original_ep_or_archive_modified": False})
    for name in ("tensor-type-conversion.json", "original-semantic-ir.json", "canonical-semantic-ir.json"):
        auxiliary[f"evidence/{name}"] = folder / name
    input_folder = folder / "inputs"
    input_folder.mkdir()
    input_identities = {}
    for index, seed in enumerate(SEEDS):
        child = input_folder / f"sample-{index}"
        child.mkdir()
        with np.load(args.capture / f"seed-{seed}" / "inputs.npz", allow_pickle=False) as pack:
            for port in module.inputs:
                path = child / f"{port.name}.bin"
                path.write_bytes(pack[port.name].tobytes())
                input_identities[str(path.relative_to(input_folder))] = identity(path)
    report = {"status": "building", "reference_identity": captured["reference_identity"], "seeds": list(SEEDS),
              "inputs": input_identities, "runs": [], "no_python_deployment": False, "original_meta_config_verified": False,
              "autonomous_cpp_rng": False, "runtime_numerical_provider_enforcement": False, "timing_is_benchmark": False,
              "rng_contract": {"algorithm": "torch-cuda-default-generator-via-libtorch-2.10",
                               "state_layout": "16-byte CUDA generator state", "draws": 11,
                               "shapes": [[1, 16, 7], [10, 2, 16, 7]],
                               "seed_environment": "VLAFORGE_RNG_SEEDS"}}
    write(folder / "report.json", report)
    bundle = folder / "bundle"
    try:
        build_artifact_compile_bundle(module, bundle, region_artifacts=contracts, artifact_sources=sources,
            validators=InvocationProgram(module, {}).cpp_validators(), runner_source=runner_source(module),
            runtime_root=SOURCE, cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={"torchscript": torch.__version__, "cuda": str(torch.version.cuda)},
            profile="verified", loop_execution="off", source_revision=args.source_revision, source_dirty=True,
            default_device="cuda:0", state_device="cuda:0",
            environment={"TORCH_CUDA_ARCH_LIST": "9.0", "CMAKE_BUILD_PARALLEL_LEVEL": "2"}, auxiliary_files=auxiliary)
    except subprocess.CalledProcessError as error:
        (folder / "build-failure.log").write_text(str(error.stdout or "") + str(error.stderr or ""))
        raise
    runner = bundle / "bin/vlaforge_generated_runner"
    ldd = subprocess.check_output(["ldd", str(runner)], text=True)
    (folder / "runner.ldd.txt").write_text(ldd)
    if any(name in ldd.lower() for name in ("libpython", "libtorch_python")):
        raise ValueError("standalone runner links Python")
    for relative, expected in input_identities.items():
        if identity(input_folder / relative) != expected:
            raise ValueError("native input bytes changed")
    command = [str(runner), str(bundle), str(input_folder), str(folder), str(len(SEEDS))]
    completed = run_monitored(command, folder / "native-monitor",
        environment={"PYTHONHOME": "/no/python/home", "PYTHONPATH": "/no/python/path",
                     "VLAFORGE_RNG_SEEDS": ",".join(str(seed) for seed in SEEDS)})
    (folder / "stdout.log").write_text(completed.stdout)
    (folder / "stderr.log").write_text(completed.stderr)
    report.update(command=command, exitcode=completed.returncode, runner=identity(runner))
    for index, seed in enumerate(SEEDS):
        metrics = {}
        for name, filename in OUTPUTS:
            path = folder / f"run-{index}-{name}.bin"
            if not path.is_file():
                continue
            expected = np.load(args.capture / f"seed-{seed}" / f"{filename}.npy", allow_pickle=False)
            actual = np.fromfile(path, dtype=expected.dtype).reshape(expected.shape)
            metrics[name] = exact_metrics(expected, actual)
        report["runs"].append({"index": index, "seed": seed, "outputs": metrics,
                               "all_outputs_exact": len(metrics) == 5 and all(row["bitwise_equal"] and row.get("all_finite") for row in metrics.values())})
    maps = folder / "process-maps.txt"
    report["loaded_python_library"] = any(name in maps.read_text().lower()
        for name in ("libpython", "libtorch_python")) if maps.exists() else None
    passed = completed.returncode == 0 and report["loaded_python_library"] is False and all(row["all_outputs_exact"] for row in report["runs"])
    report.update(status="native_full_candidate_exact" if passed else "failed_or_incomplete",
                  no_python_deployment=passed, autonomous_cpp_rng=passed,
                  rng_boundary=("C++ LibTorch default CUDA Generator creates 11 fresh draws and "
                                "12 explicit state snapshots before Session binding"))
    write(folder / "report.json", report)
    if not passed:
        raise ValueError("native candidate full-output/no-Python gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-report-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compile", "compile-one", "direct", "session", "adopt"), required=True)
    parser.add_argument("--region")
    parser.add_argument("--source-revision", default="frozen-cogact-torchscript-candidate-001")
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--reuse-compile-sha256")
    parser.add_argument("--reuse-direct-sha256")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"compile-one", "direct"}:
        from cogact_gpu_monitor import child_handshake

        child_handshake()
    try:
        module, captured, regions = verify_capture(args)
        if args.mode == "compile-one":
            if args.region not in regions:
                raise ValueError("explicit captured region required")
            compile_one(args, captured, args.region)
        elif args.mode == "compile":
            compile_regions(args, module, captured)
        elif args.mode == "adopt":
            adopt_existing(args, module, captured)
        elif args.mode == "direct":
            if "COGACT_GPU_OWNER_FOLDER" not in os.environ:
                from cogact_gpu_monitor import run_monitored

                completed = run_monitored([sys.executable, str(Path(__file__)), *sys.argv[1:]], args.output / "direct-monitor")
                completed.check_returncode()
            else:
                direct(args, module, captured)
        else:
            session(args, module, captured, regions)
    except BaseException:
        error = args.output / f"{args.mode}{'-' + args.region if args.region else ''}-failure.txt"
        error.write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
