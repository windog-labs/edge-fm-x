"""Independent native-scale extension of a verified normalized OpenPI bundle.

Direct artifact audit, CPU build and native execution are separate processes.
The original bundle, capture records and numerical observations stay unchanged.
"""

import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest


def checked_json(path, expected):
    path = Path(path)
    if file_digest(path)["sha256"] != expected:
        raise ValueError("source evidence SHA mismatch")
    return json.loads(path.read_text())


def materialized_package_path(record):
    path = Path(record["path"])
    observed = file_digest(path)
    if observed != {key: record[key] for key in observed}:
        raise ValueError("direct materialized manifest changed")
    return path


def rebind_existing_contracts(module, contracts):
    """New interfaces use new declaration IDs, not IDs from a prior serialization."""
    from vlaforge.ir.serializer import io_schema_digest

    ids = {region.name: index for index, region in enumerate(module.regions)}
    if len(ids) != len(module.regions) or len({item.region_name for item in contracts}) != len(contracts):
        raise ValueError("output extension region/contract names must be unique")
    if any(item.region_name not in ids for item in contracts):
        raise ValueError("output extension contract is not declared in the new IR")
    return {item.region_name: replace(item, region_id=ids[item.region_name], io_schema_digest=io_schema_digest(module))
            for item in contracts}


def output_binding(context, processor, region, compiled):
    from vlaforge.adapters.openpi.openpi_numerical import binding_from_observed_compile
    entry = {"region": region.name, "status": "compiled-unvalidated", "export": processor["export"],
        "observed_numerical_context_before": processor["observed_compile_before"],
        "observed_numerical_context_after": processor["observed_compile_after"]}
    return binding_from_observed_compile(region.name, context,
        {"archive": processor["export"], "capture_evidence": processor["capture_evidence"]},
        compiled, entry, processor["compile_provenance"])


def render_typed_runner(module, template, *, numerical_bindings=(), replay_policy="off"):
    from vlaforge.codegen.session_runner import render_resident_tensor_runner

    return render_resident_tensor_runner(module, template, samples=1,
        numerical_bindings=numerical_bindings,
        acknowledge_exclusive_process=bool(numerical_bindings),
        acknowledge_calling_thread=bool(numerical_bindings),
        replay_policy=replay_policy,
        outputs=[{"name": item.name, "role": "primary-action" if index == 1 else "float"}
                 for index, item in enumerate(module.outputs)])


def sources(args):
    from vlaforge.adapters.openpi.openpi_aoti import verify_compiled_region
    from vlaforge.adapters.openpi.openpi_output_ir import attach_output_stage
    from vlaforge.deployment import ValueContract
    from vlaforge.deployment.bundle import load_bundle_manifest
    from vlaforge.ir.program import TensorRegion, Value
    from vlaforge.ir.serializer import parse_canonical_json
    from vlaforge.numerical_context import NumericalContext

    old = checked_json(args.native_report, args.native_sha256)
    processor = checked_json(args.processor_report, args.processor_sha256)
    from vlaforge.adapters.openpi.openpi_native_join import SCHEMA, validate_typed_native_join
    if old.get("schema") == SCHEMA:
        old, original_bundle = validate_typed_native_join(old)
    else:
        if old.get("schema", "").startswith("vlaforge.openpi_typed_native_join/"):
            raise ValueError("unknown typed native source schema")
        if (old["status"] != "passed" or not old["official_all_exact"] or not old["same_artifact_all_exact"]
                or not old["native_numerical_policy_enforcement_verified"]):
            raise ValueError("source normalized native gate has not passed")
        original_bundle = args.native_report.parent / "bundle"
    if (processor["status"] != "passed" or processor.get("checked_publication") is not True
            or any(not item["metrics"]["exact_values"] for item in processor["checks"].values())):
        raise ValueError("source native or complete output-stage gate has not passed")
    if file_digest(original_bundle / "bundle.json") != old["bundle"]:
        raise ValueError("original native manifest changed")
    manifest = load_bundle_manifest(original_bundle / "bundle.json")
    manifest.verify_files(original_bundle)
    original = parse_canonical_json((original_bundle / manifest.semantic_ir.path).read_text())
    capture_path = Path(old["selection"]["capture"]["path"])
    capture = checked_json(capture_path, old["selection"]["capture"]["sha256"])
    if processor["source_capture"] != file_digest(capture_path):
        raise ValueError("processor is not bound to the same captured model/input")
    context = NumericalContext.from_dict(capture["numerical_context"])
    if NumericalContext.from_dict(processor["numerical_context"]) != context:
        raise ValueError("output and model numerical contexts differ")
    evidence = processor["capture_evidence"]
    inputs = tuple(ValueContract.from_dict(item) for item in evidence["inputs"])
    outputs = tuple(ValueContract.from_dict(item) for item in evidence["outputs"])
    region = TensorRegion("openpi_native_output", tuple(Value(item.name, item.type) for item in inputs),
                          tuple(item.type for item in outputs))
    module = attach_output_stage(original, region)
    compiled_path = args.processor_report.parent / "output.compile.json"
    if file_digest(compiled_path) != processor["compiler_manifest"]:
        raise ValueError("output compiler manifest changed")
    compiled = verify_compiled_region(compiled_path, {"archive": processor["export"]},
        target=processor["compile_provenance"]["target"], profile="aten-preserving")
    for filename, key in (("prepared_inputs.npz", "prepared_inputs"), ("actions.npz", "actions")):
        if file_digest(capture_path.parent / filename) != capture[key]:
            raise ValueError("actual model tensors changed")
    return old, processor, original_bundle, manifest, capture_path, capture, context, module, region, compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "direct", "build", "verify"), required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--processor-report", type=Path, required=True)
    parser.add_argument("--processor-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runner-template", type=Path, required=True)
    parser.add_argument("--direct-sha256")
    parser.add_argument("--build-sha256")
    parser.add_argument("--acknowledge-exclusive-process", action="store_true")
    parser.add_argument("--acknowledge-calling-thread", action="store_true")
    parser.add_argument("--package-extraction-root", type=Path)
    parser.add_argument("--materialize-aoti", action="store_true")
    parser.add_argument("--loop-execution", choices=("off", "batch-only", "required"), default="off",
                        help="Explicit bounded-loop execution policy for the generated Session bundle")
    args = parser.parse_args()
    if args.materialize_aoti and args.package_extraction_root is not None:
        parser.error("materialized AOTI excludes runtime extraction")
    if args.phase == "direct":
        from cogact_gpu_monitor import child_handshake
        child_handshake()
    if not args.acknowledge_exclusive_process or not args.acknowledge_calling_thread:
        parser.error("explicit worker numerical initialization needs both acknowledgements")
    import numpy as np
    import torch
    import torch._inductor
    import torch._inductor.codecache

    from vlaforge.frontend import InvocationProgram
    from vlaforge.ir.serializer import canonical_json, io_schema_digest
    from vlaforge.numerical_context import offline_restore
    from vlaforge.validation.contracts import NumericContract
    from vlaforge.validation.deployment_metrics import compare_action_chunk

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / (args.phase + ".json")
    if destination.exists():
        raise FileExistsError("phase report already exists")
    report = {"status": "started", "phase": args.phase, "pid": os.getpid(), "tool": file_digest(Path(__file__)),
              "physical_units_verified": False, "robot_calibration_verified": False,
              "performance_benchmark": False, "full_paper_acceptance": False}

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        _old, processor, old_bundle, old_manifest, capture_path, capture, context, module, region, compiled = sources(args)
        report.update(native_source=file_digest(args.native_report), processor_source=file_digest(args.processor_report),
                      numerical_context=context.to_dict(), new_io_digest=io_schema_digest(module))
        with np.load(capture_path.parent / "actions.npz", allow_pickle=False) as pack:
            references = {"normalized_action_chunk": pack["normalized_reference"], "native_action_chunk": pack["physical_reference"][None]}
        if args.phase == "prepare":
            from vlaforge.compiler import compile_module
            binding = output_binding(context, processor, region, compiled)
            bindings = [item.numerical_binding for item in old_manifest.region_artifacts] + [binding]
            if any(item.requirement.policy != binding.requirement.policy for item in bindings):
                raise ValueError("model and output native policies conflict")
            rendered, output_contract = render_typed_runner(
                module, args.runner_template.read_text(), numerical_bindings=bindings,
                replay_policy=args.loop_execution,
            )
            compile_module(module, profile="verified", loop_execution=args.loop_execution)
            (output / "invocation_ir.json").write_text(canonical_json(module))
            (output / "runner.cpp").write_text(rendered)
            report.update(ir=file_digest(output / "invocation_ir.json"), runner=file_digest(output / "runner.cpp"),
                output_contract=output_contract, output_binding=binding.to_dict(),
                no_python_model_execution=False, gpu_execution=False, source_artifacts_byte_verified=True,
                scope="CPU integrity/IR/Plan/renderer contract preparation only")
        elif args.phase == "direct":
            from vlaforge.interpreter import Interpreter, TensorView
            paths = {item.region_name: old_bundle / item.artifact_path for item in old_manifest.region_artifacts}
            paths[region.name] = Path(compiled["artifact"]["path"])
            with offline_restore(context, acknowledge_process_global=True), torch.inference_mode(), ExitStack() as owners:
                runners = {}
                identities = {item.region_name: {"sha256": item.artifact_sha256, "size_bytes": item.artifact_size_bytes}
                              for item in old_manifest.region_artifacts}
                identities[region.name] = compiled["artifact"]
                for name, path in paths.items():
                    if args.materialize_aoti:
                        from vlaforge.deployment import aoti_materialized
                        payload = aoti_materialized.materialize_aoti_package(path, output / "materialized" / name,
                            sha256=identities[name]["sha256"], size_bytes=identities[name]["size_bytes"])
                        identity = file_digest(payload)
                        owner = owners.enter_context(aoti_materialized.load_materialized_aoti(payload,
                            sha256=identity["sha256"], size_bytes=identity["size_bytes"], device=capture["device"]))
                        runners[name] = owner
                        report.setdefault("materialized_packages", {})[name] = {"path": str(payload), **identity,
                            "load_evidence": owner.extraction}
                        report["package_loader_source"] = file_digest(Path(aoti_materialized.__file__))
                    elif args.package_extraction_root is None:
                        runners[name] = torch._inductor.aoti_load_package(str(path))
                    else:
                        from vlaforge.deployment import aoti_load
                        owner = owners.enter_context(aoti_load.load_aoti_package(path,
                            extraction_root=args.package_extraction_root, sha256=identities[name]["sha256"],
                            size_bytes=identities[name]["size_bytes"], device=capture["device"]))
                        runners[name] = owner
                        report.setdefault("package_extraction", {})[name] = owner.extraction
                        report["package_loader_source"] = file_digest(Path(aoti_load.__file__))
                interpreter = Interpreter(module, regions=runners, validators=InvocationProgram(module, {}).validators)
                with np.load(capture_path.parent / "prepared_inputs.npz", allow_pickle=False) as pack:
                    for port in module.inputs:
                        tensor = torch.from_numpy(pack[port.name].copy()).to(port.device)
                        interpreter.bind_input(port.name, TensorView(tensor, tuple(tensor.shape), port.payload.dtype,
                            port.payload.layout, port.device, port.alignment))
                execution = interpreter.run()
                execution.trace.write(output / "direct-trace.json")
                outputs = {port.name: interpreter.read_output(port.name).cpu().numpy() for port in module.outputs}
                np.savez(output / "direct-outputs.npz", **outputs)
                report["outputs"] = file_digest(output / "direct-outputs.npz")
                report["fidelity"] = {name: compare_action_chunk(references[name], value, sample_id="recorded-frame0",
                    space=name, contract=NumericContract(absolute_tolerance=0, relative_tolerance=0)) for name, value in outputs.items()}
                if not all(item["metrics"]["exact_values"] for item in report["fidelity"].values()):
                    raise ValueError("complete extended IR differs from official outputs")
                report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                context.require_current()
            (output / "invocation_ir.json").write_text(canonical_json(module))
            report["ir"] = file_digest(output / "invocation_ir.json")
        else:
            direct = checked_json(output / "direct.json", args.direct_sha256)
            if (direct["status"] != "passed" or direct["native_source"] != report["native_source"]
                    or direct["processor_source"] != report["processor_source"]
                    or direct["new_io_digest"] != report["new_io_digest"]
                    or file_digest(output / "direct-outputs.npz") != direct["outputs"]):
                raise ValueError("independent full-IR audit provenance differs")
            runner, output_contract = render_typed_runner(module, args.runner_template.read_text())
            if args.phase == "build":
                from vlaforge.deployment import (
                    ArtifactIdentity,
                    ArtifactKind,
                    EffectAudit,
                    RegionArtifactContract,
                    ValueContract,
                    WorkspaceContract,
                    build_artifact_compile_bundle,
                )
                from vlaforge.deployment.capabilities import aoti_backend_capability
                from vlaforge.deployment.contract import NUMERICAL_ARTIFACT_SCHEMA
                from vlaforge.validation.session_benchmark import (
                    encode_output_reference,
                )
                contracts = rebind_existing_contracts(module, old_manifest.region_artifacts)
                binding = output_binding(context, processor, region, compiled)
                artifact, evidence = compiled["artifact"], processor["capture_evidence"]
                output_region_id = next(index for index, item in enumerate(module.regions) if item.name == region.name)
                contracts[region.name] = RegionArtifactContract(region_id=output_region_id, region_name=region.name,
                    inputs=tuple(ValueContract.from_dict(item) for item in evidence["inputs"]),
                    outputs=tuple(ValueContract.from_dict(item) for item in evidence["outputs"]),
                    io_schema_digest=io_schema_digest(module), identity=ArtifactIdentity(
                        model_name=capture["config_name"], upstream_revision=capture["checkpoint_provenance"]["source"]["revision"],
                        checkpoint_identity="sha256:" + capture["checkpoint_provenance"]["checkpoint"]["sha256"], graph_sha256=evidence["graph_digest"]),
                    artifact_kind=ArtifactKind.AOTI_PACKAGE, artifact_path="artifacts/" + region.name + ".pt2",
                    artifact_sha256=artifact["sha256"], artifact_size_bytes=artifact["size_bytes"],
                    workspace=WorkspaceContract(device=capture["device"]),
                    capability=aoti_backend_capability(compiled["target"], ("bf16", "bool", "f32", "f64", "i32", "i64")),
                    effect_audit=EffectAudit.from_dict(evidence["effect_audit"]), backend_variant="torch-" + str(torch.__version__),
                    schema=NUMERICAL_ARTIFACT_SCHEMA, numerical_binding=binding)
                artifact_paths = {item.region_name: old_bundle / item.artifact_path for item in old_manifest.region_artifacts}
                artifact_paths[region.name] = Path(artifact["path"])
                if args.materialize_aoti:
                    from vlaforge.deployment.aoti_materialized import (
                        materialized_region_contract,
                    )
                    if set(direct["materialized_packages"]) != set(contracts):
                        raise ValueError("materialized full-IR artifact set differs")
                    for name, contract in contracts.items():
                        entry = direct["materialized_packages"][name]
                        payload = materialized_package_path(entry)
                        contracts[name] = materialized_region_contract(contract, payload,
                            artifact_path=f"artifacts/{name}/model.vfaoti")
                        artifact_paths[name] = payload
                runner, output_contract = render_typed_runner(module, args.runner_template.read_text(),
                    numerical_bindings=[item.numerical_binding for item in contracts.values()],
                    replay_policy=args.loop_execution)
                folder = output / "data/0"
                folder.mkdir(parents=True)
                with np.load(capture_path.parent / "prepared_inputs.npz", allow_pickle=False) as pack:
                    for index, port in enumerate(module.inputs):
                        (folder / (str(index) + ".bin")).write_bytes(pack[port.name].tobytes())
                with np.load(output / "direct-outputs.npz", allow_pickle=False) as pack:
                    for item in output_contract["outputs"]:
                        (folder / item["direct_file"]).write_bytes(encode_output_reference(pack[item["name"]], item))
                        (folder / item["eager_file"]).write_bytes(encode_output_reference(references[item["name"]], item))
                manifest = build_artifact_compile_bundle(module, output / "bundle", region_artifacts=contracts,
                    artifact_sources=artifact_paths, validators=InvocationProgram(module, {}).cpp_validators(),
                    runner_source=runner, runtime_root=args.runtime_root, cmake_prefix_path=torch.utils.cmake_prefix_path,
                    backend_versions={"aoti": str(torch.__version__), "cuda": str(torch.version.cuda)},
                    profile="verified", loop_execution=args.loop_execution, default_device=capture["device"], state_device=capture["device"],
                    aoti_package_extraction_root=args.package_extraction_root,
                    source_revision="local:openpi-native-output-v2", source_dirty=True,
                    environment={**{key: os.environ[key] for key in ("CUDA_HOME", "CUDACXX", "CC", "CXX", "LDFLAGS") if key in os.environ},
                                 "TORCH_CUDA_ARCH_LIST": f"{int(compiled['target'][3:]) // 10}.{int(compiled['target'][3:]) % 10}",
                                 "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
                    auxiliary_files={"evidence/parent-native.json": args.native_report, "evidence/processor.json": args.processor_report,
                                     "evidence/direct.json": output / "direct.json"})
                manifest.verify_files(output / "bundle")
                binary = output / "bundle/bin/vlaforge_generated_runner"
                linked = subprocess.check_output(["ldd", str(binary)], text=True)
                (output / "runner.ldd.txt").write_text(linked)
                if any(term in linked.lower() for term in ("libpython", "libtorch_python", "not found")):
                    raise ValueError("native output runner dependency gate failed")
                report.update(bundle=file_digest(output / "bundle/bundle.json"), runner=file_digest(binary),
                              output_contract=output_contract, native_command=[str(binary), str(output / "bundle"),
                              str(output / "data"), str(output / "native"), "1", "2"])
            else:
                build = checked_json(output / "build.json", args.build_sha256)
                if build["status"] != "passed" or file_digest(output / "bundle/bundle.json") != build["bundle"]:
                    raise ValueError("native extended bundle differs")
                maps = output / "native/process-maps.txt"
                content = maps.read_text()
                provider = str(output / "bundle/lib/libvlaforge_libtorch_numerical_backend.so")
                if "libpython" in content or provider not in content:
                    raise ValueError("actual native mappings lack delivered provider")
                report["native_maps"] = file_digest(maps)
                report["outputs"] = []
                with np.load(output / "direct-outputs.npz", allow_pickle=False) as pack:
                    for item in output_contract["outputs"]:
                        raw_path = output / "native" / item["raw_file"]
                        values = np.fromfile(raw_path, dtype={"f32": np.float32, "f64": np.float64}[item["dtype"]])
                        values = values.reshape((3, *item["shape"]))
                        checks = []
                        for index, value in enumerate(values):
                            checks.append({key: compare_action_chunk(reference, value, sample_id=str(index), space=item["name"],
                                contract=NumericContract(absolute_tolerance=0, relative_tolerance=0))
                                for key, reference in (("official", references[item["name"]]), ("direct", pack[item["name"]]))})
                        report["outputs"].append({"name": item["name"], "raw": file_digest(raw_path), "checks": checks})
                        if not all(check[key]["metrics"]["exact_values"] for check in checks for key in ("official", "direct")):
                            raise ValueError("native complete output differs")
                report["native_numerical_provider_accepted"] = True
                report["no_python_complete_native_actions"] = True
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        if isinstance(error, subprocess.CalledProcessError):
            (output / (args.phase + "-error.log")).write_text(str(error.stdout or "") + str(error.stderr or ""))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
