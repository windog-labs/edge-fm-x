"""Build execution-policy variants without changing a verified model's artifacts.

The output protocol is consumed by benchmark_session.py; building is not a
numerical or performance gate. Model references remain those of the source run.
"""

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "python"))

from benchmark_session import read, sha, write
from vlaforge.codegen.session_runner import render_resident_tensor_runner
from vlaforge.deployment import build_artifact_compile_bundle, load_bundle_manifest
from vlaforge.deployment.capabilities import torchscript_backend_capability
from vlaforge.deployment.contract import ArtifactKind, ArtifactResidency
from vlaforge.frontend import InvocationProgram
from vlaforge.ir.serializer import io_schema_digest, parse_canonical_json
from vlaforge.validation.session_benchmark import validate_protocol


def source_files():
    paths = [SOURCE / "CMakeLists.txt"]
    for folder in ("include", "runtime", "backends", "cmake", "python", "tools"):
        paths.extend(path for path in (SOURCE / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix in (".py", ".h", ".c", ".cpp", ".in", ".cmake"))
    return {str(path.relative_to(SOURCE)): sha(path) for path in sorted(paths)}


def variant_protocol(protocol, bundles, gpu, evidence):
    result = dict(protocol, policies=list(bundles), bundles=bundles,
                  bundle_metadata_mode="selection-manifest",
                  monitor_gpu=gpu, cuda_visible_devices=gpu,
                  evidence=[*protocol["evidence"], *evidence])
    validate_protocol(result)
    return result


def bind_region_declarations(module, artifacts):
    contracts = {item.region_name: item for item in artifacts}
    if len(contracts) != len(artifacts) or set(contracts) != {item.name for item in module.regions}:
        raise ValueError("source Region declarations differ from the artifact manifest")
    if any(item.io_schema_digest != io_schema_digest(module) for item in artifacts):
        raise ValueError("source input/output schema identity changed")
    return {region.name: replace(contracts[region.name], region_id=index)
            for index, region in enumerate(module.regions)}


def bind_execution_profile(contracts, *, torchscript_shared_context=False):
    """Opt into the existing CUDA provider while retaining archived computation."""
    if not torchscript_shared_context:
        return dict(contracts)
    result = {}
    selected = 0
    for name, contract in contracts.items():
        capability = contract.capability
        if capability.backend != "torchscript":
            result[name] = contract
            continue
        context = contract.backend_variant == "torchscript-aten-context/1"
        if (contract.backend_variant not in ("torchscript-aten/1", "torchscript-aten-context/1")
                or contract.artifact_kind != ArtifactKind.TORCHSCRIPT_ARCHIVE
                or contract.residency != ArtifactResidency.SESSION
                or not contract.workspace.device.startswith("cuda:")
                or not capability.target.startswith("sm_")
                or capability != torchscript_backend_capability(
                    capability.target, capability.supported_dtypes, shared_context=context)):
            raise ValueError("shared context requires the static Session-resident CUDA TorchScript profile")
        result[name] = replace(contract, backend_variant="torchscript-aten-context/1",
            capability=torchscript_backend_capability(
                capability.target, capability.supported_dtypes, shared_context=True))
        selected += 1
    if not selected:
        raise ValueError("explicit TorchScript context selection matched no Regions")
    return result


def materialize_contracts(contracts, source, output):
    """Reuse checked deployment assets across policies without runtime extraction."""
    from vlaforge.deployment.aoti_materialized import (
        materialize_aoti_package,
        materialized_region_contract,
    )

    paths = {name: source / item.artifact_path for name, item in contracts.items()}
    selected, records = dict(contracts), []
    for index, (name, original) in enumerate(contracts.items()):
        if original.artifact_kind is not ArtifactKind.AOTI_PACKAGE:
            continue
        manifest = materialize_aoti_package(paths[name], output / str(index),
            sha256=original.artifact_sha256, size_bytes=original.artifact_size_bytes)
        selected[name] = materialized_region_contract(original, manifest,
            artifact_path=f"artifacts/materialized-{index}/model.vfaoti")
        paths[name] = manifest
        records.append({"region": name, "source_artifact_sha256": original.artifact_sha256,
            "manifest": str(manifest), "manifest_sha256": sha(manifest),
            "runtime_extraction": False})
    if not records:
        raise ValueError("explicit materialization matched no AOTI packages")
    return selected, paths, records


def build(args):
    source = args.bundle.resolve()
    if sha(source / "bundle.json") != args.bundle_sha256 or sha(args.protocol) != args.protocol_sha256:
        raise ValueError("source bundle or protocol SHA changed")
    protocol = read(args.protocol)
    validate_protocol(protocol)
    materialize = getattr(args, "materialize_aoti", False)
    if materialize and protocol.get("aoti_package_extraction_root") is not None:
        raise ValueError("materialized deployment excludes runtime package extraction")
    if source not in {Path(path).resolve() for path in protocol["bundles"].values()}:
        raise ValueError("source bundle is not bound by the reference protocol")
    manifest = load_bundle_manifest(source / "bundle.json")
    manifest.verify_files(source)
    module = parse_canonical_json((source / manifest.semantic_ir.path).read_text())
    if module.states:
        raise ValueError("persistent state initialization requires an explicit rebuild contract")
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_files()
    write(args.output / "source-files.json", before)
    report = {"status": "started", "source_bundle_sha256": args.bundle_sha256,
              "source_protocol_sha256": args.protocol_sha256, "source_revision": args.source_revision,
              "source_dirty": True, "command": sys.argv, "builds": [],
              "gpu_execution": False, "full_paper_acceptance": False}
    write(args.output / "report.json", report)
    try:
        contracts = bind_region_declarations(module, manifest.region_artifacts)
        report["region_id_rebindings"] = [{"region": item.region_name, "old_id": item.region_id,
            "new_id": contracts[item.region_name].region_id} for item in manifest.region_artifacts]
        selected = bind_execution_profile(contracts,
            torchscript_shared_context=args.torchscript_shared_context)
        report["execution_profile_changes"] = [{"region": name,
            "before": contracts[name].to_dict(), "after": item.to_dict()}
            for name, item in selected.items() if item != contracts[name]]
        contracts = selected
        artifact_sources = {name: source / item.artifact_path for name, item in contracts.items()}
        if materialize:
            contracts, artifact_sources, records = materialize_contracts(
                contracts, source, args.output / "materialized")
            report["materialized_packages"] = records
        write(args.output / "report.json", report)
        bindings = [item.numerical_binding for item in contracts.values()]
        if any(item is not None for item in bindings) and any(item is None for item in bindings):
            raise ValueError("partial numerical policy coverage is not rebuildable")
        bindings = bindings if all(item is not None for item in bindings) else []
        template = (SOURCE / "tools/session_benchmark_runner.cpp.in").read_text()
        bundles = {}
        for policy in args.policies:
            runner, _ = render_resident_tensor_runner(module, template,
                outputs=protocol["outputs"], samples=len(protocol["samples"]),
                numerical_bindings=bindings, acknowledge_exclusive_process=bool(bindings),
                acknowledge_calling_thread=bool(bindings), replay_policy=policy)
            destination = args.output / policy / "bundle"
            generated = build_artifact_compile_bundle(module, destination,
                region_artifacts=contracts,
                artifact_sources=artifact_sources,
                validators=InvocationProgram(module, {}).cpp_validators(), runner_source=runner,
                runtime_root=SOURCE, cmake_prefix_path=protocol["cmake_prefix_path"],
                backend_versions={item.name: item.version for item in manifest.backend_versions},
                source_revision=args.source_revision, source_dirty=True, profile="verified",
                default_device=f"cuda:{protocol['gpu_ordinal']}", state_device=f"cuda:{protocol['gpu_ordinal']}",
                loop_execution=policy,
                aoti_package_extraction_root=protocol.get("aoti_package_extraction_root"),
                environment={"TORCH_CUDA_ARCH_LIST": protocol["cuda_arch"], "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
                auxiliary_files={"evidence/source-bundle.json": source / "bundle.json",
                                 "evidence/source-protocol.json": args.protocol,
                                 "evidence/source-files.json": args.output / "source-files.json"})
            if {item.region_name: item for item in generated.region_artifacts} != contracts:
                raise ValueError("execution variant changed model artifact contracts")
            bundles[policy] = str(destination.resolve())
            report["builds"].append({"policy": policy, "status": "built-not-run",
                "bundle": bundles[policy], "bundle_sha256": sha(destination / "bundle.json"),
                "runner_sha256": sha(destination / "bin/vlaforge_generated_runner")})
            write(args.output / "report.json", report)
            print(policy + ": built-not-run", flush=True)
        manifest.verify_files(source)
        if before != source_files():
            raise ValueError("source changed during variant build")
        result = variant_protocol(protocol, bundles, args.gpu,
            [str(args.output / "source-files.json"), str(args.output / "report.json")])
        write(args.output / "protocol.json", result)
        report.update(status="built-not-run", protocol_sha256=sha(args.output / "protocol.json"),
                      source_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        if isinstance(error, subprocess.CalledProcessError):
            write(args.output / "build-error.json", {"command": error.cmd,
                "exit_code": error.returncode, "stdout": error.stdout, "stderr": error.stderr})
        raise
    finally:
        write(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--materialize-aoti", action="store_true",
                        help="Create hash-bound deployment payloads in the output directory; "
                             "no runtime AOTI package extraction")
    parser.add_argument("--torchscript-shared-context", action="store_true",
                        help="Rebuild CUDA TorchScript Regions with the opt-in shared context provider; "
                             "requires fresh full-model validation, not a capture certification")
    parser.add_argument("--policies", nargs="+", choices=("off", "batch-only", "required"),
                        default=["off", "batch-only", "required"])
    args = parser.parse_args()
    if len(args.policies) != len(set(args.policies)):
        parser.error("duplicate execution policy")
    args.output = args.output.resolve()
    args.protocol = args.protocol.resolve()
    build(args)


if __name__ == "__main__":
    main()
