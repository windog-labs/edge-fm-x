"""Build/run a real saved OpenPI artifact bundle through the generic C++ Session.

The source AOTI audit may have failed official parity: that failure remains a
separate gate. Native policy checking and worker initialization are explicit
options, never silently applied to legacy diagnostics or inside a Session.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

from vlaforge.adapters.openpi.openpi_aoti import _capture_source, verify_compiled_region
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.numerical_context import NumericalContext


def render_runner(module, template, *, device, native_policy_evidence=False):
    sizes = {"f64": 8, "f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "bool": 1}
    if re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("native audit requires an explicit CUDA device")
    declarations = []
    for port in module.inputs:
        if port.device != device or port.payload.dtype not in sizes:
            raise ValueError("runner requires fixed, contiguous CUDA tensor inputs")
        shape = tuple(port.payload.shape)
        if any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("runner requires nonempty static shapes")
        declarations.append(
            '{"%s", VLAFORGE_DTYPE_%s, {%s}, %du}'
            % (
                port.name,
                port.payload.dtype.upper(),
                ",".join(map(str, shape)),
                math.prod(shape) * sizes[port.payload.dtype],
            )
        )
    if len(module.outputs) != 1:
        raise ValueError("one complete normalized action output is required")
    output = module.outputs[0]
    if output.payload.dtype != "f32" or output.device != device:
        raise ValueError("complete output must retain its original CUDA float32 dtype")
    substitutions = {
        "@INPUTS@": ",\n".join(declarations),
        "@ORDINAL@": device.split(":")[1],
        "@SAMPLES@": "1",
        "@OUTPUT_COUNT@": str(math.prod(output.payload.shape)),
        "@REPLAY_AUDIT@": (
            """    {
      std::ifstream maps("/proc/self/maps");
      const std::string current((std::istreambuf_iterator<char>(maps)), std::istreambuf_iterator<char>());
      std::ofstream saved(std::string(argv[3]) + "/run-" + std::to_string(run) + ".maps");
      saved << current;
      if (current.empty() || current.find("libpython") != std::string::npos || !saved) {
        result = 61; break;
      }
    }
"""
            if native_policy_evidence
            else ""
        ),
        "@REPLAY_FAILURE@": "",
    }
    if set(re.findall(r"@[A-Z_]+@", template)) != set(substitutions):
        raise ValueError("runner template substitution contract differs")
    for key, value in substitutions.items():
        template = template.replace(key, value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runner-template", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--trace-native", action="store_true")
    parser.add_argument("--build-only", action="store_true",
                        help="deliver a verified bundle without launching a native worker")
    parser.add_argument("--require-numerical-policy", action="store_true")
    parser.add_argument("--initialize-numerical-worker", action="store_true")
    parser.add_argument("--acknowledge-exclusive-process", action="store_true")
    parser.add_argument("--acknowledge-calling-thread", action="store_true")
    args = parser.parse_args()
    if args.initialize_numerical_worker and not (
        args.require_numerical_policy
        and args.acknowledge_exclusive_process
        and args.acknowledge_calling_thread
    ):
        parser.error(
            "worker initialization requires a bound numerical policy and both explicit acknowledgements"
        )
    if not args.initialize_numerical_worker and (
        args.acknowledge_exclusive_process or args.acknowledge_calling_thread
    ):
        parser.error("worker acknowledgements require explicit worker initialization")
    if not 1 <= args.repetitions <= 8:
        parser.error("diagnostic repetitions must be in [1,8]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vlaforge.openpi_native_session_audit/1",
        "status": "started",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "native_numerical_policy_enforcement_verified": False,
        "physical_units_verified": False,
        "robot_calibration_verified": False,
        "full_paper_acceptance": False,
        "performance_benchmark": False,
        "tool": file_digest(Path(__file__)),
        "trace_native": args.trace_native,
        "numerical_policy_requested": args.require_numerical_policy,
        "explicit_worker_initialization": args.initialize_numerical_worker,
        "build_only": args.build_only,
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
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
        from vlaforge.deployment.capabilities import aoti_backend_capability
        from vlaforge.deployment.contract import (
            ARTIFACT_SCHEMA,
            NUMERICAL_ARTIFACT_SCHEMA,
        )
        from vlaforge.frontend import InvocationProgram
        from vlaforge.ir.serializer import io_schema_digest
        from vlaforge.validation.contracts import NumericContract
        from vlaforge.validation.deployment_metrics import compare_action_chunk

        path, source, context, module, regions = _capture_source(
            args.capture_report, args.capture_sha256
        )
        if file_digest(args.audit_report)["sha256"] != args.audit_sha256:
            raise ValueError("artifact audit report SHA mismatch")
        audit = json.loads(args.audit_report.read_text())
        if audit.get("schema") == "vlaforge.openpi_pruned_aoti_audit/1" and (
            audit.get("status") != "passed" or audit.get("complete_bitwise_equal") is not True
            or not audit.get("calls") or any(
                call.get("reference_inputs_bitwise_equal") is not True
                or not call.get("canonical_trajectory_output_metrics")
                or any(item.get("bitwise_equal") is not True for item in call["canonical_trajectory_output_metrics"])
                for call in audit["calls"]
            )
        ):
            raise ValueError("derived native build requires complete exact AOTI trace")
        if (
            audit.get("source_report", {}).get("sha256") != args.capture_sha256
            or audit.get("full_artifact_execution") is not True
            or audit.get("caller_policy_restoration_verified") is not True
            or audit.get("torch") != torch.__version__
            or audit.get("cuda") != torch.version.cuda
            or NumericalContext.from_dict(audit.get("numerical_context")) != context
        ):
            raise ValueError(
                "source artifact audit lacks complete same-context execution"
            )
        direct_path = args.audit_report.parent / "actions_and_region_outputs.npz"
        if file_digest(direct_path) != audit["outputs"]:
            raise ValueError("artifact audit output digest mismatch")
        compiled = {}
        compilation_observations = {}
        for recorded in audit["compile_reports"]:
            build_path = Path(recorded["path"])
            if file_digest(build_path) != {
                k: v for k, v in recorded.items() if k != "path"
            }:
                raise ValueError("artifact build report changed after direct audit")
            build = json.loads(build_path.read_text())
            for item in build["regions"]:
                name = item["region"]
                if name in compiled or name not in regions:
                    raise ValueError("duplicate/undeclared artifact")
                manifest = Path(item["manifest"]["path"])
                if file_digest(manifest) != {
                    k: v for k, v in item["manifest"].items() if k != "path"
                }:
                    raise ValueError("artifact compiler manifest digest mismatch")
                compiled[name] = verify_compiled_region(
                    manifest,
                    regions[name],
                    target=audit["target"],
                    profile=build["profile"],
                )
                compilation_observations[name] = (item, build)
        if set(compiled) != set(regions):
            raise ValueError("native Session requires all real compiled Regions")
        device = source["device"]
        contracts, artifact_sources = {}, {}
        numerical_bindings = []
        for index, region in enumerate(module.regions):
            name = region.name
            entry, artifact = regions[name], compiled[name]["artifact"]
            evidence = entry["capture_evidence"]
            binding = None
            if args.require_numerical_policy:
                from vlaforge.adapters.openpi.openpi_numerical import (
                    binding_from_legacy_compile,
                    binding_from_observed_compile,
                )

                observed_entry, observed_build = compilation_observations[name]
                try:
                    binding = binding_from_observed_compile(
                        name, context, entry, compiled[name], observed_entry, observed_build
                    )
                except KeyError as error:
                    if error.args != ("observed_numerical_context_before",):
                        raise
                    binding = binding_from_legacy_compile(
                        name, context, entry, compiled[name], observed_entry, observed_build
                    )
                numerical_bindings.append(binding)
            contracts[name] = RegionArtifactContract(
                region_id=index,
                region_name=name,
                inputs=tuple(
                    ValueContract.from_dict(item) for item in evidence["inputs"]
                ),
                outputs=tuple(
                    ValueContract.from_dict(item) for item in evidence["outputs"]
                ),
                io_schema_digest=io_schema_digest(module),
                identity=ArtifactIdentity(
                    model_name=source["config_name"],
                    upstream_revision=source["checkpoint_provenance"]["source"][
                        "revision"
                    ],
                    checkpoint_identity="sha256:"
                    + source["checkpoint_provenance"]["checkpoint"]["sha256"],
                    graph_sha256=evidence["graph_digest"],
                ),
                artifact_kind=ArtifactKind.AOTI_PACKAGE,
                artifact_path=f"artifacts/{name}.pt2",
                artifact_sha256=artifact["sha256"],
                artifact_size_bytes=artifact["size_bytes"],
                workspace=WorkspaceContract(device=device),
                capability=aoti_backend_capability(
                    audit["target"], ("bf16", "bool", "f32", "f64", "i32", "i64")
                ),
                effect_audit=EffectAudit.from_dict(evidence["effect_audit"]),
                backend_variant=f"torch-{torch.__version__}",
                schema=NUMERICAL_ARTIFACT_SCHEMA if binding is not None else ARTIFACT_SCHEMA,
                numerical_binding=binding,
            )
            artifact_sources[name] = Path(artifact["path"])
        input_folder = output / "inputs/sample-000000"
        input_folder.mkdir(parents=True)
        with np.load(path.parent / "prepared_inputs.npz", allow_pickle=False) as pack:
            if set(pack.files) != {port.name for port in module.inputs}:
                raise ValueError("complete native input set mismatch")
            for name in pack.files:
                target = input_folder / (name + ".bin")
                target.write_bytes(pack[name].tobytes())
                if target.read_bytes() != pack[name].tobytes():
                    raise ValueError("native input byte verification failed")
        selection = {
            "capture": {"path": str(path), **file_digest(path)},
            "direct_audit": {
                "path": str(args.audit_report),
                **file_digest(args.audit_report),
            },
            "template": file_digest(args.runner_template),
            "artifacts": {name: compiled[name]["artifact"] for name in compiled},
            "numerical_context": context.to_dict(),
            "numerical_context_scope": (
                "bound v2 LibTorch provider requirement; native acceptance still pending"
                if args.require_numerical_policy
                else "Python source observation only; native enforcement not requested"
            ),
            "numerical_bindings": [binding.to_dict() for binding in numerical_bindings],
            "explicit_worker_initialization": args.initialize_numerical_worker,
            "worker_acknowledgements": {
                "exclusive_process": args.acknowledge_exclusive_process,
                "calling_thread": args.acknowledge_calling_thread,
            },
            "loop_execution": "off",
            "same_frame_repetitions": args.repetitions,
            "trace_native": args.trace_native,
        }
        selection_path = output / "selection.json"
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        report.update(
            status="building",
            selection=selection,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        save()
        runtime_sources = {}
        for relative in (
            "CMakeLists.txt",
            "include",
            "runtime",
            "backends",
            "cmake",
            "python/vlaforge",
        ):
            item = args.runtime_root / relative
            files = [item] if item.is_file() else item.rglob("*")
            for file in files:
                if file.is_file() and "__pycache__" not in file.parts:
                    runtime_sources[str(file.relative_to(args.runtime_root))] = (
                        file_digest(file)
                    )
        report["runtime_sources"] = runtime_sources
        save()
        bundle = output / "bundle"
        runner = render_runner(
            module,
            args.runner_template.read_text(),
            device=device,
            native_policy_evidence=args.require_numerical_policy,
        )
        if args.initialize_numerical_worker:
            from vlaforge.adapters.openpi.openpi_numerical import (
                render_explicit_numerical_worker,
            )

            runner = render_explicit_numerical_worker(
                runner,
                numerical_bindings,
                acknowledge_exclusive_process=args.acknowledge_exclusive_process,
                acknowledge_calling_thread=args.acknowledge_calling_thread,
            )
        build_environment = {
            **{
                key: os.environ[key]
                for key in (
                    "OPENSSL_ROOT_DIR",
                    "CUDA_HOME",
                    "CUDACXX",
                    "CC",
                    "CXX",
                    "LDFLAGS",
                )
                if key in os.environ
            },
            "TORCH_CUDA_ARCH_LIST": f"{int(audit['target'][3:]) // 10}.{int(audit['target'][3:]) % 10}",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        }
        if args.trace_native:
            from vlaforge.adapters.openpi import openpi_native_trace

            runner = openpi_native_trace.SOURCE + runner
            build_environment["LDFLAGS"] = (
                build_environment.get("LDFLAGS", "")
                + " "
                + openpi_native_trace.LINK_FLAG
            ).strip()
            report["trace_implementation"] = file_digest(
                Path(openpi_native_trace.__file__)
            )
            report["trace_scope"] = (
                "ordinary loop_execution=off run, complete synchronized API tensor dumps; not performance or native policy enforcement"
            )
            save()
        manifest = build_artifact_compile_bundle(
            module,
            bundle,
            region_artifacts=contracts,
            artifact_sources=artifact_sources,
            validators=InvocationProgram(module, {}).cpp_validators(),
            runner_source=runner,
            runtime_root=args.runtime_root,
            cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={
                "aoti": torch.__version__,
                "cuda": str(torch.version.cuda),
            },
            profile="verified",
            loop_execution="off",
            default_device=device,
            state_device=device,
            source_revision=args.source_revision,
            source_dirty=True,
            environment=build_environment,
            auxiliary_files={
                "evidence/selection.json": selection_path,
                "evidence/capture.json": path,
                "evidence/direct-audit.json": args.audit_report,
            },
        )
        manifest.verify_files(bundle)
        if args.trace_native:
            from vlaforge.adapters.openpi.openpi_runtime_compare import runtime_region_ids

            report["trace_region_ids"], report["trace_region_id_source"] = (
                runtime_region_ids(bundle, file_digest(bundle / "bundle.json"))
            )
            save()
        executable = bundle / "bin/vlaforge_generated_runner"
        dependencies = subprocess.check_output(["ldd", str(executable)], text=True)
        (output / "runner.ldd.txt").write_text(dependencies)
        if "libpython" in dependencies.lower():
            raise ValueError("native runner links libpython")
        if args.build_only:
            report.update(status="built-unexecuted", bundle=file_digest(bundle / "bundle.json"),
                          executable=file_digest(bundle / "bin/vlaforge_generated_runner"),
                          native_executed=False, native_numerical_policy_enforcement_verified=False,
                          no_python_deployment_verified=False)
            save()
            return
        runs = output / "runs"
        runs.mkdir()
        if args.trace_native:
            (runs / "region-trace").mkdir()
        command = [
            str(executable),
            str(bundle),
            str(output / "inputs"),
            str(runs),
            str(args.repetitions),
        ]
        report.update(
            status="executing",
            executable=file_digest(executable),
            command=command,
            bundle=file_digest(bundle / "bundle.json"),
            python_linked=False,
        )
        save()
        with (
            (runs / "stdout.log").open("w") as stdout,
            (runs / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                env={
                    **os.environ,
                    "PYTHONHOME": "/no/python/home",
                    "PYTHONPATH": "/no/python/path",
                    **(
                        {"VLAFORGE_DIAGNOSTIC_TRACE_DIR": str(runs / "region-trace")}
                        if args.trace_native
                        else {}
                    ),
                },
            )
            report["native_pid"] = process.pid
            save()
            report["native_exit_code"] = process.wait()
        save()
        if report["native_exit_code"] != 0:
            raise ValueError(
                "native executable failed; original stdout/stderr retained"
            )
        report["native_numerical_policy_enforcement_verified"] = (
            args.require_numerical_policy
        )
        report["native_numerical_policy_scope"] = (
            "actual generated Session accepted complete24field LibTorch require-current checks; not a proof of output parity or all execution settings"
            if args.require_numerical_policy
            else "not requested"
        )
        if args.require_numerical_policy:
            report["native_maps"] = {}
            for index in range(args.repetitions):
                maps = runs / f"run-{index}.maps"
                if not maps.read_text() or "libpython" in maps.read_text():
                    raise ValueError(
                        "native policy run lacks Python-free actual mappings"
                    )
                report["native_maps"][maps.name] = file_digest(maps)
        if args.trace_native:
            report["trace_files"] = {
                file.name: file_digest(file)
                for file in (runs / "region-trace").iterdir()
                if file.is_file()
            }
            save()
        with np.load(direct_path, allow_pickle=False) as pack:
            official, direct = pack["normalized_reference"], pack["normalized_artifact"]
        comparisons = []
        for index in range(args.repetitions):
            filename = runs / f"run-{index}.bin"
            actual = np.fromfile(filename, dtype=np.float32).reshape(official.shape)
            compare = lambda reference, label: compare_action_chunk(
                reference,
                actual,
                sample_id=f"run-{index}",
                space=label,
                contract=NumericContract(absolute_tolerance=0, relative_tolerance=0),
            )
            comparisons.append(
                {
                    "index": index,
                    "output": file_digest(filename),
                    "vs_same_artifact": compare(direct, "normalized_cpp_vs_same_aoti"),
                    "vs_official": compare(official, "normalized_cpp_vs_official"),
                }
            )
        report["runs"] = comparisons
        report["same_artifact_all_exact"] = all(
            r["vs_same_artifact"]["metrics"]["exact_values"] for r in comparisons
        )
        report["official_all_exact"] = all(
            r["vs_official"]["metrics"]["exact_values"] for r in comparisons
        )
        report["status"] = (
            "passed"
            if report["same_artifact_all_exact"] and report["official_all_exact"]
            else "failed-numerics"
        )
        report["python_environment_disabled"] = True
        save()
        if report["status"] != "passed":
            raise SystemExit(1)
    except BaseException as error:
        if report["status"] != "failed-numerics":
            report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            (output / "build-failure.log").write_text(
                str(error.stdout or "") + str(error.stderr or "")
            )
        save()
        raise


if __name__ == "__main__":
    main()
