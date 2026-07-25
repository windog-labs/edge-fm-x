"""CLI for Invocation IR inspection, execution, codegen, and bundling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from vlaforge.adapters import (
    build_driving_ar_fixture,
    build_driving_diffusion_fixture,
    build_driving_trajectory_fixture,
    build_hybrid_external_feature_fixture,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.analysis import verify
from vlaforge.codegen import (
    CppValidatorDefinition,
    generate_compiled_cpp_session,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
    smolvla_fixture_regions,
    smolvla_fixture_runner_source,
    smolvla_fixture_validators,
)
from vlaforge.compiler import CompilerProfile, compile_module
from vlaforge.deployment import (
    RegionArtifactContract,
    build_artifact_compile_bundle,
    build_compile_bundle,
    load_bundle_manifest,
)
from vlaforge.interpreter import (
    InputBinding,
    InputStamp,
    Interpreter,
    ScalarValue,
    TensorView,
    Trace,
)
from vlaforge.ir.parser import parse_module
from vlaforge.ir.serializer import io_schema_digest, module_digest
from vlaforge.validation import NumericContract, compare_traces


_FIXTURES = {
    "openvla-fixture": build_openvla_fixture,
    "smolvla-fixture": build_smolvla_fixture,
    "driving-trajectory-fixture": build_driving_trajectory_fixture,
    "driving-ar-fixture": build_driving_ar_fixture,
    "driving-diffusion-fixture": build_driving_diffusion_fixture,
    "hybrid-external-feature-fixture": (
        build_hybrid_external_feature_fixture
    ),
}
_CPP_FIXTURES = {
    "openvla-fixture": (
        openvla_fixture_regions,
        openvla_fixture_validators,
        openvla_fixture_runner_source,
    ),
    "smolvla-fixture": (
        smolvla_fixture_regions,
        smolvla_fixture_validators,
        smolvla_fixture_runner_source,
    ),
}


def _source_provenance() -> tuple[str, bool]:
    """Return immutable package or source-checkout provenance.

    A wheel installation is not a Git checkout. An explicit revision may be
    supplied by a release builder; otherwise the installed distribution
    version is a stable, honest fallback.
    """

    explicit_revision = os.environ.get("VLAFORGE_SOURCE_REVISION")
    if explicit_revision:
        dirty = os.environ.get("VLAFORGE_SOURCE_DIRTY", "0")
        if dirty not in {"0", "1"}:
            raise ValueError("VLAFORGE_SOURCE_DIRTY must be 0 or 1")
        return explicit_revision, dirty == "1"

    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode == 0 and revision.stdout.strip():
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            raise RuntimeError("failed to inspect VLAForge Git source status")
        return revision.stdout.strip(), bool(status.stdout.strip())

    try:
        version = importlib.metadata.version("vlaforge")
    except importlib.metadata.PackageNotFoundError:
        version = "0.2.0.dev0"
    return f"package:vlaforge-{version}", False


def _runtime_root(explicit: str | None = None) -> Path:
    """Resolve the standalone C++ runtime source shipped with VLAForge."""

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get("VLAFORGE_RUNTIME_ROOT")
    if environment:
        candidates.append(Path(environment))
    candidates.extend(
        (
            Path(__file__).resolve().parents[2],
            Path(sys.prefix) / "share" / "vlaforge",
        )
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if all(
            (resolved / relative).is_file()
            for relative in (
                "CMakeLists.txt",
                "runtime/state_store.cpp",
                "include/vlaforge/runtime/session.h",
            )
        ):
            return resolved
    raise FileNotFoundError(
        "VLAForge C++ runtime source was not found; install a complete wheel "
        "or pass --runtime-root / VLAFORGE_RUNTIME_ROOT"
    )


def _load_module(path: str):
    return parse_module(Path(path).read_text(encoding="utf-8"))


def _fixture(name: str):
    try:
        return _FIXTURES[name]()
    except KeyError as error:
        raise ValueError(f"unknown adapter fixture: {name}") from error


def _cpp_fixture(name: str):
    try:
        region_builder, validator_builder, runner_builder = _CPP_FIXTURES[name]
    except KeyError as error:
        raise ValueError(
            f"fixture has no generated C++ evidence: {name}"
        ) from error
    return region_builder(), validator_builder(), runner_builder()


def _load_runs(path: str) -> tuple[dict[str, InputBinding], ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "vlaforge.invocation_inputs/2":
        raise ValueError("input file must use vlaforge.invocation_inputs/2")
    result = []
    for run in data.get("runs", ()):
        bindings = {}
        for name, item in run.get("inputs", {}).items():
            stamp = InputStamp(
                revision=(
                    None
                    if item.get("revision") is None
                    else int(item["revision"])
                ),
                timestamp_ns=(
                    None
                    if item.get("timestamp_ns") is None
                    else int(item["timestamp_ns"])
                ),
            )
            kind = str(item["kind"])
            if kind == "tensor":
                value = TensorView(
                    item["value"],
                    tuple(int(dim) for dim in item["shape"]),
                    str(item["dtype"]),
                    layout=str(item.get("layout", "contiguous")),
                    device=str(item.get("device", "cpu")),
                    alignment=int(item.get("alignment", 1)),
                )
            elif kind == "scalar":
                value = ScalarValue(item["value"], str(item["dtype"]))
            else:
                raise ValueError(f"input @{name} has unknown kind {kind!r}")
            bindings[str(name)] = InputBinding(value, stamp)
        result.append(bindings)
    return tuple(result)


def _inspect(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    data = {
        "name": module.name,
        "schema": module.schema_version,
        "semantic_digest": module_digest(module),
        "io_schema_digest": io_schema_digest(module),
        "inputs": [
            {
                "id": port.input_id,
                "name": port.name,
                "type": port.payload.to_dict(),
                "required": port.required,
                "device": port.device,
                "valid_for": port.valid_for,
            }
            for port in module.inputs
        ],
        "outputs": [
            {
                "id": port.output_id,
                "name": port.name,
                "type": port.payload.to_dict(),
                "group": port.group,
                "device": port.device,
            }
            for port in module.outputs
        ],
        "states": [
            {
                "name": state.name,
                "retention": state.retention,
                "reset_on_episode": state.reset_on_episode,
            }
            for state in module.states
        ],
        "regions": [region.name for region in module.regions],
        "invocations": [
            invocation.name for invocation in module.invocations
        ],
    }
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    diagnostics = verify(module, raise_on_error=False)
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic)
        return 1
    print(
        f"verification passed: module={module.name} "
        f"digest={module_digest(module)} "
        f"io_schema={io_schema_digest(module)}"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    fixture = _fixture(args.adapter)
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    runs = (
        _load_runs(args.inputs)
        if args.inputs
        else tuple(item.inputs for item in fixture.runs)
    )
    for inputs in runs:
        runtime.run(args.invocation, inputs=inputs)
    runtime.trace.write(args.trace)
    print(
        f"execution passed: runs={len(runs)} "
        f"events={len(runtime.trace.events)} trace={args.trace}"
    )
    return 0


def _diff(args: argparse.Namespace) -> int:
    report = compare_traces(
        Trace.read(args.expected),
        Trace.read(args.actual),
        NumericContract(args.atol, args.rtol),
    )
    print(report.format())
    return 0 if report.equal else 1


def _codegen(args: argparse.Namespace) -> int:
    fixture = _fixture(args.adapter)
    regions, validators, runner = _cpp_fixture(args.adapter)
    compilation = compile_module(
        fixture.module,
        profile=args.profile,
        allow_test_profile=args.allow_test_profile,
    )
    sources = generate_compiled_cpp_session(
        compilation,
        regions=regions,
        validators=validators,
        runner_source=runner,
        initial_state=fixture.initial_state,
    )
    output = Path(args.output)
    expected_names = {name for name, _ in sources.files}
    expected_names.add("compilation_certificate.json")
    if output.exists():
        unexpected = sorted(
            path.name
            for path in output.iterdir()
            if path.name not in expected_names
        )
        if unexpected:
            raise ValueError(
                "refusing to write generated sources into a directory "
                f"with unrelated entries: {unexpected}"
            )
    sources.write(output)
    (output / "compilation_certificate.json").write_text(
        compilation.certificate.canonical_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"C++ generation passed: adapter={args.adapter} "
        f"profile={compilation.certificate.profile.value} "
        f"io_schema={compilation.certificate.io_schema_digest} "
        f"digest={sources.digest()} output={output}"
    )
    return 0


def _compile(args: argparse.Namespace) -> int:
    fixture = _fixture(args.adapter)
    regions, validators, runner = _cpp_fixture(args.adapter)
    revision, dirty = _source_provenance()
    environment = {
        "host": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("CXX", "CUDA_VISIBLE_DEVICES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    manifest = build_compile_bundle(
        fixture.module,
        args.output,
        regions=regions,
        validators=validators,
        runner_source=runner,
        runtime_root=_runtime_root(args.runtime_root),
        profile=args.profile,
        allow_test_profile=args.allow_test_profile,
        source_revision=revision,
        source_dirty=dirty,
        environment=environment,
        initial_state=fixture.initial_state,
    )
    print(
        f"Compile Bundle passed: adapter={args.adapter} "
        f"profile={manifest.compilation_certificate.profile.value} "
        f"io_schema={manifest.io_schema_digest} "
        f"digest={manifest.digest()} output={args.output}"
    )
    return 0


def _bundle_verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_bundle_manifest(manifest_path)
    manifest.verify_files(manifest_path.parent)
    print(
        "Compile Bundle verification passed: "
        f"digest={manifest.digest()} "
        f"io_schema={manifest.io_schema_digest} "
        f"root={manifest_path.parent}"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_artifact(args: argparse.Namespace) -> int:
    import torch

    exported_path = Path(args.exported_program).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"artifact output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    program = torch.export.load(exported_path)
    target = args.target
    if target is None:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            target = f"sm_{major}{minor}"
        else:
            target = "cpu"
    configs: dict[str, object] = {
        "aot_inductor.force_mmap_weights": True,
    }
    if args.inductor_profile == "conservative":
        configs.update(
            {
                "force_same_precision": True,
                "max_autotune_gemm_backends": "ATEN",
                "mixed_mm_choice": "aten",
                "epilogue_fusion": False,
            }
        )
    started = time.perf_counter()
    actual = Path(
        torch._inductor.aoti_compile_and_package(
            program,
            package_path=str(output),
            inductor_configs=configs,
        )
    ).resolve()
    compile_seconds = time.perf_counter() - started
    if actual != output or not output.is_file():
        raise RuntimeError(f"AOTI output mismatch: {actual} != {output}")
    result = {
        "schema": "vlaforge.compile_artifact_result/1",
        "status": "passed",
        "evidence_level": "L3-candidate",
        "backend": "aoti",
        "target": target,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inductor_profile": args.inductor_profile,
        "inductor_configs": configs,
        "exported_program": {
            "path": str(exported_path),
            "sha256": _sha256(exported_path),
        },
        "artifact": {
            "path": str(output),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "compile_seconds": compile_seconds,
        "graph_nodes": len(tuple(program.graph_module.graph.nodes)),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.manifest:
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _name_value_map(
    values: list[str], *, category: str
) -> dict[str, str]:
    result = {}
    for item in values:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in result:
            raise ValueError(
                f"{category} entries must be unique NAME=VALUE pairs"
            )
        result[name] = value
    return result


def _build_artifact_bundle(args: argparse.Namespace) -> int:
    module = _load_module(args.program)
    contract_paths = _name_value_map(
        args.artifact_contract, category="artifact contract"
    )
    source_paths = _name_value_map(
        args.artifact_source, category="artifact source"
    )
    contracts = {
        name: RegionArtifactContract.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        for name, path in contract_paths.items()
    }
    validator_data = json.loads(
        Path(args.validator_definitions).read_text(encoding="utf-8")
    )
    if not isinstance(validator_data, dict):
        raise ValueError("validator definitions must be a JSON object")
    validators = {
        str(name): CppValidatorDefinition(str(name), str(body))
        for name, body in validator_data.items()
    }
    backend_versions = _name_value_map(
        args.backend_version, category="backend version"
    )
    if args.cmake_prefix_path is None or not backend_versions:
        import torch

        prefix = (
            torch.utils.cmake_prefix_path
            if args.cmake_prefix_path is None
            else args.cmake_prefix_path
        )
        if not backend_versions:
            backend_versions = {
                "aoti": f"torch-{torch.__version__}",
                "cuda": str(torch.version.cuda),
            }
    else:
        prefix = args.cmake_prefix_path
    revision, dirty = _source_provenance()
    initial_state = (
        None
        if args.initial_state is None
        else json.loads(
            Path(args.initial_state).read_text(encoding="utf-8")
        )
    )
    environment = {
        "host": platform.platform(),
        "machine": platform.machine(),
        "TORCH_CUDA_ARCH_LIST": args.cuda_arch,
    }
    manifest = build_artifact_compile_bundle(
        module,
        args.output,
        region_artifacts=contracts,
        artifact_sources=source_paths,
        validators=validators,
        runner_source=Path(args.runner_source).read_text(encoding="utf-8"),
        runtime_root=_runtime_root(args.runtime_root),
        cmake_prefix_path=prefix,
        backend_versions=backend_versions,
        profile=args.profile,
        allow_test_profile=args.allow_test_profile,
        source_revision=revision,
        source_dirty=dirty,
        environment=environment,
        initial_state=initial_state,
        default_device=args.default_device,
        state_device=args.state_device,
    )
    print(
        json.dumps(
            {
                "schema": "vlaforge.build_bundle_result/1",
                "status": "passed",
                "evidence_level": "L4-candidate",
                "manifest": str(
                    (Path(args.output) / "bundle.json").resolve()
                ),
                "bundle_digest": manifest.digest(),
                "io_schema_digest": manifest.io_schema_digest,
                "regions": [
                    item.region_name for item in manifest.region_artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=CompilerProfile.VERIFIED.value,
        choices=(
            CompilerProfile.OFF.value,
            "conservative",
            CompilerProfile.VERIFIED.value,
            "auto",
            CompilerProfile.FORCE_ON.value,
        ),
    )
    parser.add_argument("--allow-test-profile", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlaforge")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("program")
    inspect_parser.set_defaults(handler=_inspect)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("program")
    verify_parser.set_defaults(handler=_verify)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("program")
    run_parser.add_argument(
        "--adapter", required=True, choices=tuple(_FIXTURES)
    )
    run_parser.add_argument("--inputs")
    run_parser.add_argument("--invocation", default="act")
    run_parser.add_argument("--trace", required=True)
    run_parser.set_defaults(handler=_run)

    diff_parser = commands.add_parser("diff")
    diff_parser.add_argument("expected")
    diff_parser.add_argument("actual")
    diff_parser.add_argument("--atol", type=float, default=0.0)
    diff_parser.add_argument("--rtol", type=float, default=0.0)
    diff_parser.set_defaults(handler=_diff)

    codegen_parser = commands.add_parser("codegen")
    codegen_parser.add_argument(
        "--adapter", required=True, choices=tuple(_CPP_FIXTURES)
    )
    codegen_parser.add_argument("--output", required=True)
    _profile_argument(codegen_parser)
    codegen_parser.set_defaults(handler=_codegen)

    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument(
        "--adapter", required=True, choices=tuple(_CPP_FIXTURES)
    )
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--runtime-root")
    _profile_argument(compile_parser)
    compile_parser.set_defaults(handler=_compile)

    compile_artifact_parser = commands.add_parser("compile-artifact")
    compile_artifact_parser.add_argument("exported_program")
    compile_artifact_parser.add_argument("--output", required=True)
    compile_artifact_parser.add_argument("--manifest")
    compile_artifact_parser.add_argument("--target")
    compile_artifact_parser.add_argument(
        "--inductor-profile",
        choices=("default", "conservative"),
        default="default",
    )
    compile_artifact_parser.set_defaults(handler=_compile_artifact)

    build_bundle_parser = commands.add_parser("build-bundle")
    build_bundle_parser.add_argument("program")
    build_bundle_parser.add_argument(
        "--artifact-contract", action="append", default=[], required=True
    )
    build_bundle_parser.add_argument(
        "--artifact-source", action="append", default=[], required=True
    )
    build_bundle_parser.add_argument(
        "--validator-definitions", required=True
    )
    build_bundle_parser.add_argument("--runner-source", required=True)
    build_bundle_parser.add_argument("--initial-state")
    build_bundle_parser.add_argument("--runtime-root")
    build_bundle_parser.add_argument("--cmake-prefix-path")
    build_bundle_parser.add_argument(
        "--backend-version", action="append", default=[]
    )
    build_bundle_parser.add_argument("--cuda-arch", default="8.6")
    build_bundle_parser.add_argument("--default-device", default="cpu")
    build_bundle_parser.add_argument("--state-device", default="cpu")
    build_bundle_parser.add_argument("--output", required=True)
    _profile_argument(build_bundle_parser)
    build_bundle_parser.set_defaults(handler=_build_artifact_bundle)

    bundle_verify_parser = commands.add_parser("bundle-verify")
    bundle_verify_parser.add_argument("manifest")
    bundle_verify_parser.set_defaults(handler=_bundle_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
