"""Build a self-verifying no-Python fixture Compile Bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from vlaforge.codegen import (
    CppRegionDefinition,
    CppValidatorDefinition,
    generate_compiled_cpp_session,
)
from vlaforge.compiler import CompilerProfile, compile_module
from vlaforge.deployment.bundle import (
    CompileBundleManifest,
    FileRecord,
    ReproducibilityManifest,
    VersionEntry,
)
from vlaforge.deployment.contract import (
    ArtifactKind,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
)
from vlaforge.ir.program import Module
from vlaforge.ir.serializer import canonical_json
from vlaforge.ir.types import ScalarType, TensorType


def build_compile_bundle(
    module: Module,
    output: str | Path,
    *,
    regions: Mapping[str, CppRegionDefinition],
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str,
    runtime_root: str | Path,
    profile: CompilerProfile | str = CompilerProfile.VERIFIED,
    allow_test_profile: bool = False,
    source_revision: str,
    source_dirty: bool,
    environment: Mapping[str, str] | None = None,
    initial_state: Mapping[str, object] | None = None,
) -> CompileBundleManifest:
    """Compile, build, hash, and verify one standalone deployment bundle."""

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"compile bundle output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    compilation = compile_module(
        module,
        profile=profile,
        allow_test_profile=allow_test_profile,
    )
    sources = generate_compiled_cpp_session(
        compilation,
        regions=regions,
        validators=validators,
        runner_source=runner_source,
        initial_state=initial_state,
    )

    metadata = root / "metadata"
    generated = root / "generated"
    artifacts = root / "artifacts"
    binary = root / "bin"
    for directory in (metadata, generated, artifacts, binary):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(metadata / "semantic_ir.json", canonical_json(compilation.module))
    _write_json(
        metadata / "scheduled_plan.json",
        compilation.plan.canonical_json(),
    )
    _write_json(
        metadata / "state_schema.json",
        json.dumps(
            {
                "states": [
                    {
                        "state_id": item.state_id,
                        "name": item.name,
                        "payload": item.payload.to_dict(),
                        "retention": item.retention,
                        "reset_on_episode": (
                            compilation.module.states[
                                item.state_id
                            ].reset_on_episode
                        ),
                    }
                    for item in compilation.plan.states
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    assert compilation.plan.arena is not None
    _write_json(
        metadata / "physical_memory_plan.json",
        json.dumps(
            {
                "arena": compilation.plan.arena.to_dict(),
                "states": [item.to_dict() for item in compilation.plan.states],
                "compiler_arena_bytes": compilation.plan.arena.size_bytes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    _write_json(
        metadata / "input_schema.json",
        json.dumps(
            {
                "schema": "vlaforge.input_schema/2",
                "io_schema_digest": compilation.certificate.io_schema_digest,
                "inputs": [
                    {
                        "input_id": item.input_id,
                        "name": item.name,
                        "payload": item.payload.to_dict(),
                        "required": item.required,
                        "default": _json_value(item.default),
                        "device": item.device,
                        "ownership": item.ownership.value,
                        "alignment": item.alignment,
                        "extension": item.extension,
                        "value_range": item.value_range,
                        "valid_for": item.valid_for,
                    }
                    for item in compilation.module.inputs
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    _write_json(
        metadata / "output_schema.json",
        json.dumps(
            {
                "schema": "vlaforge.output_schema/2",
                "io_schema_digest": compilation.certificate.io_schema_digest,
                "outputs": [
                    {
                        "output_id": item.output_id,
                        "name": item.name,
                        "payload": item.payload.to_dict(),
                        "group": item.group,
                        "device": item.device,
                        "alignment": item.alignment,
                    }
                    for item in compilation.module.outputs
                ],
                "runtime_output": "CommittedOutputGroup",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    (metadata / "compilation_certificate.json").write_text(
        compilation.certificate.canonical_json(indent=2) + "\n",
        encoding="utf-8",
    )
    sources.write(generated)

    region_contracts = []
    region_definitions = dict(regions)
    for region_id, region in enumerate(compilation.module.regions):
        definition = region_definitions[region.name]
        relative = f"artifacts/{region_id:03d}_{region.name}.fixture.cpp.txt"
        payload = definition.body.encode("utf-8")
        (root / relative).write_bytes(payload)
        dtypes = tuple(
            sorted(
                {_dtype_name(value.type) for value in region.inputs}
                | {_dtype_name(value) for value in region.outputs}
            )
        )
        region_contracts.append(
            RegionArtifactContract(
                region_id=region_id,
                region_name=region.name,
                inputs=tuple(
                    ValueContract.from_ir(
                        value.name,
                        value.type,
                        device="cpu",
                    )
                    for value in region.inputs
                ),
                outputs=tuple(
                    ValueContract.from_ir(
                        f"output_{index}",
                        value,
                        device="cpu",
                    )
                    for index, value in enumerate(region.outputs)
                ),
                artifact_kind=ArtifactKind.CPU_FIXTURE,
                artifact_path=relative,
                artifact_sha256=hashlib.sha256(payload).hexdigest(),
                artifact_size_bytes=len(payload),
                workspace=WorkspaceContract(),
                capability=BackendCapability(
                    backend="cpu_fixture",
                    target="cxx17",
                    supported_dtypes=dtypes,
                    supports_dynamic_shapes=False,
                    supports_device_resident_io=False,
                    requires_synchronize=False,
                ),
                effect_audit=EffectAudit(),
                plugin_abi="vlaforge.region_executable/2",
            )
        )

    runtime = Path(runtime_root).resolve()
    configure = (
        "cmake -S generated -B <build> "
        f"-DVLAFORGE_RUNTIME_ROOT={runtime} "
        "-DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release"
    )
    build = "cmake --build <build> --parallel"
    with tempfile.TemporaryDirectory(prefix="vlaforge-bundle-build-") as temp:
        build_dir = Path(temp) / "build"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(generated),
                "-B",
                str(build_dir),
                f"-DVLAFORGE_RUNTIME_ROOT={runtime}",
                "-DBUILD_TESTING=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--parallel"],
            check=True,
            capture_output=True,
            text=True,
        )
        runner = build_dir / "vlaforge_generated_runner"
        if not runner.is_file():
            raise FileNotFoundError(runner)
        shutil.copy2(runner, binary / "vlaforge_generated_runner")

    required = {
        role: FileRecord.from_file(
            root,
            f"metadata/{role}.json",
            role,
        )
        for role in (
            "semantic_ir",
            "scheduled_plan",
            "state_schema",
            "physical_memory_plan",
            "input_schema",
            "output_schema",
        )
    }
    manifest = CompileBundleManifest(
        semantic_ir=required["semantic_ir"],
        scheduled_plan=required["scheduled_plan"],
        state_schema=required["state_schema"],
        physical_memory_plan=required["physical_memory_plan"],
        input_schema=required["input_schema"],
        output_schema=required["output_schema"],
        io_schema_digest=compilation.certificate.io_schema_digest,
        region_artifacts=tuple(region_contracts),
        generated_sources=(
            FileRecord.from_file(
                root,
                "metadata/compilation_certificate.json",
                "compilation_certificate",
            ),
            *tuple(
                FileRecord.from_file(
                    root,
                    f"generated/{name}",
                    "generated_source",
                )
                for name, _ in sources.files
            ),
        ),
        binaries=(
            FileRecord.from_file(
                root,
                "bin/vlaforge_generated_runner",
                "session_binary",
                executable=True,
            ),
        ),
        toolchain_versions=(
            VersionEntry("cmake", _first_version_line(["cmake", "--version"])),
            VersionEntry("cxx", _first_version_line(["c++", "--version"])),
        ),
        backend_versions=(VersionEntry("cpu_fixture", "1"),),
        reproducibility=ReproducibilityManifest(
            source_revision=source_revision,
            source_dirty=source_dirty,
            build_commands=(configure, build),
            random_seed=0,
            environment=tuple(sorted((environment or {}).items())),
        ),
        compilation_certificate=compilation.certificate,
    )
    manifest.write(root / "bundle.json")
    manifest.verify_files(root)
    return manifest


def _write_json(path: Path, payload: str) -> None:
    path.write_text(payload + "\n", encoding="utf-8")


def _first_version_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0].strip()


def _dtype_name(value: TensorType | ScalarType) -> str:
    return value.dtype if isinstance(value, TensorType) else value.name


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value
