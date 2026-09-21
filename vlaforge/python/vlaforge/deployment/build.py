"""Build a self-verifying no-Python fixture Compile Bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath

from vlaforge.codegen import (
    CppArtifactRegionDefinition,
    CppRegionDefinition,
    CppValidatorDefinition,
    generate_compiled_cpp_session,
)
from vlaforge.compiler import (
    NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
    CompilerProfile,
    compile_module,
)
from vlaforge.deployment.bundle import (
    BUNDLE_SCHEMA,
    NUMERICAL_BUNDLE_SCHEMA,
    CompileBundleManifest,
    FileRecord,
    ReproducibilityManifest,
    VersionEntry,
)
from vlaforge.deployment.contract import (
    ArtifactIdentity,
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
from vlaforge.plan import ArtifactVariant


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
                io_schema_digest=compilation.certificate.io_schema_digest,
                identity=ArtifactIdentity(
                    model_name=compilation.module.name,
                    upstream_revision=source_revision,
                    checkpoint_identity="fixture:no-checkpoint",
                    graph_sha256=hashlib.sha256(payload).hexdigest(),
                ),
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
                backend_variant="inline-cpp",
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


def build_artifact_compile_bundle(
    module: Module,
    output: str | Path,
    *,
    region_artifacts: Mapping[str, RegionArtifactContract],
    artifact_sources: Mapping[str, str | Path],
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str,
    runtime_root: str | Path,
    cmake_prefix_path: str | Path,
    backend_versions: Mapping[str, str],
    profile: CompilerProfile | str = CompilerProfile.VERIFIED,
    allow_test_profile: bool = False,
    source_revision: str,
    source_dirty: bool,
    environment: Mapping[str, str] | None = None,
    initial_state: Mapping[str, object] | None = None,
    default_device: str = "cpu",
    state_device: str = "cpu",
    auxiliary_files: Mapping[str, str | Path] | None = None,
    loop_execution: str = "source",
    libtorch_graph_memory_policy: str = "retain",
    aoti_package_extraction_root: str | Path | None = None,
) -> CompileBundleManifest:
    """Build a self-verifying bundle backed by real compiled Region artifacts."""

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"compile bundle output must be empty: {root}")
    expected_names = {region.name for region in module.regions}
    if set(region_artifacts) != expected_names:
        raise ValueError(
            "Region artifact set does not match Semantic IR Regions"
        )
    if set(artifact_sources) != expected_names:
        raise ValueError(
            "Region artifact source set does not match Semantic IR Regions"
        )
    if not backend_versions:
        raise ValueError("artifact bundle requires backend versions")
    build_configuration = _libtorch_graph_memory_configuration(
        libtorch_graph_memory_policy, region_artifacts, backend_versions,
    )
    extraction_configuration = _aoti_package_extraction_configuration(
        aoti_package_extraction_root, region_artifacts, backend_versions,
        runtime_root=Path(runtime_root),
    )
    if extraction_configuration is not None:
        build_configuration["aoti_package_extraction"] = extraction_configuration

    contracts = dict(region_artifacts)
    for contract in contracts.values():
        contract.require_runtime_deployable()
    variants = {}
    for region_id, region in enumerate(module.regions):
        contract = contracts[region.name]
        if (
            contract.region_id != region_id
            or contract.region_name != region.name
        ):
            raise ValueError(
                f"Region artifact identity mismatch for {region.name}"
            )
        variants[region.name] = ArtifactVariant(
            backend=contract.capability.backend,
            variant=contract.backend_variant or "default",
            artifact_path=contract.artifact_path,
            workspace_size_bytes=contract.workspace.size_bytes,
            workspace_alignment=contract.workspace.alignment,
            workspace_device=contract.workspace.device,
            plugin_abi=contract.plugin_abi,
        )
    compilation = compile_module(
        module,
        profile=profile,
        artifact_variants=variants,
        allow_test_profile=allow_test_profile,
        default_device=default_device,
        state_device=state_device,
        loop_execution=loop_execution,
    )
    numerical_bindings = tuple(
        contracts[name].numerical_binding for name in sorted(contracts)
        if contracts[name].numerical_binding is not None
    )
    if numerical_bindings:
        compilation = replace(compilation, certificate=replace(
            compilation.certificate,
            schema=NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
            numerical_bindings=numerical_bindings,
        ))
    for contract in contracts.values():
        if contract.io_schema_digest != compilation.certificate.io_schema_digest:
            raise ValueError(
                f"Region artifact {contract.region_name}: I/O schema mismatch"
            )

    definitions = {
        name: CppArtifactRegionDefinition(
            region_name=name,
            backend=contract.capability.backend,
            artifact_path=contract.artifact_path,
            artifact_sha256=contract.artifact_sha256,
            artifact_size_bytes=contract.artifact_size_bytes,
            io_schema_digest=contract.io_schema_digest,
            target=contract.capability.target,
            device=_artifact_execution_device(contract),
            backend_variant=contract.backend_variant,
            residency=contract.residency.value,
            callable_abi_version=contract.callable_abi_version,
            supports_external_cuda_graph=contract.capability.supports_external_cuda_graph,
            effect_audit=contract.effect_audit,
            supports_execution_context=contract.capability.supports_execution_context,
            numerical_binding=contract.numerical_binding,
        )
        for name, contract in contracts.items()
    }
    sources = generate_compiled_cpp_session(
        compilation,
        artifact_regions=definitions,
        validators=validators,
        runner_source=runner_source,
        initial_state=initial_state,
    )

    metadata = root / "metadata"
    generated = root / "generated"
    binary = root / "bin"
    for directory in (metadata, generated, binary):
        directory.mkdir(parents=True, exist_ok=True)
    _write_compilation_metadata(compilation, metadata)
    _write_json(metadata / "build_configuration.json", json.dumps(build_configuration, sort_keys=True, indent=2))
    input_ir_records = []
    if compilation.input_module is not None:
        input_ir_records.extend(FileRecord.from_file(
            root, f"metadata/{role}.json", role,
        ) for role in ("input_semantic_ir", "loop_execution"))
    sources.write(generated)

    for name, contract in contracts.items():
        source = Path(artifact_sources[name]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            source.stat().st_size != contract.artifact_size_bytes
            or _sha256_file(source) != contract.artifact_sha256
        ):
            raise ValueError(
                f"Region artifact source does not match contract: {name}"
            )
        destination = root / contract.artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    auxiliary_records = []
    all_auxiliary = dict(auxiliary_files or {})
    for name, contract in contracts.items():
        if contract.artifact_kind is not ArtifactKind.AOTI_MATERIALIZED:
            continue
        from vlaforge.deployment.aoti_materialized import MaterializedAotiPackage
        source = Path(artifact_sources[name]).resolve()
        materialized = MaterializedAotiPackage.parse(source.read_text())
        materialized.verify(source.parent)
        for member in materialized.files:
            relative = str(PurePosixPath(contract.artifact_path).parent / member.path)
            if relative in all_auxiliary:
                raise ValueError("materialized AOTI payload collides with auxiliary file: " + relative)
            all_auxiliary[relative] = source.parent / member.path
    occupied_paths = {
        PurePosixPath(contract.artifact_path) for contract in contracts.values()
    }
    for relative_text, source_text in sorted(
        all_auxiliary.items()
    ):
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != relative_text
            or not relative.parts
            or relative.parts[0] in {"bin", "generated", "metadata"}
            or relative_text == "bundle.json"
            or relative in occupied_paths
        ):
            raise ValueError(
                f"invalid or colliding auxiliary artifact path: "
                f"{relative_text}"
            )
        source = Path(source_text).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = root / relative_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        auxiliary_records.append(
            FileRecord.from_file(
                root,
                relative_text,
                "region_artifact_auxiliary",
            )
        )
        occupied_paths.add(relative)

    for contract in contracts.values():
        if contract.artifact_kind is ArtifactKind.AOTI_MATERIALIZED:
            from vlaforge.deployment.aoti_materialized import MaterializedAotiPackage
            destination = root / contract.artifact_path
            MaterializedAotiPackage.parse(destination.read_text()).verify(destination.parent)

    runtime = Path(runtime_root).resolve()
    prefix = Path(cmake_prefix_path).resolve()
    graph_memory_option = build_configuration["cmake_definition"]
    extraction_options = [] if extraction_configuration is None else [
        "-DVLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT=" + extraction_configuration["root"]
    ]
    configure_command = (
        "cmake -S generated -B <build> "
        f"-DVLAFORGE_RUNTIME_ROOT={runtime} "
        f"-DCMAKE_PREFIX_PATH={prefix} "
        f"-D{graph_memory_option} "
        + (shlex.join(extraction_options) + " " if extraction_options else "")
        +
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "
        "-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' "
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON "
        "-DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release"
    )
    build_command = "cmake --build <build> --parallel"
    build_environment = dict(environment or {})
    with tempfile.TemporaryDirectory(
        prefix="vlaforge-artifact-bundle-build-"
    ) as temp:
        build_dir = Path(temp) / "build"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(generated),
                "-B",
                str(build_dir),
                f"-DVLAFORGE_RUNTIME_ROOT={runtime}",
                f"-DCMAKE_PREFIX_PATH={prefix}",
                f"-D{graph_memory_option}",
                *extraction_options,
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib",
                "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
                "-DBUILD_TESTING=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=None if not build_environment else {
                **dict(os.environ),
                **build_environment,
            },
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--parallel"],
            check=True,
            capture_output=True,
            text=True,
            env=None if not build_environment else {
                **dict(os.environ),
                **build_environment,
            },
        )
        runner = build_dir / "vlaforge_generated_runner"
        if not runner.is_file():
            raise FileNotFoundError(runner)
        shutil.copy2(runner, binary / "vlaforge_generated_runner")
        runtime_library_records = _collect_runtime_libraries(build_dir, root)

    required = {
        role: FileRecord.from_file(root, f"metadata/{role}.json", role)
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
        region_artifacts=tuple(
            contracts[region.name] for region in module.regions
        ),
        generated_sources=(
            *input_ir_records,
            FileRecord.from_file(root, "metadata/build_configuration.json", "build_configuration"),
            FileRecord.from_file(
                root,
                "metadata/compilation_certificate.json",
                "compilation_certificate",
            ),
            *tuple(
                FileRecord.from_file(
                    root, f"generated/{name}", "generated_source"
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
            *auxiliary_records,
            *runtime_library_records,
        ),
        toolchain_versions=(
            VersionEntry("cmake", _first_version_line(["cmake", "--version"])),
            VersionEntry("cxx", _first_version_line(["c++", "--version"])),
        ),
        backend_versions=tuple(
            VersionEntry(name, version)
            for name, version in sorted(backend_versions.items())
        ),
        reproducibility=ReproducibilityManifest(
            source_revision=source_revision,
            source_dirty=source_dirty,
            build_commands=(configure_command, build_command),
            random_seed=0,
            environment=tuple(sorted(build_environment.items())),
        ),
        compilation_certificate=compilation.certificate,
        schema=NUMERICAL_BUNDLE_SCHEMA if numerical_bindings else BUNDLE_SCHEMA,
    )
    manifest.write(root / "bundle.json")
    manifest.verify_files(root)
    return manifest


def _aoti_package_extraction_configuration(
    root: str | Path | None,
    contracts: Mapping[str, RegionArtifactContract],
    backend_versions: Mapping[str, str],
    *,
    runtime_root: Path,
) -> dict[str, object] | None:
    if root is None:
        return None
    if not isinstance(root, (str, Path)):
        raise ValueError("AOTI extraction root must be an absolute literal path")  # noqa: TRY004
    value = str(root)
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) != value or ".." in path.parts
            or any(ord(ch) < 32 or ord(ch) == 127 or ch in '\\";$' for ch in value)):
        raise ValueError("AOTI extraction root must be a canonical absolute literal path")
    if not any(c.capability.backend == "aoti" for c in contracts.values()):
        raise ValueError("AOTI extraction root requires an AOTI Region contract")
    version = backend_versions.get("aoti")
    if not isinstance(version, str) or not re.fullmatch(r"2\.10\.\d+(?:\+[A-Za-z0-9_.-]+)?", version):
        raise ValueError("AOTI extraction root requires audited LibTorch 2.10")
    helper_paths = (
        "CMakeLists.txt", "backends/aoti_package_config.h.in",
        "backends/aoti_extracted_package.h", "backends/aoti_extracted_package.cpp",
        "backends/aoti_callable.h", "backends/aoti_callable.cpp",
        "backends/aoti_region_executable.cpp", "backends/aoti_sequence_runner.cpp",
        "backends/aoti_sequence_runner.h", "include/vlaforge/backends/aoti_region_executable.h",
    )
    return {
        "schema": "vlaforge.aoti_package_extraction/1", "root": value,
        "mode": "owned-private-streaming", "compiled_packages_only": True,
        "environment_modified": False, "shared_cache": False,
        "declared_libtorch_version": version,
        "source_sha256": {
            name: _sha256_file(runtime_root / name)
            for name in helper_paths
        },
    }


def _libtorch_graph_memory_configuration(
    policy: str,
    contracts: Mapping[str, RegionArtifactContract],
    backend_versions: Mapping[str, str],
) -> dict[str, object]:
    if policy not in ("retain", "scoped-reclaim"):
        raise ValueError("libtorch_graph_memory_policy must be retain or scoped-reclaim")
    selected = sorted({
        contract.capability.backend for contract in contracts.values()
        if contract.capability.backend in {"aoti", "torchscript"}
        and contract.capability.target.startswith("sm_")
    })
    declared = {name: backend_versions.get(name) for name in selected}
    if policy == "scoped-reclaim":
        if not selected:
            raise ValueError("scoped-reclaim requires a CUDA LibTorch Region contract")
        for name, version in declared.items():
            if not isinstance(version, str) or not re.fullmatch(r"2\.10\.\d+(?:\+[A-Za-z0-9_.-]+)?", version):
                raise ValueError(f"scoped-reclaim requires audited LibTorch 2.10 version for {name}")
    return {
        "schema": "vlaforge.artifact_build_configuration/1",
        "libtorch_graph_memory_policy": policy,
        "cmake_definition": "VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=" + (
            "ON" if policy == "scoped-reclaim" else "OFF"
        ),
        "declared_libtorch_backend_versions": declared,
        "actual_sdk_version_checked_by_cmake_and_header": policy == "scoped-reclaim",
        "global_cache_clear": False,
        "scoped_device_frees_possible": policy == "scoped-reclaim",
        "destructor_status_returned_to_caller": False,
    }


def _collect_runtime_libraries(build_dir: Path, root: Path) -> tuple[FileRecord, ...]:
    manifest = build_dir / "vlaforge_runtime/vlaforge-runtime-libraries-Release.txt"
    if not manifest.is_file():
        raise ValueError("runtime build did not emit its shared-library target manifest")
    records = []
    seen = {}
    for line in manifest.read_text().splitlines():
        source = Path(line)
        if not source.is_absolute() or not source.resolve(strict=True).is_relative_to(build_dir.resolve()):
            raise ValueError("runtime shared-library target is outside the isolated build")
        if not source.is_file():
            raise ValueError("runtime shared-library target is not a regular file")
        if source.name in seen:
            if source.resolve() != seen[source.name]:
                raise ValueError("runtime shared-library target basenames collide")
            continue
        seen[source.name] = source.resolve()
        relative = "lib/" + source.name
        destination = root / relative
        if destination.exists() or destination.is_symlink():
            raise ValueError("runtime shared-library path collides with an existing bundle file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(FileRecord.from_file(root, relative, "runtime_shared_library"))
    return tuple(records)


def _write_compilation_metadata(compilation: object, metadata: Path) -> None:
    if getattr(compilation, "input_module", None) is not None:
        _write_json(metadata / "input_semantic_ir.json", canonical_json(compilation.input_module))
        _write_json(metadata / "loop_execution.json", json.dumps({
            "schema": "vlaforge.loop_execution_selection/1",
            "requested": compilation.loop_execution,
            "input_semantic_digest": compilation.certificate.input_semantic_digest,
            "compiled_semantic_digest": compilation.certificate.compiled_semantic_digest,
            "loops": [{"task_id": task.id, "policy": task.attributes.get("replay", "off")}
                      for task in compilation.plan.tasks if task.opcode == "vla.for"],
        }, sort_keys=True))
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
                "states": [
                    item.to_dict() for item in compilation.plan.states
                ],
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
                "io_schema_digest": (
                    compilation.certificate.io_schema_digest
                ),
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
                ],
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
                "io_schema_digest": (
                    compilation.certificate.io_schema_digest
                ),
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


def _artifact_execution_device(
    contract: RegionArtifactContract,
) -> str:
    devices = {
        value.device
        for value in contract.inputs + contract.outputs
        if value.device.startswith("cuda:")
    }
    if contract.capability.target.startswith("sm_"):
        if not devices:
            raise ValueError(
                f"CUDA Region {contract.region_name} has no CUDA value contract"
            )
        ordinals = {device for device in devices}
        if len(ordinals) != 1:
            raise ValueError(
                f"CUDA Region {contract.region_name} spans device ordinals"
            )
        return next(iter(ordinals))
    return "cpu"


def _write_json(path: Path, payload: str) -> None:
    path.write_text(payload + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
