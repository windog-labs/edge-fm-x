from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from vlaforge.compiler import (
    NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
    ArenaCertificate,
    CompilationCertificate,
    CompilerProfile,
    PassCertificate,
)
from vlaforge.deployment import (
    ArtifactDiagnostic,
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    CompileBundleManifest,
    DiagnosticSeverity,
    EffectAudit,
    FileRecord,
    RegionArtifactContract,
    ReproducibilityManifest,
    ValueContract,
    VersionEntry,
    WorkspaceContract,
    load_bundle_manifest,
)
from vlaforge.deployment.bundle import NUMERICAL_BUNDLE_SCHEMA
from vlaforge.deployment.contract import NUMERICAL_ARTIFACT_SCHEMA
from vlaforge.deployment.numerical import (
    NumericalCompileRecord,
    NumericalContractError,
    NumericalEnforcementUnavailable,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.ir.types import TensorType


def _write(root: Path, path: str, payload: bytes) -> FileRecord:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return FileRecord.from_file(root, path, path.replace("/", "_"))


def _region(root: Path, *, region_id: int = 0) -> RegionArtifactContract:
    payload = b"fixture-region-artifact"
    path = f"artifacts/region-{region_id}.bin"
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    tensor = TensorType((2, None), "f32")
    dynamic = {1: ("tokens", 1, 4, 8)}
    return RegionArtifactContract(
        region_id=region_id,
        region_name=f"region_{region_id}",
        io_schema_digest="0" * 64,
        identity=ArtifactIdentity(
            model_name="contract-fixture",
            upstream_revision="fixture-revision",
            checkpoint_identity="fixture:no-checkpoint",
            graph_sha256="1" * 64,
        ),
        inputs=(
            ValueContract.from_ir(
                "input",
                tensor,
                device="cuda:0",
                dynamic_bounds=dynamic,
                alignment=64,
            ),
        ),
        outputs=(
            ValueContract.from_ir(
                "output",
                tensor,
                device="cuda:0",
                dynamic_bounds=dynamic,
                alignment=64,
            ),
        ),
        artifact_kind=ArtifactKind.CUDA_BINARY,
        artifact_path=path,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_size_bytes=len(payload),
        workspace=WorkspaceContract(4096, 256, "cuda:0"),
        capability=BackendCapability(
            backend="fixture_cuda",
            target="sm_86",
            supported_dtypes=("f32",),
            supports_dynamic_shapes=True,
            supports_device_resident_io=True,
            requires_synchronize=True,
        ),
        effect_audit=EffectAudit(
            explicit_rng=True,
            lifted_states=("rng",),
            diagnostics=(
                ArtifactDiagnostic(
                    "audit.explicit_rng",
                    "RNG is represented by an explicit region value",
                    DiagnosticSeverity.INFO,
                ),
            ),
        ),
        backend_variant="test",
        residency=ArtifactResidency.INVOCATION,
    )


def _bundle(root: Path) -> CompileBundleManifest:
    required = {
        role: _write(root, f"metadata/{role}.json", role.encode())
        for role in (
            "semantic_ir",
            "scheduled_plan",
            "state_schema",
            "physical_memory_plan",
            "input_schema",
            "output_schema",
        )
    }
    required = {
        role: replace(record, role=role) for role, record in required.items()
    }
    source = replace(
        _write(root, "generated/session_generated.cpp", b"int generated = 1;\n"),
        role="generated_source",
    )
    binary = replace(
        _write(root, "bin/session_runner", b"fixture executable"),
        role="session_binary",
        executable=True,
    )
    return CompileBundleManifest(
        semantic_ir=required["semantic_ir"],
        scheduled_plan=required["scheduled_plan"],
        state_schema=required["state_schema"],
        physical_memory_plan=required["physical_memory_plan"],
        input_schema=required["input_schema"],
        output_schema=required["output_schema"],
        io_schema_digest="0" * 64,
        region_artifacts=(_region(root),),
        generated_sources=(source,),
        binaries=(binary,),
        toolchain_versions=(VersionEntry("compiler", "1.0"),),
        backend_versions=(VersionEntry("fixture_cuda", "1.0"),),
        reproducibility=ReproducibilityManifest(
            source_revision="0123456789abcdef",
            source_dirty=False,
            build_commands=("vlaforge compile program.vla",),
            random_seed=7,
            environment=(("TARGET", "sm_86"),),
        ),
        compilation_certificate=CompilationCertificate(
            profile=CompilerProfile.OFF,
            test_only=False,
            input_semantic_digest="0" * 64,
            compiled_semantic_digest="0" * 64,
            io_schema_digest="0" * 64,
            plan_digest="0" * 64,
            passes=(
                PassCertificate(
                    "exact_cache_contract",
                    enabled=False,
                    applied=False,
                    reason="fixture",
                ),
            ),
            caches=(),
            loops=(),
            arena=ArenaCertificate(
                enabled=False,
                baseline_bytes=0,
                compiled_bytes=0,
                baseline_allocations=0,
                compiled_allocations=0,
            ),
        ),
    )


def test_region_artifact_round_trip_is_deterministic(tmp_path: Path) -> None:
    artifact = _region(tmp_path)
    decoded = RegionArtifactContract.from_dict(artifact.to_dict())
    assert decoded == artifact
    assert decoded.inputs[0].dimensions[1].symbol == "tokens"
    assert decoded.input_schema_digest == artifact.input_schema_digest
    assert decoded.output_schema_digest == artifact.output_schema_digest
    assert decoded.residency is ArtifactResidency.INVOCATION


def _numerical_artifact(artifact: RegionArtifactContract) -> RegionArtifactContract:
    policy = NumericalPolicy("fixture.numeric/1", (("precise", True),))
    record = NumericalCompileRecord(
        artifact.capability.backend, artifact.capability.target, "compiler-1",
        "e" * 64, artifact.identity.graph_sha256, artifact.artifact_sha256,
        artifact.artifact_size_bytes, policy, policy, policy,
        '{"options":{"fuse":false},"rewrites":[],"versions":{"runtime":"1"}}',
    )
    requirement = NumericalRequirement(policy, "same-precision", policy.digest(), record.digest())
    binding = RegionNumericalBinding(artifact.region_name, requirement, record)
    return replace(artifact, schema=NUMERICAL_ARTIFACT_SCHEMA, numerical_binding=binding)


def _numerical_bundle(root: Path) -> CompileBundleManifest:
    bundle = _bundle(root)
    artifact = _numerical_artifact(bundle.region_artifacts[0])
    certificate = replace(
        bundle.compilation_certificate, schema=NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
        numerical_bindings=(artifact.numerical_binding,),
    )
    return replace(bundle, schema=NUMERICAL_BUNDLE_SCHEMA, region_artifacts=(artifact,),
                   compilation_certificate=certificate)


def test_policy_documents_round_trip_without_claiming_enforcement(tmp_path):
    bundle = _numerical_bundle(tmp_path)
    restored = CompileBundleManifest.from_dict(bundle.to_dict())
    assert restored == bundle
    assert restored.digest() == bundle.digest()
    restored.verify_files(tmp_path)
    for document in (bundle, bundle.region_artifacts[0], bundle.compilation_certificate):
        assert document.to_dict()["runtime_enforcement"] == "unimplemented"
        with pytest.raises(NumericalEnforcementUnavailable, match="unimplemented"):
            document.require_runtime_deployable()


def test_legacy_documents_have_no_new_fields_or_numerical_claims(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle.schema == "vlaforge.compile_bundle/4"
    assert bundle.region_artifacts[0].schema == "vlaforge.region_artifact/3"
    assert bundle.compilation_certificate.schema == "vlaforge.compilation_certificate/2"
    for document in (bundle, bundle.region_artifacts[0], bundle.compilation_certificate):
        payload = document.to_dict()
        assert not any(key.startswith("numerical") or key == "runtime_enforcement" for key in payload)
        assert type(document).from_dict(payload).to_dict() == payload
        document.require_runtime_deployable()


@pytest.mark.parametrize("kind,legacy", (("artifact", "vlaforge.region_artifact/3"),
    ("certificate", "vlaforge.compilation_certificate/2"), ("bundle", "vlaforge.compile_bundle/4")))
@pytest.mark.parametrize("change", ("downgrade", "missing", "claim", "unknown"))
def test_policy_document_downgrades_and_unchecked_claims_rejected(tmp_path, kind, legacy, change):
    bundle = _numerical_bundle(tmp_path)
    documents = {"artifact": bundle.region_artifacts[0],
                 "certificate": bundle.compilation_certificate, "bundle": bundle}
    document = documents[kind]
    payload = document.to_dict()
    if change == "downgrade":
        payload["schema"] = legacy
    elif change == "missing":
        del payload[next(key for key in payload if key.startswith("numerical"))]
    elif change == "claim":
        payload["runtime_enforcement"] = "verified"
    else:
        payload["fidelity_verified"] = True
    with pytest.raises(NumericalContractError):
        type(document).from_dict(payload)


@pytest.mark.parametrize("kind,version", (("artifact", NUMERICAL_ARTIFACT_SCHEMA),
    ("certificate", NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA), ("bundle", NUMERICAL_BUNDLE_SCHEMA)))
def test_new_schema_cannot_omit_policy(tmp_path, kind, version):
    bundle = _bundle(tmp_path)
    document = {"artifact": bundle.region_artifacts[0],
                "certificate": bundle.compilation_certificate, "bundle": bundle}[kind]
    with pytest.raises(ValueError, match="requires its explicit schema"):
        replace(document, schema=version)


@pytest.mark.parametrize("field,value", (("backend", "other"), ("target", "cpu"),
    ("graph_sha256", "f" * 64), ("artifact_sha256", "f" * 64), ("artifact_size_bytes", 99)))
def test_numerical_compile_record_must_match_actual_artifact(tmp_path, field, value):
    artifact = _numerical_artifact(_region(tmp_path))
    binding = artifact.numerical_binding
    record = replace(binding.compile_record, **{field: value})
    binding = replace(binding, compile_record=record,
                      requirement=replace(binding.requirement, compile_record_sha256=record.digest()))
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(artifact, numerical_binding=binding)


def test_numerical_region_identity_and_certificate_coverage(tmp_path):
    bundle = _numerical_bundle(tmp_path)
    artifact = bundle.region_artifacts[0]
    binding = artifact.numerical_binding
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(artifact, numerical_binding=replace(binding, region_name="other"))
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(bundle.compilation_certificate, numerical_bindings=(binding, binding))
    changed = replace(binding, requirement=replace(binding.requirement, execution_lane="quantized"))
    certificate = replace(bundle.compilation_certificate, numerical_bindings=(changed,))
    with pytest.raises(ValueError, match="disagree with compilation certificate"):
        replace(bundle, compilation_certificate=certificate)


def test_numerical_bundle_rejects_forged_binding_digest(tmp_path):
    data = _numerical_bundle(tmp_path).to_dict()
    data["numerical_bindings_sha256"]["region_0"] = "f" * 64
    with pytest.raises(ValueError, match="numerical binding digest mismatch"):
        CompileBundleManifest.from_dict(data)


def test_provider_mode_roundtrip_and_outer_marker_consistency(tmp_path):
    from vlaforge.deployment.numerical import PROVIDER_REQUIRED

    bundle = _numerical_bundle(tmp_path)
    original = bundle.region_artifacts[0]
    binding = replace(original.numerical_binding, runtime_enforcement=PROVIDER_REQUIRED)
    artifact = replace(original, numerical_binding=binding)
    certificate = replace(bundle.compilation_certificate, numerical_bindings=(binding,))
    bundle = replace(bundle, region_artifacts=(artifact,), compilation_certificate=certificate)
    for document in (artifact, certificate, bundle):
        data = document.to_dict()
        assert data["runtime_enforcement"] == PROVIDER_REQUIRED
        assert type(document).from_dict(data) == document
        with pytest.raises(ValueError, match="enforcement marker mismatch"):
            type(document).from_dict({**data, "runtime_enforcement": "unimplemented"})
    # Existing AOTI/TensorRT contracts remain unsupported despite explicit mode.
    with pytest.raises(NumericalEnforcementUnavailable, match="unsupported"):
        artifact.require_runtime_deployable()


def test_certificate_rejects_mixed_record_only_and_provider_modes(tmp_path):
    from vlaforge.deployment.numerical import PROVIDER_REQUIRED

    certificate = _numerical_bundle(tmp_path).compilation_certificate
    record_only = certificate.numerical_bindings[0]
    provider = replace(record_only, region_name="z_other", runtime_enforcement=PROVIDER_REQUIRED)
    with pytest.raises(NumericalContractError, match="mixed"):
        replace(certificate, numerical_bindings=(record_only, provider))


def test_bundle_loader_rejects_duplicate_policy_keys(tmp_path):
    bundle = _numerical_bundle(tmp_path)
    text = bundle.canonical_json().replace('"precise":true', '"precise":true,"precise":false', 1)
    path = tmp_path / "bundle.json"
    path.write_text(text)
    with pytest.raises(NumericalContractError, match="duplicate JSON key"):
        load_bundle_manifest(path)


def test_policy_bearing_bundle_build_fails_before_writes_or_compilation(tmp_path, monkeypatch):
    from vlaforge.adapters import build_openvla_fixture
    from vlaforge.deployment import build, build_artifact_compile_bundle

    module = build_openvla_fixture().module
    legacy = _region(tmp_path)
    contracts = {region.name: _numerical_artifact(replace(legacy, region_id=index, region_name=region.name))
                 for index, region in enumerate(module.regions)}

    def forbidden(*args, **kwargs):
        raise AssertionError("policy requirements must be rejected before compilation")

    monkeypatch.setattr(build, "compile_module", forbidden)
    output = tmp_path / "must-not-exist"
    with pytest.raises(NumericalEnforcementUnavailable, match="unimplemented"):
        build_artifact_compile_bundle(
            module, output, region_artifacts=contracts,
            artifact_sources={name: "missing-is-not-read" for name in contracts},
            validators={}, runner_source="", runtime_root="missing", cmake_prefix_path="missing",
            backend_versions={"fixture": "1"}, source_revision="test", source_dirty=False,
        )
    assert not output.exists()


def test_policy_certificate_cannot_be_silently_dropped_by_codegen(tmp_path):
    from vlaforge.adapters import build_openvla_fixture
    from vlaforge.codegen import generate_compiled_cpp_session, generate_cpp_session
    from vlaforge.codegen.cpp import CodegenUnsupportedError
    from vlaforge.compiler import compile_module

    compilation = compile_module(build_openvla_fixture().module)
    binding = _numerical_artifact(_region(tmp_path)).numerical_binding
    certificate = replace(compilation.certificate, schema=NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
                          numerical_bindings=(binding,))
    with pytest.raises(CodegenUnsupportedError, match="numerical runtime enforcement is unimplemented"):
        generate_compiled_cpp_session(replace(compilation, certificate=certificate), validators={})
    with pytest.raises(CodegenUnsupportedError, match="numerical runtime enforcement is unimplemented"):
        generate_cpp_session(compilation.plan, compilation.module, validators={},
                             compilation_certificate=certificate)
    with pytest.raises(CodegenUnsupportedError, match="numerical runtime enforcement is unimplemented"):
        generate_cpp_session(compilation.plan, compilation.module, validators={},
                             compilation_certificate=certificate.to_dict())
    with pytest.raises(CodegenUnsupportedError, match="numerical runtime enforcement is unimplemented"):
        generate_cpp_session(compilation.plan, compilation.module, validators={},
                             compilation_certificate={"numerical_bindings": []})


def test_region_artifact_defaults_legacy_residency_to_session(
    tmp_path: Path,
) -> None:
    payload = _region(tmp_path).to_dict()
    del payload["residency"]

    decoded = RegionArtifactContract.from_dict(payload)

    assert decoded.residency is ArtifactResidency.SESSION


def test_region_artifact_rejects_unknown_residency(tmp_path: Path) -> None:
    payload = _region(tmp_path).to_dict()
    payload["residency"] = "forever"

    with pytest.raises(ValueError, match="not a valid ArtifactResidency"):
        RegionArtifactContract.from_dict(payload)


def test_region_artifact_rejects_tampered_signature_digest(
    tmp_path: Path,
) -> None:
    payload = _region(tmp_path).to_dict()
    payload["input_schema_digest"] = "f" * 64
    with pytest.raises(ValueError, match="input schema digest mismatch"):
        RegionArtifactContract.from_dict(payload)


def test_region_artifact_rejects_unknown_schema(tmp_path: Path) -> None:
    artifact = _region(tmp_path)
    with pytest.raises(ValueError, match="unsupported artifact schema"):
        replace(artifact, schema="vlaforge.region_artifact/999")


def test_region_artifact_rejects_hidden_effects(tmp_path: Path) -> None:
    artifact = _region(tmp_path)
    with pytest.raises(ValueError, match="effect audit did not pass"):
        replace(artifact, effect_audit=EffectAudit(hidden_mutation=True))


def test_region_artifact_rejects_backend_dtype_mismatch(tmp_path: Path) -> None:
    artifact = _region(tmp_path)
    with pytest.raises(ValueError, match="does not support dtypes"):
        replace(
            artifact,
            capability=replace(
                artifact.capability, supported_dtypes=("f16",)
            ),
        )


@pytest.mark.parametrize(
    "path",
    ("/absolute/artifact.bin", "../escape.bin", "a/../artifact.bin", r"a\b.bin"),
)
def test_region_artifact_rejects_unsafe_paths(
    tmp_path: Path, path: str
) -> None:
    artifact = _region(tmp_path)
    with pytest.raises(ValueError, match="artifact path"):
        replace(artifact, artifact_path=path)


def test_bundle_manifest_round_trip_and_file_verification(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    first = bundle.canonical_json()
    bundle.write(manifest_path)
    loaded = load_bundle_manifest(manifest_path)

    assert loaded == bundle
    assert loaded.canonical_json() == first
    assert loaded.digest() == bundle.digest()
    assert loaded.semantic_ir_digest == bundle.semantic_ir.sha256
    loaded.verify_files(tmp_path)


def test_bundle_rejects_unknown_schema(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="unsupported bundle schema"):
        replace(bundle, schema="vlaforge.compile_bundle/999")


def test_bundle_detects_tampered_file(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / bundle.semantic_ir.path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="semantic_ir"):
        bundle.verify_files(tmp_path)


def test_bundle_rejects_artifact_io_schema_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="I/O schema digest"):
        replace(
            bundle,
            region_artifacts=(
                replace(
                    bundle.region_artifacts[0],
                    io_schema_digest="f" * 64,
                ),
            ),
        )


def test_bundle_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "metadata" / "escaped.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes bundle root"):
        FileRecord.from_file(tmp_path, "metadata/escaped.json", "escaped")


def test_bundle_rejects_duplicate_region_ids(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    other = replace(
        _region(tmp_path, region_id=1),
        region_id=0,
        region_name="different",
    )
    with pytest.raises(ValueError, match="duplicate region ids"):
        replace(bundle, region_artifacts=(bundle.region_artifacts[0], other))


def test_bundle_rejects_wrong_required_role(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="required roles"):
        replace(
            bundle,
            semantic_ir=replace(bundle.semantic_ir, role="not_semantic_ir"),
        )


def test_reproducibility_environment_must_be_sorted() -> None:
    with pytest.raises(ValueError, match="sorted"):
        ReproducibilityManifest(
            source_revision="abc",
            source_dirty=False,
            build_commands=("build",),
            random_seed=0,
            environment=(("Z", "1"), ("A", "2")),
        )
