from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from vlaforge.compiler import (
    ArenaCertificate,
    CompilationCertificate,
    CompilerProfile,
    PassCertificate,
)
from vlaforge.deployment import (
    ArtifactDiagnostic,
    ArtifactKind,
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
