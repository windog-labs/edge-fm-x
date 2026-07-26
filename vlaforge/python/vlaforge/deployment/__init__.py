"""Versioned deployment artifact and compile-bundle contracts."""

from vlaforge.deployment.aoti_sequence import (
    AOTI_SEQUENCE_SCHEMA,
    AotiSequenceArtifact,
    AotiSequenceManifest,
    AotiSequenceNode,
    AotiSequenceValue,
)
from vlaforge.deployment.contract import (
    CALLABLE_ABI_VERSION,
    ARTIFACT_SCHEMA,
    REGION_PLUGIN_ABI,
    ArtifactDiagnostic,
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    DiagnosticSeverity,
    EffectAudit,
    RegionArtifactContract,
    ShapeDimension,
    ValueContract,
    WorkspaceContract,
)
from vlaforge.deployment.bundle import (
    BUNDLE_SCHEMA,
    CompileBundleManifest,
    FileRecord,
    ReproducibilityManifest,
    VersionEntry,
    load_bundle_manifest,
)


def build_compile_bundle(*args, **kwargs):
    """Lazily import the build pipeline to keep frontend contracts acyclic."""

    from vlaforge.deployment.build import build_compile_bundle as build

    return build(*args, **kwargs)


def build_artifact_compile_bundle(*args, **kwargs):
    """Lazily import the real-artifact bundle pipeline."""

    from vlaforge.deployment.build import (
        build_artifact_compile_bundle as build,
    )

    return build(*args, **kwargs)

__all__ = [
    "AOTI_SEQUENCE_SCHEMA",
    "AotiSequenceArtifact",
    "AotiSequenceManifest",
    "AotiSequenceNode",
    "AotiSequenceValue",
    "ARTIFACT_SCHEMA",
    "BUNDLE_SCHEMA",
    "CALLABLE_ABI_VERSION",
    "REGION_PLUGIN_ABI",
    "ArtifactDiagnostic",
    "ArtifactIdentity",
    "ArtifactKind",
    "ArtifactResidency",
    "BackendCapability",
    "CompileBundleManifest",
    "DiagnosticSeverity",
    "EffectAudit",
    "FileRecord",
    "RegionArtifactContract",
    "ReproducibilityManifest",
    "ShapeDimension",
    "ValueContract",
    "VersionEntry",
    "WorkspaceContract",
    "load_bundle_manifest",
    "build_artifact_compile_bundle",
    "build_compile_bundle",
]
