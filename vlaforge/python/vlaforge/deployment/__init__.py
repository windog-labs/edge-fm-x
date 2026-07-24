"""Versioned deployment artifact and compile-bundle contracts."""

from vlaforge.deployment.contract import (
    CALLABLE_ABI_VERSION,
    ARTIFACT_SCHEMA,
    ArtifactDiagnostic,
    ArtifactKind,
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

__all__ = [
    "ARTIFACT_SCHEMA",
    "BUNDLE_SCHEMA",
    "CALLABLE_ABI_VERSION",
    "ArtifactDiagnostic",
    "ArtifactKind",
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
]
