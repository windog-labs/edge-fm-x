"""Versioned deployment artifact and compile-bundle contracts."""

from vlaforge.deployment.contract import (
    CALLABLE_ABI_VERSION,
    ARTIFACT_SCHEMA,
    REGION_PLUGIN_ABI,
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


def build_compile_bundle(*args, **kwargs):
    """Lazily import the build pipeline to keep frontend contracts acyclic."""

    from vlaforge.deployment.build import build_compile_bundle as build

    return build(*args, **kwargs)

__all__ = [
    "ARTIFACT_SCHEMA",
    "BUNDLE_SCHEMA",
    "CALLABLE_ABI_VERSION",
    "REGION_PLUGIN_ABI",
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
    "build_compile_bundle",
]
