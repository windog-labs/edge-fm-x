"""Backend compile requests derived from verified frontend captures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from vlaforge.deployment.contract import (
    CALLABLE_ABI_VERSION,
    ArtifactKind,
    ArtifactIdentity,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
)
from vlaforge.frontend.region_capture import CaptureOutcome


COMPILE_REQUEST_SCHEMA = "vlaforge.region_compile_request/3"


@dataclass(frozen=True, slots=True)
class RegionCompileRequest:
    region_id: int
    region_name: str
    graph_digest: str
    io_schema_digest: str
    identity: ArtifactIdentity
    inputs: tuple[ValueContract, ...]
    outputs: tuple[ValueContract, ...]
    artifact_kind: ArtifactKind
    output_path: str
    workspace: WorkspaceContract
    capability: BackendCapability
    effect_audit: EffectAudit
    backend_options: tuple[tuple[str, str], ...] = ()
    backend_variant: str | None = None
    callable_abi_version: int = CALLABLE_ABI_VERSION
    schema: str = COMPILE_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPILE_REQUEST_SCHEMA:
            raise ValueError(f"unsupported compile request schema: {self.schema!r}")
        if self.region_id < 0 or not self.region_name:
            raise ValueError("compile request requires a valid region id and name")
        if len(self.graph_digest) != 64:
            raise ValueError("compile request graph digest must be SHA-256")
        if self.identity.graph_sha256 != self.graph_digest:
            raise ValueError(
                "compile request identity graph digest does not match capture"
            )
        if len(self.io_schema_digest) != 64:
            raise ValueError("compile request I/O schema digest must be SHA-256")
        if self.callable_abi_version != CALLABLE_ABI_VERSION:
            raise ValueError("unsupported compile-request callable ABI")
        candidate = PurePosixPath(self.output_path)
        if (
            not self.output_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or str(candidate) != self.output_path
        ):
            raise ValueError("compile request output path must be normalized and relative")
        if tuple(sorted(self.backend_options)) != self.backend_options:
            raise ValueError("backend options must be sorted")
        keys = [key for key, _ in self.backend_options]
        if len(keys) != len(set(keys)):
            raise ValueError("backend options contain duplicate keys")
        if not self.effect_audit.passed:
            raise ValueError("compile request requires a passing effect audit")
        if self.backend_variant is not None and not self.backend_variant:
            raise ValueError("backend variant must be non-empty when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "callable_abi_version": self.callable_abi_version,
            "region_id": self.region_id,
            "region_name": self.region_name,
            "graph_digest": self.graph_digest,
            "io_schema_digest": self.io_schema_digest,
            "identity": self.identity.to_dict(),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "artifact_kind": self.artifact_kind.value,
            "output_path": self.output_path,
            "workspace": self.workspace.to_dict(),
            "capability": self.capability.to_dict(),
            "effect_audit": self.effect_audit.to_dict(),
            "backend_options": [
                {"name": name, "value": value}
                for name, value in self.backend_options
            ],
            "backend_variant": self.backend_variant,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionCompileRequest":
        return cls(
            schema=str(data["schema"]),
            callable_abi_version=int(data["callable_abi_version"]),
            region_id=int(data["region_id"]),
            region_name=str(data["region_name"]),
            graph_digest=str(data["graph_digest"]),
            io_schema_digest=str(data["io_schema_digest"]),
            identity=ArtifactIdentity.from_dict(data["identity"]),
            inputs=tuple(
                ValueContract.from_dict(item) for item in data["inputs"]
            ),
            outputs=tuple(
                ValueContract.from_dict(item) for item in data["outputs"]
            ),
            artifact_kind=ArtifactKind(str(data["artifact_kind"])),
            output_path=str(data["output_path"]),
            workspace=WorkspaceContract.from_dict(data["workspace"]),
            capability=BackendCapability.from_dict(data["capability"]),
            effect_audit=EffectAudit.from_dict(data["effect_audit"]),
            backend_options=tuple(
                (str(item["name"]), str(item["value"]))
                for item in data.get("backend_options", ())
            ),
            backend_variant=(
                None
                if data.get("backend_variant") is None
                else str(data["backend_variant"])
            ),
        )


def make_compile_request(
    capture: CaptureOutcome,
    *,
    region_id: int,
    artifact_kind: ArtifactKind,
    output_path: str,
    capability: BackendCapability,
    io_schema_digest: str,
    identity: ArtifactIdentity,
    workspace: WorkspaceContract = WorkspaceContract(),
    backend_options: Mapping[str, str] | None = None,
    backend_variant: str | None = None,
) -> RegionCompileRequest:
    capture.require_supported()
    assert capture.evidence is not None
    return RegionCompileRequest(
        region_id=region_id,
        region_name=capture.region.name,
        graph_digest=capture.evidence.graph_digest,
        io_schema_digest=io_schema_digest,
        identity=identity,
        inputs=capture.evidence.inputs,
        outputs=capture.evidence.outputs,
        artifact_kind=artifact_kind,
        output_path=output_path,
        workspace=workspace,
        capability=capability,
        effect_audit=capture.evidence.effect_audit,
        backend_options=tuple(sorted(dict(backend_options or {}).items())),
        backend_variant=backend_variant,
    )


def finalize_region_artifact(
    request: RegionCompileRequest, root: str | Path
) -> RegionArtifactContract:
    artifact = Path(root) / request.output_path
    payload = artifact.read_bytes()
    return RegionArtifactContract(
        region_id=request.region_id,
        region_name=request.region_name,
        io_schema_digest=request.io_schema_digest,
        identity=request.identity,
        inputs=request.inputs,
        outputs=request.outputs,
        artifact_kind=request.artifact_kind,
        artifact_path=request.output_path,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_size_bytes=len(payload),
        workspace=request.workspace,
        capability=request.capability,
        effect_audit=request.effect_audit,
        backend_variant=request.backend_variant,
        callable_abi_version=request.callable_abi_version,
    )
