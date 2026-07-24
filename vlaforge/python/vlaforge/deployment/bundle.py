"""Deterministic compile-bundle manifest and on-disk verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from vlaforge.deployment.contract import RegionArtifactContract


BUNDLE_SCHEMA = "vlaforge.compile_bundle/1"
_REQUIRED_ROLES = {
    "semantic_ir",
    "scheduled_plan",
    "state_schema",
    "physical_memory_plan",
    "input_schema",
    "output_schema",
}


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def _validate_relative_path(path: str) -> None:
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in path
        or str(candidate) != path
        or path == "."
    ):
        raise ValueError(f"bundle path must be normalized and relative: {path!r}")


@dataclass(frozen=True, slots=True)
class FileRecord:
    role: str
    path: str
    sha256: str
    size_bytes: int
    executable: bool = False

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("file role must be non-empty")
        _validate_relative_path(self.path)
        _validate_sha256(self.sha256)
        if self.size_bytes < 0:
            raise ValueError("file size must be non-negative")

    @classmethod
    def from_file(
        cls,
        root: str | Path,
        path: str,
        role: str,
        *,
        executable: bool = False,
    ) -> "FileRecord":
        _validate_relative_path(path)
        payload = (Path(root) / path).read_bytes()
        return cls(
            role=role,
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            executable=executable,
        )

    def verify(self, root: str | Path) -> None:
        candidate = Path(root) / self.path
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        payload = candidate.read_bytes()
        if len(payload) != self.size_bytes:
            raise ValueError(
                f"bundle file {self.path}: size {len(payload)} != {self.size_bytes}"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.sha256:
            raise ValueError(
                f"bundle file {self.path}: sha256 {digest} != {self.sha256}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "executable": self.executable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileRecord":
        return cls(
            role=str(data["role"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            executable=bool(data.get("executable", False)),
        )


@dataclass(frozen=True, slots=True)
class VersionEntry:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("version entry name and version must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionEntry":
        return cls(str(data["name"]), str(data["version"]))


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    source_revision: str
    source_dirty: bool
    build_commands: tuple[str, ...]
    random_seed: int
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_revision:
            raise ValueError("source revision must be non-empty")
        if not self.build_commands or any(not command for command in self.build_commands):
            raise ValueError("at least one non-empty build command is required")
        names = [name for name, _ in self.environment]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("reproducibility environment keys must be unique")
        if tuple(sorted(self.environment)) != self.environment:
            raise ValueError("reproducibility environment must be sorted by key")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_revision": self.source_revision,
            "source_dirty": self.source_dirty,
            "build_commands": list(self.build_commands),
            "random_seed": self.random_seed,
            "environment": [
                {"name": name, "value": value} for name, value in self.environment
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReproducibilityManifest":
        return cls(
            source_revision=str(data["source_revision"]),
            source_dirty=bool(data["source_dirty"]),
            build_commands=tuple(str(item) for item in data["build_commands"]),
            random_seed=int(data["random_seed"]),
            environment=tuple(
                (str(item["name"]), str(item["value"]))
                for item in data.get("environment", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class CompileBundleManifest:
    semantic_ir: FileRecord
    scheduled_plan: FileRecord
    state_schema: FileRecord
    physical_memory_plan: FileRecord
    input_schema: FileRecord
    output_schema: FileRecord
    region_artifacts: tuple[RegionArtifactContract, ...]
    generated_sources: tuple[FileRecord, ...]
    binaries: tuple[FileRecord, ...]
    toolchain_versions: tuple[VersionEntry, ...]
    backend_versions: tuple[VersionEntry, ...]
    reproducibility: ReproducibilityManifest
    schema: str = BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported bundle schema: {self.schema!r}")
        required = (
            self.semantic_ir,
            self.scheduled_plan,
            self.state_schema,
            self.physical_memory_plan,
            self.input_schema,
            self.output_schema,
        )
        roles = {item.role for item in required}
        if roles != _REQUIRED_ROLES:
            raise ValueError(
                f"bundle required roles are {sorted(roles)}, expected "
                f"{sorted(_REQUIRED_ROLES)}"
            )
        if not self.region_artifacts:
            raise ValueError("compile bundle requires at least one region artifact")
        if not self.generated_sources:
            raise ValueError("compile bundle requires generated source files")
        if not self.binaries:
            raise ValueError("compile bundle requires at least one binary")
        if not self.toolchain_versions or not self.backend_versions:
            raise ValueError("bundle requires toolchain and backend versions")
        region_ids = [item.region_id for item in self.region_artifacts]
        region_names = [item.region_name for item in self.region_artifacts]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("bundle contains duplicate region ids")
        if len(region_names) != len(set(region_names)):
            raise ValueError("bundle contains duplicate region names")
        version_names = [
            item.name for item in self.toolchain_versions + self.backend_versions
        ]
        if len(version_names) != len(set(version_names)):
            raise ValueError("bundle contains duplicate version entry names")
        files = required + self.generated_sources + self.binaries
        paths = [item.path for item in files] + [
            item.artifact_path for item in self.region_artifacts
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle contains duplicate file paths")

    @property
    def semantic_ir_digest(self) -> str:
        return self.semantic_ir.sha256

    def file_records(self) -> tuple[FileRecord, ...]:
        return (
            self.semantic_ir,
            self.scheduled_plan,
            self.state_schema,
            self.physical_memory_plan,
            self.input_schema,
            self.output_schema,
            *self.generated_sources,
            *self.binaries,
        )

    def verify_files(self, root: str | Path) -> None:
        for record in self.file_records():
            record.verify(root)
        for artifact in self.region_artifacts:
            FileRecord(
                role=f"region_artifact:{artifact.region_name}",
                path=artifact.artifact_path,
                sha256=artifact.artifact_sha256,
                size_bytes=artifact.artifact_size_bytes,
            ).verify(root)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "semantic_ir_digest": self.semantic_ir_digest,
            "semantic_ir": self.semantic_ir.to_dict(),
            "scheduled_plan": self.scheduled_plan.to_dict(),
            "state_schema": self.state_schema.to_dict(),
            "physical_memory_plan": self.physical_memory_plan.to_dict(),
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "region_artifacts": [
                item.to_dict()
                for item in sorted(
                    self.region_artifacts, key=lambda artifact: artifact.region_id
                )
            ],
            "generated_sources": [
                item.to_dict()
                for item in sorted(self.generated_sources, key=lambda file: file.path)
            ],
            "binaries": [
                item.to_dict()
                for item in sorted(self.binaries, key=lambda file: file.path)
            ],
            "toolchain_versions": [
                item.to_dict()
                for item in sorted(self.toolchain_versions, key=lambda entry: entry.name)
            ],
            "backend_versions": [
                item.to_dict()
                for item in sorted(self.backend_versions, key=lambda entry: entry.name)
            ],
            "reproducibility": self.reproducibility.to_dict(),
        }

    def canonical_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.canonical_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompileBundleManifest":
        semantic_ir = FileRecord.from_dict(data["semantic_ir"])
        if str(data.get("semantic_ir_digest", semantic_ir.sha256)) != semantic_ir.sha256:
            raise ValueError("semantic_ir_digest does not match semantic IR record")
        return cls(
            schema=str(data["schema"]),
            semantic_ir=semantic_ir,
            scheduled_plan=FileRecord.from_dict(data["scheduled_plan"]),
            state_schema=FileRecord.from_dict(data["state_schema"]),
            physical_memory_plan=FileRecord.from_dict(
                data["physical_memory_plan"]
            ),
            input_schema=FileRecord.from_dict(data["input_schema"]),
            output_schema=FileRecord.from_dict(data["output_schema"]),
            region_artifacts=tuple(
                RegionArtifactContract.from_dict(item)
                for item in data["region_artifacts"]
            ),
            generated_sources=tuple(
                FileRecord.from_dict(item) for item in data["generated_sources"]
            ),
            binaries=tuple(FileRecord.from_dict(item) for item in data["binaries"]),
            toolchain_versions=tuple(
                VersionEntry.from_dict(item) for item in data["toolchain_versions"]
            ),
            backend_versions=tuple(
                VersionEntry.from_dict(item) for item in data["backend_versions"]
            ),
            reproducibility=ReproducibilityManifest.from_dict(
                data["reproducibility"]
            ),
        )


def load_bundle_manifest(path: str | Path) -> CompileBundleManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("bundle manifest root must be an object")
    return CompileBundleManifest.from_dict(data)
