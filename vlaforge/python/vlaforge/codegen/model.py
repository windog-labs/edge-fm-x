"""Inputs and deterministic outputs for static C++ session generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ZeroStateInitializer:
    """Request runtime zero initialization without embedding tensor literals."""


ZERO_STATE = ZeroStateInitializer()


@dataclass(frozen=True, slots=True)
class CppRegionDefinition:
    region_name: str
    body: str

    def __post_init__(self) -> None:
        if not self.region_name or not self.body.strip():
            raise ValueError("C++ region definition requires name and body")


@dataclass(frozen=True, slots=True)
class CppArtifactRegionDefinition:
    """Codegen-facing subset of a verified RegionArtifactContract."""

    region_name: str
    backend: str
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    io_schema_digest: str
    target: str
    device: str
    backend_variant: str | None = None
    residency: str = "session"
    callable_abi_version: int = 2

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.artifact_path)
        if (
            not self.region_name
            or self.backend not in {"aoti", "tensorrt"}
            or not self.artifact_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or str(candidate) != self.artifact_path
            or self.artifact_size_bytes < 0
            or not self.target
            or not self.device
            or self.callable_abi_version != 2
        ):
            raise ValueError("invalid C++ artifact Region definition")
        if self.backend == "tensorrt" and (
            not self.target.startswith("sm_") or self.device == "cpu"
        ):
            raise ValueError(
                "TensorRT artifact Regions require a CUDA SM target/device"
            )
        if self.backend == "tensorrt" and (
            self.backend_variant is None
            or not self.backend_variant.startswith("tensorrt-")
        ):
            raise ValueError(
                "TensorRT artifact Regions require a TensorRT backend variant"
            )
        for name, value in (
            ("artifact SHA-256", self.artifact_sha256),
            ("I/O schema digest", self.io_schema_digest),
        ):
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if self.backend_variant is not None and not self.backend_variant:
            raise ValueError("backend variant must be non-empty when provided")
        if self.residency not in {"session", "invocation"}:
            raise ValueError(
                "artifact residency must be session or invocation"
            )
        if self.device != "cpu" and not (
            self.device.startswith("cuda:")
            and self.device[5:].isdigit()
        ):
            raise ValueError(
                "artifact device must be cpu or an explicit cuda ordinal"
            )


@dataclass(frozen=True, slots=True)
class CppValidatorDefinition:
    contract_name: str
    body: str

    def __post_init__(self) -> None:
        if not self.contract_name or not self.body.strip():
            raise ValueError("C++ validator definition requires name and body")


@dataclass(frozen=True, slots=True)
class GeneratedSources:
    files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.files]
        if (
            not self.files
            or names != sorted(names)
            or len(names) != len(set(names))
        ):
            raise ValueError(
                "generated source files must be non-empty, sorted, and unique"
            )
        for name, content in self.files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not content:
                raise ValueError(f"invalid generated source file: {name}")

    def as_dict(self) -> dict[str, str]:
        return dict(self.files)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, content in self.files:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(content.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def write(self, root: str | Path) -> None:
        output = Path(root)
        output.mkdir(parents=True, exist_ok=True)
        for name, content in self.files:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def sorted_definitions(
    definitions: Mapping[str, CppRegionDefinition],
) -> tuple[CppRegionDefinition, ...]:
    return tuple(definitions[name] for name in sorted(definitions))
