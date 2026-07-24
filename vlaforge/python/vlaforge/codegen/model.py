"""Inputs and deterministic outputs for static C++ session generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CppRegionDefinition:
    region_name: str
    body: str

    def __post_init__(self) -> None:
        if not self.region_name or not self.body.strip():
            raise ValueError("C++ region definition requires name and body")


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
