"""Schema-version compatibility helpers."""

from __future__ import annotations

from vlaforge.ir.program import SCHEMA_VERSION


class UnsupportedSchemaVersion(ValueError):
    pass


def require_supported(version: str) -> None:
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported VLA IR schema {version!r}; expected {SCHEMA_VERSION!r}"
        )

