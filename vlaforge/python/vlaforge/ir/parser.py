"""Parser for the stable VLAForge textual IR form."""

from __future__ import annotations

from vlaforge.ir.printer import MAGIC
from vlaforge.ir.program import Module
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.ir.versioning import require_supported


class ParseError(ValueError):
    pass


def parse_module(text: str) -> Module:
    header, separator, payload = text.partition("\n")
    if not separator:
        raise ParseError("missing textual IR payload")
    parts = header.split()
    if len(parts) != 2 or parts[0] != MAGIC:
        raise ParseError(f"expected header '{MAGIC} <version>'")
    require_supported(parts[1])
    try:
        module = parse_canonical_json(payload)
    except Exception as exc:
        raise ParseError(f"invalid VLAForge IR payload: {exc}") from exc
    if module.schema_version != parts[1]:
        raise ParseError(
            f"header schema {parts[1]!r} does not match payload "
            f"{module.schema_version!r}"
        )
    return module

