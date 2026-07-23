"""Stable textual representation of VLAForge IR v0."""

from __future__ import annotations

from vlaforge.ir.program import Module
from vlaforge.ir.serializer import canonical_json


MAGIC = "!vlaforge.ir"


def print_module(module: Module) -> str:
    """Print a deterministic, parser-round-trippable textual module.

    The v0 syntax uses a versioned header plus canonical structural payload.
    The payload is executable IR, not a deployment manifest. A future MLIR
    dialect can be introduced without changing the normative Python semantics.
    """

    return f"{MAGIC} {module.schema_version}\n{canonical_json(module, indent=2)}\n"

