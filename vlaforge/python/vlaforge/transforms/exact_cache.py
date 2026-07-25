"""Validate and configure explicit exact-cache contracts."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Module


class ExactCacheContractError(ValueError):
    pass


def configure_exact_cache(module: Module, *, enabled: bool) -> Module:
    """Enable only pure, explicit exact memoization declarations."""

    regions = []
    for region in module.regions:
        metadata = dict(region.metadata)
        requested = bool(metadata.get("memoize", False))
        reuse_kind = str(metadata.get("reuse_kind", "exact"))
        if requested and reuse_kind != "exact":
            raise ExactCacheContractError(
                f"region={region.name} exact-cache path rejects "
                f"reuse_kind={reuse_kind!r}"
            )
        if requested and not region.pure:
            raise ExactCacheContractError(
                f"region={region.name} exact cache requires a pure region"
            )
        metadata["memoize"] = requested and enabled
        if requested:
            metadata["cache_key_contract"] = (
                "input_revision+state_version+episode+model+artifact"
            )
        regions.append(replace(region, metadata=metadata))
    transformed = replace(module, regions=tuple(regions))
    verify(transformed)
    return transformed
