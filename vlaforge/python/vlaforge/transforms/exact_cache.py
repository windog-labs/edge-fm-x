"""Validate and configure explicit exact-cache contracts."""

from __future__ import annotations

from dataclasses import replace

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Module


class ExactCacheContractError(ValueError):
    pass


def configure_exact_cache(module: Module, *, enabled: bool) -> Module:
    """Enable only pure, explicit exact memoization declarations."""

    input_names = tuple(port.name for port in module.inputs)
    state_names = tuple(state.name for state in module.states)
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
            selected_inputs = tuple(
                str(item)
                for item in metadata.get(
                    "cache_input_ports", input_names
                )
            )
            selected_states = tuple(
                str(item)
                for item in metadata.get(
                    "cache_state_slots", state_names
                )
            )
            unknown_inputs = sorted(set(selected_inputs) - set(input_names))
            unknown_states = sorted(set(selected_states) - set(state_names))
            if (
                unknown_inputs
                or unknown_states
                or len(selected_inputs) != len(set(selected_inputs))
                or len(selected_states) != len(set(selected_states))
            ):
                raise ExactCacheContractError(
                    f"region={region.name} invalid exact-cache dependencies: "
                    f"unknown_inputs={unknown_inputs}, "
                    f"unknown_states={unknown_states}"
                )
            metadata["cache_input_ports"] = selected_inputs
            metadata["cache_state_slots"] = selected_states
            metadata["cache_key_contract"] = (
                "selected_input_revision+selected_state_version+"
                "episode+model+artifact"
            )
        regions.append(replace(region, metadata=metadata))
    transformed = replace(module, regions=tuple(regions))
    verify(transformed)
    return transformed
