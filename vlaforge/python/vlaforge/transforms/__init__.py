"""VLA invocation-specific, semantics-preserving transformations."""

from vlaforge.transforms.canonicalize import canonicalize
from vlaforge.transforms.exact_cache import (
    ExactCacheContractError,
    configure_exact_cache,
)
from vlaforge.transforms.loop_invariant import (
    LoopInvariantAnalysis,
    LoopInvariantDecision,
    analyze_structured_loop_invariance,
)
from vlaforge.transforms.region_interface import project_region_inputs, reorder_region_interface
from vlaforge.transforms.region_partition import split_region_invocation

__all__ = [
    "analyze_structured_loop_invariance",
    "canonicalize",
    "configure_exact_cache",
    "ExactCacheContractError",
    "LoopInvariantAnalysis",
    "LoopInvariantDecision",
    "project_region_inputs",
    "reorder_region_interface",
    "split_region_invocation",
]
