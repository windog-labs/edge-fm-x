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

__all__ = [
    "analyze_structured_loop_invariance",
    "canonicalize",
    "configure_exact_cache",
    "ExactCacheContractError",
    "LoopInvariantAnalysis",
    "LoopInvariantDecision",
]
