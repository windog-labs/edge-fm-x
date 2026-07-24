"""Semantics-preserving IR transformations."""

from vlaforge.transforms.canonicalize import canonicalize
from vlaforge.transforms.epoch_memoize import (
    CacheDependency,
    MemoizationLegality,
    MemoizationSynthesisError,
    memoization_legality,
    synthesize_epoch_memoization,
)
from vlaforge.transforms.physicalize_state import physicalize_state
from vlaforge.transforms.temporal_licm import (
    TemporalLICMDecision,
    TemporalLICMResult,
    temporal_loop_invariant_code_motion,
)
from vlaforge.transforms.whole_program import (
    WholeProgramOptimizationResult,
    optimize_whole_program,
)

__all__ = [
    "canonicalize",
    "CacheDependency",
    "MemoizationLegality",
    "MemoizationSynthesisError",
    "memoization_legality",
    "physicalize_state",
    "synthesize_epoch_memoization",
    "TemporalLICMDecision",
    "TemporalLICMResult",
    "temporal_loop_invariant_code_motion",
    "WholeProgramOptimizationResult",
    "optimize_whole_program",
]
