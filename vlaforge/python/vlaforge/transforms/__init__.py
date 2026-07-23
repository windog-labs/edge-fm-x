"""Semantics-preserving IR transformations."""

from vlaforge.transforms.canonicalize import canonicalize
from vlaforge.transforms.epoch_memoize import (
    MemoizationSynthesisError,
    synthesize_epoch_memoization,
)
from vlaforge.transforms.physicalize_state import physicalize_state

__all__ = [
    "canonicalize",
    "MemoizationSynthesisError",
    "physicalize_state",
    "synthesize_epoch_memoization",
]
