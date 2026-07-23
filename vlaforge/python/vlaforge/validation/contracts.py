"""Validation contract definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumericContract:
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    compare_tensor_hashes: bool = True

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("numeric tolerances must be non-negative")

