"""Core VLAForge IR definitions."""

from vlaforge.ir.attrs import (
    CheckpointPolicy,
    ConsistencyPolicy,
    Effect,
    EpochExpr,
    FreshnessConstraint,
    Ownership,
    ResetPolicy,
    StateScope,
)
from vlaforge.ir.program import (
    Block,
    ClockDomain,
    InputStream,
    Module,
    Operation,
    Policy,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import *

__all__ = [
    "Block",
    "CheckpointPolicy",
    "ClockDomain",
    "ConsistencyPolicy",
    "Effect",
    "EpochExpr",
    "FreshnessConstraint",
    "InputStream",
    "Module",
    "Operation",
    "Ownership",
    "Policy",
    "ResetPolicy",
    "StateScope",
    "StateSlot",
    "TensorRegion",
    "Value",
]

