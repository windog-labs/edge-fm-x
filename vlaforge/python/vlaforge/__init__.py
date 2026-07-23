"""VLAForge executable reference IR."""

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
from vlaforge.ir.types import (
    ActionType,
    CommittedActionType,
    EpochType,
    EventType,
    FutureType,
    IRType,
    PendingType,
    ScalarType,
    SnapshotType,
    TensorType,
    TransactionType,
)

__all__ = [
    "ActionType",
    "Block",
    "ClockDomain",
    "CommittedActionType",
    "EpochType",
    "EventType",
    "FutureType",
    "IRType",
    "InputStream",
    "Module",
    "Operation",
    "PendingType",
    "Policy",
    "ScalarType",
    "SnapshotType",
    "StateSlot",
    "TensorRegion",
    "TensorType",
    "TransactionType",
    "Value",
]

__version__ = "0.1.0.dev0"

