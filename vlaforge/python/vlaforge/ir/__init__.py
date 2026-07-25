"""Core VLAForge Invocation IR definitions."""

from vlaforge.ir.attrs import Effect, Ownership
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    Module,
    Operation,
    OutputPort,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import *

__all__ = [
    "Block",
    "Effect",
    "InputPort",
    "Invocation",
    "Module",
    "Operation",
    "OutputPort",
    "Ownership",
    "StateSlot",
    "TensorRegion",
    "Value",
]
