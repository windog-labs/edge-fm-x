"""Static analyses and semantic verification."""

from vlaforge.analysis.dependency import DependencyGraph, build_dependency_graph
from vlaforge.analysis.liveness import LiveRange, analyze_liveness
from vlaforge.analysis.physical_slots import (
    PhysicalSlotPlan,
    UnsafePhysicalizationError,
    plan_physical_slots,
)
from vlaforge.analysis.verifier import Diagnostic, VerificationError, verify

__all__ = [
    "DependencyGraph",
    "Diagnostic",
    "LiveRange",
    "PhysicalSlotPlan",
    "UnsafePhysicalizationError",
    "VerificationError",
    "analyze_liveness",
    "build_dependency_graph",
    "plan_physical_slots",
    "verify",
]

