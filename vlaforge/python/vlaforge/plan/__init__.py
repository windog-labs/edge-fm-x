"""Internal lowering, verification, and simulation for Scheduled Plans."""

from vlaforge.plan.executor import PlanExecutionError, PlanExecutor
from vlaforge.plan.lowering import ArtifactVariant, lower_to_plan
from vlaforge.plan.memory import (
    MemoryPlanningError,
    StorageOverride,
    UnsafeStateCapacityError,
    emit_memory_constants,
    physicalize_plan,
    state_arena_sizes,
    storage_size_bytes,
)
from vlaforge.plan.model import (
    PLAN_SCHEMA,
    ArtifactBinding,
    BufferClass,
    DeadlineGuard,
    FallbackTarget,
    FreshnessGuard,
    LogicalBuffer,
    PhysicalBuffer,
    PlanBlock,
    PlanModule,
    PlanPolicy,
    StateBinding,
    StaticArenaPlan,
    Task,
    TaskKind,
)
from vlaforge.plan.verifier import (
    PlanDiagnostic,
    PlanVerificationError,
    verify_plan,
)

__all__ = [
    "PLAN_SCHEMA",
    "ArtifactBinding",
    "ArtifactVariant",
    "BufferClass",
    "DeadlineGuard",
    "FallbackTarget",
    "FreshnessGuard",
    "LogicalBuffer",
    "MemoryPlanningError",
    "PhysicalBuffer",
    "PlanBlock",
    "PlanDiagnostic",
    "PlanExecutionError",
    "PlanExecutor",
    "PlanModule",
    "PlanPolicy",
    "PlanVerificationError",
    "StateBinding",
    "StaticArenaPlan",
    "StorageOverride",
    "Task",
    "TaskKind",
    "UnsafeStateCapacityError",
    "emit_memory_constants",
    "lower_to_plan",
    "physicalize_plan",
    "state_arena_sizes",
    "storage_size_bytes",
    "verify_plan",
]
