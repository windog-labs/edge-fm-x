"""Lowering, verification, execution, and memory planning for Plan v2."""

from vlaforge.plan.executor import PlanExecutionError, PlanExecutor
from vlaforge.plan.lowering import ArtifactVariant, lower_to_plan
from vlaforge.plan.memory import (
    MemoryPlanningError,
    StorageOverride,
    UnsafeStateCapacityError,
    can_reuse_physical_storage,
    emit_memory_constants,
    physicalize_plan,
    state_arena_sizes,
    storage_size_bytes,
)
from vlaforge.plan.model import (
    PLAN_SCHEMA,
    ArtifactBinding,
    BufferClass,
    LogicalBuffer,
    PhysicalBuffer,
    PlanBlock,
    PlanInvocation,
    PlanModule,
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
    "LogicalBuffer",
    "MemoryPlanningError",
    "PhysicalBuffer",
    "PlanBlock",
    "PlanDiagnostic",
    "PlanExecutionError",
    "PlanExecutor",
    "PlanInvocation",
    "PlanModule",
    "PlanVerificationError",
    "StateBinding",
    "StaticArenaPlan",
    "StorageOverride",
    "Task",
    "TaskKind",
    "UnsafeStateCapacityError",
    "can_reuse_physical_storage",
    "emit_memory_constants",
    "lower_to_plan",
    "physicalize_plan",
    "state_arena_sizes",
    "storage_size_bytes",
    "verify_plan",
]
