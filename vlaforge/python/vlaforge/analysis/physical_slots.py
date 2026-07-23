"""Logical persistent-state version to bounded physical-slot planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vlaforge.ir.program import Module


class UnsafePhysicalizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhysicalSlotPlan:
    state: str
    capacity: int
    retention: int
    max_in_flight: int
    consumer_lag: int
    fallback_snapshots: int

    def slot_for(self, logical_version: int) -> int:
        if logical_version < 0:
            raise ValueError("logical version must be non-negative")
        return logical_version % self.capacity


def plan_physical_slots(
    module: Module,
    *,
    max_in_flight: int = 1,
    consumer_lag: int = 0,
    fallback_snapshots: int = 0,
    capacities: Mapping[str, int] | None = None,
) -> dict[str, PhysicalSlotPlan]:
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if consumer_lag < 0 or fallback_snapshots < 0:
        raise ValueError("consumer lag and fallback snapshots must be non-negative")
    requested = dict(capacities or {})
    plans: dict[str, PhysicalSlotPlan] = {}
    for state in module.states:
        required = max(
            state.retention,
            1 + max_in_flight + consumer_lag + fallback_snapshots,
        )
        capacity = requested.get(state.name, required)
        if capacity < required:
            raise UnsafePhysicalizationError(
                f"program={module.name} rule=physical_slot.unsafe_reuse "
                f"state={state.name} version=retention:{state.retention} "
                f"capacity={capacity} required={required} "
                f"(max_in_flight={max_in_flight}, consumer_lag={consumer_lag}, "
                f"fallback_snapshots={fallback_snapshots})"
            )
        plans[state.name] = PhysicalSlotPlan(
            state=state.name,
            capacity=capacity,
            retention=state.retention,
            max_in_flight=max_in_flight,
            consumer_lag=consumer_lag,
            fallback_snapshots=fallback_snapshots,
        )
    unknown = sorted(set(requested) - {state.name for state in module.states})
    if unknown:
        raise KeyError(f"physical capacities reference unknown states: {unknown}")
    return plans

