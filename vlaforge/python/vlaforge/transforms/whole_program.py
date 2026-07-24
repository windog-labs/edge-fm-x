"""Small VLA-specific whole-program optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.ir.program import Module
from vlaforge.plan import PlanModule, lower_to_plan, physicalize_plan
from vlaforge.transforms.epoch_memoize import (
    synthesize_epoch_memoization,
)
from vlaforge.transforms.temporal_licm import (
    TemporalLICMDecision,
    temporal_loop_invariant_code_motion,
)


@dataclass(frozen=True, slots=True)
class WholeProgramOptimizationResult:
    module: Module
    baseline_plan: PlanModule
    optimized_plan: PlanModule
    licm_decisions: tuple[TemporalLICMDecision, ...]

    @property
    def baseline_arena_bytes(self) -> int:
        assert self.baseline_plan.arena is not None
        return self.baseline_plan.arena.size_bytes

    @property
    def optimized_arena_bytes(self) -> int:
        assert self.optimized_plan.arena is not None
        return self.optimized_plan.arena.size_bytes


def optimize_whole_program(
    module: Module,
) -> WholeProgramOptimizationResult:
    """Run only the three first-round VLA deployment optimizations."""

    memoized = synthesize_epoch_memoization(module)
    licm = temporal_loop_invariant_code_motion(memoized)
    lowered = lower_to_plan(licm.module)
    baseline = physicalize_plan(lowered)
    optimized = physicalize_plan(
        lowered,
        reuse_temporaries=True,
    )
    return WholeProgramOptimizationResult(
        module=licm.module,
        baseline_plan=baseline,
        optimized_plan=optimized,
        licm_decisions=licm.decisions,
    )
