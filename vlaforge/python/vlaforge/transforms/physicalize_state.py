"""Annotate state operations with a verified bounded physical-slot plan."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from vlaforge.analysis.physical_slots import plan_physical_slots
from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module


def _rewrite_block(block: Block, capacities: Mapping[str, int]) -> Block:
    operations = []
    for operation in block.operations:
        rewritten = replace(
            operation,
            regions=tuple(
                _rewrite_block(region, capacities) for region in operation.regions
            ),
        )
        if operation.opcode in {"vla.state.read", "vla.state.stage_write"}:
            state = str(operation.attributes["state"])
            rewritten = rewritten.with_attributes(
                physical_capacity=capacities[state],
                physical_index=f"logical_version mod {capacities[state]}",
            )
        operations.append(rewritten)
    return replace(block, operations=tuple(operations))


def physicalize_state(
    module: Module,
    *,
    max_in_flight: int = 1,
    consumer_lag: int = 0,
    fallback_snapshots: int = 0,
    capacities: Mapping[str, int] | None = None,
) -> Module:
    """Lower logical state versions to a proven finite ring capacity."""

    verify(module)
    plans = plan_physical_slots(
        module,
        max_in_flight=max_in_flight,
        consumer_lag=consumer_lag,
        fallback_snapshots=fallback_snapshots,
        capacities=capacities,
    )
    capacity_map = {state: plan.capacity for state, plan in plans.items()}
    metadata = dict(module.metadata)
    metadata["physical_state_plan"] = {
        state: {
            "capacity": plan.capacity,
            "retention": plan.retention,
            "max_in_flight": plan.max_in_flight,
            "consumer_lag": plan.consumer_lag,
            "fallback_snapshots": plan.fallback_snapshots,
        }
        for state, plan in plans.items()
    }
    transformed = replace(
        module,
        policies=tuple(
            replace(policy, body=_rewrite_block(policy.body, capacity_map))
            for policy in module.policies
        ),
        metadata=metadata,
    )
    verify(transformed)
    return transformed

