import random

import pytest

from vlaforge.adapters import build_smolvla_fixture
from vlaforge.analysis import (
    UnsafePhysicalizationError,
    analyze_liveness,
    build_dependency_graph,
    plan_physical_slots,
)


def test_dependency_graph_tracks_state_and_ssa_edges():
    module = build_smolvla_fixture().module
    graph = build_dependency_graph(module)
    assert "rng" in graph.state_readers
    assert "rng" in graph.state_writers
    assert "prefix_cache" in graph.state_writers
    assert graph.value_producers["sample_final"].endswith("vla.for")
    assert any("vla.invoke" in item for item in graph.value_consumers["prefix"])


def test_liveness_covers_loop_and_commit_values():
    ranges = {
        live_range.value: live_range
        for live_range in analyze_liveness(
            build_smolvla_fixture().module, "act"
        )
    }
    assert ranges["tick"].first_definition == -1
    assert ranges["txn"].last_use > ranges["txn"].first_definition
    assert ranges["decoded_action"].last_use > ranges["decoded_action"].first_definition


def test_physical_slot_plan_rejects_unsafe_capacity():
    module = build_smolvla_fixture().module
    with pytest.raises(UnsafePhysicalizationError, match="unsafe_reuse.*rng"):
        plan_physical_slots(
            module,
            max_in_flight=2,
            consumer_lag=1,
            fallback_snapshots=1,
            capacities={"rng": 3, "prefix_cache": 8},
        )


def test_physical_slot_mapping_is_bounded_and_collision_free_in_live_window():
    module = build_smolvla_fixture().module
    rng = random.Random(20260723)
    for _ in range(100):
        in_flight = rng.randint(1, 4)
        lag = rng.randint(0, 3)
        fallback = rng.randint(0, 2)
        plans = plan_physical_slots(
            module,
            max_in_flight=in_flight,
            consumer_lag=lag,
            fallback_snapshots=fallback,
        )
        for plan in plans.values():
            live_window = min(
                plan.capacity,
                max(
                    plan.retention,
                    1 + in_flight + lag + fallback,
                ),
            )
            versions = list(range(17, 17 + live_window))
            assert len({plan.slot_for(version) for version in versions}) == len(
                versions
            )
            assert all(
                0 <= plan.slot_for(version) < plan.capacity for version in versions
            )

