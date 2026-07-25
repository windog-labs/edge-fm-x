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
    assert "action_queue" in graph.state_readers
    assert "action_queue" in graph.state_writers
    assert graph.value_producers["sample_final"].endswith("vla.for")
    assert any("vla.invoke" in item for item in graph.value_consumers["prefix"])


def test_liveness_covers_loop_and_commit_values():
    ranges = {
        live_range.value: live_range
        for live_range in analyze_liveness(
            build_smolvla_fixture().module, "act"
        )
    }
    assert ranges["image_value"].first_definition == 0
    assert ranges["txn"].last_use > ranges["txn"].first_definition
    assert ranges["selected_action"].last_use > ranges["selected_action"].first_definition


def test_physical_slot_plan_rejects_unsafe_capacity():
    module = build_smolvla_fixture().module
    with pytest.raises(UnsafePhysicalizationError, match="unsafe_reuse.*rng"):
        plan_physical_slots(
            module,
            max_in_flight=2,
            consumer_lag=1,
            fallback_snapshots=1,
            capacities={
                "rng": 3,
                "action_queue": 8,
                "queue_cursor": 8,
            },
        )


@pytest.mark.parametrize(
    ("in_flight", "lag", "fallback", "first_version"),
    (
        (1, 0, 0, 0),
        (2, 1, 1, 17),
        (4, 3, 2, 1_001),
        (8, 6, 4, 10_000),
    ),
)
def test_physical_slot_mapping_is_bounded_and_collision_free_in_live_window(
    in_flight,
    lag,
    fallback,
    first_version,
):
    module = build_smolvla_fixture().module
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
        versions = list(range(first_version, first_version + live_window))
        assert len({plan.slot_for(version) for version in versions}) == len(
            versions
        )
        assert all(
            0 <= plan.slot_for(version) < plan.capacity for version in versions
        )
