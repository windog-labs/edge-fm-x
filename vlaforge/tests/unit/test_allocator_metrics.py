from copy import deepcopy

import pytest
from vlaforge.validation.allocator_metrics import (
    COUNTS,
    GROUPS,
    PHASES,
    allocator_report,
    parse_allocator_json,
    parse_allocator_snapshots,
)


def records():
    rows = []
    for index, phase in enumerate(PHASES):
        counters = {
            name: {
                "current": 10,
                "peak": 20,
                "allocated": 100 + 3 * index,
                "freed": 90 + 3 * index,
            }
            for name in GROUPS
        }
        counters.update(dict.fromkeys(COUNTS, 0))
        rows.append(
            {
                "schema": "vlaforge.libtorch_allocator_snapshot/1",
                "phase": phase,
                "torch_release": "2.10.0",
                "allocator": "native",
                "ordinal": 0,
                "device_free_bytes": 100,
                "device_total_bytes": 200,
                "initialized": True,
                "counters": counters,
            }
        )
    return rows


def test_allocator_requests_are_not_device_malloc_or_complete_zero_allocation():
    report = allocator_report(records(), ordinal=0)
    assert report["steady_interval"]["allocator_requests"] == 3
    assert report["steady_interval"]["no_allocator_requests_observed"] is False
    assert report["steady_interval"]["no_device_allocation_calls_observed"] is True
    assert report["zero_allocation_claim_verified"] is False


def test_no_observed_requests_does_not_assert_allocation_outside_coverage():
    rows = records()
    rows[3]["counters"] = deepcopy(rows[2]["counters"])
    report = allocator_report(rows, ordinal=0)
    assert report["steady_interval"]["no_allocator_requests_observed"] is True
    assert report["zero_allocation_claim_verified"] is False
    assert report["excludes"]


def test_uninitialized_before_session_remains_unknown_not_zero():
    rows = records()
    rows[0].update(initialized=False, counters=None)
    assert allocator_report(rows, ordinal=0)["snapshots"][0]["counters"] is None
    rows[1].update(initialized=False, counters=None)
    with pytest.raises(ValueError, match="uninitialized"):
        allocator_report(rows, ordinal=0)


@pytest.mark.parametrize("group", GROUPS)
def test_zero_requests_with_unchanged_live_count_and_new_frees_is_impossible(group):
    rows = records()
    rows[3]["counters"][group]["allocated"] = rows[2]["counters"][group]["allocated"]
    with pytest.raises(ValueError, match="conservation"):
        allocator_report(rows, ordinal=0)


@pytest.mark.parametrize(
    "text", ['{"phase":1,"phase":2}', '{"counters":{"n":1,"n":2}}']
)
def test_duplicate_json_keys_cannot_silently_replace_evidence(text):
    with pytest.raises(ValueError, match="duplicate"):
        parse_allocator_json(text)
    with pytest.raises(ValueError, match="duplicate"):
        parse_allocator_snapshots(text + "\n")


@pytest.mark.parametrize("ordinal", [False, -1, 0.0])
def test_invalid_expected_device_cannot_match_ordinal_zero(ordinal):
    with pytest.raises(ValueError, match="nonnegative integers"):
        allocator_report(records(), ordinal=ordinal)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reordered",
        "wrong_allocator",
        "wrong_release",
        "wrong_device",
        "boolean_counter",
        "reset",
        "peak",
        "memory",
        "unknown_counter",
    ],
)
def test_malformed_or_reset_observations_cannot_be_reported(mutation):
    rows = records()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "wrong_allocator":
        rows[1]["allocator"] = "cudaMallocAsync"
    elif mutation == "wrong_release":
        rows[1]["torch_release"] = "2.7.1"
    elif mutation == "wrong_device":
        rows[1]["ordinal"] = 1
    elif mutation == "boolean_counter":
        rows[1]["counters"]["num_ooms"] = False
    elif mutation == "reset":
        rows[3]["counters"]["allocation"]["allocated"] = 0
    elif mutation == "peak":
        rows[1]["counters"]["allocation"]["peak"] = 9
    elif mutation == "memory":
        rows[1]["device_free_bytes"] = 201
    else:
        rows[1]["counters"]["invented"] = 0
    with pytest.raises(ValueError):
        allocator_report(rows, ordinal=0)
