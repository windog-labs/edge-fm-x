"""Cross-Session allocator observations and complete diagnostic output gates."""

from __future__ import annotations

import copy
import csv
import io
from itertools import pairwise

from vlaforge.validation.allocator_metrics import COUNTS, GROUPS, allocator_report


def _count(value, name, minimum):
    if type(value) is not int or not minimum <= value <= 100000:
        raise ValueError(f"invalid lifecycle {name}")


def _continuous(previous, current):
    if not current["initialized"]:
        raise ValueError("allocator became uninitialized between Sessions")
    left, right = previous["counters"], current["counters"]
    for group in GROUPS:
        if any(right[group][key] < left[group][key] for key in ("peak", "allocated", "freed")):
            raise ValueError("allocator counters reset between Sessions")
        if right[group]["current"] - left[group]["current"] != (
            right[group]["allocated"] - left[group]["allocated"]
            - right[group]["freed"] + left[group]["freed"]
        ):
            raise ValueError("allocator conservation failed between Sessions")
    if any(right[name] < left[name] for name in COUNTS):
        raise ValueError("allocator cumulative counters reset between Sessions")


def allocator_lifecycle_report(records, *, expected_cycles, ordinal):
    """Validate each cycle and every inter-cycle interval without resetting peaks."""
    _count(expected_cycles, "cycle count", 2)
    if len(records) != expected_cycles:
        raise ValueError("lifecycle allocator cycles are incomplete")
    snapshots = copy.deepcopy(records)
    reports = [allocator_report(values, ordinal=ordinal) for values in snapshots]
    total = snapshots[0][0]["device_total_bytes"]
    if any(item["device_total_bytes"] != total for values in snapshots for item in values):
        raise ValueError("lifecycle device total memory changed")
    for previous, current in pairwise(snapshots):
        _continuous(previous[-1], current[0])
    retained = {}
    for group in GROUPS:
        values = [cycle[-1]["counters"][group]["current"] for cycle in snapshots]
        increments = [right - left for left, right in pairwise(values)]
        retained[group] = {
            "current_after_destroy": values,
            "successive_changes": increments,
            "net_change_after_first_destroy": values[-1] - values[0],
            "maximum_growth_above_first_destroy": max(values) - values[0],
            "unchanged_after_first_destroy": len(set(values)) == 1,
            "strictly_growing_each_cycle": all(value > 0 for value in increments),
        }
    return {
        "schema": "vlaforge.session_lifecycle_allocator/1",
        "status": "validated_observations",
        "cycles": expected_cycles,
        "coverage": reports[0]["coverage"],
        "excludes": reports[0]["excludes"],
        "peak_scope": reports[0]["peak_scope"],
        "device_free_bytes_scope": reports[0]["device_free_bytes_scope"],
        "zero_allocation_claim_verified": False,
        "leak_attribution_verified": False,
        "interpretation": "retained allocated/active/requested storage and reserved caching are separate observations, not ownership or leak proof",
        "template_phase_semantics": {
            "after_warmup": "after first validated invocation; not a performance warmup",
            "after_measured": "after remaining validated invocations; not a measurement interval",
        },
        "performance_measurement": False,
        "retained_after_destroy": retained,
        "snapshots_by_cycle": snapshots,
    }


def split_lifecycle_rows(text, *, cycles, samples):
    """Parse repeated CSV headers as explicit cycle boundaries, never skip rows."""
    _count(cycles, "cycle count", 2)
    _count(samples, "sample count", 2)
    records = list(csv.reader(io.StringIO(text)))
    if not records:
        raise ValueError("lifecycle output rows are missing")
    header = records[0]
    if len(header) != len(set(header)) or not {
        "run", "sample", "measured", "revision", "finite", "direct_exact",
    }.issubset(header):
        raise ValueError("lifecycle CSV header is invalid")
    stride = samples + 1
    if len(records) != cycles * stride:
        raise ValueError("lifecycle output cycle/row count differs")
    result = []
    for cycle in range(cycles):
        group = records[cycle * stride:(cycle + 1) * stride]
        if group[0] != header or any(len(row) != len(header) for row in group[1:]):
            raise ValueError("lifecycle CSV cycle header or row width differs")
        rows = [dict(zip(header, row, strict=True)) for row in group[1:]]
        for index, row in enumerate(rows):
            if (int(row["run"]) != index or int(row["sample"]) != index
                    or int(row["revision"]) != index + 1
                    or int(row["measured"]) != int(index >= 1)
                    or row["finite"] != "1" or row["direct_exact"] != "1"):
                raise ValueError("lifecycle sample order/revision/output gate differs")
        result.append(rows)
    return result


def lifecycle_scope(cycles, samples):
    _count(cycles, "cycle count", 2)
    _count(samples, "sample count", 2)
    return {
        "cycles": cycles, "validated_calls_per_cycle": samples,
        "validated_calls": cycles * samples,
        "template_call_partition": [1, samples - 1],
        "warmup_calls": None, "measured_calls": None,
        "performance_measurement": False, "formal_latency_or_cdf": False,
        "caller_input_buffers": "allocated once outside all Session lifetimes",
        "synchronization": "CUDA completion before every allocator snapshot, including after Session destruction",
        "allocator_reset_or_empty_cache_requested": False,
    }
