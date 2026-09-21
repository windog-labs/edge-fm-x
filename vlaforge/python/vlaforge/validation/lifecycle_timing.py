"""Descriptive Session destruction costs, separate from inference latency."""

from collections.abc import Sequence
from statistics import median


def destruction_timing_report(rows: Sequence[dict], *, expected_cycles: int) -> dict:
    if type(expected_cycles) is not int or expected_cycles < 1 or len(rows) != expected_cycles:
        raise ValueError("destruction timing cycle count differs")
    fields = ("api_destroy_ns", "post_destroy_drain_ns", "destroy_through_drain_ns")
    owned = []
    for cycle, row in enumerate(rows):
        if set(row) != {"cycle", *fields} or type(row["cycle"]) is not int or row["cycle"] != cycle:
            raise ValueError("destruction timing cycle identity differs")
        if any(type(row[key]) is not int or row[key] < 0 for key in fields):
            raise ValueError("destruction durations must be nonnegative integer nanoseconds")
        if row["api_destroy_ns"] + row["post_destroy_drain_ns"] != row["destroy_through_drain_ns"]:
            raise ValueError("destruction timing intervals do not conserve total")
        owned.append(dict(row))
    return {
        "schema": "vlaforge.session_destruction_timing/1",
        "clock": "std::chrono::steady_clock",
        "cycles": expected_cycles,
        "raw_nanoseconds": owned,
        "descriptive_nanoseconds": {
            key: {"minimum": min(row[key] for row in owned),
                  "median": median(row[key] for row in owned),
                  "maximum": max(row[key] for row in owned)}
            for key in fields
        },
        "boundary": "Session API destroy followed by explicit CUDA completion drain; excludes snapshots and file I/O",
        "inference_latency_or_cdf": False,
        "confidence_or_generalization_claim": False,
    }
