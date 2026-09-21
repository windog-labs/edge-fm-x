"""Validate observed LibTorch native-allocator counters without guessing coverage."""

import json

PHASES = (
    "before_session",
    "after_load",
    "after_warmup",
    "after_measured",
    "after_destroy",
)
GROUPS = (
    "allocation",
    "segment",
    "active",
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "requested_bytes",
)
COUNTS = (
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
)


def _nonnegative(value):
    if type(value) is not int or value < 0:
        raise ValueError("allocator counters require nonnegative integers")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate allocator JSON key")
        result[key] = value
    return result


def parse_allocator_json(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def parse_allocator_snapshots(text):
    return [parse_allocator_json(line) for line in text.splitlines()]


def allocator_report(records, *, ordinal):
    _nonnegative(ordinal)
    if (
        len(records) != len(PHASES)
        or tuple(item.get("phase") for item in records) != PHASES
    ):
        raise ValueError("allocator snapshots are incomplete, duplicated or reordered")
    previous = None
    for index, record in enumerate(records):
        if (
            record.get("schema") != "vlaforge.libtorch_allocator_snapshot/1"
            or record.get("torch_release") != "2.10.0"
            or record.get("allocator") != "native"
            or type(record.get("ordinal")) is not int
            or record["ordinal"] != ordinal
            or type(record.get("initialized")) is not bool
        ):
            raise ValueError("unsupported allocator snapshot domain")
        for name in ("device_free_bytes", "device_total_bytes"):
            _nonnegative(record.get(name))
        if (
            not 0 < record["device_total_bytes"]
            or record["device_free_bytes"] > record["device_total_bytes"]
        ):
            raise ValueError("invalid device memory observation")
        if not record["initialized"]:
            if index != 0 or record.get("counters") is not None:
                raise ValueError("uninitialized allocator cannot provide counters")
            continue
        counters = record.get("counters")
        if not isinstance(counters, dict) or set(counters) != set(GROUPS + COUNTS):
            raise ValueError("missing or unknown allocator counters")
        for group in GROUPS:
            values = counters[group]
            if not isinstance(values, dict) or set(values) != {
                "current",
                "peak",
                "allocated",
                "freed",
            }:
                raise ValueError("allocator stat group is incomplete")
            for value in values.values():
                _nonnegative(value)
            if values["peak"] < values["current"]:
                raise ValueError(
                    "allocator current value exceeds observed lifetime peak"
                )
            if previous is not None and any(
                values[key] < previous[group][key]
                for key in ("peak", "allocated", "freed")
            ):
                raise ValueError("allocator counters were reset or regressed")
            if previous is not None and (
                values["current"] - previous[group]["current"]
                != values["allocated"]
                - previous[group]["allocated"]
                - values["freed"]
                + previous[group]["freed"]
            ):
                raise ValueError("allocator counter changes violate conservation")
        for name in COUNTS:
            _nonnegative(counters[name])
            if previous is not None and counters[name] < previous[name]:
                raise ValueError("allocator counters were reset or regressed")
        previous = counters
    begin, end = records[2]["counters"], records[3]["counters"]
    requests = end["allocation"]["allocated"] - begin["allocation"]["allocated"]
    device_allocs = end["num_device_alloc"] - begin["num_device_alloc"]
    return {
        "schema": "vlaforge.allocator_observation_report/1",
        "status": "validated_observations",
        "coverage": "process/device LibTorch native caching allocator counters",
        "excludes": [
            "runtime arena and caller cudaMalloc",
            "CPU heap allocations",
            "CUDA/library internal allocations outside this allocator",
        ],
        "zero_allocation_claim_verified": False,
        "device_free_bytes_scope": "whole device instantaneous free memory, not process ownership or peak",
        "peak_scope": "cumulative allocator lifetime; counters are never reset",
        "steady_interval": {
            "begin": "after_warmup",
            "end": "after_measured",
            "allocator_requests": requests,
            "allocator_frees": end["allocation"]["freed"]
            - begin["allocation"]["freed"],
            "device_allocation_calls": device_allocs,
            "device_free_calls": end["num_device_free"] - begin["num_device_free"],
            "allocation_retries": end["num_alloc_retries"] - begin["num_alloc_retries"],
            "ooms": end["num_ooms"] - begin["num_ooms"],
            "current_reserved_bytes_change": end["reserved_bytes"]["current"]
            - begin["reserved_bytes"]["current"],
            "reserved_peak_growth_bytes": end["reserved_bytes"]["peak"]
            - begin["reserved_bytes"]["peak"],
            "no_allocator_requests_observed": requests == 0,
            "no_device_allocation_calls_observed": device_allocs == 0,
        },
        "snapshots": records,
    }
