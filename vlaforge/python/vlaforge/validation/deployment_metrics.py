"""Model-independent action fidelity and raw-sample latency reporting.

Reports are evidence, not a claim of bitwise or task-behavior equivalence.
The caller owns the observation/noise pairing and the measurement boundary.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from vlaforge.validation.contracts import NumericContract


@dataclass(frozen=True, slots=True)
class ActionMetrics:
    count: int
    cosine_similarity: float | None
    cosine_status: str
    mean_squared_error: float
    root_mean_squared_error: float
    maximum_absolute_error: float
    mean_absolute_error: float
    reference_norm: float
    candidate_norm: float
    exact_values: bool
    mismatch_count: int
    within_tolerance: bool


def _array(value: Any) -> tuple[tuple[int, ...], Any, list[int | float]]:
    # Optional tensor/array libraries are not dependencies of the IR package.
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("action tensors must not have empty dimensions")
        children = [_array(item) for item in value]
        if any(child[0] != children[0][0] for child in children):
            raise ValueError("action tensors must be rectangular")
        return (
            (len(children), *children[0][0]),
            [child[1] for child in children],
            [element for child in children for element in child[2]],
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("action values must be real numbers, not booleans")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("action values must be finite float64-representable numbers")
    return (), value, [value]


def _metrics(
    reference: list[int | float],
    candidate: list[int | float],
    contract: NumericContract,
    near_zero_norm: float,
    *,
    discrete: bool = False,
) -> ActionMetrics:
    try:
        differences = [float(left - right) for left, right in zip(reference, candidate)]
    except OverflowError as error:
        raise ValueError(
            "action metric overflow; rescale the values before comparison"
        ) from error
    reference_norm = math.hypot(*reference)
    candidate_norm = math.hypot(*candidate)
    rmse = math.hypot(*differences) / math.sqrt(len(reference))
    mse = rmse * rmse
    maximum = max(abs(value) for value in differences)
    mean = math.fsum(abs(value) / len(reference) for value in differences)
    if not all(
        math.isfinite(value)
        for value in (reference_norm, candidate_norm, rmse, mse, maximum, mean)
    ):
        raise ValueError("action metric overflow; rescale the values before comparison")
    if reference_norm <= near_zero_norm or candidate_norm <= near_zero_norm:
        cosine = None
        if reference_norm <= near_zero_norm and candidate_norm <= near_zero_norm:
            status = "both_near_zero"
        elif reference_norm <= near_zero_norm:
            status = "reference_near_zero"
        else:
            status = "candidate_near_zero"
    else:
        cosine = max(
            -1.0,
            min(
                1.0,
                math.fsum(
                    (left / reference_norm) * (right / candidate_norm)
                    for left, right in zip(reference, candidate)
                ),
            ),
        )
        status = "defined"
    exact = all(left == right for left, right in zip(reference, candidate))
    if discrete or (
        contract.absolute_tolerance == 0 and contract.relative_tolerance == 0
    ):
        mismatches = sum(left != right for left, right in zip(reference, candidate))
    else:
        mismatches = sum(
            not math.isclose(
                left,
                right,
                abs_tol=contract.absolute_tolerance,
                rel_tol=contract.relative_tolerance,
            )
            for left, right in zip(reference, candidate)
        )
    return ActionMetrics(
        count=len(reference),
        cosine_similarity=cosine,
        cosine_status=status,
        mean_squared_error=mse,
        root_mean_squared_error=rmse,
        maximum_absolute_error=maximum,
        mean_absolute_error=mean,
        reference_norm=reference_norm,
        candidate_norm=candidate_norm,
        exact_values=exact,
        mismatch_count=mismatches,
        within_tolerance=mismatches == 0,
    )


def compare_action_chunk(
    reference: Any,
    candidate: Any,
    *,
    sample_id: str,
    space: str,
    contract: NumericContract = NumericContract(),
    near_zero_norm: float = 1e-12,
    discrete_dimensions: Iterable[int] = (),
) -> dict[str, Any]:
    """Compare every element; the last axis is the action dimension.

    Accept nested sequences, NumPy arrays or PyTorch tensors. Each call is one
    logical sample, including any leading batch axis; no flatten-and-truncate
    or broadcasting is allowed. ``space`` names e.g. normalized or physical
    units; no normalization or gripper thresholding happens here.
    """
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(space, str) or not space:
        raise ValueError("space must name the coordinate space or physical units")
    for name, value in (
        ("near_zero_norm", near_zero_norm),
        ("absolute_tolerance", contract.absolute_tolerance),
        ("relative_tolerance", contract.relative_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    shape, reference_raw, left = _array(reference)
    actual_shape, candidate_raw, right = _array(candidate)
    if shape != actual_shape:
        raise ValueError(f"action shape mismatch: {shape} != {actual_shape}")
    if len(shape) < 2:
        raise ValueError("action chunks require at least horizon and action axes")
    discrete = tuple(discrete_dimensions)
    if any(type(index) is not int or not 0 <= index < shape[-1] for index in discrete):
        raise ValueError("discrete dimensions must be valid last-axis indices")
    if len(set(discrete)) != len(discrete):
        raise ValueError("discrete dimensions must be unique")
    dimensions = [
        {
            "dimension": index,
            "discrete": index in discrete,
            **asdict(
                _metrics(
                    left[index :: shape[-1]],
                    right[index :: shape[-1]],
                    contract,
                    near_zero_norm,
                    discrete=index in discrete,
                )
            ),
        }
        for index in range(shape[-1])
    ]
    overall = asdict(_metrics(left, right, contract, near_zero_norm))
    overall["mismatch_count"] = sum(item["mismatch_count"] for item in dimensions)
    overall["within_tolerance"] = overall["mismatch_count"] == 0
    worst_offset = max(
        range(len(left)), key=lambda index: abs(left[index] - right[index])
    )
    offset = worst_offset
    worst_index = []
    for size in reversed(shape):
        worst_index.append(offset % size)
        offset //= size
    return {
        "sample_id": sample_id,
        "space": space,
        "shape": list(shape),
        "reference_dtype": str(reference.dtype)
        if hasattr(reference, "dtype")
        else None,
        "candidate_dtype": str(candidate.dtype)
        if hasattr(candidate, "dtype")
        else None,
        "contract": asdict(contract),
        "near_zero_norm": near_zero_norm,
        "discrete_dimensions": list(discrete),
        "metrics": overall,
        "per_dimension": dimensions,
        "worst_element": {
            "index": list(reversed(worst_index)),
            "reference": left[worst_offset],
            "candidate": right[worst_offset],
        },
        "reference": reference_raw,
        "candidate": candidate_raw,
    }


def action_fidelity_report(
    samples: Iterable[Mapping[str, Any]],
    *,
    space: str,
    contract: NumericContract = NumericContract(),
    near_zero_norm: float = 1e-12,
    discrete_dimensions: Iterable[int] = (),
) -> dict[str, Any]:
    """Retain all raw chunks and per-sample metrics, including worst samples.

    Each input contains sample_id, reference and candidate. This report uses
    one locked profile/space/contract; different profiles require separate
    reports. Aggregate worst cases do not average away failing dimensions.
    """
    discrete = tuple(discrete_dimensions)
    results = [
        compare_action_chunk(
            sample["reference"],
            sample["candidate"],
            sample_id=sample["sample_id"],
            space=space,
            contract=contract,
            near_zero_norm=near_zero_norm,
            discrete_dimensions=discrete,
        )
        for sample in samples
    ]
    if not results:
        raise ValueError("at least one action sample is required")
    if len({item["sample_id"] for item in results}) != len(results):
        raise ValueError("action sample IDs must be unique")
    if any(item["shape"] != results[0]["shape"] for item in results):
        raise ValueError("action samples must use one locked shape profile")
    defined = [
        item for item in results if item["metrics"]["cosine_similarity"] is not None
    ]
    worst = {
        metric: max(results, key=lambda item: item["metrics"][metric])["sample_id"]
        for metric in (
            "mean_squared_error",
            "root_mean_squared_error",
            "maximum_absolute_error",
        )
    }
    worst["cosine_similarity"] = (
        min(defined, key=lambda item: item["metrics"]["cosine_similarity"])["sample_id"]
        if defined
        else None
    )
    return {
        "schema": "vlaforge.action_fidelity.v1",
        "space": space,
        "shape": results[0]["shape"],
        "contract": asdict(contract),
        "near_zero_norm": near_zero_norm,
        "discrete_dimensions": list(discrete),
        "sample_count": len(results),
        "failed_sample_ids": [
            item["sample_id"]
            for item in results
            if not item["metrics"]["within_tolerance"]
        ],
        "within_tolerance": all(
            item["metrics"]["within_tolerance"] for item in results
        ),
        "exact_values": all(item["metrics"]["exact_values"] for item in results),
        "cosine_status_counts": dict(
            Counter(item["metrics"]["cosine_status"] for item in results)
        ),
        "worst_sample_ids": worst,
        "samples": results,
    }


def _latency_summary(values: list[int], deadline_ns: int | None) -> dict[str, Any]:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    summary = {
        "count": len(values),
        "mean_ns": mean,
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "std_ns": statistics.pstdev(values),
        "sequential_calls_per_second": 1e9 / mean,
    }
    for name, quantile in (
        ("p50_ns", 0.50),
        ("p90_ns", 0.90),
        ("p95_ns", 0.95),
        ("p99_ns", 0.99),
    ):
        summary[name] = ordered[math.ceil(len(ordered) * quantile) - 1]
    summary["deadline_ns"] = deadline_ns
    summary["deadline_miss_count"] = (
        sum(value > deadline_ns for value in values)
        if deadline_ns is not None
        else None
    )
    summary["deadline_miss_rate"] = (
        summary["deadline_miss_count"] / len(values)
        if deadline_ns is not None
        else None
    )
    return summary


def latency_report(
    records: Iterable[Mapping[str, Any]],
    *,
    deadline_ns: int | None = None,
) -> dict[str, Any]:
    """Build an exact empirical CDF from measured index/latency_ns records.

    Records retain their original order and extra fields. Optional repeat_id
    distinguishes independently collected runs; missing IDs mean one run.
    IDs alone do not prove independent processes or a fresh-action workload.
    """
    if deadline_ns is not None and (
        type(deadline_ns) is not int or not 0 < deadline_ns <= 2**63 - 1
    ):
        raise ValueError("deadline_ns must be a positive int64")
    raw = []
    repeats: dict[str, list[int]] = {}
    identities = set()
    for record in records:
        item = dict(record)
        if (
            type(item.get("latency_ns")) is not int
            or not 0 < item["latency_ns"] <= 2**63 - 1
        ):
            raise ValueError("latency_ns must be a positive int64")
        if type(item.get("index")) is not int or item["index"] < 0:
            raise ValueError("latency index must be a non-negative integer")
        repeat_id = item.get("repeat_id", "0")
        if (
            isinstance(repeat_id, bool)
            or not isinstance(repeat_id, (str, int))
            or repeat_id == ""
        ):
            raise ValueError("repeat_id must be a non-empty string or integer")
        item["repeat_id"] = str(repeat_id)
        identity = (item["repeat_id"], item["index"])
        if identity in identities:
            raise ValueError(f"duplicate latency sample {identity}")
        identities.add(identity)
        raw.append(item)
        repeats.setdefault(item["repeat_id"], []).append(item["latency_ns"])
    if not raw:
        raise ValueError("at least one latency sample is required")
    values = [item["latency_ns"] for item in raw]
    cumulative = 0
    cdf = []
    for value, count in sorted(Counter(values).items()):
        cumulative += count
        cdf.append(
            {
                "latency_ns": value,
                "count": count,
                "cumulative_count": cumulative,
                "cdf": cumulative / len(values),
                "exceedance_probability": (len(values) - cumulative) / len(values),
            }
        )
    return {
        "schema": "vlaforge.latency_distribution.v1",
        "quantile_method": "nearest_rank",
        "std_method": "population",
        "summary": _latency_summary(values, deadline_ns),
        "per_repeat": [
            {"repeat_id": key, **_latency_summary(value, deadline_ns)}
            for key, value in repeats.items()
        ],
        "worst_sample": max(raw, key=lambda item: item["latency_ns"]),
        "cdf": cdf,
        "raw_samples": raw,
    }
