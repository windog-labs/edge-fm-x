from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from vlaforge.validation.contracts import NumericContract
from vlaforge.validation.deployment_metrics import (
    action_fidelity_report,
    compare_action_chunk,
    latency_report,
)


def compare(reference, candidate, **kwargs):
    return compare_action_chunk(
        reference, candidate, sample_id="frame-0", space="normalized", **kwargs
    )


def test_full_chunk_catches_error_beyond_output_probe() -> None:
    result = compare([[1.0, 0.0], [3.0, 4.0]], [[1.0, 0.0], [3.0, 2.0]])
    metrics = result["metrics"]
    assert metrics["count"] == 4
    assert metrics["mean_squared_error"] == 1.0
    assert metrics["root_mean_squared_error"] == 1.0
    assert metrics["maximum_absolute_error"] == 2.0
    assert metrics["mean_absolute_error"] == 0.5
    assert metrics["cosine_similarity"] == pytest.approx(18 / math.sqrt(26 * 14))
    assert metrics["mismatch_count"] == 1
    assert result["per_dimension"][0]["within_tolerance"]
    assert not result["per_dimension"][1]["within_tolerance"]
    assert result["per_dimension"][1]["mean_squared_error"] == pytest.approx(2.0)
    assert result["worst_element"] == {
        "index": [1, 1],
        "reference": 4.0,
        "candidate": 2.0,
    }
    assert result["reference"] == [[1.0, 0.0], [3.0, 4.0]]


def test_cosine_one_does_not_hide_scale_error() -> None:
    result = compare([[1.0, 2.0]], [[2.0, 4.0]])
    assert result["metrics"]["cosine_similarity"] == pytest.approx(1.0)
    assert not result["metrics"]["within_tolerance"]
    assert not result["metrics"]["exact_values"]


@pytest.mark.parametrize(
    "reference,candidate,status,passed",
    [
        ([[0.0, 0.0]], [[0.0, 0.0]], "both_near_zero", True),
        ([[1e-14, 0.0]], [[-1e-14, 0.0]], "both_near_zero", False),
        ([[0.0, 0.0]], [[1.0, 0.0]], "reference_near_zero", False),
        ([[1.0, 0.0]], [[0.0, 0.0]], "candidate_near_zero", False),
    ],
)
def test_near_zero_cosine_is_undefined_not_fabricated(
    reference, candidate, status, passed
) -> None:
    result = compare(reference, candidate)
    assert result["metrics"]["cosine_similarity"] is None
    assert result["metrics"]["cosine_status"] == status
    assert result["metrics"]["within_tolerance"] is passed
    json.dumps(result, allow_nan=False)


def test_discrete_dimensions_require_exact_values_even_with_float_tolerance() -> None:
    result = compare(
        [[0.0, 0.0]],
        [[0.01, 0.01]],
        contract=NumericContract(absolute_tolerance=0.1),
        discrete_dimensions=[1],
    )
    assert result["per_dimension"][0]["within_tolerance"]
    assert not result["per_dimension"][1]["within_tolerance"]
    assert result["metrics"]["mismatch_count"] == 1
    assert not result["metrics"]["within_tolerance"]


def test_large_discrete_integer_differences_are_not_lost_to_float_conversion() -> None:
    result = compare([[2**60]], [[2**60 + 1]], discrete_dimensions=[0])
    assert not result["metrics"]["exact_values"]
    assert result["metrics"]["mismatch_count"] == 1
    assert result["metrics"]["maximum_absolute_error"] == 1.0


@pytest.mark.parametrize(
    "reference,candidate,message",
    [
        ([[1.0, 2.0]], [[1.0], [2.0]], "shape mismatch"),
        ([[1.0, 2.0], [3.0]], [[1.0, 2.0]], "rectangular"),
        ([[]], [[]], "empty"),
        ([1.0], [1.0], "horizon"),
        ([[float("nan")]], [[1.0]], "finite"),
        ([[1.0]], [[float("inf")]], "finite"),
        ([[float("inf")]], [[float("inf")]], "finite"),
        ([[1.0]], [[True]], "real numbers"),
        ([[1e308]], [[-1e308]], "overflow"),
        ([[2**1023]], [[-(2**1023)]], "overflow"),
    ],
)
def test_malformed_or_nonfinite_chunks_fail_closed(
    reference, candidate, message
) -> None:
    with pytest.raises(ValueError, match=message):
        compare(reference, candidate)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"near_zero_norm": float("nan")},
        {"near_zero_norm": -1.0},
        {"contract": NumericContract(absolute_tolerance=float("inf"))},
        {"discrete_dimensions": [1]},
        {"discrete_dimensions": [0, 0]},
        {"discrete_dimensions": [False]},
    ],
)
def test_invalid_contract_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        compare([[1.0]], [[1.0]], **kwargs)


def test_reports_keep_raw_samples_and_worst_cases_in_each_space() -> None:
    samples = [
        {"sample_id": "a", "reference": [[1.0, 1.0]], "candidate": [[1.0, 1.0]]},
        {"sample_id": "b", "reference": [[1.0, 1.0]], "candidate": [[1.0, -1.0]]},
        {"sample_id": "zero", "reference": [[0.0, 0.0]], "candidate": [[0.0, 0.0]]},
    ]
    report = action_fidelity_report(samples, space="physical:m,rad")
    assert report["space"] == "physical:m,rad"
    assert report["failed_sample_ids"] == ["b"]
    assert all(value == "b" for value in report["worst_sample_ids"].values())
    assert report["samples"][1]["candidate"] == [[1.0, -1.0]]
    assert report["cosine_status_counts"] == {"defined": 2, "both_near_zero": 1}


def test_action_report_rejects_mixed_profiles_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        action_fidelity_report([], space="normalized")
    sample = {"sample_id": "a", "reference": [[1.0]], "candidate": [[1.0]]}
    with pytest.raises(ValueError, match="unique"):
        action_fidelity_report([sample, sample], space="normalized")
    with pytest.raises(ValueError, match="locked shape"):
        action_fidelity_report(
            [
                sample,
                {
                    "sample_id": "b",
                    "reference": [[1.0, 1.0]],
                    "candidate": [[1.0, 1.0]],
                },
            ],
            space="normalized",
        )


def test_numpy_and_torch_tensors_keep_dtype_and_all_dimensions() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    reference = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    candidate = torch.tensor(reference, requires_grad=True)
    result = compare(reference, candidate)
    assert result["shape"] == [1, 3, 4]
    assert result["reference_dtype"] == "float32"
    assert result["candidate_dtype"] == "torch.float32"
    assert result["metrics"]["exact_values"]
    assert all(item["count"] == 3 for item in result["per_dimension"])


def test_latency_cdf_preserves_order_ties_repeats_and_tail() -> None:
    raw = [
        {"index": 0, "latency_ns": 30, "repeat_id": "a", "input_hash": "first"},
        {"index": 1, "latency_ns": 10, "repeat_id": "a"},
        {"index": 0, "latency_ns": 20, "repeat_id": "b"},
        {"index": 1, "latency_ns": 20, "repeat_id": "b"},
    ]
    report = latency_report(raw, deadline_ns=20)
    assert report["raw_samples"] == raw
    assert report["worst_sample"] == raw[0]
    assert report["summary"]["mean_ns"] == 20
    assert report["summary"]["std_ns"] == pytest.approx(math.sqrt(50))
    assert report["summary"]["p50_ns"] == 20
    assert report["summary"]["p95_ns"] == 30
    assert report["summary"]["p99_ns"] == 30
    assert report["summary"]["deadline_miss_rate"] == 0.25
    assert report["summary"]["sequential_calls_per_second"] == 50_000_000
    assert report["cdf"] == [
        {
            "latency_ns": 10,
            "count": 1,
            "cumulative_count": 1,
            "cdf": 0.25,
            "exceedance_probability": 0.75,
        },
        {
            "latency_ns": 20,
            "count": 2,
            "cumulative_count": 3,
            "cdf": 0.75,
            "exceedance_probability": 0.25,
        },
        {
            "latency_ns": 30,
            "count": 1,
            "cumulative_count": 4,
            "cdf": 1.0,
            "exceedance_probability": 0.0,
        },
    ]
    assert report["per_repeat"][0]["std_ns"] == 10
    assert report["per_repeat"][1]["std_ns"] == 0


def test_latency_nearest_rank_matches_existing_benchmark_convention() -> None:
    report = latency_report(
        {"index": index, "latency_ns": index + 1} for index in range(100)
    )
    assert report["summary"]["p50_ns"] == 50
    assert report["summary"]["p90_ns"] == 90
    assert report["summary"]["p95_ns"] == 95
    assert report["summary"]["p99_ns"] == 99
    assert report["summary"]["deadline_miss_rate"] is None
    assert len(report["per_repeat"]) == 1
    assert report["raw_samples"][0]["repeat_id"] == "0"


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"index": 0, "latency_ns": 0}],
        [{"index": 0, "latency_ns": -1}],
        [{"index": 0, "latency_ns": float("nan")}],
        [{"index": 0, "latency_ns": 1.5}],
        [{"index": 0, "latency_ns": True}],
        [{"index": 0, "latency_ns": 2**63}],
        [{"index": -1, "latency_ns": 1}],
        [{"index": 0, "latency_ns": 1}, {"index": 0, "latency_ns": 2}],
    ],
)
def test_invalid_latencies_are_not_dropped_or_hidden(records) -> None:
    with pytest.raises(ValueError):
        latency_report(records)


@pytest.mark.parametrize("deadline", [0, -1, True, 0.5, 2**63])
def test_latency_deadline_requires_positive_int64(deadline) -> None:
    with pytest.raises(ValueError, match="positive int64"):
        latency_report([{"index": 0, "latency_ns": 1}], deadline_ns=deadline)


def test_cli_retains_failed_fidelity_report_and_emits_raw_and_cdf_csv(
    tmp_path: Path,
) -> None:
    path = (
        Path(__file__).resolve().parents[2] / "tools" / "report_deployment_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("report_deployment_metrics", path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    actions = tmp_path / "actions.json"
    actions.write_text(
        json.dumps(
            {
                "space": "normalized",
                "contract": {"absolute_tolerance": 0.0},
                "samples": [
                    {
                        "sample_id": "frame-0",
                        "reference": [[1, 2]],
                        "candidate": [[1, 3]],
                    }
                ],
            }
        )
    )
    first = tmp_path / "first.csv"
    first.write_text("index,latency_ns,output_probe\n0,30,1.0\n1,10,2.0\n")
    second = tmp_path / "second.csv"
    second.write_text("index,latency_ns\n0,20\n")
    output = tmp_path / "report"
    status = tool.main(
        [
            "--actions",
            str(actions),
            "--latencies",
            str(first),
            "--latencies",
            str(second),
            "--deadline-ns",
            "20",
            "--output",
            str(output),
        ]
    )
    assert status == 1
    report = json.loads((output / "report.json").read_text())
    assert report["actions"]["failed_sample_ids"] == ["frame-0"]
    assert len(report["sources"]) == 3
    assert all(len(item["sha256"]) == 64 for item in report["sources"])
    assert len(report["latency"]["per_repeat"]) == 2
    with (output / "latency_cdf.csv").open() as stream:
        assert [row["cdf"] for row in csv.DictReader(stream)] == [
            str(1 / 3),
            str(2 / 3),
            "1.0",
        ]
    with (output / "latency_raw.csv").open() as stream:
        assert [row["latency_ns"] for row in csv.DictReader(stream)] == [
            "30",
            "10",
            "20",
        ]
