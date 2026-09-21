import pytest
from vlaforge.validation.lifecycle_timing import destruction_timing_report


def rows():
    return [{"cycle": i, "api_destroy_ns": 10 + i, "post_destroy_drain_ns": 2,
             "destroy_through_drain_ns": 12 + i} for i in range(5)]


def test_raw_intervals_are_owned_and_descriptive_only():
    data = rows()
    report = destruction_timing_report(data, expected_cycles=5)
    assert report["descriptive_nanoseconds"]["api_destroy_ns"] == {"minimum": 10, "median": 12, "maximum": 14}
    data[0]["api_destroy_ns"] = 99
    assert report["raw_nanoseconds"][0]["api_destroy_ns"] == 10
    assert not report["inference_latency_or_cdf"] and not report["confidence_or_generalization_claim"]


@pytest.mark.parametrize("field,value", [("cycle", True), ("cycle", 3), ("api_destroy_ns", -1),
                                        ("api_destroy_ns", 1.5), ("api_destroy_ns", True),
                                        ("destroy_through_drain_ns", 100)])
def test_invalid_intervals_rejected(field, value):
    data = rows()
    data[0][field] = value
    with pytest.raises(ValueError):
        destruction_timing_report(data, expected_cycles=5)


def test_missing_extra_and_incomplete_records_rejected():
    for change in (lambda data: data.pop(), lambda data: data[0].pop("api_destroy_ns"),
                   lambda data: data[0].update(extra=1)):
        data = rows()
        change(data)
        with pytest.raises(ValueError):
            destruction_timing_report(data, expected_cycles=5)
