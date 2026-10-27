import copy

import pytest

from vlaforge.validation.session_benchmark import validate_raw_observation_timing


def rows():
    return [{"run": index, "sample": index % 2, "measured": int(index >= 1),
             "preprocess_ns": 11, "h2d_ns": 12, "bind_ns": 3, "model_ns": 44, "d2h_ns": 5,
             "latency_ns": 75, "finite": 1, "direct_exact": 1} for index in range(3)]


def test_original_observation_timing_validates_every_complete_contiguous_call():
    values = rows()
    validate_raw_observation_timing(values, warmup=1, measured=2, samples=2)
    validate_raw_observation_timing([{key: str(value) for key, value in row.items()} for row in values],
                                    warmup=1, measured=2, samples=2)


@pytest.mark.parametrize("change", ["missing", "boundary", "sample", "measured", "quality", "preprocess", "model", "d2h", "float", "bool", "extra"])
def test_raw_timing_rejects_partial_or_mislabeled_measurements(change):
    values = copy.deepcopy(rows())
    if change == "missing":
        values.pop()
    elif change == "boundary":
        values[1]["latency_ns"] = 74
    elif change == "sample":
        values[1]["sample"] = 0
    elif change == "measured":
        values[0]["measured"] = 1
    elif change == "quality":
        values[1]["direct_exact"] = 0
    elif change in ("preprocess", "model", "d2h"):
        values[1][change + "_ns"] = 0
    elif change == "float":
        values[1]["latency_ns"] = 75.0
    elif change == "bool":
        values[1]["finite"] = True
    else:
        values[1]["unrecognized"] = 1
    with pytest.raises(ValueError):
        validate_raw_observation_timing(values, warmup=1, measured=2, samples=2)
