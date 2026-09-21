"""Unit coverage for generic region calibration input mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/quantize_region_hmct.py"
_SPEC = importlib.util.spec_from_file_location("quantize_region_hmct", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _spec(
    name: str, element_type: int, shape: list[int]
) -> dict[str, object]:
    return {"name": name, "element_type": element_type, "shape": shape}


def test_map_calibration_inputs_renames_by_manifest_order() -> None:
    data = {
        "image": [np.zeros((1, 3, 4, 4), dtype=np.float32)],
        "mask": [np.ones((1,), dtype=np.bool_)],
    }

    mapped, mapping = tool.map_calibration_inputs(
        data,
        ["image", "mask"],
        [_spec("input_0", 1, [1, 3, 4, 4]), _spec("input_1", 9, [1])],
    )

    assert list(mapped) == ["input_0", "input_1"]
    assert mapped["input_0"][0] is data["image"][0]
    assert mapping == {"image": "input_0", "mask": "input_1"}


def test_map_calibration_inputs_accepts_matching_names() -> None:
    data = {"input_0": [np.zeros((1,), dtype=np.float32)]}

    mapped, mapping = tool.map_calibration_inputs(
        data,
        ["input_0"],
        [_spec("input_0", 1, [1])],
    )

    assert mapped == data
    assert mapping == {"input_0": "input_0"}


def test_map_calibration_inputs_rejects_shape_mismatch() -> None:
    data = {"image": [np.zeros((1, 3, 8, 8), dtype=np.float32)]}

    with pytest.raises(ValueError, match="shape"):
        tool.map_calibration_inputs(
            data,
            ["image"],
            [_spec("input_0", 1, [1, 3, 4, 4])],
        )


def test_map_calibration_inputs_rejects_partial_name_overlap() -> None:
    data = {
        "image": [np.zeros((1,), dtype=np.float32)],
        "mask": [np.ones((1,), dtype=np.bool_)],
    }

    with pytest.raises(ValueError, match="partially overlap"):
        tool.map_calibration_inputs(
            data,
            ["image", "mask"],
            [_spec("image", 1, [1]), _spec("input_1", 9, [1])],
        )
