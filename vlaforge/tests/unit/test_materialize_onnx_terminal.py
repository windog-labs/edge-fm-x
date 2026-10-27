import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
SPEC = importlib.util.spec_from_file_location("materialize_onnx_terminal",
    Path(__file__).resolve().parents[2] / "tools/materialize_onnx_terminal.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def run(spec, values, accepted=True):
    model = TOOL.materialize(spec)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {"input_0": values}
    if spec["kind"] == "affine-finite":
        feeds["input_1"] = np.asarray([accepted], np.bool_)
    return session.run(None, feeds)


@pytest.mark.parametrize("value", [0.0, 1e10, np.finfo(np.float32).max, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", [0, 3])
def test_slice_checks_entire_input_not_only_retained_coordinates(value, position):
    values = np.ones((1, 2, 4), np.float32)
    values[0, 1, position] = value
    result, accepted = run({"kind": "slice-finite", "shape": [1, 2, 4], "width": 2}, values)
    np.testing.assert_array_equal(result, values[..., :2])
    assert accepted.dtype == np.bool_ and accepted.shape == (1,)
    assert accepted.item() == np.isfinite(values).all()


@pytest.mark.parametrize("incoming", [True, False])
@pytest.mark.parametrize("case", ["finite", "nan", "inf", "native-overflow"])
def test_affine_preserves_incoming_and_checks_both_complete_tensors(incoming, case):
    values = np.ones((1, 2, 3), np.float32)
    values[0, 0, 2] = {"finite": 1, "nan": np.nan, "inf": np.inf,
                       "native-overflow": np.finfo(np.float32).max}[case]
    spec = {"kind": "affine-finite", "shape": [1, 2, 3], "scale": [2, 3, 4], "bias": [1, 2, 3]}
    result, accepted = run(spec, values, incoming)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = values * np.asarray(spec["scale"], np.float32) + np.asarray(spec["bias"], np.float32)
    np.testing.assert_array_equal(result, expected)
    assert accepted.item() == (incoming and np.isfinite(values).all() and np.isfinite(expected).all())


@pytest.mark.parametrize("spec", [
    {"kind": "slice-finite", "shape": [1, 0, 4], "width": 2},
    {"kind": "slice-finite", "shape": [True, 4], "width": 2},
    {"kind": "slice-finite", "shape": [1, 4], "width": 5},
    {"kind": "affine-finite", "shape": [1, 3], "scale": [1, 2], "bias": [0, 0]},
    {"kind": "affine-finite", "shape": [1, 1], "scale": [float("inf")], "bias": [0]},
])
def test_incomplete_or_unsupported_contracts_fail(spec):
    with pytest.raises(ValueError):
        TOOL.materialize(spec)
