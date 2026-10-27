import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import helper, TensorProto


@pytest.fixture
def tool():
    file = Path(__file__).resolve().parents[2] / "tools/lower_onnx_trilu_batch_axes.py"
    spec = importlib.util.spec_from_file_location("trilu_batch", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(shape, upper):
    return helper.make_model(helper.make_graph([
        helper.make_node("Trilu", ["x", "k"], ["y"], name="triangle", upper=upper)
    ], "triangle", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
                     helper.make_tensor_value_info("k", TensorProto.INT64, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (1, 2, 3, 5, 4), (2, 1, 2, 3, 4, 5)])
@pytest.mark.parametrize("upper", [0, 1])
@pytest.mark.parametrize("diagonal", [-2, 0, 2])
def test_flatten_preserves_every_batch_value_exactly(tool, shape, upper, diagonal):
    original = model(shape, upper)
    before = original.SerializeToString()
    lowered, report = tool.lower(original)
    assert original.SerializeToString() == before
    assert len(report["rewritten"]) == 1 and not report["skipped"]
    assert [port.SerializeToString() for port in lowered.graph.input] == [port.SerializeToString() for port in original.graph.input]
    values = np.random.default_rng(661).standard_normal(shape).astype(np.float32)
    feeds = {"x": values, "k": np.asarray(diagonal, dtype=np.int64)}
    expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_already_supported_rank_is_unchanged(tool, shape):
    source = model(shape, 1)
    lowered, report = tool.lower(source)
    assert not report["rewritten"] and lowered.SerializeToString() == source.SerializeToString()


def test_dynamic_batch_uses_runtime_volume_without_guessing_a_batch_size(tool):
    source = model((None, 2, 3, 4), 0)
    lowered, report = tool.lower(source)
    assert len(report["rewritten"]) == 1 and not report["skipped"]
    assert report["rewritten"][0]["flattened_shape"] == [-1, 3, 4]
    assert report["rewritten"][0]["batch_inferred_from_runtime_volume"]
    original = ort.InferenceSession(source.SerializeToString(), providers=["CPUExecutionProvider"])
    rewritten = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        feeds = {"x": np.random.default_rng(batch).normal(size=(batch, 2, 3, 4)).astype(np.float32),
                 "k": np.asarray(-1, dtype=np.int64)}
        np.testing.assert_array_equal(original.run(None, feeds)[0].view(np.uint32), rewritten.run(None, feeds)[0].view(np.uint32))


def test_unknown_matrix_dimension_is_not_guessed(tool):
    source = model((2, 2, None, 4), 0)
    lowered, report = tool.lower(source)
    assert not report["rewritten"] and len(report["skipped"]) == 1
    assert lowered.SerializeToString() == source.SerializeToString()
