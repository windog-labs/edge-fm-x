import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx_graphsurgeon")
from onnx import TensorProto, helper, numpy_helper


@pytest.fixture
def tool(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("integer_shapes", directory / "fold_onnx_integer_shapes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make(nodes, inputs, outputs, initializers=(), infos=()):
    return helper.make_model(helper.make_graph(nodes, "integer_shapes", inputs, outputs,
                                               list(initializers), value_info=list(infos)),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def compare(before, after, feeds):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    old = ort.InferenceSession(before.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(after.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for x, y in zip(old.run(None, feeds), new.run(None, feeds), strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("start,end", [(0, 2), (-2, 10), (1, -1), (-9, -1)])
def test_shape_slices_follow_opset_semantics_without_computing_float_data(tool, start, end):
    source = make([helper.make_node("Shape", ["x"], ["y"], start=start, end=end)],
                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [len((2, 3, 4)[start:end])])])
    original = source.SerializeToString()
    result, report = tool.fold(source)
    assert source.SerializeToString() == original
    assert not result.graph.node and report["passes"][0]["materialized_static_shapes"] == 1
    compare(source, result, {"x": np.zeros((2, 3, 4), np.float32)})


def test_integer_boolean_control_chain_folds_but_float_arithmetic_does_not(tool):
    nodes = [helper.make_node("Shape", ["x"], ["s"]),
             helper.make_node("Add", ["s", "delta"], ["t"]),
             helper.make_node("Equal", ["t", "expected"], ["condition"]),
             helper.make_node("Where", ["condition", "t", "s"], ["dimensions"]),
             helper.make_node("Mul", ["a", "b"], ["float_constant"]),
             helper.make_node("Add", ["x", "float_constant"], ["y"])]
    source = make(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                  [helper.make_tensor_value_info("dimensions", TensorProto.INT64, [2]),
                   helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
                  [numpy_helper.from_array(np.asarray([1, 1], np.int64), "delta"),
                   numpy_helper.from_array(np.asarray([3, 4], np.int64), "expected"),
                   numpy_helper.from_array(np.asarray(0.125, np.float32), "a"),
                   numpy_helper.from_array(np.asarray(0.5, np.float32), "b")])
    result, report = tool.fold(source)
    assert [n.op_type for n in result.graph.node] == ["Mul", "Add"]
    assert not report["floating_arithmetic_evaluated"]
    compare(source, result, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)})


def test_dynamic_shape_and_stale_internal_annotations_are_not_specialized(tool):
    source = make([helper.make_node("Identity", ["x"], ["internal"]),
                   helper.make_node("Shape", ["internal"], ["y"])],
                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
                  infos=[helper.make_tensor_value_info("internal", TensorProto.FLOAT, [7, 3])])
    result, _ = tool.fold(source)
    assert any(n.op_type == "Shape" for n in result.graph.node)
    for batch in (1, 4):
        compare(source, result, {"x": np.zeros((batch, 3), np.float32)})


def test_initializer_also_exposed_as_input_is_not_constant(tool):
    source = make([helper.make_node("Add", ["x", "x"], ["y"])],
                  [helper.make_tensor_value_info("x", TensorProto.INT64, [1])],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [1])],
                  [numpy_helper.from_array(np.asarray([4], np.int64), "x")])
    with pytest.raises(ValueError, match="overridable"):
        tool.fold(source)


def test_narrowing_cast_is_retained(tool):
    source = make([helper.make_node("Cast", ["x"], ["y"], to=TensorProto.INT32)], [],
                  [helper.make_tensor_value_info("y", TensorProto.INT32, [1])],
                  [numpy_helper.from_array(np.asarray([2**40], np.int64), "x")])
    result, _ = tool.fold(source)
    assert result.graph.node[0].op_type == "Cast"
    compare(source, result, {})


def scatter_shape_model(indices=(2,), *, dynamic=False, reduction="none"):
    inputs = [helper.make_tensor_value_info("indices", TensorProto.INT64, [len(indices)])] if dynamic else []
    literals = [numpy_helper.from_array(np.asarray([1, 2, 342, 256], np.int64), "shape"),
                numpy_helper.from_array(np.ones(len(indices), np.int64), "updates")]
    if not dynamic:
        literals.append(numpy_helper.from_array(np.asarray(indices, np.int64), "indices"))
    return make([helper.make_node("ScatterElements", ["shape", "indices", "updates"], ["result"],
                                   axis=0, reduction=reduction)], inputs,
                [helper.make_tensor_value_info("result", TensorProto.INT64, [4])], literals)


@pytest.mark.parametrize("indices", [(2,), (-2,), (0, 2)])
def test_constant_unique_integer_scatter_shape_is_folded(tool, indices):
    source = scatter_shape_model(indices)
    original = source.SerializeToString()
    result, report = tool.fold(source)
    assert source.SerializeToString() == original
    assert not result.graph.node
    assert report["floating_arithmetic_evaluated"] is False
    compare(source, result, {})


@pytest.mark.parametrize("indices,reduction", [((2, 2), "none"), ((2, -2), "none"), ((2,), "add")])
def test_ambiguous_or_reducing_integer_scatter_is_not_folded(tool, indices, reduction):
    source = scatter_shape_model(indices, reduction=reduction)
    result, _ = tool.fold(source)
    assert any(node.op_type == "ScatterElements" for node in result.graph.node)


def test_runtime_integer_scatter_indices_are_not_specialized(tool):
    source = scatter_shape_model(dynamic=True)
    result, _ = tool.fold(source)
    assert any(node.op_type == "ScatterElements" for node in result.graph.node)
    for indices in (np.asarray([0], np.int64), np.asarray([2], np.int64)):
        compare(source, result, {"indices": indices})


def test_floating_scatter_payload_is_not_evaluated(tool):
    source = make([helper.make_node("ScatterElements", ["x", "indices", "updates"], ["y"], axis=0)], [],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
                  [numpy_helper.from_array(np.asarray([1., 2., 3.], np.float32), "x"),
                   numpy_helper.from_array(np.asarray([1], np.int64), "indices"),
                   numpy_helper.from_array(np.asarray([4.], np.float32), "updates")])
    result, _ = tool.fold(source)
    assert any(node.op_type == "ScatterElements" for node in result.graph.node)
    compare(source, result, {})


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("duplicate", [False, True])
def test_scatter_uniqueness_is_checked_per_destination_slice(tool, axis, duplicate):
    indices = np.asarray([[1, -2] if duplicate else [1, 0], [1, 2]], np.int32)
    source = make([helper.make_node("ScatterElements", ["x", "indices", "updates"], ["y"], axis=axis)], [],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [2, 3])],
                  [numpy_helper.from_array(np.arange(6, dtype=np.int64).reshape(2, 3), "x"),
                   numpy_helper.from_array(indices, "indices"),
                   numpy_helper.from_array(np.ones((2, 2), np.int64), "updates")])
    result, _ = tool.fold(source)
    assert bool(result.graph.node) is duplicate
    if not duplicate:
        compare(source, result, {})


def test_external_float_weights_stay_external_and_unread(tool):
    tensor = helper.make_tensor("weight", TensorProto.FLOAT, [2, 3], [], raw=False)
    tensor.ClearField("float_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in [("location", "not-present.bin"), ("offset", "0"), ("length", "24")]:
        tensor.external_data.add(key=key, value=value)
    source = make([helper.make_node("Add", ["x", "weight"], ["y"])],
                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])], [tensor])
    result, report = tool.fold(source)
    assert result.graph.initializer[0].SerializeToString() == tensor.SerializeToString()
    assert report["floating_literals_unchanged"]


def test_bfloat16_input_is_excluded_without_evaluating_float_data(tool):
    identity = helper.make_node("Identity", ["x"], ["y"])
    source = make(
        [identity, helper.make_node("Shape", ["x"], ["shape"]),
         helper.make_node("Add", ["shape", "offset"], ["dimensions"])],
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [2, 3]),
         helper.make_tensor_value_info("dimensions", TensorProto.INT64, [2])],
        [numpy_helper.from_array(np.asarray([1, 1], np.int64), "offset")])
    before = source.SerializeToString()
    result, report = tool.fold(source)
    assert source.SerializeToString() == before
    assert len(result.graph.node) == 1 and result.graph.node[0].SerializeToString() == identity.SerializeToString()
    assert result.graph.input[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    assert report["floating_arithmetic_evaluated"] is False
    dimensions = next(value for value in result.graph.initializer if value.name == "dimensions")
    np.testing.assert_array_equal(numpy_helper.to_array(dimensions), np.asarray([3, 4], np.int64))


def test_unmapped_external_bfloat16_literal_stays_unread_and_unchanged(tool):
    tensor = helper.make_tensor("weight", TensorProto.BFLOAT16, [2, 3], [], raw=False)
    tensor.ClearField("int32_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in [("location", "must-not-read-bf16.bin"), ("offset", "0"), ("length", "12")]:
        tensor.external_data.add(key=key, value=value)
    identity = helper.make_node("Identity", ["weight"], ["y"])
    source = make([identity], [], [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [2, 3])], [tensor])
    before = source.SerializeToString()
    result, report = tool.fold(source)
    assert source.SerializeToString() == before
    assert result.graph.initializer[0].SerializeToString() == tensor.SerializeToString()
    assert result.graph.node[0].SerializeToString() == identity.SerializeToString()
    assert report["floating_literals_unchanged"] and not report["floating_arithmetic_evaluated"]


def test_integer_expansion_exceeding_budget_is_not_evaluated(tool):
    source = make([helper.make_node("Expand", ["x", "shape"], ["y"])], [],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [100])],
                  [numpy_helper.from_array(np.asarray([1], np.int64), "x"),
                   numpy_helper.from_array(np.asarray([100], np.int64), "shape")])
    result, _ = tool.fold(source, max_elements=32)
    assert result.graph.node[0].op_type == "Expand"
    compare(source, result, {})


@pytest.mark.parametrize("length", [16, 8])
def test_external_integer_controls_use_exact_bounded_bytes(tool, tmp_path, length):
    payload = np.asarray([2, 3], np.int64)
    (tmp_path / "controls.bin").write_bytes(payload.tobytes())
    tensor = numpy_helper.from_array(payload, "control")
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in [("location", "controls.bin"), ("offset", "0"), ("length", str(length))]:
        tensor.external_data.add(key=key, value=value)
    source = make([helper.make_node("Add", ["control", "control"], ["y"])], [],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [2])], [tensor])
    before = source.SerializeToString()
    if length == 8:
        with pytest.raises(ValueError, match="byte count"):
            tool.fold(source, external_base_dir=tmp_path)
    else:
        result, report = tool.fold(source, external_base_dir=tmp_path)
        assert not result.graph.node and report["materialized_external_integer_controls"] == 1
        assert np.array_equal(onnx.numpy_helper.to_array(result.graph.initializer[0]), payload * 2)
    assert source.SerializeToString() == before


def test_float_cast_sandwich_is_not_elided(tool):
    source = make([helper.make_node("Cast", ["x"], ["f"], to=TensorProto.FLOAT),
                   helper.make_node("Add", ["f", "half"], ["sum"]),
                   helper.make_node("Cast", ["sum"], ["y"], to=TensorProto.INT64)], [],
                  [helper.make_tensor_value_info("y", TensorProto.INT64, [1])],
                  [numpy_helper.from_array(np.asarray([2**24 + 1], np.int64), "x"),
                   numpy_helper.from_array(np.asarray([0.5], np.float32), "half")])
    result, _ = tool.fold(source)
    assert [n.op_type for n in result.graph.node] == ["Cast", "Add", "Cast"]
    compare(source, result, {})


@pytest.mark.parametrize("trusted", [False, True])
def test_plugin_requires_explicit_reviewed_shape_contract_and_is_not_replaced(tool, trusted):
    source = make([helper.make_node("VLAForgeBFloat16Round", ["x"], ["y"], name="rounding",
                                   domain="vlaforge.trt", plugin_version="1", plugin_namespace=""),
                   helper.make_node("Shape", ["y"], ["s"])],
                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
                   helper.make_tensor_value_info("s", TensorProto.INT64, [2])])
    source.opset_import.append(helper.make_opsetid("vlaforge.trt", 1))
    if not trusted:
        with pytest.raises(ValueError, match="reviewed plugin"):
            tool.fold(source)
    else:
        result, report = tool.fold(source, trusted_plugins=["bf16-round/1"])
        assert tool._node_key(result.graph.node[0]) == tool._node_key(source.graph.node[0])
        assert report["retained_computations_unchanged"]


def test_floating_constant_of_shape_is_not_folded_or_reencoded(tool):
    source = make([helper.make_node("ConstantOfShape", ["s"], ["y"], name="float_zeros",
                                   value=numpy_helper.from_array(np.asarray([0.0], np.float32)))], [],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
                  [numpy_helper.from_array(np.asarray([2, 3], np.int64), "s")])
    result, _ = tool.fold(source)
    assert result.graph.node[0].SerializeToString() == source.graph.node[0].SerializeToString()
    compare(source, result, {})
