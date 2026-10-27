import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
SPEC = importlib.util.spec_from_file_location(
    "lower_onnx_opset18_to17",
    Path(__file__).resolve().parents[2] / "tools/lower_onnx_opset18_to17.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def tensor(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def save(tmp_path, nodes, inputs, outputs, initializers=()):
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "conversion", inputs, outputs, list(initializers)),
        opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9,
    )
    source = tmp_path / "source.onnx"
    onnx.save(model, source)
    return source, tmp_path / "converted.onnx"


@pytest.mark.parametrize("op", ["ReduceMean", "ReduceMin", "ReduceMax", "ReduceProd", "ReduceL2"])
@pytest.mark.parametrize("noop", [False, True])
def test_constant_reduction_axes_and_empty_noop_preserve_full_outputs(tmp_path, op, noop):
    axes = onnx.numpy_helper.from_array(np.asarray([] if noop else [-1], dtype=np.int64), "axes")
    source, output = save(tmp_path, [onnx.helper.make_node(
        op, ["x", "axes"], ["y"], keepdims=1, noop_with_empty_axes=int(noop)
    )], [tensor("x", [2, 3])], [tensor("y", [2, 3] if noop else [2, 1])], [axes])
    result = TOOL.lower(source, output)
    feeds = {"x": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert not result["precision_changed"]
    assert onnx.load(output).opset_import[0].version == 17


def test_resize_and_reduction_graph_missing_from_official_converter(tmp_path):
    sizes = onnx.numpy_helper.from_array(np.asarray([1, 1, 4, 4], np.int64), "sizes")
    axes = onnx.numpy_helper.from_array(np.asarray([-1], np.int64), "axes")
    source, output = save(tmp_path, [
        onnx.helper.make_node("Resize", ["x", "", "", "sizes"], ["resized"],
                              mode="linear", coordinate_transformation_mode="half_pixel"),
        onnx.helper.make_node("ReduceMean", ["resized", "axes"], ["y"], keepdims=1),
    ], [tensor("x", [1, 1, 2, 2])], [tensor("y", [1, 1, 4, 1])], [sizes, axes])
    TOOL.lower(source, output)
    feeds = {"x": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


def test_dynamic_axes_are_not_frozen_to_initializer_default(tmp_path):
    axes = onnx.numpy_helper.from_array(np.asarray([1], np.int64), "axes")
    source, output = save(tmp_path, [onnx.helper.make_node("ReduceMean", ["x", "axes"], ["y"])],
                          [tensor("x", [2, 3]), onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])],
                          [tensor("y", [2, 1])], [axes])
    with pytest.raises(ValueError, match="constant int64"):
        TOOL.lower(source, output)
    assert not output.exists()


def test_reshaped_constant_axes_preserve_complete_reduction_output(tmp_path):
    axes = onnx.numpy_helper.from_array(np.asarray([-1], np.int64), "literal_axes")
    shape = onnx.numpy_helper.from_array(np.asarray([-1], np.int64), "axis_shape")
    source, output = save(tmp_path, [
        onnx.helper.make_node("Reshape", ["literal_axes", "axis_shape"], ["axes"], allowzero=0),
        onnx.helper.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=1),
    ], [tensor("x", [2, 3])], [tensor("y", [2, 1])], [axes, shape])
    result = TOOL.lower(source, output)
    assert result["constant_integer_dataflow_resolved"] == 1
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    assert actual.tobytes() == expected.tobytes()
    assert not result["precision_changed"]


def test_reshaped_overridable_axes_are_not_treated_as_constant(tmp_path):
    axes = onnx.numpy_helper.from_array(np.asarray([1], np.int64), "runtime_axes")
    shape = onnx.numpy_helper.from_array(np.asarray([-1], np.int64), "axis_shape")
    source, output = save(tmp_path, [
        onnx.helper.make_node("Reshape", ["runtime_axes", "axis_shape"], ["axes"]),
        onnx.helper.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=1),
    ], [tensor("x", [2, 3]), onnx.helper.make_tensor_value_info("runtime_axes", onnx.TensorProto.INT64, [1])],
        [tensor("y", [2, 1])], [axes, shape])
    with pytest.raises(ValueError, match="constant int64"):
        TOOL.lower(source, output)
    assert not output.exists()


def test_external_axes_are_read_without_writing_shared_weight_file(tmp_path):
    axes = onnx.numpy_helper.from_array(np.asarray([-1], np.int64), "axes")
    source, output = save(tmp_path, [onnx.helper.make_node("ReduceMean", ["x", "axes"], ["y"])],
                          [tensor("x", [2, 3])], [tensor("y", [2, 1])], [axes])
    model = onnx.load(source)
    onnx.save_model(model, source, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="shared.weights", size_threshold=0)
    weights = tmp_path / "shared.weights"
    before = weights.read_bytes(), weights.stat().st_mtime_ns, weights.stat().st_ctime_ns
    TOOL.lower(source, output)
    after = weights.read_bytes(), weights.stat().st_mtime_ns, weights.stat().st_ctime_ns
    assert after == before


@pytest.mark.parametrize("mode", ["constant", "edge", "reflect"])
def test_pad_without_new_axes_input_preserves_outputs(tmp_path, mode):
    pads = onnx.numpy_helper.from_array(np.asarray([0, 1, 0, 1], np.int64), "pads")
    source, output = save(tmp_path, [onnx.helper.make_node("Pad", ["x", "pads"], ["y"], mode=mode)],
                          [tensor("x", [2, 3])], [tensor("y", [2, 5])], [pads])
    TOOL.lower(source, output)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


def test_explicit_uneven_split_preserves_every_output(tmp_path):
    sizes = onnx.numpy_helper.from_array(np.asarray([1, 2], np.int64), "sizes")
    source, output = save(tmp_path, [onnx.helper.make_node("Split", ["x", "sizes"], ["a", "b"], axis=-1)],
                          [tensor("x", [2, 3])], [tensor("a", [2, 1]), tensor("b", [2, 2])], [sizes])
    TOOL.lower(source, output)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_existing_scatter_reductions_preserve_outputs(tmp_path, reduction):
    indices = onnx.numpy_helper.from_array(np.asarray([[0]], np.int64), "indices")
    updates = onnx.numpy_helper.from_array(np.asarray([[3, 2, 1]], np.float32), "updates")
    source, output = save(tmp_path, [onnx.helper.make_node("ScatterND", ["x", "indices", "updates"], ["y"], reduction=reduction)],
                          [tensor("x", [2, 3])], [tensor("y", [2, 3])], [indices, updates])
    TOOL.lower(source, output)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_existing_scatter_elements_reductions_preserve_full_state(tmp_path, reduction):
    indices = onnx.numpy_helper.from_array(np.asarray([[0, 0, 0]], np.int64), "indices")
    updates = onnx.numpy_helper.from_array(np.asarray([[3, 2, 1]], np.float32), "updates")
    source, output = save(tmp_path, [onnx.helper.make_node(
        "ScatterElements", ["x", "indices", "updates"], ["y"], axis=0, reduction=reduction,
    )], [tensor("x", [2, 3])], [tensor("y", [2, 3])], [indices, updates])
    before = onnx.load(source).graph.initializer
    result = TOOL.lower(source, output)
    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)
    assert actual[0].tobytes() == expected[0].tobytes()
    assert result["rewrites"]["unchanged_scatter_elements"] == 1
    assert not result["precision_changed"]
    assert [v.SerializeToString() for v in before] == [
        v.SerializeToString() for v in onnx.load(output).graph.initializer]


@pytest.mark.parametrize("reduction", ["min", "max"])
def test_opset18_scatter_elements_reductions_fail_closed(tmp_path, reduction):
    source, output = save(tmp_path, [onnx.helper.make_node(
        "ScatterElements", ["x", "indices", "updates"], ["y"], reduction=reduction,
    )], [tensor("x", [2, 3])], [tensor("y", [2, 3])])
    with pytest.raises(ValueError):
        TOOL.lower(source, output)
    assert not output.exists()


@pytest.mark.parametrize("op,attrs", [("Pad", {"mode": "wrap"}), ("ScatterND", {"reduction": "min"})])
def test_new_pad_or_scatter_semantics_rejected(tmp_path, op, attrs):
    source, output = save(tmp_path, [onnx.helper.make_node(op, ["x"], ["y"], **attrs)],
                          [tensor("x", [2, 3])], [tensor("y", [2, 3])])
    with pytest.raises(ValueError):
        TOOL.lower(source, output)
    assert not output.exists()


def test_unknown_changed_operator_fails_closed(tmp_path):
    source, output = save(tmp_path, [onnx.helper.make_node("Mish", ["x"], ["y"])],
                          [tensor("x", [2, 3])], [tensor("y", [2, 3])])
    with pytest.raises(ValueError, match="unsupported opset-18 change"):
        TOOL.lower(source, output)
    assert not output.exists()


@pytest.mark.parametrize("attrs", [{"antialias": 1}, {"axes": [2, 3]}, {"keep_aspect_ratio_policy": "not_larger"}])
def test_non_default_resize_semantics_rejected(tmp_path, attrs):
    source, output = save(tmp_path, [onnx.helper.make_node("Resize", ["x"], ["y"], **attrs)],
                          [tensor("x", [1, 1, 2, 2])], [tensor("y", [1, 1, 4, 4])])
    with pytest.raises(ValueError, match="non-default"):
        TOOL.lower(source, output)
    assert not output.exists()


def test_uneven_split_num_outputs_not_changed_to_equal_split(tmp_path):
    source, output = save(tmp_path, [onnx.helper.make_node("Split", ["x"], ["a", "b"], axis=0, num_outputs=2)],
                          [tensor("x", [5])], [tensor("a", [3]), tensor("b", [2])])
    with pytest.raises(ValueError, match="explicit split sizes"):
        TOOL.lower(source, output)


def test_external_weights_are_not_rewritten_or_relocated(tmp_path, monkeypatch):
    weights = onnx.numpy_helper.from_array(np.ones((8, 8), dtype=np.float32), "weights")
    bias = onnx.numpy_helper.from_array(np.ones((1, 8), dtype=np.float32), "bias")
    source, output = save(tmp_path, [onnx.helper.make_node("MatMul", ["x", "weights"], ["product"]),
                                    onnx.helper.make_node("Add", ["product", "bias"], ["y"])],
                          [tensor("x", [1, 8])], [tensor("y", [1, 8])], [weights, bias])
    model = onnx.load(source)
    onnx.save_model(model, source, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    before = (tmp_path / "weights.bin").read_bytes()
    original_digest = TOOL.digest
    weight_reads = []

    def count_digest(path):
        if path.name == "weights.bin":
            weight_reads.append(path)
        return original_digest(path)

    monkeypatch.setattr(TOOL, "digest", count_digest)
    result = TOOL.lower(source, output)
    assert len(weight_reads) == 2, "hash each shared weight file once before and once after"
    assert (tmp_path / "weights.bin").read_bytes() == before
    assert "weights.bin" in result["external_weights_sha256"]
    with pytest.raises(ValueError, match="source directory"):
        TOOL.lower(source, tmp_path / "other/converted.onnx")
