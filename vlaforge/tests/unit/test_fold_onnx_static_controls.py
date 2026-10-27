import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

PATH = Path(__file__).resolve().parents[2] / "tools/fold_onnx_static_controls.py"
SPEC = importlib.util.spec_from_file_location("static_controls", PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def tensor(name, value, dtype=np.int64):
    return h.make_node("Constant", [], [name], name=name,
                       value=nh.from_array(np.asarray(value, dtype=dtype)))


def model(nodes, inputs=None, outputs=None, initializers=()):
    return h.make_model(h.make_graph(nodes, "controls", inputs or [
        h.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [8])], outputs or [
        h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [3])], list(initializers)),
        opset_imports=[h.make_opsetid("", 17)], ir_version=8)


def slice_model(*, dtype=np.int64, shared=False):
    nodes = [tensor("vector_shape", [-1])]
    for name, value in (("starts", [1]), ("ends", [7]), ("axes", [0]), ("steps", [2])):
        nodes += [tensor(name + "_raw", value, dtype),
                  h.make_node("Cast", [name + "_raw"], [name + "_cast"], to=onnx.TensorProto.INT64),
                  h.make_node("Reshape", [name + "_cast", "vector_shape"], [name], name=name + "_reshape")]
    nodes.append(h.make_node("Slice", ["data", "starts", "ends", "axes", "steps"], ["out"], name="slice"))
    outputs = [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [3])]
    if shared:
        nodes.append(h.make_node("Identity", ["starts"], ["observed"], name="observer"))
        outputs.append(h.make_tensor_value_info("observed", onnx.TensorProto.INT64, [1]))
    return model(nodes, outputs=outputs)


def assert_same(a, b, feeds):
    left, right = ReferenceEvaluator(a).run(None, feeds), ReferenceEvaluator(b).run(None, feeds)
    assert len(left) == len(right)
    for x, y in zip(left, right):
        assert (x.dtype, x.shape, x.tobytes()) == (y.dtype, y.shape, y.tobytes())


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_actual_slice_cast_reshape_controls_and_shared_observer(dtype):
    source = slice_model(dtype=dtype, shared=True)
    before = source.SerializeToString()
    result, report = TOOL.fold(source)
    assert source.SerializeToString() == before
    assert {r["value"] for r in report["rewritten"]} == {"starts", "ends", "axes", "steps"}
    assert not report["unresolved"]
    assert result.graph.input == source.graph.input and result.graph.output == source.graph.output
    assert result.graph.initializer == source.graph.initializer
    onnx.checker.check_model(result)
    assert_same(source, result, {"data": np.arange(8, dtype=np.float32)})
    changed = {r["value"] for r in report["rewritten"]}
    for old, new in zip(source.graph.node, result.graph.node):
        if old.output[0] not in changed:
            assert old.SerializeToString() == new.SerializeToString()
    again, second = TOOL.fold(result)
    assert again.SerializeToString() == result.SerializeToString() and not second["rewritten"]


@pytest.mark.parametrize("op", ["Identity", "Cast", "Concat", "Unsqueeze", "Squeeze"])
def test_allowed_integer_forms(op):
    nodes = [tensor("raw", [0]), tensor("zero_axis", [0])]
    if op == "Identity":
        nodes.append(h.make_node(op, ["raw"], ["axes"], name="final"))
    elif op == "Cast":
        nodes.append(h.make_node(op, ["raw"], ["axes"], name="final", to=onnx.TensorProto.INT32))
    elif op == "Concat":
        nodes.append(h.make_node(op, ["raw"], ["axes"], name="final", axis=0))
    elif op == "Unsqueeze":
        nodes[0] = tensor("raw", 0)
        nodes.append(h.make_node(op, ["raw", "zero_axis"], ["axes"], name="final"))
    else:
        nodes[0] = tensor("raw", [[0]])
        nodes.append(h.make_node(op, ["raw", "zero_axis"], ["axes"], name="final"))
    nodes += [tensor("starts", [0]), tensor("ends", [1]),
              h.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"], name="slice")]
    source = model(nodes, outputs=[h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])])
    result, report = TOOL.fold(source)
    assert "axes" in {r["value"] for r in report["rewritten"]}
    assert not report["unresolved"]
    assert_same(source, result, {"data": np.arange(8, dtype=np.float32)})


def test_integer_dtype_and_extreme_payload_preserved():
    values = np.array([-(2**63), 2**63 - 1], np.int64)
    source = model([tensor("raw", values), h.make_node("Identity", ["raw"], ["shape"], name="final"),
                    h.make_node("Reshape", ["data", "shape"], ["out"], name="consumer")])
    result, report = TOOL.fold(source)
    constant = result.graph.node[1].attribute[0].t
    actual = nh.to_array(constant)
    assert actual.dtype == values.dtype and actual.tobytes() == values.tobytes()
    assert not report["unresolved"]


@pytest.mark.parametrize("mode", ["input", "overridable", "float", "mean", "custom", "narrow-overflow"])
def test_unproved_dataflow_is_retained_and_reported(mode):
    source = slice_model()
    axes_index = next(i for i, n in enumerate(source.graph.node) if list(n.output) == ["axes"])
    if mode in ("input", "overridable"):
        del source.graph.node[axes_index]
        source.graph.input.append(h.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1]))
        if mode == "overridable":
            source.graph.initializer.append(nh.from_array(np.array([0], np.int64), name="axes"))
    elif mode == "float":
        source.graph.node[axes_index].CopyFrom(h.make_node("Cast", ["data"], ["axes"], to=onnx.TensorProto.INT64))
    elif mode == "mean":
        source.graph.node.insert(axes_index, h.make_node("ReduceMean", ["data"], ["mean"], axes=[0], keepdims=1))
        source.graph.node[axes_index + 1].CopyFrom(h.make_node("Cast", ["mean"], ["axes"], to=onnx.TensorProto.INT64))
    elif mode == "custom":
        source.graph.node[axes_index].domain = "vendor"
    else:
        source.graph.node[axes_index - 2].CopyFrom(tensor("axes_raw", [2**40]))
        source.graph.node[axes_index].CopyFrom(h.make_node("Cast", ["axes_raw"], ["axes"], to=onnx.TensorProto.INT32))
    original = copy.deepcopy(source)
    result, report = TOOL.fold(source)
    assert "axes" not in {r["value"] for r in report["rewritten"]}
    assert any(r["value"] == "axes" for r in report["unresolved"])
    old = next(n for n in original.graph.node if "axes" in n.output) if mode not in ("input", "overridable") else None
    if old:
        assert next(n for n in result.graph.node if "axes" in n.output).SerializeToString() == old.SerializeToString()
    assert result.graph.initializer == original.graph.initializer


@pytest.mark.parametrize("shape,allowzero", [([2**50], 0), ([-1, -1], 0), ([-2], 0),
                                           ([0, -1], 1), ([4096, 0, 4096], 1), ([2, 2], 0)])
def test_reshape_preflight_rejects_before_reference_execution(shape, allowzero, monkeypatch):
    import onnx.reference
    real = onnx.reference.ReferenceEvaluator
    calls = []

    def track(node, *args, **kwargs):
        calls.append(node.op_type)
        return real(node, *args, **kwargs)

    monkeypatch.setattr(onnx.reference, "ReferenceEvaluator", track)
    source = model([tensor("raw", [0]), tensor("shape", shape),
                    h.make_node("Reshape", ["raw", "shape"], ["axes"], allowzero=allowzero),
                    tensor("start", [0]), tensor("end", [1]),
                    h.make_node("Slice", ["data", "start", "end", "axes"], ["out"], name="slice")])
    _, report = TOOL.fold(source)
    assert any(r["value"] == "axes" for r in report["unresolved"])
    assert "Reshape" not in calls


@pytest.mark.parametrize("shape,allowzero", [([-1], 0), ([0, -1], 0), ([0, 3], 1)])
def test_valid_reshape_inference_and_zero_semantics(shape, allowzero):
    raw = np.arange(6, dtype=np.int64).reshape(2, 3) if allowzero == 0 else np.empty((0, 3), np.int64)
    source = model([tensor("raw", raw), tensor("shape", shape),
                    h.make_node("Reshape", ["raw", "shape"], ["intermediate"], allowzero=allowzero),
                    tensor("flat_shape", [-1]), h.make_node("Reshape", ["intermediate", "flat_shape"], ["control"]),
                    h.make_node("Reshape", ["data", "control"], ["out"], name="consumer")])
    result, report = TOOL.fold(source)
    assert "control" in {r["value"] for r in report["rewritten"]}
    expected = raw.reshape(-1)
    actual = nh.to_array(next(n for n in result.graph.node if list(n.output) == ["control"]).attribute[0].t)
    assert actual.shape == expected.shape and actual.tobytes() == expected.tobytes()


def test_budgeted_constant_and_concat_stay_unresolved():
    for nodes in ([h.make_node("Constant", [], ["control"], value_ints=[0] * 4097)],
                  [tensor("half", [0] * 3000), h.make_node("Concat", ["half", "half"], ["control"], axis=0)]):
        source = model([*nodes, h.make_node("Reshape", ["data", "control"], ["out"], name="consumer")])
        _, report = TOOL.fold(source)
        assert any(r["value"] == "control" for r in report["unresolved"])
        assert not report["rewritten"]


def test_cycle_missing_axes_and_nested_graph_guards():
    source = model([h.make_node("Identity", ["b"], ["a"]), h.make_node("Identity", ["a"], ["b"]),
                    tensor("start", [0]), tensor("end", [1]),
                    h.make_node("Slice", ["data", "start", "end", "a"], ["out"], name="slice")])
    _, report = TOOL.fold(source)
    assert any("cyclic" in r["reason"] for r in report["unresolved"])
    source.graph.node[-1].input.pop()
    _, report = TOOL.fold(source)
    assert any("axes absent" in r["reason"] for r in report["unresolved"])
    source.functions.append(onnx.FunctionProto())
    with pytest.raises(ValueError, match="flat"):
        TOOL.fold(source)


def test_literal_empty_vector_and_integer_reshape_shape_fold_without_float_math():
    source = model([tensor("empty", []), h.make_node("Identity", ["empty"], ["control"]),
                    h.make_node("Reshape", ["data", "control"], ["out"], name="consumer")])
    result, report = TOOL.fold(source)
    assert report["rewritten"][0]["shape"] == [0]
    assert list(result.graph.node[1].attribute[0].t.dims) == [0]


def test_cli_retains_external_weights_and_never_overwrites(tmp_path):
    source = slice_model()
    weight = nh.from_array(np.array([1.0, -0.0], np.float32), name="learned_unused_weight")
    weight.ClearField("raw_data")
    weight.data_location = onnx.TensorProto.EXTERNAL
    weight.external_data.add(key="location", value="weights.data")
    weight.external_data.add(key="offset", value="0")
    weight.external_data.add(key="length", value="8")
    source.graph.initializer.append(weight)
    weights = tmp_path / "weights.data"
    weights.write_bytes(np.array([1.0, -0.0], np.float32).tobytes())
    original_weight = weights.read_bytes()
    path, output = tmp_path / "source.onnx", tmp_path / "folded.onnx"
    path.write_bytes(source.SerializeToString())
    original = path.read_bytes()
    command = [sys.executable, str(PATH), "--input", str(path), "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert path.read_bytes() == original and weights.read_bytes() == original_weight
    folded = onnx.load(output, load_external_data=False)
    assert folded.graph.initializer == source.graph.initializer
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before
