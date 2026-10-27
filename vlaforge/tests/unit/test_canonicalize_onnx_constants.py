import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx.reference import ReferenceEvaluator

SPEC = importlib.util.spec_from_file_location(
    "literal_constants", Path(__file__).resolve().parents[2] / "tools/canonicalize_onnx_constants.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def model(nodes, inputs, outputs, initializers=()):
    return onnx.helper.make_model(onnx.helper.make_graph(
        nodes, "literals", inputs, outputs, list(initializers)),
        opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)


@pytest.mark.parametrize("attr,value,dtype,shape", [
    ("value_int", -(2 ** 63), onnx.TensorProto.INT64, []),
    ("value_ints", [0, 2 ** 63 - 1, -1], onnx.TensorProto.INT64, [3]),
    ("value_ints", [], onnx.TensorProto.INT64, [0]),
    ("value_float", -0.0, onnx.TensorProto.FLOAT, []),
    ("value_floats", [0.0, -0.0, 2. ** -149, 1.25], onnx.TensorProto.FLOAT, [4]),
    ("value_floats", [], onnx.TensorProto.FLOAT, [0]),
])
def test_literals_preserve_complete_values_and_contracts(attr, value, dtype, shape):
    node = onnx.helper.make_node("Constant", [], ["out"], name="literal")
    if isinstance(value, list) and not value:
        node.attribute.append(onnx.helper.make_attribute(attr, value,
            attr_type=onnx.AttributeProto.INTS if attr == "value_ints" else onnx.AttributeProto.FLOATS))
    else:
        node.attribute.append(onnx.helper.make_attribute(attr, value))
    source = model([node], [], [onnx.helper.make_tensor_value_info("out", dtype, shape)])
    before = source.SerializeToString()
    result, report = TOOL.canonicalize(source)
    assert source.SerializeToString() == before
    onnx.checker.check_model(result)
    a = ReferenceEvaluator(source).run(None, {})[0]
    b = ReferenceEvaluator(result).run(None, {})[0]
    assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert report["converted"][0]["shape"] == shape
    assert result.graph.output == source.graph.output
    assert not report["target_execution_verified"]


def test_scalar_expand_then_unsqueeze_matches_source_shape_and_value():
    h = onnx.helper
    source = model([
        h.make_node("Constant", [], ["one"], value_ints=[1]),
        h.make_node("Concat", ["one"], ["shape"], axis=0),
        h.make_node("Expand", ["time", "shape"], ["expanded"]),
        h.make_node("Constant", [], ["axes"], value_ints=[1]),
        h.make_node("Unsqueeze", ["expanded", "axes"], ["out"]),
    ], [h.make_tensor_value_info("time", onnx.TensorProto.FLOAT, [])],
        [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 1])])
    candidate, _ = TOOL.canonicalize(source)
    for time in [0.0, -0.0, 1.0, -0.5, 2. ** -149]:
        inputs = {"time": np.array(time, dtype=np.float32)}
        a = ReferenceEvaluator(source).run(None, inputs)[0]
        b = ReferenceEvaluator(candidate).run(None, inputs)[0]
        assert a.shape == b.shape == (1, 1)
        assert a.tobytes() == b.tobytes()


def test_tensor_constants_custom_domains_and_external_initializers_are_untouched():
    h = onnx.helper
    node = h.make_node("Constant", [], ["literal"],
                       value=onnx.numpy_helper.from_array(np.array([-0.], np.float32)))
    custom = h.make_node("Constant", [], ["custom"], domain="example", value_ints=[2])
    weight = h.make_tensor("weight", onnx.TensorProto.FLOAT, [2], [1, 2])
    weight.ClearField("float_data")
    weight.data_location = onnx.TensorProto.EXTERNAL
    weight.external_data.add(key="location", value="unchanged.data")
    source = model([node, custom], [], [], [weight])
    candidate, report = TOOL.canonicalize(source)
    assert candidate.SerializeToString() == source.SerializeToString()
    assert not report["converted"]


@pytest.mark.parametrize("kind", ["duplicate", "wrong-type", "nan"])
def test_reject_ambiguous_or_unproved_literal(kind):
    h = onnx.helper
    node = h.make_node("Constant", [], ["out"], value_float=1.0)
    if kind == "duplicate":
        node.attribute.append(h.make_attribute("value_int", 1))
    elif kind == "wrong-type":
        node.attribute[0].type = onnx.AttributeProto.INT
    else:
        node.attribute[0].f = float("nan")
    source = model([node], [], [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [])])
    with pytest.raises(ValueError):
        TOOL.canonicalize(source)
