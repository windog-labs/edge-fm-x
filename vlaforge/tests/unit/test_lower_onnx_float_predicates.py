import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper

SPEC = importlib.util.spec_from_file_location("float_predicates", Path(__file__).resolve().parents[2] / "tools/lower_onnx_float_predicates.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def graph(kind=TensorProto.FLOAT):
    nodes = [helper.make_node("IsNaN", ["x"], ["nan"], name="nan")]
    for name, positive, negative in (("inf", 1, 1), ("positive", 1, 0), ("negative", 0, 1), ("none", 0, 0)):
        nodes.append(helper.make_node("IsInf", ["x"], [name], name=name,
                                      detect_positive=positive, detect_negative=negative))
    return helper.make_model(helper.make_graph(nodes, "nonfinite", [helper.make_tensor_value_info("x", kind, [327680])],
        [helper.make_tensor_value_info(node.output[0], TensorProto.BOOL, [327680]) for node in nodes]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def patterns():
    high = np.arange(65536, dtype=np.uint32)
    lows = np.asarray([0, 0x7fff, 0x8000, 0x8001, 0xffff], dtype=np.uint32)
    return ((high[:, None] << 16) | lows[None, :]).reshape(-1).view(np.float32)


def test_all_bf16_classes_and_float_tie_patterns_preserve_nan_and_inf_without_reflexive_equal():
    source = graph()
    before = source.SerializeToString()
    lowered, report = TOOL.lower(source)
    assert source.SerializeToString() == before and len(report["rewritten"]) == 5
    assert all(node.op_type not in ("IsNaN", "IsInf") for node in lowered.graph.node)
    assert all(len(node.input) != 2 or node.input[0] != node.input[1]
               for node in lowered.graph.node if node.op_type == "Equal")
    expected = ort.InferenceSession(before, providers=["CPUExecutionProvider"]).run(None, {"x": patterns()})
    actual = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": patterns()})
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("kind", [TensorProto.FLOAT16, TensorProto.DOUBLE, TensorProto.INT32])
def test_unproven_input_dtypes_are_not_silently_cast(kind):
    with pytest.raises(ValueError, match="explicit FLOAT"):
        TOOL.lower(graph(kind))


def test_invalid_inf_flags_are_not_ignored():
    model = graph()
    model.graph.node[1].attribute[0].i = 2
    with pytest.raises(ValueError, match="flags"):
        TOOL.lower(model)


def test_supported_native_operators_can_be_left_unchanged():
    source = graph()
    lowered, report = TOOL.lower(source, operators=("IsInf",))
    assert [node.op_type for node in lowered.graph.node].count("IsNaN") == 1
    assert len(report["rewritten"]) == 4 and report["selected_operators"] == ["IsInf"]
    with pytest.raises(ValueError, match="explicit"):
        TOOL.lower(source, operators=("Other",))


@pytest.mark.parametrize("observable", [False, True])
def test_only_classification_can_bypass_exact_float_to_double_widening(observable):
    source = graph()
    source.graph.node.insert(0, helper.make_node("Cast", ["x"], ["wide"], name="widen", to=TensorProto.DOUBLE))
    source.graph.node[2].input[0] = "wide"
    if observable:
        source.graph.output.append(helper.make_tensor_value_info("wide", TensorProto.DOUBLE, [327680]))
    before = source.SerializeToString()
    lowered, report = TOOL.lower(source, operators=("IsInf",))
    assert source.SerializeToString() == before
    assert report["classification_widening_casts"][0]["from"] == "FLOAT"
    assert any(node.name == "widen" for node in lowered.graph.node) == observable
    expected = ort.InferenceSession(before, providers=["CPUExecutionProvider"]).run(None, {"x": patterns()})
    actual = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": patterns()})
    for left, right in zip(expected[:5], actual[:5], strict=True):
        np.testing.assert_array_equal(left, right)
