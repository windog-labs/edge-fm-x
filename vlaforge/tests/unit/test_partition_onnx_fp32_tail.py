import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

SPEC = importlib.util.spec_from_file_location("partition_onnx_fp32_tail",
    Path(__file__).resolve().parents[2] / "tools/partition_onnx_fp32_tail.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def source():
    inputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
        ("state", TensorProto.FLOAT, [1, 2]), ("features", TensorProto.FLOAT, [1, 2]),
        ("index", TensorProto.INT64, [1]))]
    nodes = [helper.make_node("Add", ["index", "one"], ["next_index"], name="counter"),
             helper.make_node("MatMul", ["features", "weight"], ["velocity"], name="predict"),
             helper.make_node("Mul", ["velocity", "fraction"], ["scaled"], name="scale"),
             helper.make_node("Add", ["state", "scaled"], ["result"], name="update")]
    outputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
        ("result", TensorProto.FLOAT, [1, 2]), ("next_index", TensorProto.INT64, [1]))]
    constants = [numpy_helper.from_array(np.asarray([1], np.int64), "one"),
                 numpy_helper.from_array(np.asarray([[1.5, -3.0], [2.0, 1.0]], np.float32), "weight"),
                 numpy_helper.from_array(np.asarray(-0.125, np.float32), "fraction")]
    return helper.make_model(helper.make_graph(nodes, "functional-update", inputs, outputs, constants),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


def test_real_onnx_composition_preserves_formula_constants_and_counter():
    model = source()
    before = model.SerializeToString()
    core, tail, ledger = TOOL.partition(model)
    assert model.SerializeToString() == before
    assert ledger["original_tail_nodes"] == ["scale", "update"]
    assert len(core.graph.node) == 2 and len(tail.graph.node) == 3
    assert [value.name for value in core.graph.input] == [value.name for value in model.graph.input]
    assert [value.name for value in tail.graph.output] == [value.name for value in model.graph.output]
    assert not set(value.name for value in tail.graph.input) & set(value.name for value in tail.graph.output)
    assert next(value for value in tail.graph.initializer if value.name == "fraction").SerializeToString() == model.graph.initializer[2].SerializeToString()
    original, first, second = session(model), session(core), session(tail)
    for index in range(4):
        inputs = {"state": np.asarray([[index + 0.12345, -index - 4.5]], np.float32),
                  "features": np.asarray([[index + 1.25, index - 2.5]], np.float32),
                  "index": np.asarray([index], np.int64)}
        intermediate = first.run(None, inputs)
        original_inputs = list(inputs.values())
        feeds = {route["name"]: (original_inputs if route["source"] == "input" else intermediate)[route["index"]]
                 for route in ledger["tail_inputs"]}
        actual, expected = second.run(None, feeds), original.run(None, inputs)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)
    assert not ledger["formula_rewritten"] and not ledger["arithmetic_node_duplication"]


@pytest.mark.parametrize("change", ["custom", "random", "no-original-input", "duplicate-arithmetic", "wrong-output"])
def test_unsafe_or_unhelpful_partition_is_rejected(change):
    model = source()
    if change == "custom":
        model.graph.node[1].domain = "unreviewed"
    elif change == "random":
        model.graph.node[1].op_type = "RandomNormalLike"
    elif change == "no-original-input":
        model.graph.node[3].input[0] = "velocity"
    elif change == "duplicate-arithmetic":
        model.graph.output.append(helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 2]))
    with pytest.raises(ValueError):
        TOOL.partition(model, ["missing"] if change == "wrong-output" else ["result"])
