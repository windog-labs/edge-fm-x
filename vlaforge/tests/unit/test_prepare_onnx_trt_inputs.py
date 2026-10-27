import copy
import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx.reference import ReferenceEvaluator

SPEC = importlib.util.spec_from_file_location(
    "prepare_trt_inputs", Path(__file__).resolve().parents[2] / "tools/prepare_onnx_trt_inputs.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def cast_model():
    h = onnx.helper
    return h.make_model(h.make_graph([
        h.make_node("Cast", ["state"], ["converted"], to=onnx.TensorProto.FLOAT),
        h.make_node("Add", ["converted", "one"], ["output"])
    ], "cast", [h.make_tensor_value_info("state", onnx.TensorProto.DOUBLE, [2]),
                h.make_tensor_value_info("unused", onnx.TensorProto.DOUBLE, [3])],
        [h.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2])],
        [onnx.numpy_helper.from_array(np.ones(2, dtype=np.float32), "one")],
        value_info=[h.make_tensor_value_info("converted", onnx.TensorProto.FLOAT, [2])]),
        opset_imports=[h.make_opsetid("", 18)], ir_version=8)


def test_externalized_cast_has_explicit_route_and_preserves_complete_result():
    model = cast_model()
    before = model.SerializeToString()
    candidate, report = TOOL.prepare(model, drop_unused=True, externalize_casts=["converted"])
    assert model.SerializeToString() == before
    onnx.checker.check_model(candidate)
    assert [value.name for value in candidate.graph.input] == ["converted"]
    assert report["external_casts"][0]["checked_runtime_cast_required"]
    assert not report["external_casts"][0]["hoisting_proved"]
    state = np.array([1 + 2. ** -40, -0.5], dtype=np.float64)
    old = ReferenceEvaluator(model).run(None, {"state": state, "unused": np.zeros(3)})[0]
    new = ReferenceEvaluator(candidate).run(None, {"converted": state.astype(np.float32)})[0]
    assert old.tobytes() == new.tobytes()
    assert not report["runtime_boundaries_implemented"]


@pytest.mark.parametrize("kind", ["observer", "overridable", "not-cast", "duplicate"])
def test_externalization_rejects_unproved_boundary(kind):
    model = cast_model()
    if kind == "observer":
        model.graph.output.append(onnx.helper.make_tensor_value_info("state", onnx.TensorProto.DOUBLE, [2]))
    elif kind == "overridable":
        model.graph.initializer.append(onnx.numpy_helper.from_array(np.zeros(2), "state"))
    elif kind == "not-cast":
        model.graph.node[0].op_type = "Identity"
    with pytest.raises(ValueError):
        TOOL.prepare(model, externalize_casts=["converted"] * (2 if kind == "duplicate" else 1))


def gather_model():
    h = onnx.helper
    return h.make_model(h.make_graph([h.make_node("Gather", ["weight", "indices"], ["output"], axis=0)],
        "gather", [h.make_tensor_value_info("indices", onnx.TensorProto.INT64, [3])],
        [h.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3, 2])],
        [onnx.numpy_helper.from_array(np.arange(8, dtype=np.float32).reshape(4, 2), "weight")]),
        opset_imports=[h.make_opsetid("", 18)], ir_version=8)


def test_gather_index_projection_requires_checked_runtime_range():
    model = gather_model()
    candidate, report = TOOL.prepare(model, narrow_gather_indices=["indices"])
    onnx.checker.check_model(candidate)
    indices = np.array([0, 2, -1], dtype=np.int64)
    assert ReferenceEvaluator(model).run(None, {"indices": indices})[0].tobytes() == ReferenceEvaluator(candidate).run(
        None, {"indices": indices.astype(np.int32)})[0].tobytes()
    assert report["narrowed_gather_indices"][0]["range"] == [-2147483648, 2147483647]
    assert report["narrowed_gather_indices"][0]["checked_runtime_cast_required"]


def test_index_narrowing_rejects_other_integer_observers():
    model = gather_model()
    model.graph.node.append(onnx.helper.make_node("Identity", ["indices"], ["observed"]))
    model.graph.output.append(onnx.helper.make_tensor_value_info("observed", onnx.TensorProto.INT64, [3]))
    with pytest.raises(ValueError, match="only as standard Gather"):
        TOOL.prepare(model, narrow_gather_indices=["indices"])


def test_unused_projection_keeps_observed_input_and_all_nodes():
    model = cast_model()
    model.graph.output.append(copy.deepcopy(model.graph.input[1]))
    candidate, report = TOOL.prepare(model, drop_unused=True)
    assert [node.SerializeToString() for node in candidate.graph.node] == [node.SerializeToString() for node in model.graph.node]
    assert [value.name for value in candidate.graph.input] == ["state", "unused"]
    assert not report["dropped_unused_inputs"]
