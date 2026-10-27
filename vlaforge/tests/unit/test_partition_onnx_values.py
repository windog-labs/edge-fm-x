import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

SPEC = importlib.util.spec_from_file_location("partition_onnx_values",
    Path(__file__).resolve().parents[2] / "tools/partition_onnx_values.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def source(shared=False):
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3]) for name in ("position", "features")]
    nodes = [helper.make_node("Mul", ["position", "frequency"], ["phase"], name="phase"),
             helper.make_node("Sin", ["phase"], ["sine"], name="sine"),
             helper.make_node("Cos", ["phase"], ["cosine"], name="cosine"),
             helper.make_node("Add", ["sine", "cosine"], ["sum"], name="sum"),
             helper.make_node("Add", ["sum", "phase" if shared else "features"], ["result"], name="consumer")]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3]) for name in ("result", "sine")]
    constant = numpy_helper.from_array(np.asarray([[1.0, 0.3245321, 0.0789123]], np.float32), "frequency")
    return helper.make_model(helper.make_graph(nodes, "producer-cut", inputs, outputs, [constant]),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("shared", [False, True])
def test_dependency_closed_partition_is_byte_exact_and_has_no_shared_computation(shared):
    model = source(shared)
    before = model.SerializeToString()
    first, second, ledger = TOOL.partition(model, ["sine", "cosine"])
    assert model.SerializeToString() == before
    assert set(node.name for node in first.graph.node).isdisjoint(node.name for node in second.graph.node)
    assert ledger["computation_duplication"] is False
    assert ledger["formula_rewritten"] is False
    assert ledger["passthrough_aliases"]["sine"] != "sine"
    assert not set(value.name for value in second.graph.input) & set(value.name for value in second.graph.output)
    assert ("phase" in ledger["first_outputs"]) == shared
    original, a, b = session(model), session(first), session(second)
    for positions in ([0, 1, 2], [128, 139, 175], [-40, 60, 90]):
        inputs = {"position": np.asarray([positions], np.float32), "features": np.asarray([[0.2, 0.3, 0.4]], np.float32)}
        middle = a.run(None, inputs)
        values = list(inputs.values())
        feeds = {route["name"]: (values if route["source"] == "input" else middle)[route["index"]]
                 for route in ledger["second_inputs"]}
        for actual, expected in zip(b.run(None, feeds), original.run(None, inputs), strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("values", [[], ["position"], ["missing"], ["sine", "sine"], ["result"]])
def test_ambiguous_unproduced_or_whole_model_cuts_are_rejected(values):
    with pytest.raises(ValueError):
        TOOL.partition(source(), values)


@pytest.mark.parametrize('dtype,value', [(np.int64, -7), (np.float32, .125)])
def test_direct_initializer_output_retains_exact_original_constant(dtype, value):
    model = source()
    constant = numpy_helper.from_array(np.asarray([[value]], dtype=dtype), 'source_offset')
    model.graph.initializer.append(constant)
    model.graph.output.insert(0, helper.make_tensor_value_info('source_offset', constant.data_type, [1, 1]))
    before = model.SerializeToString()
    first, second, ledger = TOOL.partition(model, ['sine', 'cosine'])
    assert model.SerializeToString() == before
    restored = next(tensor for tensor in second.graph.initializer if tensor.name == 'source_offset')
    assert restored.SerializeToString() == constant.SerializeToString()
    assert ledger['retained_output_initializers']['consumer'] == ['source_offset']
    inputs = {'position': np.asarray([[0, 1, 2]], np.float32), 'features': np.asarray([[.2, .3, .4]], np.float32)}
    intermediate = session(first).run(None, inputs)
    original_inputs = list(inputs.values())
    feeds = {route['name']: (original_inputs if route['source'] == 'input' else intermediate)[route['index']]
             for route in ledger['second_inputs']}
    for expected, actual in zip(session(model).run(None, inputs), session(second).run(None, feeds), strict=True):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape and expected.tobytes() == actual.tobytes()


def test_unreviewed_domain_is_not_assumed_functional():
    model = source()
    model.graph.node[3].domain = "unreviewed"
    with pytest.raises(ValueError, match="functional"):
        TOOL.partition(model, ["sine", "cosine"])


def test_external_weights_survive_partition_publication_and_source_unlink(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    root = tmp_path / "source"
    root.mkdir()
    path = root / "model.onnx"
    onnx.save_model(source(), str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    metadata = onnx.load(str(path), load_external_data=False)
    before = metadata.SerializeToString()
    with pytest.raises(ValueError, match="base directory"):
        TOOL.partition(metadata, ["sine", "cosine"])
    TOOL.partition(metadata, ["sine", "cosine"], external_base_dir=root)
    assert metadata.SerializeToString() == before
    destination = tmp_path / "partitions"
    monkeypatch.setattr("sys.argv", ["partition", "--input", str(path), "--output", str(destination),
                                     "--value-name", "sine", "--value-name", "cosine"])
    TOOL.main()
    assert (destination / "weights.bin").stat().st_nlink == 1
    assert (root / "weights.bin").stat().st_nlink == 1
    (root / "weights.bin").unlink()
    first = ort.InferenceSession(str(destination / "producer.onnx"), providers=["CPUExecutionProvider"])
    second = ort.InferenceSession(str(destination / "consumer.onnx"), providers=["CPUExecutionProvider"])
    import json
    ledger = json.loads((destination / "partition.json").read_text())
    inputs = {"position": np.asarray([[1, 2, 3]], np.float32),
              "features": np.asarray([[.2, .3, .4]], np.float32)}
    middle = first.run(None, inputs)
    feeds = {route["name"]: (list(inputs.values()) if route["source"] == "input" else middle)[route["index"]]
             for route in ledger["second_inputs"]}
    for actual, expected in zip(second.run(None, feeds), session(source()).run(None, inputs), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert len(ledger["external_data"]) == 1


def test_deep_dependency_chain_does_not_use_python_recursion():
    nodes = [helper.make_node("Identity", ["input" if index == 0 else "v" + str(index - 1)],
                              ["v" + str(index)], name="copy" + str(index)) for index in range(1502)]
    model = helper.make_model(helper.make_graph(nodes, "deep",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("v1501", TensorProto.FLOAT, [1])]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    first, second, report = TOOL.partition(model, ["v1400"])
    assert report["first_nodes"] == 1401
    assert len(first.graph.node) + len(second.graph.node) == len(nodes)
