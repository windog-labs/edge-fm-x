import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip('onnx')
np = pytest.importorskip('numpy')
from onnx import TensorProto as T, helper as h, numpy_helper as nh

SPEC = importlib.util.spec_from_file_location('partition_plugins',
    Path(__file__).resolve().parents[2] / 'tools/partition_onnx_values.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
TRUSTED = ('bf16-round/1', 'layernorm/1')


def source():
    nodes = [h.make_node('VLAForgeBFloat16Round', ['x'], ['rounded'], name='arbitrary_round',
                         domain='vlaforge.trt', plugin_version='1', plugin_namespace=''),
             h.make_node('LayerNorm', ['rounded', 'gamma', 'beta'], ['normalized'], name='arbitrary_norm',
                         domain='trt.plugins', epsilon=1e-6, plugin_version='1', plugin_namespace=''),
             h.make_node('Add', ['normalized', 'x'], ['result'], name='tail')]
    return h.make_model(h.make_graph(nodes, 'unbranded',
        [h.make_tensor_value_info('x', T.FLOAT, [1, 2, 4])],
        [h.make_tensor_value_info('result', T.FLOAT, [1, 2, 4])],
        [nh.from_array(np.ones(4, np.float32), 'gamma'), nh.from_array(np.zeros(4, np.float32), 'beta')]),
        opset_imports=[h.make_opsetid('', 17), h.make_opsetid('vlaforge.trt', 1), h.make_opsetid('trt.plugins', 1)], ir_version=8)


def test_reviewed_plugins_keep_real_nodes_and_independent_static_frontier():
    model = source()
    before = model.SerializeToString()
    a, b, report = TOOL.partition(model, ['normalized'], trusted_plugins=TRUSTED)
    assert model.SerializeToString() == before
    assert [n.SerializeToString() for n in a.graph.node] == [n.SerializeToString() for n in model.graph.node[:2]]
    assert b.graph.node[0].SerializeToString() == model.graph.node[2].SerializeToString()
    assert list(a.graph.output[0].type.tensor_type.shape.dim)[-1].dim_value == 4
    assert report['shape_proof']['internal_annotations_seeded'] is False
    assert report['shape_proof']['plugin_counts'] == {'bf16-round/1': 1, 'layernorm/1': 1}


def test_custom_plugins_remain_rejected_without_explicit_reviewed_contract():
    with pytest.raises(ValueError, match='functional'):
        TOOL.partition(source(), ['normalized'])


@pytest.mark.parametrize('change', ['version', 'namespace', 'attribute', 'width', 'half', 'domain', 'rank', 'output-type'])
def test_unproved_plugin_contract_rejects_instead_of_trusting_annotations(change):
    model = source()
    if change in ('version', 'namespace'):
        key = 'plugin_' + change
        next(a for a in model.graph.node[0].attribute if a.name == key).s = b'wrong'
    elif change == 'attribute':
        model.graph.node[0].attribute.append(h.make_attribute('hidden_rng', 1))
    elif change == 'width':
        model.graph.initializer[1].CopyFrom(nh.from_array(np.zeros(3, np.float32), 'beta'))
    elif change == 'half':
        model.graph.input[0].type.tensor_type.elem_type = T.FLOAT16
    elif change == 'domain':
        model.graph.node[0].domain = 'not_reviewed'
    elif change == 'rank':
        model.graph.input[0].type.tensor_type.shape.dim.pop()
        model.graph.output[0].type.tensor_type.shape.dim.pop()
    else:
        model.graph.value_info.append(h.make_tensor_value_info('rounded', T.DOUBLE, [1, 2, 4]))
    with pytest.raises((ValueError, onnx.shape_inference.InferenceError)):
        TOOL.partition(model, ['normalized'], trusted_plugins=TRUSTED)


def test_literal_hoisting_avoids_constant_frontier_without_rewriting_values():
    model = h.make_model(h.make_graph([
        h.make_node('Constant', [], ['offset'], name='literal', value=nh.from_array(np.ones(3, np.float32))),
        h.make_node('Add', ['x', 'offset'], ['middle'], name='producer'),
        h.make_node('Mul', ['middle', 'offset'], ['out'], name='consumer')], 'literal_partition',
        [h.make_tensor_value_info('x', T.FLOAT, [3])], [h.make_tensor_value_info('out', T.FLOAT, [3])]),
        opset_imports=[h.make_opsetid('', 17)], ir_version=8)
    before = model.SerializeToString()
    a, b, report = TOOL.partition(model, ['middle'], hoist_constants=True)
    assert model.SerializeToString() == before
    assert report['first_outputs'] == ['middle']
    assert len(report['constant_hoists']) == 1
    assert all(n.op_type != 'Constant' for part in (a, b) for n in part.graph.node)
    assert all(nh.to_array(part.graph.initializer[0]).tobytes() == np.ones(3, np.float32).tobytes() for part in (a, b))
