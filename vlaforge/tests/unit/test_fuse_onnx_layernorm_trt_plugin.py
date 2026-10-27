import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip('onnx')
np = pytest.importorskip('numpy')
ort = pytest.importorskip('onnxruntime')
from onnx import TensorProto, helper, numpy_helper

SPEC = importlib.util.spec_from_file_location('fuse_layernorm',
    Path(__file__).resolve().parents[2] / 'tools/fuse_onnx_layernorm_trt_plugin.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def graph(shape=(1, 3, 4), *, positive_axis=False, swapped=False):
    axis = len(shape) - 1 if positive_axis else -1
    nodes = [helper.make_node('ReduceMean', ['x'], ['mean'], name='arbitrary_mean', axes=[axis], keepdims=1),
        helper.make_node('Sub', ['x', 'mean'], ['centered'], name='center'),
        helper.make_node('Mul', ['centered', 'centered'], ['square'], name='square'),
        helper.make_node('ReduceMean', ['square'], ['variance'], name='variance', axes=[axis], keepdims=1),
        helper.make_node('Add', ['variance', 'eps'], ['stabilized'], name='epsilon'),
        helper.make_node('Sqrt', ['stabilized'], ['deviation'], name='deviation'),
        helper.make_node('Div', ['centered', 'deviation'], ['normalized'], name='divide'),
        helper.make_node('Mul', ['normalized', 'gamma'], ['scaled'], name='scale'),
        helper.make_node('Add', ['scaled', 'beta'], ['y'], name='shift')]
    if swapped:
        for index in (4, 7, 8):
            nodes[index].input.reverse()
    initializers = [numpy_helper.from_array(np.linspace(0.3, 1.2, shape[-1], dtype=np.float32), 'gamma'),
        numpy_helper.from_array(np.linspace(-0.1, 0.1, shape[-1], dtype=np.float32), 'beta'),
        numpy_helper.from_array(np.asarray(1e-6, dtype=np.float32), 'eps')]
    ports = [helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)]
    return helper.make_model(helper.make_graph(nodes, 'unbranded_normalization', ports, outputs, initializers),
        opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


@pytest.mark.parametrize('shape', [(4,), (2, 4), (1, 3, 4), (2, 2, 3, 4)])
@pytest.mark.parametrize('positive_axis', [False, True])
@pytest.mark.parametrize('swapped', [False, True])
def test_dataflow_match_preserves_source_and_complete_cpu_outputs(shape, positive_axis, swapped):
    model = graph(shape, positive_axis=positive_axis, swapped=swapped)
    before = model.SerializeToString()
    reference, report = TOOL.rewrite_layer_norms(model, plugin=False)
    deployment, deployed = TOOL.rewrite_layer_norms(model)
    assert model.SerializeToString() == before
    assert report['rewritten'] == deployed['rewritten'] == 1
    assert [n.op_type for n in reference.graph.node] == ['LayerNormalization']
    assert [n.op_type for n in deployment.graph.node] == (['LayerNorm'] if len(shape) == 3 else ['Reshape', 'LayerNorm', 'Reshape'])
    assert deployment.graph.node[0].name != 'shift'
    assert [value.SerializeToString() for value in deployment.graph.initializer[:3]] == [
        value.SerializeToString() for value in model.graph.initializer]
    assert deployed['groups'][0]['epsilon'] == float(np.float32(1e-6))
    onnx.checker.check_model(deployment)
    data = np.random.default_rng(17).normal(size=shape).astype(np.float32)
    expected = ort.InferenceSession(before, providers=['CPUExecutionProvider']).run(None, {'x': data})[0]
    actual = ort.InferenceSession(reference.SerializeToString(), providers=['CPUExecutionProvider']).run(None, {'x': data})[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('value', ['mean', 'centered', 'variance', 'normalized', 'scaled'])
def test_externally_observable_intermediates_are_not_deleted(value):
    model = graph()
    model.graph.output.append(helper.make_tensor_value_info(value, TensorProto.FLOAT,
        [1, 3, 1 if value in ('mean', 'variance') else 4]))
    before = model.SerializeToString()
    result, report = TOOL.rewrite_layer_norms(model)
    assert report['rewritten'] == 0 and result.SerializeToString() == before


def test_external_internal_consumer_is_preserved():
    model = graph()
    model.graph.node.append(helper.make_node('Identity', ['mean'], ['observed']))
    model.graph.output.append(helper.make_tensor_value_info('observed', TensorProto.FLOAT, [1, 3, 1]))
    result, report = TOOL.rewrite_layer_norms(model)
    assert report['rewritten'] == 0 and result.SerializeToString() == model.SerializeToString()


@pytest.mark.parametrize('epsilon', [0.0, -1e-6, float('nan'), float('inf')])
def test_nonpositive_or_nonfinite_epsilon_is_not_reinterpreted(epsilon):
    model = graph()
    model.graph.initializer[2].CopyFrom(numpy_helper.from_array(np.asarray(epsilon, np.float32), 'eps'))
    assert TOOL.rewrite_layer_norms(model)[1]['rewritten'] == 0


@pytest.mark.parametrize('change', ['axis', 'variance_axis', 'keepdims', 'half', 'dynamic', 'scale_shape', 'epsilon_override', 'epsilon_rank', 'not_square'])
def test_unproven_normalization_contract_is_left_unchanged(change):
    model = graph()
    if change in ('axis', 'variance_axis'):
        index = 0 if change == 'axis' else 3
        model.graph.node[index].attribute[0].ints[:] = [1]
    elif change == 'keepdims':
        next(a for a in model.graph.node[0].attribute if a.name == 'keepdims').i = 0
    elif change == 'half':
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    elif change == 'dynamic':
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = 'sequence'
    elif change == 'scale_shape':
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((1, 4), np.float32), 'gamma'))
    elif change == 'epsilon_override':
        model.graph.input.append(helper.make_tensor_value_info('eps', TensorProto.FLOAT, []))
    elif change == 'epsilon_rank':
        model.graph.initializer[2].CopyFrom(numpy_helper.from_array(np.full((1, 1, 1, 1), 1e-6, np.float32), 'eps'))
    else:
        model.graph.node[2].input[1] = 'x'
    before = model.SerializeToString()
    result, report = TOOL.rewrite_layer_norms(model)
    assert report['rewritten'] == 0 and result.SerializeToString() == before


def test_reshape_names_do_not_collide_with_existing_symbols():
    model = graph((2, 4))
    model.graph.node[0].name = 'vf_layer_norm_0'
    result, report = TOOL.rewrite_layer_norms(model)
    assert report['groups'][0]['replacement'] == 'vf_layer_norm_0_'
    names = [node.name for node in result.graph.node]
    assert len(names) == len(set(names))


def test_multiple_disjoint_norms_do_not_duplicate_or_drop_nodes():
    model = graph()
    second = graph()
    for node in second.graph.node:
        node.name += '_second'
        for index, value in enumerate(node.input):
            node.input[index] = 'y' if value == 'x' else value + '_second'
        for index, value in enumerate(node.output):
            node.output[index] = value + '_second'
    for tensor in second.graph.initializer:
        tensor.name += '_second'
    model.graph.node.extend(second.graph.node)
    model.graph.initializer.extend(second.graph.initializer)
    model.graph.output[0].name = 'y_second'
    model.graph.value_info.append(helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 4]))
    result, report = TOOL.rewrite_layer_norms(model)
    assert report['rewritten'] == 2 and len(result.graph.node) == 2
    onnx.checker.check_model(result)


@pytest.mark.parametrize('shape', [(4,), (2, 4), (1, 3, 4), (2, 2, 3, 4)])
def test_standard_layernormalization_reuses_the_same_typed_plugin_contract(shape):
    standard, _ = TOOL.rewrite_layer_norms(graph(shape), plugin=False)
    before = standard.SerializeToString()
    result, report = TOOL.rewrite_layer_norms(standard)
    assert standard.SerializeToString() == before
    assert report['rewritten'] == 1
    assert [node.op_type for node in result.graph.node] == (['LayerNorm'] if len(shape) == 3 else ['Reshape', 'LayerNorm', 'Reshape'])
    assert report['groups'][0]['epsilon'] == float(np.float32(1e-6))
    onnx.checker.check_model(result)


@pytest.mark.parametrize('change', ['axis', 'stash', 'epsilon', 'statistics'])
def test_standard_layernorm_does_not_drop_statistics_or_reinterpret_contract(change):
    standard, _ = TOOL.rewrite_layer_norms(graph(), plugin=False)
    node = standard.graph.node[0]
    if change == 'axis':
        next(attribute for attribute in node.attribute if attribute.name == 'axis').i = 1
    elif change == 'stash':
        node.attribute.append(helper.make_attribute('stash_type', TensorProto.BFLOAT16))
    elif change == 'epsilon':
        next(attribute for attribute in node.attribute if attribute.name == 'epsilon').f = float('nan')
    else:
        node.output.append('mean')
        standard.graph.output.append(helper.make_tensor_value_info('mean', TensorProto.FLOAT, [1, 3, 1]))
    assert TOOL.rewrite_layer_norms(standard)[1]['rewritten'] == 0


def test_internal_unknown_dimension_must_be_proven_from_static_graph_not_guessed():
    standard, _ = TOOL.rewrite_layer_norms(graph(), plugin=False)
    standard.graph.input[0].name = 'raw'
    standard.graph.node.insert(0, helper.make_node('Reshape', ['raw', 'shape'], ['x']))
    standard.graph.initializer.append(numpy_helper.from_array(np.asarray([-1, 3, 4], np.int64), 'shape'))
    value = helper.make_tensor_value_info('x', TensorProto.FLOAT, ['unknown_batch', 3, 4])
    standard.graph.value_info.append(value)
    before = standard.SerializeToString()
    result, report = TOOL.rewrite_layer_norms(standard)
    assert report['rewritten'] == 1 and report['groups'][0]['input_shape'] == [1, 3, 4]
    assert standard.SerializeToString() == before
    onnx.checker.check_model(result)


@pytest.mark.parametrize('version,rewritten', [('1', 1), ('2', 0)])
def test_shape_shadow_uses_only_the_known_rounding_plugin_without_removing_real_rounding(version, rewritten):
    standard, _ = TOOL.rewrite_layer_norms(graph(), plugin=False)
    standard.graph.input[0].name = 'raw'
    standard.graph.node.insert(0, helper.make_node('VLAForgeBFloat16Round', ['raw'], ['x'],
        name='round', domain='vlaforge.trt', plugin_version=version, plugin_namespace=''))
    standard.opset_import.append(helper.make_opsetid('vlaforge.trt', 1))
    standard.graph.value_info.append(helper.make_tensor_value_info('x', TensorProto.FLOAT, ['unknown_batch', 3, 4]))
    result, report = TOOL.rewrite_layer_norms(standard)
    assert report['rewritten'] == rewritten
    assert result.graph.node[0].SerializeToString() == standard.graph.node[0].SerializeToString()
    if rewritten:
        assert report['groups'][0]['input_shape'] == [1, 3, 4]


@pytest.mark.parametrize('external_axes', [False, True])
def test_external_numeric_metadata_reads_leave_source_and_sidecar_immutable(tmp_path, external_axes):
    source = graph()
    if external_axes:
        source.opset_import[0].version = 18
        source.graph.initializer.append(numpy_helper.from_array(np.asarray([-1], np.int64), 'axes'))
        for index in (0, 3):
            node = source.graph.node[index]
            retained = [attribute for attribute in node.attribute if attribute.name != 'axes']
            node.ClearField('attribute')
            node.attribute.extend(retained)
            node.input.append('axes')
    path = tmp_path / 'source.onnx'
    onnx.save_model(source, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weights.data', size_threshold=0)
    loaded = onnx.load(path, load_external_data=False)
    before = loaded.SerializeToString()
    sidecar = tmp_path / 'weights.data'
    weight_bytes, mtime = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
    candidate, report = TOOL.rewrite_layer_norms(loaded, base_dir=str(tmp_path))
    assert report['rewritten'] == 1
    assert loaded.SerializeToString() == before
    assert sidecar.read_bytes() == weight_bytes and sidecar.stat().st_mtime_ns == mtime
    output = tmp_path / 'candidate.onnx'
    output.write_bytes(candidate.SerializeToString())
    onnx.checker.check_model(str(output))
