import copy
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest

PATH = Path(__file__).resolve().parents[2] / 'tools/deduplicate_onnx_initializer_transposes.py'
SPEC = importlib.util.spec_from_file_location('initializer_transpose_sharing', PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def model(*, public_weight=False, overridable=False, second_perm=(1, 0), dtype=np.float32):
    weight = numpy_helper.from_array(np.arange(6, dtype=dtype).reshape(2, 3), name='w')
    onnx_dtype = weight.data_type
    inputs = [helper.make_tensor_value_info('x', onnx_dtype, [3, 2])]
    if overridable: inputs.append(helper.make_tensor_value_info('w', onnx_dtype, [2, 3]))
    nodes = [helper.make_node('Transpose', ['w'], ['a'], name='one', perm=[1, 0]),
             helper.make_node('Transpose', ['w'], ['b'], name='two', perm=list(second_perm)),
             helper.make_node('Add', ['a', 'x'], ['y'], name='use_one'),
             helper.make_node('Add', ['b', 'x'], ['z'], name='use_two')]
    outputs = [helper.make_tensor_value_info(name, onnx_dtype, [3, 2]) for name in ('y', 'z')]
    if public_weight: outputs.append(helper.make_tensor_value_info('b', onnx_dtype, [3, 2]))
    return helper.make_model(helper.make_graph(nodes, 'test', inputs, outputs, [weight]),
                             opset_imports=[helper.make_opsetid('', 17)])


def test_only_readonly_initializer_transposes_are_shared_without_mutating_source():
    source = model(); original = source.SerializeToString()
    result, report = TOOL.deduplicate(source)
    assert source.SerializeToString() == original
    assert len(result.graph.node) == 3 and report['removed_node_count'] == 1
    assert report['node_aliases'] == {'two': 'one'}
    assert report['value_aliases'] == {'b': 'a'}
    assert list(result.graph.node[-1].input) == ['a', 'x']
    assert [x.SerializeToString() for x in source.graph.initializer] == [x.SerializeToString() for x in result.graph.initializer]
    assert [x.SerializeToString() for x in source.graph.output] == [x.SerializeToString() for x in result.graph.output]
    onnx.checker.check_model(result)


@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.float64, np.int64])
def test_complete_outputs_byte_exact(dtype):
    source = model(dtype=dtype); result, _ = TOOL.deduplicate(source)
    inputs = {'x': np.arange(6, dtype=dtype).reshape(3, 2)}
    old = ReferenceEvaluator(source).run(None, inputs)
    new = ReferenceEvaluator(result).run(None, inputs)
    assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes() for a, b in zip(old, new, strict=True))


def test_public_output_and_overridable_weight_not_removed():
    for source in (model(public_weight=True), model(overridable=True)):
        result, report = TOOL.deduplicate(source)
        assert len(result.graph.node) == 4 and report['removed_node_count'] == 0


def test_default_reverse_perm_and_explicit_reverse_are_equivalent():
    source = model(); source.graph.node[1].ClearField('attribute')
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 1
    assert len(result.graph.node) == 3


def test_different_permutations_and_different_initializers_not_merged():
    source = model(second_perm=(0, 1))
    source.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(2, 2), name='w'))
    source.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    for output in source.graph.output: output.type.tensor_type.shape.dim[0].dim_value = 2
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 0
    source = model(); other = copy.deepcopy(source.graph.initializer[0]); other.name = 'other'
    source.graph.initializer.append(other); source.graph.node[1].input[0] = 'other'
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 0


@pytest.mark.parametrize('key', ['layer_precisions', 'layer_output_types'])
def test_incompatible_or_absent_precision_requests_not_merged(key):
    for values in ({'one': 'fp32', 'two': 'fp16'}, {'one': 'fp32'}):
        result, report = TOOL.deduplicate(model(), **{key: values})
        assert report['removed_node_count'] == 0


def test_identical_precision_requests_preserved_by_alias_projection():
    precision = {'one': 'fp32', 'two': 'fp32', 'use_one': 'fp16'}
    output = {'one': 'fp32', 'two': 'fp32'}
    result, report = TOOL.deduplicate(model(), layer_precisions=precision, layer_output_types=output)
    projected = TOOL.project_requests({'layer_precisions': precision, 'layer_output_types': output}, report)
    assert projected['layer_precisions'] == {'one': 'fp32', 'use_one': 'fp16'}
    assert projected['layer_output_types'] == {'one': 'fp32'}
    assert report['node_aliases'] == {'two': 'one'}


def test_dynamic_transposes_not_merged():
    source = model(); source.graph.input.append(helper.make_tensor_value_info('dynamic', TensorProto.FLOAT, [2, 3]))
    source.graph.node[0].input[0] = 'dynamic'; source.graph.node[1].input[0] = 'dynamic'
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 0


def test_invalid_permutation_rejected():
    with pytest.raises(ValueError, match='permutation'):
        TOOL.deduplicate(model(second_perm=(0, 0)))


def test_external_weight_payload_not_read_or_modified():
    source = model(); weight = source.graph.initializer[0]
    weight.ClearField('raw_data'); weight.data_location = TensorProto.EXTERNAL
    for key, value in [('location', 'missing.data'), ('offset', '0'), ('length', '24')]:
        item = weight.external_data.add(); item.key = key; item.value = value
    original = weight.SerializeToString()
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 1 and result.graph.initializer[0].SerializeToString() == original
    assert report['floating_values_evaluated'] is False and report['external_payloads_read'] is False


def test_missing_names_fail_closed_for_auditable_candidates():
    source = model(); source.graph.node[1].name = ''
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 0


def test_transpose_sharing_preserves_nan_payloads_and_signed_zero_bits():
    source = model()
    bits = np.array([0x7fc01234, 0xffa54321, 0x00000001, 0x80000000, 0x3f800000, 0x7f800000], dtype=np.uint32)
    source.graph.initializer[0].CopyFrom(numpy_helper.from_array(bits.view(np.float32).reshape(2, 3), name='w'))
    for index in (2, 3):
        node = source.graph.node[index]
        node.op_type = 'Identity'; del node.input[1:]
    result, report = TOOL.deduplicate(source)
    old = ReferenceEvaluator(source).run(None, {'x': np.zeros((3, 2), dtype=np.float32)})
    new = ReferenceEvaluator(result).run(None, {'x': np.zeros((3, 2), dtype=np.float32)})
    assert report['removed_node_count'] == 1 and all(a.tobytes() == b.tobytes() for a, b in zip(old, new, strict=True))


def test_overload_or_device_specific_nodes_not_merged():
    source = model(); source.graph.node[1].overload = 'custom'
    result, report = TOOL.deduplicate(source)
    assert report['removed_node_count'] == 0


def test_projection_rejects_inconsistent_alias_policy():
    _, report = TOOL.deduplicate(model())
    with pytest.raises(ValueError, match='presence'):
        TOOL.project_requests({'layer_precisions': {'two': 'fp32'}, 'layer_output_types': {}}, report)
    with pytest.raises(ValueError, match='value'):
        TOOL.project_requests({'layer_precisions': {'one': 'fp16', 'two': 'fp32'}, 'layer_output_types': {}}, report)
