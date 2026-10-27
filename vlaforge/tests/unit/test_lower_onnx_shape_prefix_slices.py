import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import pytest
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('shape_prefix_slices', ROOT / 'tools/lower_onnx_shape_prefix_slices.py')
bounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounds)

def fixture(*, data_shape=(3,), selector_shape=(None,), axes=(0,), steps=None, starts=None, raw_end=False, int32=False):
    integer = onnx.TensorProto.INT32 if int32 else onnx.TensorProto.INT64
    dtype = np.int32 if int32 else np.int64
    initializers = [numpy_helper.from_array(np.asarray(axes, dtype=dtype), 'axes'), numpy_helper.from_array(np.asarray(starts or [0] * len(axes), dtype=dtype), 'starts')]
    inputs = [helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT, list(data_shape))]
    nodes = []
    if raw_end:
        inputs.append(helper.make_tensor_value_info('ends', integer, [len(axes)]))
    else:
        inputs.append(helper.make_tensor_value_info('selector', onnx.TensorProto.FLOAT, list(selector_shape)))
        nodes.append(helper.make_node('Shape', ['selector'], ['ends_i64'], name='source_shape'))
        nodes.append(helper.make_node('Cast', ['ends_i64'], ['ends'], to=integer, name='source_cast'))
    ports = ['data', 'starts', 'ends', 'axes']
    if steps is not None:
        initializers.append(numpy_helper.from_array(np.asarray(steps, dtype=dtype), 'steps'))
        ports.append('steps')
    nodes.append(helper.make_node('Slice', ports, ['result'], name='source_slice'))
    graph = helper.make_graph(nodes, 'shape_derived_slice', inputs, [helper.make_tensor_value_info('result', onnx.TensorProto.FLOAT, [None] * len(data_shape)), helper.make_tensor_value_info('ends', integer, [len(axes)])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)

def run(model, feeds):
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, feeds)

def test_actual_bug_pattern_emits_a_runtime_bound_without_changing_source_end():
    source = fixture()
    before = source.SerializeToString()
    lowered, report = bounds.lower(source)
    assert len(report['rewritten']) == 1
    assert source.SerializeToString() == before
    assert not any((node.op_type == 'Slice' for node in lowered.graph.node))
    assert any((node.op_type == 'Range' for node in lowered.graph.node))
    minimum = next((node for node in lowered.graph.node if node.op_type == 'Min'))
    assert minimum.op_type == 'Min' and 'ends' in minimum.input
    assert [value.SerializeToString() for value in source.graph.input] == [value.SerializeToString() for value in lowered.graph.input]
    assert [value.SerializeToString() for value in source.graph.output] == [value.SerializeToString() for value in lowered.graph.output]
    for expected, actual in zip(run(source, {'data': np.arange(3, dtype=np.float32), 'selector': np.ones(9, np.float32)}), run(lowered, {'data': np.arange(3, dtype=np.float32), 'selector': np.ones(9, np.float32)}), strict=True):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape and (expected.tobytes() == actual.tobytes())

@pytest.mark.parametrize('length', [0, 1, 3, 8])
@pytest.mark.parametrize('end', [0, 1, 3, 12])
def test_shape_ends_below_equal_and_above_actual_data(length, end):
    source = fixture(data_shape=(None,))
    lowered, report = bounds.lower(source)
    assert len(report['rewritten']) == 1
    feeds = {'data': np.arange(length, dtype=np.float32), 'selector': np.ones(end, np.float32)}
    for expected, actual in zip(run(source, feeds), run(lowered, feeds), strict=True):
        assert expected.shape == actual.shape and expected.tobytes() == actual.tobytes()

@pytest.mark.parametrize('axes', [(1, 2), (1, -1), (-2, -1)])
@pytest.mark.parametrize('steps', [(1, 1), (2, 3)])
def test_nonprefix_multiple_axes_cases_are_untouched(axes, steps):
    source = fixture(data_shape=(2, 3, 4), selector_shape=(None, None), axes=axes, steps=steps, starts=(-9, 1))
    lowered, report = bounds.lower(source)
    assert report['rewritten'] == []
    assert lowered.SerializeToString() == source.SerializeToString()
    feeds = {'data': np.arange(24, dtype=np.float32).reshape(2, 3, 4), 'selector': np.ones((9, 2), np.float32)}
    for expected, actual in zip(run(source, feeds), run(lowered, feeds), strict=True):
        assert expected.shape == actual.shape and expected.tobytes() == actual.tobytes()

@pytest.mark.parametrize('arguments', [dict(raw_end=True), dict(steps=(-1,)), dict(int32=True)])
def test_unproved_signed_or_reverse_cases_are_untouched(arguments):
    source = fixture(**arguments)
    lowered, report = bounds.lower(source)
    assert report['rewritten'] == [] and lowered.SerializeToString() == source.SerializeToString()
    assert report['skipped']

def test_idempotent_and_initializers_are_unchanged():
    source = fixture()
    lowered, first = bounds.lower(source)
    again, second = bounds.lower(lowered)
    assert len(first['rewritten']) == 1 and second['rewritten'] == []
    assert lowered.SerializeToString() == again.SerializeToString()
    assert [value.SerializeToString() for value in source.graph.initializer] == [value.SerializeToString() for value in lowered.graph.initializer[:len(source.graph.initializer)]]
    assert all((value.data_type == onnx.TensorProto.INT64 for value in lowered.graph.initializer[len(source.graph.initializer):]))

def test_invalid_axis_and_unknown_rank_are_conservatively_skipped():
    for source in [fixture(axes=(2,)), fixture(axes=(0, 0), selector_shape=(None, None))]:
        lowered, report = bounds.lower(source)
        assert report['rewritten'] == [] and lowered.SerializeToString() == source.SerializeToString()
    source = fixture()
    source.graph.input[0].type.tensor_type.ClearField('shape')
    lowered, report = bounds.lower(source)
    assert report['rewritten'] == []

def test_custom_ops_are_not_matched_as_standard_slice():
    source = fixture()
    source.graph.node[-1].domain = 'custom'
    lowered, report = bounds.lower(source)
    assert report['rewritten'] == [] and lowered.SerializeToString() == source.SerializeToString()

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int32, np.int64, np.bool_])
def test_data_bits_and_dtype_are_never_recomputed(dtype):
    source = fixture(data_shape=(7,))
    tensor_type = numpy_helper.from_array(np.zeros(1, dtype=dtype)).data_type
    source.graph.input[0].type.tensor_type.elem_type = tensor_type
    source.graph.output[0].type.tensor_type.elem_type = tensor_type
    if dtype == np.float32:
        values = np.asarray([2147483648, 2143294004, 1065353216, 0, 3212836864, 2139095040, 4286578688], np.uint32).view(np.float32)
    else:
        values = np.asarray([0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    lowered, report = bounds.lower(source)
    assert len(report['rewritten']) == 1
    feeds = {'data': values, 'selector': np.ones(9, np.float32)}
    for expected, actual in zip(run(source, feeds), run(lowered, feeds), strict=True):
        assert expected.dtype == actual.dtype and expected.tobytes() == actual.tobytes()

def test_cli_never_overwrites_source_or_existing_output(tmp_path):
    source = tmp_path / 'source.onnx'
    source.write_bytes(fixture().SerializeToString())
    before = source.read_bytes()
    output = tmp_path / 'bounded.onnx'
    command = [sys.executable, str(ROOT / 'tools/lower_onnx_shape_prefix_slices.py'), '--input', str(source), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == before
    assert len(json.loads(output.with_suffix('.report.json').read_text())['rewritten']) == 1
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert source.read_bytes() == before
