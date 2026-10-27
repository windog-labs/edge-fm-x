import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import pytest

PATH = Path(__file__).resolve().parents[2] / 'tools/lower_onnx_crop_pads.py'
SPEC = importlib.util.spec_from_file_location('crop_pads', PATH)
tool = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(tool)


def model(shape, pads, dtype=np.float32, mode='constant'):
    kind = numpy_helper.from_array(np.zeros(1, dtype=dtype)).data_type
    output_shape = [dimension + pads[index] + pads[index+len(shape)] for index, dimension in enumerate(shape)]
    graph = helper.make_graph([helper.make_node('Pad', ['x', 'pads', ''], ['y'], name='crop', mode=mode)],
        'crop_only', [helper.make_tensor_value_info('x', kind, shape)],
        [helper.make_tensor_value_info('y', kind, output_shape)],
        [numpy_helper.from_array(np.asarray(pads, np.int64), 'pads')])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


def evaluate(graph, value):
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, {'x': value})[0]


def test_real_state_crop_geometry_is_rewritten_without_changing_output():
    source = model([1, 6144, 327], [0, 0, -323, 0, 0, 0])
    original = source.SerializeToString(); result, report = tool.lower(source)
    assert len(report['rewritten']) == 1 and source.SerializeToString() == original
    assert not any(node.op_type == 'Pad' for node in result.graph.node)
    assert any(node.op_type == 'Slice' for node in result.graph.node)
    values = np.arange(6144 * 327, dtype=np.float32).reshape(1, 6144, 327)
    assert evaluate(source, values).tobytes() == evaluate(result, values).tobytes()


@pytest.mark.parametrize('shape,pads', [([7], [-2, -1]), ([7], [0, -3]),
    ([2, 3, 7], [0, -1, -2, 0, 0, -1]), ([3, 4], [-1, 0, 0, -2]), ([3], [-3, 0])])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64, np.bool_])
def test_all_crop_axes_shapes_and_data_bits(shape, pads, dtype):
    source = model(shape, pads, dtype); result, report = tool.lower(source)
    assert len(report['rewritten']) == 1
    values = (np.arange(np.prod(shape)) % 2).astype(dtype).reshape(shape)
    expected, actual = evaluate(source, values), evaluate(result, values)
    assert expected.dtype == actual.dtype and expected.shape == actual.shape and expected.tobytes() == actual.tobytes()


def test_nan_payload_and_signed_zero_are_preserved():
    source = model([5], [-1, -1]); result, _ = tool.lower(source)
    values = np.asarray([0x3f800000, 0x80000000, 0x7fc01234, 0x7f800000, 0], np.uint32).view(np.float32)
    assert evaluate(source, values).tobytes() == evaluate(result, values).tobytes()


@pytest.mark.parametrize('shape,pads,mode', [([7], [2, 0], 'constant'),
    ([7], [-1, 2], 'constant'), ([7], [0, 0], 'constant'), ([7], [-8, 0], 'constant'),
    ([7], [-1, 0], 'reflect')])
def test_unsupported_or_invalid_crop_is_left_unchanged(shape, pads, mode):
    source = model(shape, pads, mode=mode); result, report = tool.lower(source)
    assert report['rewritten'] == [] and result.SerializeToString() == source.SerializeToString()


def test_unknown_extent_and_dynamic_pads_are_not_guessed():
    for unknown in ('shape', 'pads'):
        source = model([7], [-1, 0])
        if unknown == 'shape': source.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
        else:
            source.graph.ClearField('initializer')
            source.graph.input.append(helper.make_tensor_value_info('pads', onnx.TensorProto.INT64, [2]))
        result, report = tool.lower(source)
        assert report['rewritten'] == [] and result.SerializeToString() == source.SerializeToString()


def test_second_application_does_not_change_the_graph():
    result, _ = tool.lower(model([7], [-2, 0])); again, report = tool.lower(result)
    assert report['rewritten'] == [] and result.SerializeToString() == again.SerializeToString()
