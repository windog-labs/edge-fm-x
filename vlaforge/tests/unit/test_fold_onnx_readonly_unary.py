import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest

onnx=pytest.importorskip('onnx')
ort=pytest.importorskip('onnxruntime')
from onnx import TensorProto,helper,numpy_helper


@pytest.fixture
def tool():
    path=Path(__file__).resolve().parents[2]/'tools/fold_onnx_readonly_unary.py'
    spec=importlib.util.spec_from_file_location('readonly_unary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def model(dtype=TensorProto.FLOAT,values=None,overridable=False):
    values=np.asarray([.25,-2.],np.float32)if values is None else values
    weights=numpy_helper.from_array(values,'parameter')
    inputs=[helper.make_tensor_value_info('x',dtype,list(values.shape))]
    if overridable:inputs.append(helper.make_tensor_value_info('parameter',dtype,list(values.shape)))
    nodes=[helper.make_node('Cast',['parameter'],['cast'],name='parameter_cast',to=dtype),
           helper.make_node('Exp',['cast'],['coefficient'],name='static_exp'),
           helper.make_node('Mul',['x','coefficient'],['product'],name='dynamic_product'),
           helper.make_node('Exp',['product'],['output'],name='dynamic_exp')]
    return helper.make_model(helper.make_graph(nodes,'readonly',inputs,
        [helper.make_tensor_value_info('output',dtype,list(values.shape)),
         helper.make_tensor_value_info('coefficient',dtype,list(values.shape))],[weights]),
        opset_imports=[helper.make_opsetid('',18)],ir_version=8)


def compare(before,after,feeds):
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;opts.log_severity_level=3
    a=ort.InferenceSession(before.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    b=ort.InferenceSession(after.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    for left,right in zip(a.run(None,feeds),b.run(None,feeds),strict=True):
        assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()


def test_readonly_exp_is_constant_but_dynamic_math_and_parameters_stay_exact(tool):
    source=model();original=source.SerializeToString()
    result,report=tool.fold(source)
    assert source.SerializeToString()==original
    assert len(report['rewritten'])==1 and report['rewritten'][0]['node']=='static_exp'
    assert result.graph.node[1].op_type=='Constant'
    assert result.graph.initializer[0].SerializeToString()==source.graph.initializer[0].SerializeToString()
    for index in (0,2,3):assert result.graph.node[index].SerializeToString()==source.graph.node[index].SerializeToString()
    for value in (-.25,0.,.75):compare(source,result,{'x':np.full(2,value,np.float32)})


def test_overridable_parameter_is_not_frozen(tool):
    source=model(overridable=True);result,report=tool.fold(source)
    assert not report['rewritten'] and result.SerializeToString()==source.SerializeToString()
    compare(source,result,{'x':np.ones(2,np.float32),'parameter':np.asarray([-3.,.5],np.float32)})


@pytest.mark.parametrize('dtype,array',[(TensorProto.FLOAT16,np.array([.5],np.float16)),
                                      (TensorProto.DOUBLE,np.array([.5],np.float64))])
def test_other_floating_precision_is_not_silently_promoted(tool,dtype,array):
    source=model(dtype=dtype,values=array);result,report=tool.fold(source)
    assert not report['rewritten'] and result.SerializeToString()==source.SerializeToString()


def test_payload_bound_and_nonfinite_source_skip_without_clamping(tool):
    for source,limit in ((model(values=np.zeros(17,np.float32)),16),
                         (model(values=np.array([np.nan],np.float32)),1024)):
        result,report=tool.fold(source,max_elements=limit)
        assert not report['rewritten'] and result.SerializeToString()==source.SerializeToString()


def test_new_nonfinite_constant_fails_closed(tool):
    source=model(values=np.array([1000.],np.float32));original=source.SerializeToString()
    with pytest.raises(ValueError,match='nonfinite'):tool.fold(source)
    assert source.SerializeToString()==original


def test_unrelated_dead_computation_is_not_cleaned_up(tool):
    source=model();source.graph.node.append(helper.make_node('Add',['x','x'],['dead'],name='unrelated'))
    result,_=tool.fold(source)
    assert result.graph.node[-1].SerializeToString()==source.graph.node[-1].SerializeToString()


def test_scalar_readonly_parameter_uses_source_precision(tool):
    source=model(values=np.asarray(.25,np.float32));result,report=tool.fold(source)
    assert report['rewritten'][0]['shape']==[]
    compare(source,result,{'x':np.asarray(.5,np.float32)})


def test_profile_discharges_only_folded_layers_into_typed_float_literals(tool):
    source_model=model();result,report=tool.fold(source_model)
    profile=dict(schema='edgefm.trt_precision_profile/1',source_sha256=hashlib.sha256(source_model.SerializeToString()).hexdigest(),
        layer_precisions={'static_exp':'fp32','dynamic_exp':'fp32','dense':'fp16'},
        layer_output_types={'static_exp':'fp32','dynamic_exp':'fp32','dense':'fp16'},counts={'fp32_Exp':2})
    source=dict(path='/new/model.onnx',bytes=123,sha256='2'*64)
    projected=tool.project_profile(profile,report,source)
    assert projected['layer_precisions']=={'dynamic_exp':'fp32','dense':'fp16'}
    assert projected['layer_output_types']=={'dynamic_exp':'fp32','dense':'fp16'}
    assert projected['readonly_constant_fold']['typed_float_literals'][0]['tensor']=='coefficient'
    assert projected['readonly_constant_fold']['typed_float_literals'][0]['dtype']=='float32'
    assert profile['layer_output_types']['static_exp']=='fp32'
    assert profile['layer_precisions']['static_exp']=='fp32'
    assert projected['source_sha256']=='2'*64 and projected['counts']['fp32_Exp']==1
    profile['layer_precisions']['static_exp']='fp16'
    with pytest.raises(ValueError,match='explicit FP16'):tool.project_profile(profile,report,source)


def test_mismatched_profile_cannot_be_rebound(tool):
    _,report=tool.fold(model())
    with pytest.raises(ValueError,match='exact source graph'):
        tool.project_profile(dict(schema='edgefm.trt_precision_profile/1',source_sha256='a'*64),report,
                             dict(path='/new.onnx',bytes=1,sha256='b'*64))


def test_external_extent_is_bounded_and_source_descriptor_is_not_mutated(tool,tmp_path):
    source=model()
    value=source.graph.initializer[0]
    payload=value.raw_data;(tmp_path/'parameter.bin').write_bytes(payload)
    value.ClearField('raw_data');value.data_location=TensorProto.EXTERNAL
    for key,text in [('location','parameter.bin'),('offset','0'),('length',str(len(payload)))]:
        value.external_data.add(key=key,value=text)
    original=source.SerializeToString()
    result,report=tool.fold(source,external_base_dir=tmp_path)
    assert len(report['rewritten'])==1 and source.SerializeToString()==original
    assert result.graph.initializer[0].SerializeToString()==value.SerializeToString()
    value.external_data[-1].value=str(len(payload)+4)
    with pytest.raises(ValueError,match='extent'):tool.fold(source,external_base_dir=tmp_path)


def test_large_external_literal_is_not_read_to_resolve_a_constant(tool):
    source=model(values=np.zeros(2048,np.float32));value=source.graph.initializer[0]
    value.ClearField('raw_data');value.data_location=TensorProto.EXTERNAL
    value.external_data.add(key='location',value='must-not-read.bin')
    original=source.SerializeToString()
    result,report=tool.fold(source,max_elements=1024)
    assert not report['rewritten'] and result.SerializeToString()==original


def test_unknown_local_function_is_not_evaluated(tool):
    source=model();source.functions.append(helper.make_function('custom','f',[],[],[],[helper.make_opsetid('',18)]))
    with pytest.raises(ValueError,match='flat'):tool.fold(source)
