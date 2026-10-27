"""Independent shape proofs for explicitly reviewed, pure TensorRT plugins.

Shadow Identity nodes are used only for inference. Deployment nodes and tensor
payloads are copied unchanged; no shadow is an executable replacement.
"""
from __future__ import annotations

from collections import Counter
import copy
import math

CONTRACTS = {
    ('vlaforge.trt', 'VLAForgeBFloat16Round'): 'bf16-round/1',
    ('vlaforge.trt', 'VLAForgeBFloat16MatMul'): 'bf16-matmul/1',
    ('trt.plugins', 'LayerNorm'): 'layernorm/1',
}


def infer_reviewed_shapes(model, trusted_plugins):
    import onnx
    from onnx import helper

    if isinstance(trusted_plugins, (str, bytes)):
        raise ValueError('explicit sequence of reviewed plugin contracts required')
    selected = tuple(trusted_plugins)
    if len(set(selected)) != len(selected) or set(selected) - set(CONTRACTS.values()):
        raise ValueError('unknown or duplicate reviewed plugin contract')
    shadow = copy.deepcopy(model)
    shadow.graph.ClearField('value_info')
    versions = {item.domain: item.version for item in model.opset_import}
    if len(versions) != len(model.opset_import):
        raise ValueError('unique opset imports required')
    reviewed = []
    for index, node in enumerate(model.graph.node):
        if node.domain in ('', 'ai.onnx'):
            continue
        contract = CONTRACTS.get((node.domain, node.op_type))
        if contract not in selected or versions.get(node.domain) != 1:
            raise ValueError('functional partition requires an explicitly reviewed plugin contract')
        if len({a.name for a in node.attribute}) != len(node.attribute) or any(a.ref_attr_name for a in node.attribute):
            raise ValueError('ambiguous plugin attributes')
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        keys = {'plugin_version', 'plugin_namespace'} | ({'epsilon'} if contract == 'layernorm/1' else set())
        if (set(attrs) != keys or attrs['plugin_version'] != b'1' or attrs['plugin_namespace'] != b''
                or len(node.output) != 1 or len(node.input) != (3 if contract == 'layernorm/1' else 2 if contract == 'bf16-matmul/1' else 1)):
            raise ValueError('reviewed plugin arity/namespace/version/attributes differ')
        if contract == 'layernorm/1' and (not isinstance(attrs['epsilon'], float)
                or not math.isfinite(attrs['epsilon']) or attrs['epsilon'] <= 0):
            raise ValueError('finite positive FLOAT LayerNorm epsilon required')
        shadow.graph.node[index].CopyFrom(helper.make_node(
            'MatMul' if contract == 'bf16-matmul/1' else 'Identity',
            list(node.input) if contract == 'bf16-matmul/1' else [node.input[0]], list(node.output), name=node.name))
        reviewed.append((node, contract))
    inferred = onnx.shape_inference.infer_shapes(shadow, strict_mode=True, data_prop=True)
    info = {v.name: v for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    for tensor in inferred.graph.initializer:
        info[tensor.name] = helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims))
    declared = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}

    def typed(name):
        value = info.get(name)
        if (value is None or value.type.WhichOneof('value') != 'tensor_type'
                or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or not value.type.tensor_type.HasField('shape')):
            raise ValueError('reviewed plugin requires independently inferred FLOAT tensor types')
        tensor = value.type.tensor_type
        original = declared.get(name)
        if original is not None and (original.type.WhichOneof('value') != 'tensor_type'
                or original.type.tensor_type.elem_type not in (0, tensor.elem_type)):
            raise ValueError('source annotation contradicts reviewed plugin FLOAT contract')
        return tensor

    counts = Counter()
    for node, contract in reviewed:
        x, y = typed(node.input[0]), typed(node.output[0])
        if contract == 'bf16-matmul/1':
            import numpy as np
            right = typed(node.input[1])
            tensors = (x, right, y)
            if any(not 2 <= len(t.shape.dim) <= 8 or any(
                    not d.HasField('dim_value') or not 0 < d.dim_value <= 2147483647 for d in t.shape.dim) for t in tensors):
                raise ValueError('BF16 MatMul shape proof requires static positive rank2..8 tensors')
            left_shape, right_shape, output_shape = [tuple(d.dim_value for d in t.shape.dim) for t in tensors]
            if left_shape[-1] != right_shape[-2]:
                raise ValueError('BF16 MatMul contraction dimensions differ')
            expected = (*np.broadcast_shapes(left_shape[:-2], right_shape[:-2]), left_shape[-2], right_shape[-1])
            if output_shape != expected:
                raise ValueError('BF16 MatMul shadow output does not match broadcast/contraction rules')
            counts[contract] += 1
            continue
        if x.shape != y.shape or len(x.shape.dim) > 8:
            raise ValueError('plugin shadow did not preserve its admitted tensor shape')
        if any(d.HasField('dim_value') and d.dim_value <= 0 for d in x.shape.dim):
            raise ValueError('reviewed plugin does not admit empty tensor extents')
        if contract == 'layernorm/1':
            dimensions = x.shape.dim
            if len(dimensions) != 3 or not dimensions[-1].HasField('dim_value') or dimensions[-1].dim_value <= 0:
                raise ValueError('LayerNorm plugin requires rank three and proved channel width')
            for parameter in node.input[1:]:
                t = typed(parameter)
                if (len(t.shape.dim) != 1 or not t.shape.dim[0].HasField('dim_value')
                        or t.shape.dim[0].dim_value != dimensions[-1].dim_value):
                    raise ValueError('LayerNorm gamma/beta width differs from proved channels')
        counts[contract] += 1
    result = copy.deepcopy(model)
    result.graph.ClearField('value_info')
    result.graph.value_info.extend(inferred.graph.value_info)
    # Refinements come only from actual static inputs, initializer descriptors
    # and standard shape inference after validating the reviewed shape rules.
    for old, proven in zip(result.graph.output, inferred.graph.output, strict=True):
        old.type.CopyFrom(proven.type)
    return result, {'plugin_counts': dict(counts), 'internal_annotations_seeded': False,
                    'strict_mode': True, 'data_prop': True, 'deployment_nodes_rewritten': False,
                    'trusted_contracts': list(selected), 'numerical_equivalence_verified': False}
