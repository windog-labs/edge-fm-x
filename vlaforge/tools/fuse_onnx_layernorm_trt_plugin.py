#!/usr/bin/env python3
"""Replace proven last-axis FLOAT LayerNorm primitives with a TensorRT plugin."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def rewrite_layer_norms(model, *, plugin=True, base_dir=None):
    """Match dataflow, not node names; require static typed FLOAT interfaces.

    The deployed plugin always receives rank-three data, avoiding older
    LayerNorm plugin releases' rank-dependent axis convention. ``plugin=False``
    emits standard ONNX LayerNormalization for independent CPU shadow checks.
    """
    import onnx
    import numpy as np
    from onnx import helper, numpy_helper

    if model.functions or model.training_info or any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node for attr in node.attribute
    ):
        raise ValueError("LayerNorm rewrite requires a flat inference graph")
    if next((item.version for item in model.opset_import if item.domain in ('', 'ai.onnx')), 0) < 17:
        raise ValueError("LayerNorm rewrite requires ONNX opset 17 or later")
    nodes = list(model.graph.node)
    producers, consumers = {}, {}
    for index, node in enumerate(nodes):
        for name in node.output:
            if name in producers:
                raise ValueError("LayerNorm rewrite requires unique SSA producers")
            producers[name] = index
        for name in node.input:
            consumers.setdefault(name, set()).add(index)
    ports = {value.name: value for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    inferred_ports = None

    def typed(name):
        nonlocal inferred_ports
        if name in ports:
            tensor = ports[name].type.tensor_type
            if any(not dim.HasField('dim_value') or dim.dim_value <= 0 for dim in tensor.shape.dim):
                if name in graph_inputs:
                    return None
                if inferred_ports is None:
                    shape_shadow = copy.deepcopy(model)
                    for candidate in shape_shadow.graph.node:
                        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in candidate.attribute}
                        if (candidate.domain == 'vlaforge.trt' and candidate.op_type == 'VLAForgeBFloat16Round'
                                and len(candidate.input) == len(candidate.output) == 1
                                and attributes == {'plugin_version': b'1', 'plugin_namespace': b''}):
                            candidate.CopyFrom(helper.make_node('Identity', list(candidate.input),
                                                                list(candidate.output), name=candidate.name))
                    inferred = onnx.shape_inference.infer_shapes(shape_shadow, strict_mode=True, data_prop=True)
                    inferred_ports = {value.name: value for value in
                                      [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
                proven = inferred_ports.get(name)
                if proven is None or proven.type.tensor_type.elem_type != tensor.elem_type:
                    return None
                tensor = proven.type.tensor_type
                if any(not dim.HasField('dim_value') or dim.dim_value <= 0 for dim in tensor.shape.dim):
                    return None
            return tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim)
        if name in constants:
            tensor = constants[name]
            return tensor.data_type, tuple(tensor.dims)
        return None

    def operation(value, kind):
        index = producers.get(value)
        if index is None:
            return None
        node = nodes[index]
        return index if node.domain in ('', 'ai.onnx') and node.op_type == kind and len(node.output) == 1 else None

    def axes(index, rank):
        node = nodes[index]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if attrs.get('keepdims', 1) != 1:
            return None
        values = attrs.get('axes')
        if len(node.input) == 2:
            tensor = constants.get(node.input[1])
            if tensor is None or node.input[1] in graph_inputs or tensor.data_type != onnx.TensorProto.INT64:
                return None
            values = numpy_helper.to_array(copy.deepcopy(tensor), base_dir=base_dir).tolist()
        if values is None or len(values) != 1 or not -rank <= values[0] < rank:
            return None
        return values[0] % rank

    def match(last_index):
        last = nodes[last_index]
        if (last.domain in ('', 'ai.onnx') and last.op_type == 'LayerNormalization'
                and len(last.input) == 3 and len(last.output) == 1):
            attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in last.attribute}
            interface = typed(last.input[0])
            if interface is None or interface[0] != onnx.TensorProto.FLOAT:
                return None
            shape = interface[1]
            axis, epsilon = attrs.get('axis', -1), attrs.get('epsilon', 1e-5)
            if (not shape or len(shape) > 8 or any(dimension <= 0 for dimension in shape)
                    or not -len(shape) <= axis < len(shape) or axis % len(shape) != len(shape) - 1
                    or set(attrs) - {'axis', 'epsilon', 'stash_type'} or attrs.get('stash_type', 1) != 1
                    or not np.isfinite(epsilon) or epsilon <= 0
                    or typed(last.input[1]) != (onnx.TensorProto.FLOAT, (shape[-1],))
                    or typed(last.input[2]) != (onnx.TensorProto.FLOAT, (shape[-1],))
                    or typed(last.output[0]) not in (None, interface)):
                return None
            return {'indices': {last_index}, 'internal': set(), 'x': last.input[0],
                    'gamma': last.input[1], 'beta': last.input[2], 'shape': shape,
                    'epsilon': float(epsilon), 'output': last.output[0]}
        if last.domain not in ('', 'ai.onnx') or last.op_type != 'Add' or len(last.input) != 2 or len(last.output) != 1:
            return None
        for scale_value, beta in (last.input, list(reversed(last.input))):
            scale_index = operation(scale_value, 'Mul')
            if scale_index is None or len(nodes[scale_index].input) != 2:
                continue
            for normalized, gamma in (nodes[scale_index].input, list(reversed(nodes[scale_index].input))):
                divide_index = operation(normalized, 'Div')
                if divide_index is None or len(nodes[divide_index].input) != 2:
                    continue
                centered, deviation = nodes[divide_index].input
                centered_index, sqrt_index = operation(centered, 'Sub'), operation(deviation, 'Sqrt')
                if centered_index is None or sqrt_index is None or len(nodes[centered_index].input) != 2 or len(nodes[sqrt_index].input) != 1:
                    continue
                x, mean_value = nodes[centered_index].input
                mean_index = operation(mean_value, 'ReduceMean')
                add_index = operation(nodes[sqrt_index].input[0], 'Add')
                if mean_index is None or add_index is None or nodes[mean_index].input[0] != x or len(nodes[add_index].input) != 2:
                    continue
                interface = typed(x)
                if interface is None or interface[0] != onnx.TensorProto.FLOAT:
                    continue
                shape = interface[1]
                if (not shape or len(shape) > 8 or any(dim <= 0 for dim in shape)
                        or axes(mean_index, len(shape)) != len(shape) - 1
                        or typed(last.output[0]) not in (None, interface)):
                    continue
                if typed(gamma) != (onnx.TensorProto.FLOAT, (shape[-1],)) or typed(beta) != typed(gamma):
                    continue
                for variance, epsilon_name in (nodes[add_index].input, list(reversed(nodes[add_index].input))):
                    variance_index = operation(variance, 'ReduceMean')
                    epsilon_tensor = constants.get(epsilon_name)
                    if variance_index is None or epsilon_tensor is None or epsilon_name in graph_inputs:
                        continue
                    if (epsilon_tensor.data_type != onnx.TensorProto.FLOAT
                            or math.prod(epsilon_tensor.dims) != 1 or len(epsilon_tensor.dims) > len(shape)):
                        continue
                    square_index = operation(nodes[variance_index].input[0], 'Mul')
                    if (square_index is None or list(nodes[square_index].input) != [centered, centered]
                            or axes(variance_index, len(shape)) != len(shape) - 1):
                        continue
                    epsilon = float(numpy_helper.to_array(copy.deepcopy(epsilon_tensor), base_dir=base_dir).reshape(-1)[0])
                    if not np.isfinite(epsilon) or epsilon <= 0:
                        continue
                    selected = {last_index, scale_index, divide_index, centered_index, sqrt_index,
                                mean_index, add_index, variance_index, square_index}
                    if len(selected) != 9:
                        continue
                    internal = {value for index in selected - {last_index} for value in nodes[index].output}
                    if any(value in graph_outputs or not consumers.get(value, set()) <= selected for value in internal):
                        continue
                    return {'indices': selected, 'internal': internal, 'x': x, 'gamma': gamma,
                            'beta': beta, 'shape': shape, 'epsilon': epsilon, 'output': last.output[0]}
        return None

    result = copy.deepcopy(model)
    occupied = set(producers) | set(constants) | set(ports) | {node.name for node in nodes}

    def unique(prefix):
        name = prefix
        while name in occupied:
            name += '_'
        occupied.add(name)
        return name

    removed, replacements, internals, records = set(), {}, set(), []
    for index in range(len(nodes)):
        found = match(index)
        if found is None or found['indices'] & removed:
            continue
        shape = found['shape']
        if math.prod(shape) > 2**31 - 1:
            continue
        name = unique('vf_layer_norm_' + str(len(records)))
        operands = [found[key] for key in ('x', 'gamma', 'beta')]
        output, additions = found['output'], []
        if plugin and len(shape) != 3:
            reshape_shape = unique(name + '_input_shape')
            restore_shape = unique(name + '_output_shape')
            reshaped = unique(name + '_input')
            plugin_output = unique(name + '_output')
            result.graph.initializer.extend([
                numpy_helper.from_array(np.asarray([1, math.prod(shape[:-1]), shape[-1]], dtype=np.int64), reshape_shape),
                numpy_helper.from_array(np.asarray(shape, dtype=np.int64), restore_shape)])
            additions.append(helper.make_node('Reshape', [operands[0], reshape_shape], [reshaped], name=unique(name + '_reshape')))
            operands[0], output = reshaped, plugin_output
        node = helper.make_node('LayerNorm' if plugin else 'LayerNormalization', operands, [output],
            name=name, domain='trt.plugins' if plugin else '', epsilon=found['epsilon'],
            **({'plugin_version': '1', 'plugin_namespace': ''} if plugin else {'axis': -1}))
        additions.append(node)
        if output != found['output']:
            additions.append(helper.make_node('Reshape', [output, restore_shape], [found['output']], name=unique(name + '_restore')))
        replacements[index] = additions
        removed.update(found['indices'])
        internals.update(found['internal'])
        records.append({'replacement': name, 'source_nodes': [nodes[item].name for item in sorted(found['indices'])],
                        'source_node_indices': sorted(found['indices']), 'epsilon': found['epsilon'],
                        'input_shape': list(shape), 'output': found['output'], 'reshaped_rank_three': len(shape) != 3 and plugin})
    result.graph.ClearField('node')
    for index, node in enumerate(nodes):
        if index in replacements:
            result.graph.node.extend(replacements[index])
        elif index not in removed:
            result.graph.node.append(node)
    retained_info = [value for value in result.graph.value_info if value.name not in internals]
    result.graph.ClearField('value_info')
    result.graph.value_info.extend(retained_info)
    if plugin and records:
        domain = next((item for item in result.opset_import if item.domain == 'trt.plugins'), None)
        if domain is not None and domain.version != 1:
            raise ValueError("conflicting TensorRT plugin opset")
        if domain is None:
            result.opset_import.append(helper.make_opsetid('trt.plugins', 1))
    return result, {'schema': 'vlaforge.trt_layer_norm_rewrite/1',
        'status': 'rewritten-not-target-qualified', 'plugin': plugin, 'rewritten': len(records), 'groups': records,
        'scope': 'typed static last-axis FLOAT LayerNorm only; weights/epsilon unchanged, numerical verification required',
        'plugin_contract': {'name': 'LayerNorm', 'version': '1', 'namespace': '',
                            'rank_three_input': True, 'required_precision': 'fp32'} if plugin else None}


def main():
    import onnx
    import os
    import tempfile
    from export_torchscript_region_onnx import digest, external_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--standard-reference', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('LayerNorm rewrite never overwrites evidence')
    source_pin = digest(args.input)
    model = onnx.load(args.input, load_external_data=False)
    sidecars = external_files(model, args.input.resolve().parent)
    if sidecars and args.output.resolve().parent != args.input.resolve().parent:
        raise ValueError('external-weight rewrites require a sibling output graph; payloads are never silently relocated')
    external_state = [{'location': str(path), 'size': (args.input.parent / path).stat().st_size,
                       'sha256': digest(args.input.parent / path)} for path in sidecars]
    result, report = rewrite_layer_norms(model, plugin=not args.standard_reference, base_dir=str(args.input.parent))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.layernorm-', suffix='.onnx', dir=args.output.parent) as staged:
        Path(staged.name).write_bytes(result.SerializeToString())
        onnx.checker.check_model(staged.name)
        if digest(args.input) != source_pin or any(
                digest(args.input.parent / row['location']) != row['sha256'] for row in external_state):
            raise ValueError('source graph or external weights changed during LayerNorm preparation')
        os.link(staged.name, args.output)
    try:
        report.update(source_sha256=source_pin, output_sha256=digest(args.output),
                      output=str(args.output.resolve()), generator_sha256=digest(Path(__file__)),
                      external_data=external_state)
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    except Exception:
        args.output.unlink()
        raise
    print(json.dumps({'rewritten': report['rewritten'], 'output_sha256': report['output_sha256']}))


if __name__ == '__main__':
    main()
