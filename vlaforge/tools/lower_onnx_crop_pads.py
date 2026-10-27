#!/usr/bin/env python3
"""Represent constant, crop-only Pad operations as value-preserving Slice."""
import copy
import argparse
import json
import math
from pathlib import Path
import tempfile


def lower(model):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if model.functions or model.training_info or any(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attribute in node.attribute):
        raise ValueError('flat inference graph required')
    version = next((item.version for item in model.opset_import if item.domain in ('', 'ai.onnx')), 0)
    if version < 13:
        raise ValueError('standard opset 13 or newer required')
    result = copy.deepcopy(model)
    ports = {value.name: value for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    tensors = {value.name: value for value in model.graph.initializer}
    producers = {value: node for node in model.graph.node for value in node.output}
    occupied = set(ports) | set(tensors)
    occupied.update(value for node in model.graph.node for value in [node.name, *node.input, *node.output])

    def integer_vector(name):
        tensor = tensors.get(name)
        node = producers.get(name)
        if (tensor is None and node is not None and node.domain in ('', 'ai.onnx')
                and node.op_type == 'Constant'):
            tensor = next((attribute.t for attribute in node.attribute if attribute.name == 'value'), None)
        if (tensor is None or tensor.data_location == onnx.TensorProto.EXTERNAL
                or tensor.data_type != onnx.TensorProto.INT64 or len(tensor.dims) != 1
                or math.prod(tensor.dims) > 128):
            return None
        return numpy_helper.to_array(tensor)

    def shape(name):
        if name in tensors:
            return tuple(tensors[name].dims)
        value = ports.get(name)
        if value is None or not value.type.tensor_type.HasField('shape'):
            return None
        dimensions = value.type.tensor_type.shape.dim
        if any(not dimension.HasField('dim_value') for dimension in dimensions):
            return None
        return tuple(dimension.dim_value for dimension in dimensions)

    def fresh(name):
        while name in occupied:
            name += '_'
        occupied.add(name)
        return name

    report = dict(schema='vlaforge.onnx_crop_pads/1', rewritten=[], skipped=[],
        changes_public_io=False, changes_data_dtype=False, changes_float_arithmetic=False,
        bounds_computed_from_runtime_shape=True,
        semantics_reference='https://onnx.ai/onnx/operators/onnx__Pad.html')
    nodes = []
    for index, node in enumerate(result.graph.node):
        if node.domain not in ('', 'ai.onnx') or node.op_type != 'Pad':
            nodes.append(node)
            continue
        reason = None
        if len(node.input) not in (2, 3) or not all(node.input[:2]) or len(node.output) != 1:
            reason = 'Pad without an axes operand required'
        else:
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            pads, dimensions = integer_vector(node.input[1]), shape(node.input[0])
            if (len(attributes) != len(node.attribute) or set(attributes) - {'mode'}
                    or attributes.get('mode', b'constant') != b'constant'):
                reason = 'reviewed constant mode required'
            elif dimensions is None or not 1 <= len(dimensions) <= 8 or any(value < 0 for value in dimensions):
                reason = 'known nonnegative profile extents and rank1..8 required'
            elif pads is None or pads.size != 2 * len(dimensions):
                reason = 'constant INT64 full-rank pads required'
            elif any(int(value) > 0 for value in pads) or not any(int(value) < 0 for value in pads):
                reason = 'crop-only pads with at least one negative extent required'
            elif any(-int(pads[axis]) - int(pads[axis + len(dimensions)]) > dimension
                     for axis, dimension in enumerate(dimensions)):
                reason = 'crop exceeds the declared input extent'
        if reason is not None:
            report['skipped'].append(dict(node=node.name, reason=reason))
            nodes.append(node)
            continue
        rank = len(dimensions)
        axes = [axis for axis in range(rank) if pads[axis] or pads[axis + rank]]
        starts = [-int(pads[axis]) for axis in axes]
        offsets = [int(pads[axis + rank]) for axis in axes]
        label = node.name or 'crop_pad_' + str(index)
        start_name, axis_name = fresh(label + '__crop_starts'), fresh(label + '__crop_axes')
        step_name = fresh(label + '__crop_steps')
        data_shape, extents = fresh(label + '__data_shape'), fresh(label + '__crop_extents')
        result.graph.initializer.extend([
            numpy_helper.from_array(np.asarray(starts, np.int64), start_name),
            numpy_helper.from_array(np.asarray(axes, np.int64), axis_name),
            numpy_helper.from_array(np.ones(len(axes), np.int64), step_name)])
        nodes.append(helper.make_node('Shape', [node.input[0]], [data_shape], name=fresh(label + '__shape')))
        nodes.append(helper.make_node('Gather', [data_shape, axis_name], [extents], axis=0, name=fresh(label + '__extents')))
        ends = extents
        if any(offsets):
            offset_name, ends = fresh(label + '__crop_offsets'), fresh(label + '__crop_ends')
            result.graph.initializer.append(numpy_helper.from_array(np.asarray(offsets, np.int64), offset_name))
            nodes.append(helper.make_node('Add', [extents, offset_name], [ends], name=fresh(label + '__ends')))
        nodes.append(helper.make_node('Slice', [node.input[0], start_name, ends, axis_name, step_name],
                                      list(node.output), name=node.name))
        report['rewritten'].append(dict(node=node.name, input_profile_shape=list(dimensions),
            source_pads=[int(value) for value in pads], crop_axes=axes, starts=starts,
            data_values_recomputed=False, constant_fill_value_unused=True))
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return result, report


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.absolute()
    report_path = output.with_suffix('.report.json')
    if source.parent != output.parent.resolve() or output.exists() or report_path.exists():
        raise ValueError('fresh sibling output required; external weights stay immutable')
    model = onnx.load_model(source, load_external_data=False)
    before = digest(source)
    weights = [(source.parent / value, digest(source.parent / value)) for value in external_files(model, source.parent)]
    result, report = lower(model)
    with tempfile.NamedTemporaryFile(prefix='.crop-pads-', suffix='.onnx', dir=source.parent) as temporary:
        staged = Path(temporary.name); staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        assert digest(source) == before and all(digest(path) == value for path, value in weights)
        report.update(source_sha256=before, output_sha256=digest(staged),
                      generator_sha256=digest(Path(__file__)), target_execution_verified=False)
        publish_files(staged, output, [])
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(rewritten=len(report['rewritten']), skipped=len(report['skipped']))))


if __name__ == '__main__':
    main()
