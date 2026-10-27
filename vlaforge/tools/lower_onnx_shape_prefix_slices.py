#!/usr/bin/env python3
"""Lower proved shape-derived vector prefix slices to bounded Range/Gather."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile


def lower(model):
    """Return a metadata-only copy; unproved signs and unsupported slices stay put."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    if model.functions or model.training_info or any(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attribute in node.attribute):
        raise ValueError('flat inference graph required')
    version = next((op.version for op in model.opset_import if op.domain in ('', 'ai.onnx')), 0)
    if version < 13:
        raise ValueError('standard opset 13 or newer required')
    result = copy.deepcopy(model)
    producers = {value: node for node in model.graph.node for value in node.output}
    tensors = {value.name: value for value in model.graph.initializer}
    ports = {value.name: value for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    occupied = set(ports) | set(tensors)
    occupied.update((value for node in model.graph.node for value in [node.name, *node.input, *node.output]))
    arrays = {}

    def array(name):
        if name in arrays:
            return arrays[name]
        tensor = tensors.get(name)
        producer = producers.get(name)
        if (tensor is None and producer is not None
                and producer.domain in ('', 'ai.onnx') and producer.op_type == 'Constant'):
            tensor = next((attribute.t for attribute in producer.attribute if attribute.name == 'value'), None)
        value = None
        if (tensor is not None and tensor.data_location != onnx.TensorProto.EXTERNAL
                and tensor.data_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64)
                and math.prod(tensor.dims) <= 4096):
            value = numpy_helper.to_array(tensor)
        arrays[name] = value
        return value
    moves = {'Identity', 'Gather', 'Slice', 'Reshape', 'Transpose', 'Unsqueeze', 'Squeeze', 'Flatten'}
    dtype_cache = {}

    def dtype(name, trail=frozenset()):
        if name in dtype_cache:
            return dtype_cache[name]
        if name in trail or len(trail) > 128:
            return None
        trail = trail | {name}
        node = producers.get(name)
        value = None
        if name in tensors:
            value = tensors[name].data_type
        elif node is not None and node.domain in ('', 'ai.onnx'):
            if node.op_type == 'Shape':
                value = onnx.TensorProto.INT64
            elif node.op_type == 'Cast':
                value = next((attribute.i for attribute in node.attribute if attribute.name == 'to'), None)
            elif node.op_type in moves and node.input:
                value = dtype(node.input[0], trail)
            elif node.op_type in {'Concat', 'Min', 'Max'}:
                kinds = {dtype(operand, trail) for operand in node.input}
                value = next(iter(kinds)) if len(kinds) == 1 else None
            elif node.op_type == 'Constant':
                constant = array(name)
                if constant is not None:
                    value = onnx.TensorProto.INT64 if constant.dtype == np.int64 else onnx.TensorProto.INT32
        elif name in ports:
            value = ports[name].type.tensor_type.elem_type
        dtype_cache[name] = value
        return value
    positive_cache = {}

    def nonnegative(name, trail=frozenset()):
        """Track sign through shape producers and value-preserving integer views."""
        if name in positive_cache:
            return positive_cache[name]
        if name in trail or len(trail) > 128 or dtype(name) != onnx.TensorProto.INT64:
            return False
        trail = trail | {name}
        constant = array(name)
        node = producers.get(name)
        proved = False
        if constant is not None:
            proved = bool(np.all(constant >= 0))
        elif node is not None and node.domain in ('', 'ai.onnx'):
            if node.op_type == 'Shape':
                proved = True
            elif node.op_type in moves and node.input:
                proved = nonnegative(node.input[0], trail)
            elif node.op_type == 'Cast' and node.input and (dtype(node.input[0]) == onnx.TensorProto.INT64):
                proved = nonnegative(node.input[0], trail)
            elif node.op_type in {'Concat', 'Min'}:
                proved = bool(node.input) and all((nonnegative(operand, trail) for operand in node.input))
            elif node.op_type == 'Max':
                proved = any((nonnegative(operand, trail) for operand in node.input))
        positive_cache[name] = proved
        return proved

    def already_bounded(node):
        minimum = producers.get(node.input[2])
        if minimum is None or minimum.domain not in ('', 'ai.onnx') or minimum.op_type != 'Min':
            return False
        for operand in minimum.input:
            gather = producers.get(operand)
            if (gather is None or gather.domain not in ('', 'ai.onnx')
                    or gather.op_type != 'Gather' or len(gather.input) != 2
                    or gather.input[1] != node.input[3]
                    or any(attribute.name == 'axis' and attribute.i != 0 for attribute in gather.attribute)):
                continue
            shape = producers.get(gather.input[0])
            if (shape is not None and shape.domain in ('', 'ai.onnx')
                    and shape.op_type == 'Shape' and list(shape.input) == [node.input[0]]
                    and not shape.attribute):
                return True
        return False

    def fresh(name):
        while name in occupied:
            name += '_'
        occupied.add(name)
        return name
    report = dict(
        schema='vlaforge.onnx_shape_prefix_slices/1', rewritten=[], skipped=[],
        changes_public_io=False, changes_data_dtype=False, changes_float_arithmetic=False,
        bounds_computed_from_runtime_shape=True,
        semantics_reference='https://onnx.ai/onnx/operators/onnx__Slice.html')
    nodes = []
    for index, node in enumerate(result.graph.node):
        if node.domain not in ('', 'ai.onnx') or node.op_type != 'Slice':
            nodes.append(node)
            continue
        reason = None
        if len(node.input) not in (4, 5) or not all(node.input[:4]) or len(node.output) != 1:
            reason = 'explicit axes and four or five inputs required'
        else:
            axes = array(node.input[3])
            starts_type = dtype(node.input[1])
            end_type = dtype(node.input[2])
            port = ports.get(node.input[0])
            tensor = tensors.get(node.input[0])
            if tensor is not None:
                rank = len(tensor.dims)
            elif port is not None and port.type.tensor_type.HasField('shape'):
                rank = len(port.type.tensor_type.shape.dim)
            else:
                rank = None
            if (axes is None or axes.dtype != np.int64 or axes.ndim != 1 or axes.size == 0
                    or rank is None or rank == 0
                    or any(not -rank <= int(axis) < rank for axis in axes)
                    or len({int(axis) % rank for axis in axes}) != axes.size):
                reason = 'known data rank and constant distinct valid INT64 axes required'
            elif starts_type != onnx.TensorProto.INT64 or end_type != onnx.TensorProto.INT64:
                reason = 'INT64 index operands required; no implicit narrowing'
            elif len(node.input) == 5 and node.input[4] and (
                    (steps := array(node.input[4])) is None or steps.dtype != np.int64
                    or steps.ndim != 1 or steps.size != axes.size or np.any(steps <= 0)):
                reason = 'known strictly positive steps required'
            elif (rank != 1 or axes.tolist() not in ([0], [-1]) or array(node.input[1]) is None
                    or array(node.input[1]).tolist() != [0]
                    or (len(node.input) == 5 and node.input[4] and array(node.input[4]).tolist() != [1])
                    or node.attribute):
                reason = 'rank-one zero-start unit-step prefix Slice required'
            elif array(node.input[2]) is not None:
                reason = 'constant ends do not need dynamic bounds'
            elif already_bounded(node):
                reason = 'runtime end bound already explicit'
            elif not nonnegative(node.input[2]):
                reason = 'nonnegative dynamic ends not structurally proved'
        if reason is not None:
            report['skipped'].append(dict(node=node.name, reason=reason))
            nodes.append(node)
            continue
        label = node.name or 'slice_' + str(index)
        source_end = node.input[2]
        shape_name = fresh(label + '__vf_data_shape')
        extent = fresh(label + '__vf_axis_extents')
        bounded = fresh(label + '__vf_bounded_ends')
        zero, one = (fresh(label + '__range_zero'), fresh(label + '__range_one'))
        end, indices = (fresh(label + '__end_scalar'), fresh(label + '__range_indices'))
        result.graph.initializer.extend([
            numpy_helper.from_array(np.asarray(0, np.int64), zero),
            numpy_helper.from_array(np.asarray(1, np.int64), one)])
        # Derive the clamp from runtime Shape, never a sampled extent annotation.
        nodes.append(helper.make_node('Shape', [node.input[0]], [shape_name], name=fresh(label + '__vf_shape')))
        nodes.append(helper.make_node('Gather', [shape_name, node.input[3]], [extent], name=fresh(label + '__vf_gather_bounds'), axis=0))
        nodes.append(helper.make_node('Min', [source_end, extent], [bounded], name=fresh(label + '__vf_clamp_ends')))
        nodes.append(helper.make_node('Gather', [bounded, zero], [end], axis=0, name=fresh(label + '__end_scalar_gather')))
        nodes.append(helper.make_node('Range', [zero, end, one], [indices], name=fresh(label + '__prefix_indices')))
        nodes.append(helper.make_node('Gather', [node.input[0], indices], list(node.output), axis=0, name=node.name))
        report['rewritten'].append(dict(
            node=node.name, original_ends=source_end, bounded_ends=bounded,
            unchanged_inputs=[node.input[0], node.input[1], node.input[3], *list(node.input[4:])],
            positive_steps=True, nonnegative_end_proved=True, prefix_gather_replacement=True,
            original_output_name_preserved=True))
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return (result, report)


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = (args.input.resolve(strict=True), args.output.absolute())
    report_path = output.with_suffix('.report.json')
    if source.parent != output.parent.resolve() or output.exists() or report_path.exists():
        raise ValueError('fresh sibling output required; external weights stay immutable')
    model = onnx.load_model(source, load_external_data=False)
    weights = [dict(location=str(path), bytes=(source.parent / path).stat().st_size,
                    sha256=digest(source.parent / path))
               for path in external_files(model, source.parent)]
    before = digest(source)
    result, report = lower(model)
    with tempfile.NamedTemporaryFile(prefix='.explicit-slice-', suffix='.onnx', dir=source.parent) as temporary:
        staged = Path(temporary.name)
        staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        assert digest(source) == before
        assert all((digest(source.parent / value['location']) == value['sha256'] for value in weights))
        report.update(source_sha256=before, output_sha256=digest(staged), external_data=weights,
                      generator_sha256=digest(Path(__file__)), target_execution_verified=False)
        publish_files(staged, output, [])
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(rewritten=len(report['rewritten']), skipped=len(report['skipped']))))


if __name__ == '__main__':
    main()
