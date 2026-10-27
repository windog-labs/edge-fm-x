"""Share identical readonly initializer transposes with identical precision requests."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import onnx


def _require(value, message):
    if not value:
        raise ValueError(message)


def _requests(values):
    result = dict(values or {})
    _require(all(isinstance(name, str) and name and isinstance(value, str) and value
                 for name, value in result.items()), 'explicit named precision strings required')
    return result


def deduplicate(model, *, layer_precisions=None, layer_output_types=None):
    """No float evaluation, parameter repacking, arithmetic or dynamic CSE.

    Initializers also exposed as inputs are overridable and remain untouched.
    Public outputs and node names are retained; only uniquely named internal
    standard Transpose nodes can be removed. Callers with precision policies
    must supply them: missing and explicit requests are not interchangeable.
    """
    _require(not model.functions and not model.training_info and not model.graph.sparse_initializer,
             'functional flat dense-initializer graph required')
    _require(all(attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                 for node in model.graph.node for attr in node.attribute), 'nested graphs are not supported')
    _require(not model.graph.quantization_annotation, 'quantization annotations require separate alias handling')
    precision, output_types = _requests(layer_precisions), _requests(layer_output_types)
    result = copy.deepcopy(model)
    graph = result.graph
    initializers = {value.name: value for value in graph.initializer}
    _require(len(initializers) == len(graph.initializer), 'duplicate initializer names')
    public_inputs = {value.name for value in graph.input}
    public_outputs = {value.name for value in graph.output}
    counts = Counter(node.name for node in graph.node)
    canonical, aliases, node_aliases, kept, removed, skipped = {}, {}, {}, [], [], []
    widths = {onnx.TensorProto.FLOAT:4, onnx.TensorProto.FLOAT16:2, onnx.TensorProto.BFLOAT16:2,
        onnx.TensorProto.DOUBLE:8, onnx.TensorProto.INT64:8, onnx.TensorProto.INT32:4,
        onnx.TensorProto.UINT8:1, onnx.TensorProto.INT8:1, onnx.TensorProto.BOOL:1}
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
        candidate = (node.domain in ('', 'ai.onnx') and node.op_type == 'Transpose'
                     and not getattr(node, 'overload', '') and not getattr(node, 'device_configurations', [])
                     and len(node.input) == len(node.output) == 1 and node.input[0] in initializers)
        if not candidate:
            kept.append(node); continue
        weight = initializers[node.input[0]]
        if weight.name in public_inputs or not node.name or counts[node.name] != 1:
            skipped.append(dict(node=node.name, reason='overridable initializer or ambiguous node name'))
            kept.append(node); continue
        _require(all(attr.name == 'perm' and attr.type == onnx.AttributeProto.INTS for attr in node.attribute)
                 and len(node.attribute) <= 1, 'unexpected Transpose attributes')
        rank = len(weight.dims)
        permutation = tuple(node.attribute[0].ints) if node.attribute else tuple(reversed(range(rank)))
        _require(len(permutation) == rank and set(permutation) == set(range(rank)), 'invalid Transpose permutation')
        _require(all(dim >= 0 for dim in weight.dims), 'invalid initializer dimensions')
        policy = (node.name in precision, precision.get(node.name), node.name in output_types, output_types.get(node.name))
        key = (node.domain, weight.name, permutation, policy)
        previous = canonical.get(key)
        if previous is None:
            canonical[key] = node; kept.append(node); continue
        if node.output[0] in public_outputs:
            skipped.append(dict(node=node.name, reason='public output name must remain'))
            kept.append(node); continue
        aliases[node.output[0]] = previous.output[0]
        node_aliases[node.name] = previous.name
        removed.append(dict(node=node.name, canonical_node=previous.name,
            output=node.output[0], canonical_output=previous.output[0], initializer=weight.name,
            permutation=list(permutation), initializer_dtype=weight.data_type,
            logical_output_bytes=math.prod(weight.dims) * widths.get(weight.data_type, 0),
            precision_request=precision.get(node.name), output_type_request=output_types.get(node.name)))
    graph.ClearField('node'); graph.node.extend(kept)
    info = [value for value in graph.value_info if value.name not in aliases]
    graph.ClearField('value_info'); graph.value_info.extend(info)
    for field in ('input', 'output', 'initializer'):
        _require([value.SerializeToString() for value in getattr(graph, field)]
                 == [value.SerializeToString() for value in getattr(model.graph, field)], 'public I/O or initializer changed')
    report = dict(schema='vlaforge.readonly_initializer_transpose_sharing/1',
        source_node_count=len(model.graph.node), output_node_count=len(graph.node), removed_node_count=len(removed),
        node_aliases=node_aliases, value_aliases=aliases, removed=removed, skipped=skipped,
        public_io_proto_unchanged=True, initializer_proto_unchanged=True,
        floating_values_evaluated=False, external_payloads_read=False,
        arithmetic_nodes_added_or_removed=False, casts_added_or_removed=False,
        logical_redundant_transpose_bytes=sum(row['logical_output_bytes'] for row in removed),
        logical_bytes_are_not_measured_peak_memory=True,
        target_execution_or_numerical_certification=False)
    return result, report


def project_requests(profile, report):
    """An eliminated request must be satisfied by an identical canonical one."""
    result = copy.deepcopy(profile)
    lineage = {}
    for key in ('layer_precisions', 'layer_output_types'):
        old = _requests(profile.get(key)); new = dict(old); names = {}
        for removed, canonical in report['node_aliases'].items():
            _require((removed in old) == (canonical in old), 'alias policy presence differs')
            if removed in old:
                _require(old[removed] == old[canonical], 'alias policy value differs')
                del new[removed]; names[removed] = canonical
        result[key] = new; lineage[key] = names
    _require(not any(row.get('name') in report['node_aliases'] for row in profile.get('source_dtype_ledger', [])),
             'source dtype ledger entry needs explicit alias support')
    result['equivalent_initializer_transpose_request_aliases'] = lineage
    return result


def _pin(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision-profile', type=Path)
    parser.add_argument('--output-profile', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    _require((args.precision_profile is None) == (args.output_profile is None), 'input/output precision paths must be paired')
    _require(not args.output.exists() and not args.report.exists()
             and (args.output_profile is None or not args.output_profile.exists()), 'fresh outputs required')
    model = onnx.load(args.input, load_external_data=False)
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in tensors):
        _require(args.input.resolve().parent == args.output.resolve().parent, 'external tensors require their unchanged base directory')
    profile = json.loads(args.precision_profile.read_text()) if args.precision_profile else {}
    source_pin = _pin(args.input)
    if args.precision_profile is not None:
        _require(isinstance(profile, dict) and 'layer_precisions' in profile and 'layer_output_types' in profile
                 and profile.get('source_sha256') == source_pin['sha256'], 'complete precision profile for this source required')
    result, report = deduplicate(model, layer_precisions=profile.get('layer_precisions'),
                                 layer_output_types=profile.get('layer_output_types'))
    projected = project_requests(profile, report) if args.precision_profile is not None else None
    with args.output.open('xb') as stream: stream.write(result.SerializeToString())
    onnx.checker.check_model(str(args.output))
    if projected is not None:
        projected.update(source_model=_pin(args.output), source_sha256=_pin(args.output)['sha256'],
                         parent_profile=_pin(args.precision_profile), correctness_verified=False)
        with args.output_profile.open('x') as stream: json.dump(projected, stream, indent=2); stream.write('\n')
    report.update(source=_pin(args.input), output=_pin(args.output), tool=_pin(__file__),
                  precision_profile=_pin(args.precision_profile) if args.precision_profile else None,
                  projected_profile=_pin(args.output_profile) if args.output_profile else None)
    with args.report.open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({key:report[key] for key in ('source_node_count','output_node_count','removed_node_count',
                                                'logical_redundant_transpose_bytes')}))


if __name__ == '__main__':
    main()
