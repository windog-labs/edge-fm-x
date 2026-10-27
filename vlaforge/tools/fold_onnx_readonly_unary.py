#!/usr/bin/env python3
"""Evaluate bounded readonly FLOAT Exp expressions with the existing CPU reference.

Only immutable initializers/Constants and identity FLOAT Cast/Identity aliases
are eligible. Runtime data, other precisions and source parameters are untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def fold(model, *, external_base_dir=None, max_elements=1024):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    if type(max_elements) is not int or not 1 <= max_elements <= 1 << 20:
        raise ValueError("bounded positive constant evaluation budget required")
    forbidden = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                 "Bernoulli", "Multinomial", "Dropout", "QuantizeLinear", "DequantizeLinear"}
    if (model.functions or model.training_info or model.graph.sparse_initializer
            or model.graph.quantization_annotation or any(node.op_type in forbidden or any(
                attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attr in node.attribute) for node in model.graph.node)):
        raise ValueError("flat dense inference graph without local functions or quantization required")
    inputs = {value.name for value in model.graph.input}
    initializers = {value.name: value for value in model.graph.initializer}
    if len(initializers) != len(model.graph.initializer):
        raise ValueError("unique initializer names required")
    producers = {}
    for node in model.graph.node:
        for value in node.output:
            if value and (value in producers or value in initializers or value in inputs):
                raise ValueError("unique source value definitions required")
            if value:
                producers[value] = node
    evaluated_nodes, leaves, known, visiting = {}, {}, {}, set()

    def leaf(tensor, name):
        if (tensor.data_type != onnx.TensorProto.FLOAT or any(dim <= 0 for dim in tensor.dims)
                or math.prod(tensor.dims) > max_elements):
            return None
        tensor = copy.deepcopy(tensor)
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            if external_base_dir is None:
                raise ValueError("readonly external parameter requires its explicit base directory")
            entries = {item.key: item.value for item in tensor.external_data}
            location_text = entries.get("location", "")
            location = Path(location_text)
            if (len(entries) != len(tensor.external_data) or not location_text
                    or location.is_absolute() or ".." in location.parts):
                raise ValueError("readonly external parameter location must stay inside its base")
            base = Path(external_base_dir).resolve(strict=True)
            original = base / location
            if any(path.is_symlink() for path in [original, *original.parents] if path != base.parent):
                raise ValueError("readonly external parameters must not use symlinks")
            original.resolve(strict=True).relative_to(base)
            expected_bytes = math.prod(tensor.dims) * 4
            offset_text = entries.get("offset", "0")
            length_text = entries.get("length")
            if not offset_text.isdecimal() or (length_text is not None and not length_text.isdecimal()):
                raise ValueError("bounded decimal external parameter extents required")
            offset = int(offset_text)
            length = int(length_text) if length_text is not None else original.stat().st_size - offset
            if length != expected_bytes or offset < 0 or offset + length > original.stat().st_size:
                raise ValueError("external readonly tensor extent disagrees with its bounded FLOAT shape")
        array = numpy_helper.to_array(tensor, base_dir=str(external_base_dir) if external_base_dir is not None else "")
        if array.dtype != np.dtype("float32") or not np.isfinite(array).all():
            return None
        leaves[name] = numpy_helper.from_array(array, name)
        return tuple(array.shape)

    def resolve(name):
        if name in inputs:
            return None
        if name in known:
            return known[name]
        if name in visiting:
            raise ValueError("cyclic source expression")
        visiting.add(name)
        shape = None
        if name in initializers:
            shape = leaf(initializers[name], name)
        else:
            node = producers.get(name)
            if node is not None and node.domain in ("", "ai.onnx") and len(node.output) == 1:
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                canonical = len(attrs) == len(node.attribute) and not any(a.ref_attr_name for a in node.attribute)
                if canonical and node.op_type == "Constant" and not node.input and set(attrs) == {"value"}:
                    shape = leaf(attrs["value"], name)
                elif (canonical and len(node.input) == 1 and (node.op_type == "Exp" and not attrs
                        or node.op_type == "Identity" and not attrs
                        or node.op_type == "Cast" and attrs == {"to": onnx.TensorProto.FLOAT})):
                    shape = resolve(node.input[0])
                    if shape is not None:
                        evaluated_nodes[name] = node
        visiting.remove(name)
        known[name] = shape
        return shape

    selected = []
    for node in model.graph.node:
        if node.op_type == "Exp" and len(node.output) == 1 and resolve(node.output[0]) is not None:
            selected.append(node)
    result = copy.deepcopy(model)
    report = dict(schema="vlaforge.readonly_float_unary_folding/1", status="unchanged-no-eligible-expression",
                  source_model_proto_sha256=hashlib.sha256(model.SerializeToString()).hexdigest(),
                  rewritten=[], max_elements=max_elements, source_parameters_unchanged=True,
                  runtime_data_evaluated=False, other_precisions_promoted=False,
                  retained_computations_unchanged=True, target_inference_executed=False,
                  full_model_fidelity_verified=False)
    if not selected:
        return result, report
    ordered = []
    for node in model.graph.node:
        if any(name in evaluated_nodes for name in node.output):
            item = copy.deepcopy(node)
            item.domain = ""
            ordered.append(item)
    names = [node.output[0] for node in selected]
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, known[name]) for name in names]
    version = next((entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")), None)
    if version is None:
        raise ValueError("source standard ONNX opset required")
    evaluation = helper.make_model(helper.make_graph(ordered, "readonly_float_expression", [], outputs,
        list(leaves.values())), opset_imports=[helper.make_opsetid("", version)], ir_version=model.ir_version)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(evaluation.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    values = session.run(names, {})
    replacements = {}
    for node, value in zip(selected, values, strict=True):
        if value.dtype != np.dtype("float32") or tuple(value.shape) != known[node.output[0]] or not np.isfinite(value).all():
            raise ValueError("readonly expression produced a nonfinite or unexpected FLOAT constant")
        constant = helper.make_node("Constant", [], list(node.output), name=node.name,
                                    value=numpy_helper.from_array(value, node.output[0]))
        replacements[node.output[0]] = constant
        report["rewritten"].append(dict(node=node.name, output=node.output[0], shape=list(value.shape), dtype="float32",
            original_node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest(),
            constant_node_sha256=hashlib.sha256(constant.SerializeToString()).hexdigest(),
            payload_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
    for node in result.graph.node:
        if len(node.output) == 1 and node.output[0] in replacements:
            node.CopyFrom(replacements[node.output[0]])
    report.update(status="folded-not-full-model-validated", evaluation_provider="CPUExecutionProvider",
                  evaluation_runtime_version=ort.__version__, graph_optimization="disabled",
                  static_evaluation_model_sha256=hashlib.sha256(evaluation.SerializeToString()).hexdigest(),
                  readonly_inputs=[dict(name=name, tensor_proto_sha256=hashlib.sha256(t.SerializeToString()).hexdigest())
                                   for name, t in sorted(leaves.items())])
    return result, report


def project_profile(profile, report, output_model):
    if profile.get("schema") != "edgefm.trt_precision_profile/1" or report.get("schema") != "vlaforge.readonly_float_unary_folding/1":
        raise ValueError("typed precision profile and readonly folding report required")
    if profile.get("source_sha256") != report.get("source_model_proto_sha256"):
        raise ValueError("precision profile is not bound to this exact source graph")
    result = copy.deepcopy(profile)
    names = [row["node"] for row in report["rewritten"]]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("profile projection requires unique named rewritten operators")
    removed, removed_storage = [], []
    for name in names:
        compute = result["layer_precisions"].get(name)
        storage = result["layer_output_types"].get(name)
        if compute not in (None, "fp32") or storage not in (None, "fp32"):
            raise ValueError("cannot discharge explicit FP16 or other non-source arithmetic/storage")
        if compute is not None:
            removed.append(name)
            del result["layer_precisions"][name]
        # Importers may represent literals as weights without a source-named layer.
        # Their FLOAT TensorProto carries the source type, not a runtime request.
        if storage is not None:
            removed_storage.append(name)
            del result["layer_output_types"][name]
    if "counts" in result and "fp32_Exp" in result["counts"]:
        if result["counts"]["fp32_Exp"] < len(removed):
            raise ValueError("precision profile counts disagree with actual compute obligations")
        result["counts"]["fp32_Exp"] -= len(removed)
    result.update(readonly_constant_parent_source=profile.get("source_model"), source_model=copy.deepcopy(output_model),
                  source_sha256=output_model["sha256"], full_output_revalidation_required=True,
                  readonly_constant_fold=dict(discharged_fp32_compute=removed,
                                              discharged_source_named_storage=removed_storage,
                                              typed_float_literals=[dict(node=row['node'],tensor=row['output'],dtype='float32',
                                                  payload_sha256=row['payload_sha256'])for row in report['rewritten']],
                                              source_parameters_unchanged=True, target_quality_verified=False))
    return result


def main():
    import onnx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-elements", type=int, default=1024)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve() if args.report else output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("readonly fold requires fresh output/report")
    model = onnx.load(args.input, load_external_data=False)
    from export_torchscript_region_onnx import external_files
    if external_files(model, args.input.resolve().parent) and output.parent != args.input.resolve().parent:
        raise ValueError("external weights stay readonly: publish beside the source or use the Python API")
    result, report = fold(model, external_base_dir=args.input.resolve().parent, max_elements=args.max_elements)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result.SerializeToString())
    onnx.checker.check_model(str(output))
    report.update(source=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], rewritten=len(report["rewritten"]))))


if __name__ == "__main__":
    main()
