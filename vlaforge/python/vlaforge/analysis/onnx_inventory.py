"""Static logical ONNX workloads, without inferring physical kernel behavior."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _digest(message: Any) -> str:
    return hashlib.sha256(message.SerializeToString(deterministic=True)).hexdigest()


def _tensor_type(value: Any) -> dict:
    import onnx

    if not value.type.HasField("tensor_type"):
        return {"missing_tensor_metadata": True, "type_proto_sha256": _digest(value.type)}
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = []
        for dimension in tensor.shape.dim:
            if dimension.HasField("dim_value"):
                shape.append(dimension.dim_value)
            elif dimension.HasField("dim_param"):
                shape.append({"symbolic": dimension.dim_param})
            else:
                shape.append(None)
    return {"tensor": True, "shape": shape,
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type) if tensor.elem_type else None}


def _bounded(value: dict) -> bool:
    shape = value.get("shape")
    return (value.get("tensor") is True and value.get("dtype") is not None and shape is not None
            and all(type(size) is int and size > 0 for size in shape))


def onnx_operator_inventory(model: Any, *, region_name: str, infer_shapes: bool = True) -> dict:
    """Group logical signatures; keep raw instances and unresolved metadata.

    No external weights are loaded, model executed, or target layout/compute
    precision invented. Composite roles need independent semantic matching.
    """
    import onnx

    if not isinstance(model, onnx.ModelProto):
        raise TypeError("an ONNX ModelProto is required")
    if not isinstance(region_name, str) or not region_name:
        raise ValueError("an explicit region name is required")
    if type(infer_shapes) is not bool:
        raise TypeError("infer_shapes must be explicit BOOL")
    if model.functions:
        raise ValueError("local functions require expansion before flat operator inventory")
    if any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in model.graph.node for attribute in node.attribute):
        raise ValueError("nested graph requires a recursive operator inventory")
    opsets = {}
    for item in model.opset_import:
        domain = item.domain or "ai.onnx"
        if domain in opsets:
            raise ValueError("duplicate domain opset declaration")
        opsets[domain] = item.version
    facts, origins = {}, {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        facts[value.name] = _tensor_type(value)
        origins[value.name] = "declared"
    initializer_names = {value.name for value in model.graph.initializer}
    for value in model.graph.initializer:
        facts[value.name] = {"tensor": True, "shape": list(value.dims), "dtype": onnx.TensorProto.DataType.Name(value.data_type)}
        origins[value.name] = "initializer-dimensions"
    inference = {"requested": infer_shapes, "status": "not-requested", "data_prop": False}
    if infer_shapes:
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
            inference["status"] = "completed-not-a-model-validity-check"
            for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
                candidate = _tensor_type(value)
                if value.name not in facts or (not _bounded(facts[value.name]) and _bounded(candidate)):
                    facts[value.name], origins[value.name] = candidate, "onnx-shape-inference"
        except (onnx.shape_inference.InferenceError, ValueError, RuntimeError) as error:
            inference.update(status="failed-declared-metadata-retained", error=str(error))
    groups = {}
    graph_inputs = {value.name for value in model.graph.input}

    def metadata(name):
        if not name:
            return {"omitted_optional_port": True}
        return facts.get(name, {"missing_tensor_metadata": True})

    for index, node in enumerate(model.graph.node):
        domain = node.domain or "ai.onnx"
        if domain not in opsets:
            raise ValueError("node domain has no explicit opset: " + domain)
        if len({attribute.name for attribute in node.attribute}) != len(node.attribute):
            raise ValueError("duplicate node attributes")
        # Attribute payload hashes preserve constants/axes/epsilon without
        # decoding tensor data or turning external initializers into arrays.
        attributes = [{"name": item.name, "type": onnx.AttributeProto.AttributeType.Name(item.type),
                       "payload_sha256": _digest(item)} for item in sorted(node.attribute, key=lambda item: item.name)]
        inputs, outputs = [metadata(name) for name in node.input], [metadata(name) for name in node.output]
        roles = ["omitted" if not name else "initializer" if name in initializer_names
                 else "graph-input" if name in graph_inputs else "intermediate" for name in node.input]
        signature = {"domain": domain, "operator": node.op_type, "opset": opsets[domain],
            "inputs": inputs, "input_roles": roles, "outputs": outputs, "attributes": attributes,
            "physical_layout": "not-known-from-ONNX", "target_compute_precision": "not-known-from-logical-IO"}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        family = "unclassified"
        if domain == "ai.onnx":
            if node.op_type in ("MatMul", "Gemm"):
                family = "gemm"
            elif node.op_type in ("LayerNormalization", "RMSNormalization", "BatchNormalization", "InstanceNormalization"):
                family = "norm"
            elif node.op_type == "Attention":
                family = "attention"
            elif node.op_type in ("Gather", "GatherElements", "GatherND"):
                family = "indexing-not-verified-embedding"
        elif domain == "trt.plugins":
            family = "plugin-requires-provider-contract"
        group = groups.setdefault(key, {"signature_sha256": key, "signature": signature, "family": family,
            "all_logical_tensor_ports_static_positive": all(_bounded(value) or value.get("omitted_optional_port") for value in (*inputs, *outputs)),
            "static_nodes_per_region_call": 0, "instances": []})
        group["static_nodes_per_region_call"] += 1
        group["instances"].append({"index": index, "node": node.name, "inputs": list(node.input), "outputs": list(node.output),
            "input_metadata_origins": [origins.get(name, "omitted" if not name else "unresolved") for name in node.input],
            "output_metadata_origins": [origins.get(name, "omitted" if not name else "unresolved") for name in node.output]})
    return {"schema": "vlaforge.onnx_operator_inventory/1", "region": region_name,
        "evidence_kind": "static_logical_ONNX_workload", "source_model_proto_sha256": _digest(model),
        "opsets": opsets, "shape_inference": inference, "static_nodes": len(model.graph.node),
        "unique_signatures": len(groups), "operators": list(groups.values()),
        "measured_runtime_counts": False, "gpu_profiling_performed": False, "kernel_launch_counts_known": False,
        "external_weight_data_loading_performed": False, "tuning_or_performance_claim": False,
        "composite_semantics": "RoPE, decomposed attention/norm, embedding and VLA roles require verified subgraph matching; unclassified is not N/A"}
