"""Model-independent, static operator workloads from a captured torch graph.

This inventory seeds profiling and candidate selection. Node multiplicity is
not a measured launch count, invocation count, runtime or hotspot ranking.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _target_name(target: Any) -> str:
    if hasattr(target, "_schema"):
        return str(target)
    return f"{target.__module__}.{target.__qualname__}"


def _metadata(value: Any) -> Any:
    import torch
    from torch.fx import Node

    if isinstance(value, Node):
        if "val" not in value.meta:
            return {"missing_tensor_metadata": True}
        return _metadata(value.meta["val"])
    if isinstance(value, torch.Tensor):

        def dimension(size):
            return size if type(size) is int else {"symbolic": str(size)}

        return {
            "tensor": True,
            "shape": [dimension(size) for size in value.shape],
            "stride": [dimension(size) for size in value.stride()],
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "storage_offset": dimension(value.storage_offset()),
        }
    if isinstance(value, tuple | list):
        return [_metadata(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("operator keyword keys must be strings")
        return {key: _metadata(item) for key, item in sorted(value.items())}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"floating_constant": str(value)}
    if isinstance(
        value, torch.dtype | torch.device | torch.layout | torch.memory_format
    ):
        return {"torch_constant": str(value)}
    if isinstance(value, torch.SymInt | torch.SymFloat | torch.SymBool):
        return {"symbolic": str(value)}
    raise TypeError(f"unsupported operator argument metadata: {type(value).__name__}")


def _family(target: str) -> str:
    operation = target.rsplit(".", 1)[0]
    if operation in {"aten.linear", "aten.mm", "aten.bmm", "aten.addmm", "aten.matmul"}:
        return "gemm"
    if operation in {"aten.layer_norm", "aten.native_layer_norm", "aten.rms_norm"}:
        return "norm"
    if operation in {"aten.embedding", "aten.embedding_bag", "aten._embedding_bag"}:
        return "embedding"
    if operation in {
        "aten.scaled_dot_product_attention",
        "aten._scaled_dot_product_flash_attention",
        "aten._scaled_dot_product_efficient_attention",
        "aten._scaled_dot_product_attention_math",
        "aten._scaled_dot_product_cudnn_attention",
    }:
        return "attention"
    return "unclassified"


def exported_operator_inventory(exported_program: Any, *, region_name: str) -> dict:
    """Preserve operator signatures and provenance without guessing model roles.

    Composite RoPE, normalization, attention and solver expressions remain raw
    nodes until a separately verified subgraph matcher assigns their semantics.
    """
    import torch

    if not isinstance(region_name, str) or not region_name:
        raise ValueError("an explicit region name is required")
    groups = {}
    for node in exported_program.graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        if node.op != "call_function":
            raise ValueError(f"unsupported graph node operation: {node.op}")
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            raise ValueError(
                "nested higher-order graphs require a separate recursive inventory"
            )
        target = _target_name(node.target)
        signature = {
            "target": target,
            "schema": str(node.target._schema)
            if hasattr(node.target, "_schema")
            else None,
            "arguments": _metadata(node.args),
            "keyword_arguments": _metadata(node.kwargs),
            "outputs": _metadata(node.meta["val"])
            if "val" in node.meta
            else {"missing_tensor_metadata": True},
        }
        key = hashlib.sha256(
            json.dumps(
                signature, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        entry = groups.setdefault(
            key,
            {
                "signature_sha256": key,
                "signature": signature,
                "family": _family(target),
                "static_nodes_per_region_call": 0,
                "instances": [],
            },
        )
        entry["static_nodes_per_region_call"] += 1
        entry["instances"].append(
            {
                "node": node.name,
                "module_stack": [
                    {"path": str(path), "type": str(kind)}
                    for path, kind in node.meta.get("nn_module_stack", {}).values()
                ],
                "stack_trace": node.meta.get("stack_trace", ""),
            }
        )
    return {
        "schema": "vlaforge.exported_operator_inventory/1",
        "region": region_name,
        "evidence_kind": "static_captured_graph_workload",
        "measured_runtime_counts": False,
        "gpu_profiling_performed": False,
        "kernel_launch_counts_known": False,
        "static_call_function_nodes": sum(
            group["static_nodes_per_region_call"] for group in groups.values()
        ),
        "operators": list(groups.values()),
        "composite_semantics": "RoPE, decomposed attention/norm and VLA roles require a verified subgraph matcher; unclassified does not mean absent",
    }
