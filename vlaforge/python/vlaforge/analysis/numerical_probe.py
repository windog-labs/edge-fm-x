"""Instrument a separate graph for diagnosis, never mutate a deployment artifact."""

from __future__ import annotations

import copy
from typing import Any


def tensor_value_bytes(value) -> bytes:
    """Canonical logical bytes, including non-unit strides on singleton axes."""
    import torch

    owned = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
    owned.copy_(value.detach())
    return owned.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_probe_module(exported_program: Any, node_names: tuple[str, ...]):
    """Expose selected tensor nodes with their original dependencies and weights.

    Additional outputs change compiler fusion decisions. Measurements from this
    graph diagnose possible error sources; they do not prove original-artifact
    intermediate values or performance.
    """
    import torch

    if not node_names or len(node_names) != len(set(node_names)):
        raise ValueError("probe nodes must be nonempty and unique")
    module = exported_program.module()
    graph = copy.deepcopy(module.graph)
    nodes = {node.name: node for node in graph.nodes}
    if set(node_names) - nodes.keys():
        raise ValueError("a selected probe node is absent from the captured graph")
    for name in node_names:
        node = nodes[name]
        if node.op != "call_function" or not isinstance(
            node.meta.get("val"), torch.Tensor
        ):
            raise ValueError("probe outputs must select tensor-valued function nodes")
    outputs = [node for node in graph.nodes if node.op == "output"]
    if len(outputs) != 1:
        raise ValueError("probe requires a single graph output node")
    snapshots = []
    for name in node_names:
        node = nodes[name]
        # A later in-place operation may change a selected tensor or its aliases.
        with graph.inserting_after(node):
            snapshot = graph.call_function(torch.ops.aten.clone.default, (node,))
            snapshot.meta = copy.copy(node.meta)
        snapshots.append(snapshot)
    outputs[0].args = (tuple(snapshots),)
    graph.set_codegen(torch.fx.graph.CodeGen())
    probe = torch.fx.GraphModule(module, graph)
    graph.eliminate_dead_code()
    graph.lint()
    probe.recompile()
    return probe


def tensor_difference(reference, candidate) -> dict:
    import torch

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("probe output shape/dtype changed")
    left, right = reference.detach().cpu(), candidate.detach().cpu()
    if not left.numel() or not bool(
        torch.isfinite(left).all() & torch.isfinite(right).all()
    ):
        raise ValueError("probe requires finite nonempty tensors")
    delta = left.double() - right.double()
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "bitwise_equal": tensor_value_bytes(left) == tensor_value_bytes(right),
        "changed_elements": int(torch.count_nonzero(left != right)),
        "mean_squared_error": float(delta.square().mean()),
        "maximum_absolute_error": float(delta.abs().max()),
        "mean_absolute_error": float(delta.abs().mean()),
    }
