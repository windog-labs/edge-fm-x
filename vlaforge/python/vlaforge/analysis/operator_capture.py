"""Extract independent, read-only ATen examples from actual exported execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlaforge.analysis.operator_inventory import _metadata


@dataclass(frozen=True)
class OperatorExample:
    node: str
    target: Any
    args: tuple
    kwargs: dict
    reference: Any
    signature: dict

    def module(self):
        import torch

        operation = self.target

        class Operation(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return operation(*args, **kwargs)

        return Operation()


def _snapshot_arguments(args, kwargs):
    import torch
    from torch.utils._pytree import tree_map

    storages = set()

    def snapshot(value):
        if not isinstance(value, torch.Tensor):
            return value
        if value.layout != torch.strided or value.numel() == 0:
            raise ValueError("operator examples require nonempty strided inputs")
        if torch._debug_has_internal_overlap(value) != 0:
            raise ValueError("overlapping or unresolved-stride input is unsupported")
        storage = (str(value.device), value.untyped_storage().data_ptr())
        if storage in storages:
            raise ValueError("shared input storage requires an explicit alias contract")
        storages.add(storage)
        offset = value.storage_offset()
        span = (
            offset
            + 1
            + sum(
                (size - 1) * stride
                for size, stride in zip(value.shape, value.stride(), strict=True)
            )
        )
        buffer = torch.empty(span, dtype=value.dtype, device=value.device)
        result = buffer.as_strided(value.shape, value.stride(), offset)
        result.copy_(value.detach())
        return result

    def ordinary_containers(value):
        # FX immutable containers carry the same values but are not export inputs.
        if isinstance(value, tuple):
            return tuple(ordinary_containers(item) for item in value)
        if isinstance(value, list):
            return [ordinary_containers(item) for item in value]
        if isinstance(value, dict):
            return {key: ordinary_containers(item) for key, item in value.items()}
        return value

    return tree_map(snapshot, ordinary_containers((args, kwargs)))


def capture_operator_examples(
    exported_program: Any, args: tuple, *, node_names: tuple[str, ...]
) -> tuple[OperatorExample, ...]:
    """Run the actual graph once and snapshot named operators before execution.

    This is an offline diagnostic run with extra allocations and synchronization,
    never a latency measurement. Parameters, arguments and outputs stay on their
    actual device. Independent snapshots preserve strides and storage offsets,
    but allocator address alignment/cache contents are not the original ones.
    Shared storage, ambiguous strides, mutable operators and implicit RNG are
    rejected instead of silently changing the microbenchmark workload.
    """
    import torch
    from torch.utils._pytree import tree_map

    if not node_names or len(node_names) != len(set(node_names)):
        raise ValueError("operator nodes must be nonempty and unique")
    module = exported_program.module()
    nodes = {node.name: node for node in module.graph.nodes}
    if set(node_names) - nodes.keys():
        raise ValueError("selected operator node is absent")
    for name in node_names:
        node = nodes[name]
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            raise ValueError("operator examples require an explicit ATen overload")
        if node.target.namespace != "aten" or node.target._schema.is_mutable:
            raise ValueError("only read-only ATen operators can be extracted")
        if torch.Tag.nondeterministic_seeded in node.target.tags:
            # SDPA is conservatively tagged stochastic even when dropout is off.
            schema_args = node.target._schema.arguments
            if str(node.target) != "aten.scaled_dot_product_attention.default":
                raise ValueError(
                    "implicit RNG requires an explicit stochastic contract"
                )
            dropout_index = next(
                index
                for index, item in enumerate(schema_args)
                if item.name == "dropout_p"
            )
            dropout = node.kwargs.get(
                "dropout_p",
                node.args[dropout_index]
                if len(node.args) > dropout_index
                else schema_args[dropout_index].default_value,
            )
            if type(dropout) not in (int, float) or dropout != 0:
                raise ValueError(
                    "implicit RNG requires an explicit stochastic contract"
                )

    results = {}

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            if node.name not in node_names:
                return super().run_node(node)
            positional, keywords = self.fetch_args_kwargs_from_env(node)
            copied_args, copied_kwargs = _snapshot_arguments(positional, keywords)
            actual = super().run_node(node)
            reference = tree_map(
                lambda value: (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                ),
                actual,
            )
            results[node.name] = OperatorExample(
                node.name,
                node.target,
                copied_args,
                copied_kwargs,
                reference,
                {
                    "target": str(node.target),
                    "schema": str(node.target._schema),
                    "arguments": _metadata(positional),
                    "keyword_arguments": _metadata(keywords),
                    "outputs": _metadata(actual),
                },
            )
            return actual

    with torch.inference_mode():
        Recorder(module).run(*args)
    return tuple(results[name] for name in node_names)
