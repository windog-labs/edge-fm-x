"""Audited backend preparation for AOTI's typed C++ dispatcher."""

from __future__ import annotations

import copy
import math


def normalize_scalar_graph(graph):
    """Select the matching Scalar overload when export used Tensor plus a literal.

    Python accepts these wrapped-number calls through Tensor overloads, whereas
    the AOTI proxy executor requires schema-correct C++ arguments. No tensor is
    synthesized and no numerical expression is decomposed. The supplied backend
    graph is modified in place; every change is returned as evidence.
    """
    import torch

    rewrites = []
    for node in graph.nodes:
        target = node.target
        if node.op != "call_function" or not isinstance(target, torch._ops.OpOverload):
            continue
        schema = target._schema
        if (
            not schema.name.startswith("aten::")
            or schema.is_mutable
            or target._overloadname not in ("Tensor", "Tensor_mode")
            or len(schema.arguments) < 2
            or schema.arguments[1].name != "other"
            or schema.arguments[1].type.kind() != "TensorType"
        ):
            continue
        other = node.args[1] if len(node.args) > 1 else node.kwargs.get("other")
        if type(other) not in (int, float, bool):
            continue
        overload = target._overloadname.replace("Tensor", "Scalar", 1)
        candidate = getattr(target.overloadpacket, overload, None)
        if candidate is None:
            raise ValueError(
                f"no scalar overload for exported call {node.name}: {target}"
            )
        replacement = candidate._schema
        if (
            replacement.is_mutable
            or len(replacement.arguments) != len(schema.arguments)
            or replacement.arguments[1].type.kind() != "NumberType"
            or [str(item.type) for item in replacement.returns]
            != [str(item.type) for item in schema.returns]
            or any(
                before.name != after.name
                or (index != 1 and str(before.type) != str(after.type))
                for index, (before, after) in enumerate(
                    zip(schema.arguments, replacement.arguments, strict=True)
                )
            )
        ):
            raise ValueError(
                f"scalar overload schema differs for {node.name}: {target}"
            )
        node.target = candidate
        rewrites.append(
            {"node": node.name, "before": str(target), "after": str(candidate)}
        )
    graph.lint()
    return tuple(rewrites)


def normalize_scalar_overloads(program):
    """Return a schema-normalized EP without modifying the input EP or weights."""
    import torch

    graph = torch.fx.Graph()
    environment = {}
    for node in program.graph.nodes:
        environment[node] = graph.node_copy(node, lambda value: environment[value])
    rewrites = normalize_scalar_graph(graph)
    if not rewrites:
        return program, ()
    module = torch.fx.GraphModule(program.graph_module, graph)
    result = program._update(module, copy.deepcopy(program.graph_signature))
    return result, tuple(rewrites)


def mark_empty_proxy_lists(graph):
    """Route empty dynamic proxy lists through native ATen/Inductor lowering.

    Torch 2.10's proxy skips zero-length dynamic arguments, leaving an IValue
    None instead of an empty list. Preserve the actual argument and scalar
    shape; mark both before selective decomposition and after AOT retracing.
    """
    import torch

    rewrites = []
    for node in graph.nodes:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        affected = []
        for index, argument in enumerate(node.target._schema.arguments):
            if index < len(node.args):
                value = node.args[index]
            elif argument.name in node.kwargs:
                value = node.kwargs[argument.name]
            else:
                value = argument.default_value if argument.has_default_value() else None
            if not isinstance(value, (tuple, list)) or value:
                continue
            kind = argument.type
            if kind.kind() == "OptionalType":
                kind = kind.getElementType()
            if kind.kind() != "ListType":
                continue
            element = kind.getElementType()
            dynamic = element.kind() in (
                "IntType", "SymIntType", "TensorType", "NumberType"
            )
            if element.kind() == "OptionalType":
                dynamic = element.getElementType().kind() == "TensorType"
            if dynamic:
                affected.append(argument.name)
        if affected:
            custom = dict(node.meta.get("custom", {}))
            custom.setdefault("compile_with_inductor", {})
            node.meta["custom"] = custom
            rewrites.append(
                {
                    "node": node.name,
                    "target": str(node.target),
                    "arguments": affected,
                    "kind": "native_lowering_empty_dynamic_proxy_list",
                }
            )
    return rewrites


def backend_program_pass_records(configs):
    """Identify preparation required before AOT's selective decomposition."""
    records = backend_pass_records(configs)
    if not records:
        return []
    if not configs.get("selective_decompose"):
        raise ValueError("ATen-preserving requires selective decomposition")
    return [
        {
            "name": "preserve_empty_dynamic_proxy_lists",
            "stage": "pre_aot_exported_program",
            "source_sha256": records[0]["source_sha256"],
        }
    ]


def prepare_backend_program(program, configs):
    """Prepare a graph-only EP copy, retaining the original weight storage.

    A post-grad-only marker is too late for factories with decompositions but
    no native lowering (e.g. ones). The pre-AOT marker permits their official
    decomposition, without changing the user's graph, metadata or tensors.
    """
    audit = {"passes": backend_program_pass_records(configs), "rewrites": []}
    if not audit["passes"]:
        return program, audit
    import torch

    graph = torch.fx.Graph()
    environment = {}
    for node in program.graph.nodes:
        copied = graph.node_copy(node, lambda value: environment[value])
        # Backend passes may extend custom metadata while compiling this copy.
        if "custom" in copied.meta:
            copied.meta["custom"] = copy.deepcopy(copied.meta["custom"])
        environment[node] = copied
    audit["rewrites"] = mark_empty_proxy_lists(graph)
    graph.lint()
    module = torch.fx.GraphModule(program.graph_module, graph)
    result = program._update(module, copy.deepcopy(program.graph_signature))
    return result, audit


def normalize_dispatch_graph(graph):
    """Make proxy arguments explicit; lower nonfinite literals without JSON scalars."""
    import torch
    from torch.fx.operator_schemas import normalize_function
    from torch.utils._pytree import tree_leaves

    rewrites = list(normalize_scalar_graph(graph))
    for node in graph.nodes:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        normalized = normalize_function(node.target, node.args, node.kwargs)
        if normalized is None:
            raise ValueError(f"cannot normalize backend arguments for {node.name}")
        if node.args != normalized.args or node.kwargs != normalized.kwargs:
            node.args, node.kwargs = normalized.args, normalized.kwargs
            rewrites.append({"node": node.name, "kind": "explicit_schema_defaults"})
        result_value = node.meta.get("val")
        integral_scalar = (
            isinstance(result_value, torch.Tensor)
            and not result_value.dtype.is_floating_point
            and not result_value.dtype.is_complex
            and bool(node.target._schema.arguments)
            and node.target._schema.arguments[0].type.kind() == "TensorType"
            and any(
                argument.type.kind() == "NumberType"
                and type(
                    node.args[index]
                    if index < len(node.args)
                    else node.kwargs.get(argument.name)
                )
                in (int, bool)
                for index, argument in enumerate(node.target._schema.arguments)
            )
        )
        if integral_scalar:
            # AOTI's C shims encode Scalar as double, promoting integer arithmetic.
            custom = dict(node.meta.get("custom", {}))
            custom.setdefault("compile_with_inductor", {})
            node.meta["custom"] = custom
            rewrites.append(
                {
                    "node": node.name,
                    "kind": "native_lowering_integral_scalar",
                    "target": str(node.target),
                    "dtype": str(result_value.dtype),
                }
            )
        if any(
            type(value) is float and not math.isfinite(value)
            for value in tree_leaves((node.args, node.kwargs))
        ):
            # The proxy JSON reader cannot consume Infinity/NaN string scalars.
            # Native Inductor lowering represents the original literal directly.
            custom = dict(node.meta.get("custom", {}))
            custom.setdefault("compile_with_inductor", {})
            node.meta["custom"] = custom
            rewrites.append(
                {
                    "node": node.name,
                    "kind": "native_lowering_nonfinite_literal",
                    "target": str(node.target),
                }
            )
    rewrites.extend(mark_empty_proxy_lists(graph))
    graph.lint()
    return rewrites


def backend_pass_records(configs):
    """Identify required backend passes without importing or initializing Torch."""
    if not configs.get("fallback_by_default"):
        return []
    if not configs.get("use_post_grad_passes"):
        raise ValueError("ATen-preserving requires post-grad schema normalization")
    import hashlib
    from pathlib import Path

    return [
        {
            "name": "normalize_aten_dispatch_arguments",
            "stage": "post_grad_custom_post_pass",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "graph_cache": "disabled to retain the actual per-compilation rewrite ledger",
        }
    ]


def prepare_backend_options(configs):
    """Attach schema normalization after AOT retracing, with a compile audit ledger."""
    options = dict(configs)
    audit = {"passes": backend_pass_records(configs), "rewrites": []}
    if not audit["passes"]:
        return options, audit
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class ScalarNormalizationPass(CustomGraphPass):
        def __call__(self, graph):
            audit["rewrites"].extend(normalize_dispatch_graph(graph))

        def uuid(self):
            return None

    options["post_grad_custom_post_pass"] = ScalarNormalizationPass()
    return options, audit
