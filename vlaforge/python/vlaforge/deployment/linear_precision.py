"""Explicit selected-Linear INT8 replacement in an existing static Region."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from vlaforge.analysis.constant_precompute import graph_sha256
from vlaforge.analysis.precision_calibration import PrecisionPlan
from vlaforge.deployment.int8_linear import (
    _tensor_identity,
    lower_scheduled_int8_linear,
)


@dataclass(frozen=True)
class LinearPrecisionResult:
    program: object
    ledger: dict


def _node_metadata(node):
    result = dict(node.meta)
    if "custom" in result:
        result["custom"] = copy.deepcopy(result["custom"])
    return result


def lower_exported_linear(
    program,
    plan,
    calibration_report,
    *,
    site,
    linear_node,
    step_input,
    region_artifact_sha256,
    numerical_context,
    precision_kind="int8",
):
    """Replace exactly one declared producer-to-Linear edge, preserving the ABI.

    The caller verifies the source archive digest before loading and binds the
    explicit int64 [1] input to its actual scheduler. A fresh artifact and new
    precision-specific contracts are mandatory; old certificates do not apply.
    This transformation preserves original state by reference, as other Region
    transforms do. Callers must keep that state immutable until serialization.
    """
    import torch
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    from vlaforge.frontend.effect_audit import audit_exported_program

    if not isinstance(plan, PrecisionPlan):
        raise TypeError("Linear replacement requires a typed PrecisionPlan")
    if not isinstance(program, torch.export.ExportedProgram) or program.range_constraints:
        raise ValueError("Linear replacement requires a static ExportedProgram")
    audit = audit_exported_program(program)
    if not audit.passed:
        raise ValueError("source Region effect audit failed")
    source_hash = graph_sha256(program)
    declaration = next((item for item in plan.sites if item.name == site), None)
    nodes = {node.name: node for node in program.graph.nodes}
    selected = nodes.get(linear_node)
    if (
        selected is None
        or selected.op != "call_function"
        or selected.target is not torch.ops.aten.linear.default
        or declaration is None
        or declaration.artifact_sha256 != region_artifact_sha256
    ):
        raise ValueError("replacement requires an exact declared aten.linear node")
    arguments = {}
    for index, argument in enumerate(selected.target._schema.arguments):
        arguments[argument.name] = (
            selected.args[index]
            if index < len(selected.args)
            else selected.kwargs.get(argument.name, argument.default_value)
        )
    value = arguments["input"]
    if not isinstance(value, torch.fx.Node) or value.name != declaration.node:
        raise ValueError("selected Linear does not consume the calibrated producer")
    specs = {spec.arg.name: spec for spec in program.graph_signature.input_specs}

    def state(argument):
        if argument is None:
            return None, None
        if not isinstance(argument, torch.fx.Node) or argument.op != "placeholder":
            raise ValueError("Linear weights must directly name immutable lifted state")
        spec = specs[argument.name]
        if spec.kind != InputKind.PARAMETER and not (
            spec.kind == InputKind.BUFFER and spec.persistent is True
        ):
            raise ValueError("Linear weights must name parameters or persistent buffers")
        return spec.target, program.state_dict[spec.target]

    weight_name, weight = state(arguments["weight"])
    bias_name, bias = state(arguments["bias"])
    if weight is None:
        raise ValueError("Linear replacement requires a weight Tensor")
    index = nodes.get(step_input)
    meta = index.meta.get("val") if index is not None else None
    if (
        index is None or index.op != "placeholder"
        or specs[index.name].kind != InputKind.USER_INPUT
        or not isinstance(meta, torch.Tensor)
        or meta.dtype != torch.int64 or tuple(meta.shape) != (1,)
        or meta.device != weight.device
    ):
        raise ValueError("schedule input must be a user int64 [1] Tensor on the weight device")
    activation_meta = value.meta.get("val")
    if not isinstance(activation_meta, torch.Tensor):
        raise TypeError("calibrated producer must have complete Tensor metadata")
    if precision_kind == "int8":
        lowered = lower_scheduled_int8_linear(
            plan, calibration_report, site=site,
            region_artifact_sha256=region_artifact_sha256,
            numerical_context=numerical_context, weight=weight, bias=bias,
        )
        expected_profile = lowered.to_data()["step_lowerings"][0]
    elif precision_kind in ("float16", "bfloat16"):
        from vlaforge.deployment.half_linear import lower_scheduled_half_linear

        lowered = lower_scheduled_half_linear(
            plan.step_keys,
            site=site,
            region_artifact_sha256=region_artifact_sha256,
            numerical_context=numerical_context,
            weight=weight,
            bias=bias,
            input_dtype=str(activation_meta.dtype).removeprefix("torch."),
            input_shape=tuple(activation_meta.shape),
            precision=precision_kind,
        )
        expected_profile = lowered.to_data()
    else:
        raise ValueError("precision_kind must be int8, float16 or bfloat16")
    if (
        list(activation_meta.shape) != expected_profile["input_shape"]
        or str(activation_meta.dtype) != "torch." + expected_profile["input_dtype"]
        or activation_meta.device != weight.device
    ):
        raise ValueError("calibrated producer metadata differs from the fitted profile")
    example = torch.zeros(tuple(activation_meta.shape), dtype=weight.dtype, device=weight.device)
    replacement = torch.export.export(
        lowered.module,
        (example, torch.zeros((1,), dtype=torch.int64, device=weight.device)),
        strict=True,
    )
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    if not isinstance(activation_meta, FakeTensor):
        raise TypeError("source Region requires export FakeTensor metadata")
    # The small export has a separate FakeTensor domain; merge metadata without
    # re-executing or re-exporting the original model's floating computations.
    fake_mode = activation_meta.fake_mode
    fake_inputs = []
    user_metadata = iter((activation_meta, meta))
    for spec in replacement.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            fake_inputs.append(next(user_metadata))
        elif spec.kind == InputKind.BUFFER and spec.persistent is True:
            fake_inputs.append(fake_mode.from_tensor(replacement.state_dict[spec.target], static_shapes=True))
        else:
            raise ValueError("integer replacement must only lift persistent buffers")
    FakeTensorProp(replacement.graph_module, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
    graph = torch.fx.Graph()
    copies = {}
    for node in program.graph.nodes:
        copied = graph.node_copy(node, lambda original: copies[original])
        copied.meta = _node_metadata(node)
        copies[node] = copied
    signature = copy.deepcopy(program.graph_signature)
    combined_state = dict(program.state_dict)
    existing = {node.name for node in graph.nodes} | set(combined_state) | set(program.constants)
    replacement_specs = {spec.arg.name: spec for spec in replacement.graph_signature.input_specs}
    replacement_nodes = {}
    user_inputs = iter((copies[value], copies[index]))
    inserted = {}
    position = next((i for i, spec in enumerate(signature.input_specs)
                     if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER)), len(signature.input_specs))
    before = list(graph.nodes)[position]
    for node in replacement.graph.nodes:
        if node.op != "placeholder":
            continue
        spec = replacement_specs[node.name]
        if spec.kind == InputKind.USER_INPUT:
            replacement_nodes[node] = next(user_inputs)
            continue
        if spec.kind != InputKind.BUFFER or spec.persistent is not True:
            raise ValueError("integer replacement must only lift persistent buffers")
        name = "_vlaforge_int8_" + node.name
        while name in existing:
            name += "_"
        existing.add(name)
        with graph.inserting_before(before):
            copied = graph.placeholder(name)
        copied.meta = _node_metadata(node)
        owned = replacement.state_dict[spec.target].detach().clone()
        combined_state[name] = owned
        inserted[name] = _tensor_identity(owned)
        signature.input_specs.insert(position, InputSpec(InputKind.BUFFER, TensorArgument(copied.name), name, True))
        position += 1
        replacement_nodes[node] = copied
    output = None
    with graph.inserting_before(copies[selected]):
        for node in replacement.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                if len(node.args[0]) != 1:
                    raise ValueError("integer replacement requires exactly one output")
                output = replacement_nodes[node.args[0][0]]
            elif node.op == "call_function":
                copied = graph.node_copy(node, lambda original: replacement_nodes[original])
                copied.meta = _node_metadata(node)
                replacement_nodes[node] = copied
            else:
                raise ValueError("integer replacement contains unlifted state or calls")
    if output is None:
        raise ValueError("integer replacement did not produce an output")
    expected_meta = selected.meta.get("val")
    actual_meta = output.meta.get("val")

    def profile(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Linear output requires complete Tensor metadata")
        return (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.layout,
                tuple(tensor.stride()), tensor.storage_offset())

    if profile(actual_meta) != profile(expected_meta):
        raise ValueError("integer replacement changes the selected output profile")
    copies[selected].replace_all_uses_with(output)
    graph.erase_node(copies[selected])
    output.name = selected.name
    graph.lint()
    result = torch.export.ExportedProgram(
        root=program.graph_module, graph=graph, graph_signature=signature,
        state_dict=combined_state, range_constraints=copy.deepcopy(program.range_constraints),
        module_call_graph=copy.deepcopy(program.module_call_graph),
        example_inputs=program.example_inputs, constants=dict(program.constants),
        verifiers=program.verifiers,
    )
    if not audit_exported_program(result).passed:
        raise ValueError("rewritten Region effect audit failed")
    if source_hash != graph_sha256(program):
        raise ValueError("source graph changed during Linear replacement")
    for name, key in ((weight_name, "source_weight"), (bias_name, "source_bias")):
        if name is not None and _tensor_identity(program.state_dict[name]) != expected_profile[key]:
            raise ValueError("source Linear state changed during replacement")
    numerical_context.require_current()
    return LinearPrecisionResult(result, {
        "schema": "vlaforge.exported_linear_precision/1",
        "source_artifact_sha256": region_artifact_sha256,
        "source_graph_sha256": source_hash,
        "result_graph_sha256": graph_sha256(result),
        "linear_node": selected.name, "activation_node": value.name,
        "weight_target": weight_name, "bias_target": bias_name,
        "step_input": step_input, "step_keys": list(plan.step_keys),
        "lowering": lowered.to_data(), "inserted_state": inserted,
        "source_state_retained": True, "source_state_storage_shared": True,
        "region_rewritten": True, "scheduler_binding_verified": False,
        "full_model_output_verified": False, "held_out_validation_complete": False,
        "real_low_precision_kernel_verified": False, "lossless_verified": False,
        "old_artifact_certificates_reusable": False,
    })


def lower_exported_all_linear_halves(
    program,
    precision_kind,
    numerical_context,
):
    """Replace every aten.linear compute with explicit half casts.

    This targets exported Regions whose Linear nodes do not expose a SmolVLA
    scheduler step argument. The ABI, state storage and output dtype remain
    unchanged; each Linear reads half inputs/weights/bias and casts its result
    back to the original float dtype. It is a real half compute candidate, not
    a claim of quality, speed, memory or lossless acceptance.
    """
    import copy

    import torch
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    from vlaforge.analysis.constant_precompute import graph_sha256
    from vlaforge.frontend.effect_audit import audit_exported_program
    from vlaforge.numerical_context import NumericalContext

    if precision_kind not in ("float16", "bfloat16"):
        raise ValueError("precision_kind must be float16 or bfloat16")
    if not isinstance(numerical_context, NumericalContext):
        raise TypeError("all-Linear half lowering requires a typed NumericalContext")
    if numerical_context.partial:
        raise ValueError("all-Linear half lowering requires an enforceable numerical context")
    if not isinstance(program, torch.export.ExportedProgram):
        raise ValueError("all-Linear half lowering requires an ExportedProgram")
    audit = audit_exported_program(program)
    if not audit.passed:
        raise ValueError("source Region effect audit failed")
    source_hash = graph_sha256(program)
    linears = [
        node for node in program.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.linear.default
    ]
    if not linears:
        raise ValueError("all-Linear half lowering found no aten.linear nodes")
    half_dtype = getattr(torch, precision_kind)
    graph = torch.fx.Graph()
    copies = {}
    for node in program.graph.nodes:
        if node.op == "placeholder":
            copied = graph.placeholder(node.name)
            copied.meta = _node_metadata(node)
            copies[node] = copied
    for node in program.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.target is torch.ops.aten.linear.default:
            expected = node.meta.get("val")
            if not isinstance(expected, torch.Tensor) or not expected.is_floating_point():
                raise ValueError("Linear output must have complete floating Tensor metadata")
            arguments = [copies[item] if isinstance(item, torch.fx.Node) else item for item in node.args]
            if len(arguments) not in (2, 3) or not all(isinstance(item, torch.fx.Node) for item in arguments[:2]):
                raise ValueError("Linear must consume Tensor input, weight and optional bias nodes")
            value, weight = arguments[:2]
            bias = arguments[2] if len(arguments) == 3 and arguments[2] is not None else None

            def cast(value):
                return graph.call_function(torch.ops.aten.to.dtype, (value, half_dtype), {})

            args = [cast(value), cast(weight)]
            if bias is not None:
                args.append(cast(bias))
            lowered = graph.call_function(node.target, tuple(args), node.kwargs)
            if expected.dtype != half_dtype:
                result = graph.call_function(torch.ops.aten.to.dtype, (lowered, expected.dtype), {})
            else:
                result = lowered
            copies[node] = result
            continue
        copied = graph.node_copy(node, lambda original: copies[original])
        copied.meta = _node_metadata(node)
        copies[node] = copied
    graph.lint()
    graph_module = torch.fx.GraphModule(program.graph_module, graph)
    fake_inputs = []
    fake_mode = None
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        meta = node.meta.get("val")
        if not isinstance(meta, torch.Tensor):
            raise ValueError("all-Linear half lowering requires Tensor placeholders only")
        fake_inputs.append(meta)
        if fake_mode is None:
            fake_mode = getattr(meta, "fake_mode", None)
    if fake_mode is None:
        raise ValueError("all-Linear half lowering requires FakeTensor graph metadata")
    FakeTensorProp(graph_module, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
    for original, rewritten in zip(linears, (copies[node] for node in linears), strict=True):
        expected = original.meta["val"]
        actual = rewritten.meta.get("val")
        profile = lambda value: (tuple(value.shape), value.dtype, value.device, value.layout,
                                 tuple(value.stride()), value.storage_offset())
        if not isinstance(actual, torch.Tensor) or profile(actual) != profile(expected):
            raise ValueError("all-Linear half rewrite changed a declared output profile")
    result = torch.export.ExportedProgram(
        root=program.graph_module, graph=graph,
        graph_signature=copy.deepcopy(program.graph_signature),
        state_dict=dict(program.state_dict),
        range_constraints=copy.deepcopy(program.range_constraints),
        module_call_graph=copy.deepcopy(program.module_call_graph),
        example_inputs=program.example_inputs,
        constants=dict(program.constants),
        verifiers=program.verifiers,
    )
    if not audit_exported_program(result).passed:
        raise ValueError("all-Linear half rewrite failed recursive effect audit")
    if source_hash != graph_sha256(program):
        raise ValueError("source graph changed during all-Linear half rewrite")
    numerical_context.require_current()
    return LinearPrecisionResult(result, {
        "schema": "vlaforge.exported_all_linear_half/1",
        "source_graph_sha256": source_hash,
        "result_graph_sha256": graph_sha256(result),
        "precision": precision_kind,
        "linear_count": len(linears),
        "linear_implementation": "aten.linear.default after explicit half input/weight/bias casts",
        "source_state_retained": True,
        "region_rewritten": True,
        "scheduler_binding": "not-applicable",
        "full_model_output_verified": False,
        "held_out_validation_complete": False,
        "real_low_precision_kernel_verified": False,
        "lossless_verified": False,
        "old_artifact_certificates_reusable": False,
    })
