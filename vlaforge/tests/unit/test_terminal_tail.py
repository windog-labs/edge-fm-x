import copy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from vlaforge.analysis.constant_precompute import graph_sha256
from vlaforge.analysis.terminal_tail import (
    partition_terminal_tail,
    save_terminal_tail_partition,
)


class Integrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        self.register_buffer("offset", torch.tensor(0.125))

    def forward(self, sample, features, index, *, scale=-0.1):
        velocity = torch.nn.functional.linear(features, self.weight) + self.offset
        torch.ops.aten._assert_async.msg(torch.isfinite(velocity).all(), "finite velocity")
        velocity.cos()
        return {"action": sample + scale * velocity, "state": (index + 1,)}


def integrator():
    args = (torch.randn(2, 4), torch.randn(2, 4), torch.tensor([3]))
    kwargs = {"scale": -0.1}
    program = torch.export.export(Integrator(), args, kwargs)
    output = list(program.graph.nodes)[-1].args[0]
    product = output[0].args[1]
    velocity = next(arg for arg in product.args if isinstance(arg, torch.fx.Node))
    inputs = ("sample", velocity.name, "index")
    outputs = tuple(spec.arg.name for spec in program.graph_signature.output_specs)
    return program, args, kwargs, inputs, outputs


def partition(program, inputs, outputs):
    return partition_terminal_tail(
        program, inputs=inputs, outputs=outputs, source_artifact_sha256="a" * 64,
    )


def joined(result, args, kwargs, prefix=None, tail=None):
    from torch.utils import _pytree

    prefix_values = (result.prefix.module() if prefix is None else prefix)(*args, **kwargs)
    flat_inputs, _ = _pytree.tree_flatten((args, kwargs))
    values = tuple((prefix_values if route.source == "prefix_output" else flat_inputs)[route.index]
                   for route in result.routes)
    result.tail.validate_inputs(values)
    return (result.tail.module if tail is None else tail)(*values), values


def test_integrator_roundtrip_routes_guards_and_exact_output(tmp_path):
    from torch.utils import _pytree

    program, args, kwargs, inputs, outputs = integrator()
    original = graph_sha256(program)
    result = partition(program, inputs, outputs)
    actual, tail_inputs = joined(result, args, kwargs)
    expected, _ = _pytree.tree_flatten(program.module()(*args, **kwargs))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert [(route.source, route.index) for route in result.routes] == [
        ("user_input", 0), ("prefix_output", 0), ("user_input", 2),
    ]
    assert result.frontier_names == (inputs[1],)
    assert result.prefix.graph_signature.input_specs == program.graph_signature.input_specs
    assert result.prefix.call_spec.in_spec == program.call_spec.in_spec
    assert result.control.call_spec == program.call_spec
    assert graph_sha256(program) == original == graph_sha256(result.control)
    assert "cos" in result.ledger["retained_no_output_nodes"]
    assert "_assert_async" in result.ledger["retained_no_output_nodes"]
    for source, candidate in ((program.state_dict, result.prefix.state_dict),
                              (program.state_dict, result.control.state_dict)):
        assert source.keys() == candidate.keys()
        assert all(source[name] is candidate[name] for name in source)
    selected = {entry["node"] for entry in result.tail.ledger["nodes"]}
    retained = [node for node in program.graph.nodes if node.name not in selected and node.op != "output"]
    prefix_nodes = list(result.prefix.graph.nodes)[:-1]
    assert [str(node) for node in retained] == [str(node) for node in prefix_nodes]
    for left, right in zip(retained, prefix_nodes, strict=True):
        assert left.op == right.op and left.target == right.target
        assert str(left.args) == str(right.args) and str(left.kwargs) == str(right.kwargs)
        assert left.meta.get("val") is right.meta.get("val")
    report = save_terminal_tail_partition(result, tmp_path / "split")
    for record in report["artifacts"].values():
        path = tmp_path / "split" / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert json.loads((tmp_path / "split" / "partition.json").read_text()) == report
    prefix = torch.export.load(tmp_path / "split" / "prefix.pt2").module()
    tail_path = tmp_path / "tail.pt2"
    torch.export.save(torch.export.export(result.tail.module, tail_inputs, strict=True), tail_path)
    reloaded, _ = joined(result, args, kwargs, prefix, torch.export.load(tail_path).module())
    assert all(torch.equal(a, b) for a, b in zip(reloaded, expected, strict=True))
    assert not result.ledger["full_model_verified"]
    assert not result.ledger["performance_measured"]
    assert not report["tail_artifact_emitted"]
    bad_args = (args[0], torch.full_like(args[1], float("nan")), args[2])
    for module in (program.module(), result.prefix.module(), prefix):
        with pytest.raises(RuntimeError, match="finite velocity"):
            module(*bad_args, **kwargs)


def test_distinct_gated_projection_formula_and_literal_slot():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, value, shift, gain):
            projection = self.linear(value)
            return projection.sigmoid() * gain + shift

    args = (torch.randn(2, 3), 2.0, torch.randn(2, 2))
    program = torch.export.export(Projection(), args)
    result = partition(program, ("gain", "linear"), ("add",))
    actual, _ = joined(result, args, {})
    assert torch.equal(actual[0], program.module()(*args))
    assert [(route.source, route.index) for route in result.routes] == [
        ("user_input", 2), ("prefix_output", 0),
    ]
    assert len(result.prefix.graph_signature.input_specs) == len(program.graph_signature.input_specs)


@pytest.mark.parametrize("change", ["trailing_assert", "interleaved_math", "partial_outputs", "reversed_outputs"])
def test_nonterminal_or_incomplete_outputs_rejected(change):
    program, _, _, inputs, outputs = integrator()
    if change in ("trailing_assert", "interleaved_math"):
        by_name = {node.name: node for node in program.graph.nodes}
        if change == "trailing_assert":
            target = next(node for node in program.graph.nodes if node.target is torch.ops.aten._assert_async.msg)
            list(program.graph.nodes)[-1].prepend(target)
        else:
            by_name[outputs[1]].prepend(by_name["cos"])
        program.graph_module.recompile()
    elif change == "partial_outputs":
        outputs = (outputs[0],)
        inputs = inputs[:2]
    else:
        outputs = tuple(reversed(outputs))
    with pytest.raises(ValueError, match="terminal suffix|every original output"):
        partition(program, inputs, outputs)


def test_computed_frontier_required():
    class AllTail(torch.nn.Module):
        def forward(self, value):
            return value.sin()
    program = torch.export.export(AllTail(), (torch.randn(3),))
    with pytest.raises(ValueError, match="computed frontier"):
        partition(program, ("value",), ("sin",))


@pytest.mark.parametrize("kind", ["parameter", "input_view", "shared_frontiers", "strided_frontier", "tail_view"])
def test_state_and_alias_boundaries_rejected(kind):
    class Aliases(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, value):
            if kind == "parameter":
                return value.sin() + self.weight
            if kind == "input_view":
                return value.view(2, 2).sin()
            copied = value.clone()
            if kind == "shared_frontiers":
                return copied + copied.view(2, 2)
            if kind == "strided_frontier":
                return copied.t().sin()
            return copied.sin().view(2, 2)

    program = torch.export.export(Aliases(), (torch.randn(2, 2),))
    options = {
        "parameter": (("sin", "p_weight"), ("add",), "lifted state"),
        "input_view": (("view",), ("sin",), "aliases input/state"),
        "shared_frontiers": (("clone", "view"), ("add",), "another frontier"),
        "strided_frontier": (("t",), ("sin",), "contiguous"),
        "tail_view": (("sin",), ("view",), "tail output aliases"),
    }
    inputs, outputs, message = options[kind]
    with pytest.raises(ValueError, match=message):
        partition(program, inputs, outputs)


def test_local_mutation_of_frontier_alias_rejected():
    class Write(torch.nn.Module):
        def forward(self, value):
            copied = value.clone()
            copied.add_(1)
            return copied.sin()
    program = torch.export.export(Write(), (torch.randn(3),))
    clone = next(node for node in program.graph.nodes if node.name == "clone")
    sine = next(node for node in program.graph.nodes if node.name == "sin")
    sine.args = (clone,)
    program.graph_module.recompile()
    with pytest.raises(ValueError, match="mutation aliases"):
        partition(program, ("clone",), ("sin",))


def test_schema_hidden_unsafe_view_rejected():
    class UnsafeView(torch.nn.Module):
        def forward(self, value):
            viewed = torch.ops.aten._unsafe_view.default(value, [4])
            return viewed.sin()
    program = torch.export.export(UnsafeView(), (torch.randn(2, 2),))
    with pytest.raises(ValueError, match="schema-hidden storage aliases"):
        partition(program, ("_unsafe_view",), ("sin",))


def test_higher_order_prefix_rejected():
    class Conditional(torch.nn.Module):
        def forward(self, value, condition):
            selected = torch.cond(condition, lambda x: x.sin(), lambda x: x.cos(), (value,))
            return selected + 1
    program = torch.export.export(Conditional(), (torch.randn(3), torch.tensor(True)))
    with pytest.raises(ValueError, match="HOPs"):
        partition(program, ("getitem",), ("add",))


def test_multiple_computed_frontiers_and_private_workspace_write():
    class Pair(torch.nn.Module):
        def forward(self, value, other):
            private = value.clone()
            private.add_(1)
            left = value.sin()
            right = other.cos()
            return left + right
    args = (torch.randn(3), torch.randn(3))
    program = torch.export.export(Pair(), args)
    result = partition(program, ("cos", "sin"), ("add",))
    actual, _ = joined(result, args, {})
    assert torch.equal(actual[0], program.module()(*args))
    assert result.frontier_names == ("cos", "sin")
    assert [route.source for route in result.routes] == ["prefix_output", "prefix_output"]
    assert "add_" in result.ledger["retained_nodes"]


def test_unknown_prefix_call_rejected():
    program, _, _, inputs, outputs = integrator()
    next(node for node in program.graph.nodes if node.name == "cos").target = torch.cos
    program.graph_module.recompile()
    with pytest.raises(ValueError, match="unknown calls"):
        partition(program, inputs, outputs)


def test_nested_module_call_signature_referencing_tail_rejected():
    from torch.export.exported_program import ModuleCallEntry
    from torch.export.graph_signature import TensorArgument

    program, _, _, inputs, outputs = integrator()
    signature = copy.deepcopy(program.module_call_graph[0].signature)
    signature.outputs = [TensorArgument(outputs[0])]
    program.module_call_graph.append(ModuleCallEntry("nested", signature))
    with pytest.raises(ValueError, match="nested module-call"):
        partition(program, inputs, outputs)


@pytest.mark.parametrize("change", ["state", "ledger", "prefix", "tail", "state_identity", "tail_contract", "guards"])
def test_save_rejects_changed_partition_without_creating_directory(tmp_path, change):
    program, _, _, inputs, outputs = integrator()
    result = partition(program, inputs, outputs)
    if change == "state":
        with torch.no_grad():
            result.prefix.state_dict["weight"].add_(1)
    elif change == "ledger":
        result.ledger["selected_for_deployment"] = True
    elif change == "prefix":
        next(node for node in result.prefix.graph.nodes if node.name == "cos").target = torch.ops.aten.sin.default
    elif change == "tail":
        next(node for node in result.tail.module.graph.nodes if node.target is torch.ops.aten.mul.Tensor).args = (
            next(node for node in result.tail.module.graph.nodes if node.name == inputs[1]), 0.5,
        )
    elif change == "state_identity":
        result.prefix.state_dict["weight"] = result.prefix.state_dict["weight"].detach().clone()
    elif change == "tail_contract":
        result.tail.ledger["inputs"][0]["metadata"]["dtype"] = "torch.float64"
    else:
        result.prefix._guards_code = ["False"]
    with pytest.raises(ValueError, match="changed|original state"):
        save_terminal_tail_partition(result, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
