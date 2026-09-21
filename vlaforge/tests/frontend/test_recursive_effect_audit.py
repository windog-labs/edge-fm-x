import operator
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph, GraphModule
from vlaforge.frontend.effect_audit import audit_exported_program


def program(graph, *, children=None, signature=None):
    return SimpleNamespace(
        graph_module=GraphModule(children or {}, graph),
        graph_signature=signature or SimpleNamespace(input_specs=()),
    )


def nested(body):
    outer = Graph()
    x = outer.placeholder("x")
    y = outer.call_module("body", (x,))
    outer.output(y)
    return program(outer, children={"body": GraphModule({}, body)})


def test_nested_rng_is_rejected_with_qualified_location():
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.ops.aten.rand_like.default, (x,))
    graph.output(y)
    audit = audit_exported_program(nested(graph))
    assert audit.hidden_rng and not audit.passed
    assert any(item.source.startswith("body.") for item in audit.diagnostics)


@pytest.mark.parametrize("style", ["keyword", "list", "view_keyword"])
def test_nested_external_mutation_is_not_private_workspace(style):
    graph = Graph()
    x = graph.placeholder("x")
    if style == "keyword":
        result = graph.call_function(
            torch.ops.aten.add_.Tensor, (), {"self": x, "other": x}
        )
    elif style == "list":
        result = graph.call_function(torch.ops.aten._foreach_add_.Scalar, ([x], 1))
    else:
        alias = graph.call_function(
            torch.ops.aten.view.default, (), {"self": x, "size": [2]}
        )
        result = graph.call_function(torch.ops.aten.add_.Tensor, (alias, x))
    graph.output(result)
    audit = audit_exported_program(nested(graph))
    assert audit.hidden_mutation and not audit.passed


def test_nested_private_clone_mutation_and_eval_dropout_are_allowed():
    graph = Graph()
    x = graph.placeholder("x")
    private = graph.call_function(torch.ops.aten.clone.default, (x,))
    graph.call_function(torch.ops.aten.add_.Tensor, (private, x))
    output = graph.call_function(torch.ops.aten.dropout.default, (private, 0.5, False))
    graph.output(output)
    audit = audit_exported_program(nested(graph))
    assert audit.passed and not audit.hidden_mutation and not audit.hidden_rng
    assert {item.code for item in audit.diagnostics} == {
        "frontend.local_workspace_mutation",
        "frontend.eval_dropout",
    }


def test_unknown_nested_return_alias_cannot_enable_external_write():
    inner = Graph()
    x = inner.placeholder("x")
    inner.output((x,))
    outer = Graph()
    x = outer.placeholder("x")
    result = outer.call_module("body", (x,))
    alias = outer.call_function(operator.getitem, (result, 0))
    outer.call_function(torch.ops.aten.add_.Tensor, (alias, x))
    outer.output(alias)
    audit = audit_exported_program(
        program(outer, children={"body": GraphModule({}, inner)})
    )
    assert audit.hidden_mutation and not audit.passed


@pytest.mark.parametrize("field", ["buffers_to_mutate", "user_inputs_to_mutate"])
def test_functionalized_signature_write_is_not_omitted(field):
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (x, x)))
    audit = audit_exported_program(
        program(
            graph,
            signature=SimpleNamespace(input_specs=(), **{field: {"add": "x"}}),
        )
    )
    assert audit.hidden_mutation and not audit.passed


def test_random_operator_tag_is_checked_without_a_name_substring():
    target = torch.ops.aten.rrelu_with_noise.default
    assert torch.Tag.nondeterministic_seeded in target.tags
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(target, (x, x, 0.1, 0.3, True)))
    audit = audit_exported_program(program(graph))
    assert audit.hidden_rng and not audit.passed


@pytest.mark.parametrize("dropout", [None, 0.0, 0.2])
def test_nested_sdpa_uses_explicit_or_schema_default_dropout(dropout):
    graph = Graph()
    x = graph.placeholder("x")
    kwargs = {} if dropout is None else {"dropout_p": dropout}
    output = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default, (x, x, x), kwargs
    )
    graph.output(output)
    audit = audit_exported_program(nested(graph))
    assert audit.hidden_rng == (dropout == 0.2)
    assert audit.passed == (dropout != 0.2)


def test_actual_exported_cond_retains_and_audits_random_branch():
    class Conditional(torch.nn.Module):
        def forward(self, x, predicate):
            return torch.cond(
                predicate,
                lambda value: value + torch.rand_like(value),
                lambda value: value.clone(),
                (x,),
            )

    exported = torch.export.export(
        Conditional(), (torch.ones(2), torch.tensor(True)), strict=True
    )
    assert (
        sum(
            isinstance(module, GraphModule)
            for module in exported.graph_module.modules()
        )
        > 1
    )
    audit = audit_exported_program(exported)
    assert audit.hidden_rng and not audit.passed


def test_actual_exported_autocast_subgraph_is_recursively_audited():
    class AutocastRandom(torch.nn.Module):
        def forward(self, x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return x @ x + torch.rand_like(x)

    exported = torch.export.export(AutocastRandom(), (torch.ones(2, 2),), strict=True)
    if (
        sum(
            isinstance(module, GraphModule)
            for module in exported.graph_module.modules()
        )
        == 1
    ):
        pytest.skip("this Torch release does not retain the autocast HOP")
    audit = audit_exported_program(exported)
    assert audit.hidden_rng and not audit.passed
