import pytest
import torch
from vlaforge.analysis.operator_capture import capture_operator_examples


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8.0).reshape(4, 2))

    def forward(self, x):
        linear = torch.nn.functional.linear(x, self.weight)
        return linear.sin()


def test_actual_inputs_and_weights_replay_and_export_with_original_strides():
    model = Model()
    x = torch.arange(6.0).reshape(2, 3).t()
    program = torch.export.export(model, (x,))
    original_graph = str(program.graph)
    (example,) = capture_operator_examples(program, (x,), node_names=("linear",))
    assert example.args[0].stride() == x.stride()
    assert example.args[0].data_ptr() != x.data_ptr()
    assert torch.equal(example.args[1], model.weight)
    assert example.signature["arguments"][0]["stride"] == [1, 3]
    actual = example.module()(*example.args, **example.kwargs)
    assert torch.equal(actual, example.reference)
    with torch.inference_mode():
        independent = torch.export.export(
            example.module(), example.args, example.kwargs
        )
        assert torch.equal(independent.module()(*example.args), example.reference)
    assert str(program.graph) == original_graph


def test_snapshot_precedes_later_alias_mutation():
    class Mutating(torch.nn.Module):
        def forward(self, x):
            value = x.sin()
            before = value.cos()
            value.add_(3)
            return before + value

    x = torch.ones(2, 3)
    program = torch.export.export(Mutating(), (x,))
    (example,) = capture_operator_examples(program, (x,), node_names=("cos",))
    assert torch.equal(example.args[0], x.sin())
    assert torch.equal(example.reference, x.sin().cos())


@pytest.mark.parametrize("names", [(), ("linear", "linear"), ("missing",), ("x",)])
def test_bad_selection_is_rejected(names):
    program = torch.export.export(Model(), (torch.ones(3, 2),))
    with pytest.raises(ValueError):
        capture_operator_examples(program, (torch.ones(3, 2),), node_names=names)


def test_mutating_operator_is_rejected():
    class Mutating(torch.nn.Module):
        def forward(self, x):
            return x.sin().add_(2)

    program = torch.export.export(Mutating(), (torch.ones(2),))
    with pytest.raises(ValueError, match="read-only"):
        capture_operator_examples(program, (torch.ones(2),), node_names=("add_",))


def test_shared_storage_is_not_silently_cloned_as_independent_inputs():
    class Aliasing(torch.nn.Module):
        def forward(self, x):
            return x + x

    program = torch.export.export(Aliasing(), (torch.ones(2),))
    with pytest.raises(ValueError, match="alias contract"):
        capture_operator_examples(program, (torch.ones(2),), node_names=("add",))


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_attention_requires_explicit_zero_dropout(dropout):
    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout
            )

    args = tuple(torch.randn(1, 2, 3, 4) for _ in range(3))
    program = torch.export.export(Attention(), args)
    if dropout:
        with pytest.raises(ValueError, match="stochastic"):
            capture_operator_examples(
                program, args, node_names=("scaled_dot_product_attention",)
            )
    else:
        (example,) = capture_operator_examples(
            program, args, node_names=("scaled_dot_product_attention",)
        )
        assert torch.equal(example.module()(*example.args), example.reference)


def test_nonzero_offset_is_preserved():
    x = torch.arange(10.0)[4:].reshape(3, 2)
    program = torch.export.export(Model(), (x,))
    (example,) = capture_operator_examples(program, (x,), node_names=("linear",))
    assert example.args[0].storage_offset() == 4
    assert torch.equal(example.module()(*example.args), example.reference)


def test_broadcast_input_is_rejected_instead_of_materializing_different_strides():
    x = torch.ones(1, 2).expand(3, 2)
    program = torch.export.export(Model(), (x,))
    with pytest.raises(ValueError, match="stride"):
        capture_operator_examples(program, (x,), node_names=("linear",))


def test_fx_immutable_shape_list_becomes_a_standard_export_input():
    class Normalizing(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.layer_norm(x, [4], weight, bias)

    args = (torch.randn(2, 4), torch.ones(4), torch.zeros(4))
    program = torch.export.export(Normalizing(), args)
    (example,) = capture_operator_examples(program, args, node_names=("layer_norm",))
    assert type(example.args[1]) is list
    with torch.inference_mode():
        independent = torch.export.export(
            example.module(), example.args, example.kwargs
        )
        assert torch.equal(independent.module()(*example.args), example.reference)
