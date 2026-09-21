import pytest
import torch
from vlaforge.analysis.numerical_probe import (
    tensor_difference,
    tensor_probe_module,
    tensor_value_bytes,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8.0).reshape(4, 2))

    def forward(self, x):
        value = torch.nn.functional.linear(x, self.weight)
        return value.sin() + 1


def test_probe_preserves_weights_and_dependencies_without_mutating_original():
    model = Model()
    x = torch.ones(3, 2)
    original = torch.export.export(model, (x,))
    original_graph = str(original.graph)
    module = tensor_probe_module(original, ("linear", "sin"))
    with torch.inference_mode():
        linear, sinusoid = module(x)
        expected = torch.nn.functional.linear(x, model.weight)
        assert torch.equal(linear, expected)
        assert torch.equal(sinusoid, expected.sin())
        exported = torch.export.export(module, (x,), strict=True)
        assert torch.equal(exported.module()(x)[1], sinusoid)
    assert str(original.graph) == original_graph
    assert all(str(node.target) != "aten.add.Tensor" for node in module.graph.nodes)


@pytest.mark.parametrize("names", [(), ("linear", "linear"), ("missing",), ("x",)])
def test_invalid_probe_selection_is_rejected(names):
    program = torch.export.export(Model(), (torch.ones(3, 2),))
    with pytest.raises(ValueError):
        tensor_probe_module(program, names)


def test_probe_reports_bitwise_difference_and_nonfinite_failures():
    assert (
        tensor_difference(torch.tensor(0.0), torch.tensor(-0.0))["bitwise_equal"]
        is False
    )
    assert (
        tensor_difference(torch.tensor([1.0]), torch.tensor([2.0]))[
            "mean_squared_error"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="finite"):
        tensor_difference(torch.tensor([float("nan")]), torch.ones(1))


def test_probe_snapshots_a_node_before_later_alias_mutation():
    class Mutating(torch.nn.Module):
        def forward(self, x):
            value = x.sin()
            value.view(-1).add_(3)
            return value

    x = torch.ones(2, 3)
    program = torch.export.export(Mutating(), (x,))
    probe = tensor_probe_module(program, ("sin",))
    assert torch.equal(probe(x)[0], x.sin())
    assert torch.equal(program.module()(x), x.sin() + 3)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bfloat16, torch.float16])
def test_singleton_nonunit_stride_is_canonicalized_without_changing_values(dtype):
    value = torch.tensor([1], dtype=dtype).as_strided((1,), (1600,))
    expected = torch.tensor([1], dtype=dtype)
    assert tensor_value_bytes(value) == tensor_value_bytes(expected)
    assert tensor_difference(value, expected)["bitwise_equal"]
    assert value.stride() == (1600,)
