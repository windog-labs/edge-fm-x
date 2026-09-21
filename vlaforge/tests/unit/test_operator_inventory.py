import json

import pytest
import torch
from vlaforge.analysis.operator_inventory import exported_operator_inventory


class RepeatedLinear(torch.nn.Module):
    def forward(self, x, weight):
        a = torch.nn.functional.linear(x, weight)
        b = torch.nn.functional.linear(a, weight)
        return b + 1.0


def test_real_export_signatures_preserve_shape_dtype_layout_and_multiplicity():
    program = torch.export.export(
        RepeatedLinear(), (torch.ones(2, 4), torch.ones(4, 4))
    )
    record = exported_operator_inventory(program, region_name="decode")
    assert record["static_call_function_nodes"] == 3
    linear = [item for item in record["operators"] if item["family"] == "gemm"]
    assert len(linear) == 1
    assert linear[0]["static_nodes_per_region_call"] == 2
    assert linear[0]["signature"]["arguments"][0]["shape"] == [2, 4]
    assert linear[0]["signature"]["arguments"][0]["stride"] == [4, 1]
    assert linear[0]["signature"]["outputs"]["dtype"] == "torch.float32"
    assert record["gpu_profiling_performed"] is False
    assert record["measured_runtime_counts"] is False
    json.dumps(record, allow_nan=False)


def test_signature_changes_for_noncontiguous_layout_and_scalar_semantics():
    class Add(torch.nn.Module):
        def forward(self, x):
            return x + 1, x + 2

    first = exported_operator_inventory(
        torch.export.export(Add(), (torch.ones(2, 4),)), region_name="first"
    )
    second = exported_operator_inventory(
        torch.export.export(Add(), (torch.ones(4, 2).t(),)), region_name="second"
    )
    keys = [
        {item["signature_sha256"] for item in record["operators"]}
        for record in (first, second)
    ]
    assert len(keys[0]) == len(keys[1]) == 2
    assert not keys[0] & keys[1]


def test_dynamic_shape_remains_symbolic_not_a_guessed_fixed_profile():
    class Add(torch.nn.Module):
        def forward(self, x):
            return x + 1

    program = torch.export.export(
        Add(),
        (torch.ones(3, 4),),
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
    )
    record = exported_operator_inventory(program, region_name="dynamic")
    shape = record["operators"][0]["signature"]["arguments"][0]["shape"]
    assert "symbolic" in shape[0]
    assert shape[1] == 4


def test_missing_metadata_is_explicit_and_missing_region_is_rejected():
    program = torch.export.export(
        RepeatedLinear(), (torch.ones(2, 4), torch.ones(4, 4))
    )
    node = next(node for node in program.graph.nodes if node.op == "call_function")
    del node.meta["val"]
    record = exported_operator_inventory(program, region_name="missing")
    assert record["operators"][0]["signature"]["outputs"] == {
        "missing_tensor_metadata": True
    }
    with pytest.raises(ValueError, match="region name"):
        exported_operator_inventory(program, region_name="")


def test_attention_embedding_and_norm_use_operator_semantics():
    class Operators(torch.nn.Module):
        def forward(self, q, tokens, weight):
            attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            norm = torch.nn.functional.layer_norm(attention, (4,))
            return norm, torch.nn.functional.embedding(tokens, weight)

    program = torch.export.export(
        Operators(),
        (torch.ones(1, 2, 3, 4), torch.ones(2, dtype=torch.int64), torch.ones(8, 4)),
    )
    record = exported_operator_inventory(program, region_name="ar")
    assert {item["family"] for item in record["operators"]} == {
        "attention",
        "norm",
        "embedding",
    }


def test_nested_graph_is_rejected_instead_of_omitting_its_operators():
    class Conditional(torch.nn.Module):
        def forward(self, predicate, x):
            return torch.cond(predicate, lambda x: x + 1, lambda x: x - 1, (x,))

    program = torch.export.export(Conditional(), (torch.tensor(True), torch.ones(2)))
    with pytest.raises(ValueError, match="higher-order"):
        exported_operator_inventory(program, region_name="branch")
