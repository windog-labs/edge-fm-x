"""Subgraph extraction contract tests, not pretrained model evidence."""

import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.openpi.openpi_operator import _fingerprint, _read_only_subgraph


def test_tensor_fingerprint_retains_layout_and_offset():
    value = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8)[1:4, 1:7:2]
    record = _fingerprint(value)
    assert record["shape"] == [3, 3]
    assert record["stride"] == [8, 2]
    assert record["storage_offset"] == 9
    assert record["dtype"] == "torch.bfloat16"
    assert record["device"] == "cpu"
    assert record["sha256"] == _fingerprint(value.clone())["sha256"]


def graph_for(target):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    output = graph.call_function(target, (value,))
    graph.output(output)
    return torch.fx.GraphModule({}, graph)


def test_read_only_graph_records_actual_code_and_digest():
    (record,) = _read_only_subgraph(graph_for(torch.ops.aten.cos.default))
    assert record["path"] == ""
    assert "aten.cos" in record["graph"]
    assert len(record["sha256"]) == 64


def test_mutation_is_rejected():
    with pytest.raises(ValueError, match="mutable"):
        _read_only_subgraph(graph_for(torch.ops.aten.zero_.default))


def test_implicit_rng_is_rejected():
    with pytest.raises(ValueError, match="implicit RNG"):
        _read_only_subgraph(graph_for(torch.ops.aten.rand_like.default))
