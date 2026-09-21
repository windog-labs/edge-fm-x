import importlib.util
import os
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[2] / "tools/diagnose_aoti_empty_lists.py"
spec = importlib.util.spec_from_file_location("empty_list_diagnostic", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize("op,args", [
    (torch.ops.aten.ones.default, ([],)),
    (torch.ops.aten.view.default, (None, [])),
    (torch.ops.aten.cat.default, ([],)),
])
def test_empty_dynamic_proxy_lists_request_native_lowering_without_rewriting_values(op, args):
    graph = torch.fx.Graph()
    node = graph.call_function(op, args)
    before = node.args
    changes = diagnostic.mark_empty_proxy_lists(graph)
    assert changes and node.args == before
    assert node.meta["custom"]["compile_with_inductor"] == {}


@pytest.mark.parametrize("op,args", [
    (torch.ops.aten.ones.default, ([1],)),
    (torch.ops.aten.sum.dim_IntList, (None, None)),
])
def test_nonempty_or_none_lists_are_not_treated_as_empty_lists(op, args):
    graph = torch.fx.Graph()
    node = graph.call_function(op, args)
    assert diagnostic.mark_empty_proxy_lists(graph) == []
    assert "custom" not in node.meta


def test_empty_optional_list_is_distinct_from_none():
    graph = torch.fx.Graph()
    node = graph.call_function(torch.ops.aten.sum.dim_IntList, (None, []))
    assert diagnostic.mark_empty_proxy_lists(graph)
    assert node.args[1] == []


def test_real_scalar_factory_semantics_unchanged_on_cpu():
    class Module(torch.nn.Module):
        def forward(self, unused):
            return torch.ones([], dtype=torch.float32)
    exported = torch.export.export(Module(), (torch.ones(1, 50, 32),))
    before = exported.module()(torch.ones(1, 50, 32))
    diagnostic.mark_empty_proxy_lists(exported.graph)
    after = exported.module()(torch.ones(1, 50, 32))
    assert before.shape == after.shape == torch.Size([])
    assert before.dtype == after.dtype == torch.float32
    assert torch.equal(before, after) and after.item() == 1.0


def test_native_stderr_capture_retains_c_abi_messages_on_exception(tmp_path):
    path = tmp_path / "native.log"
    with pytest.raises(RuntimeError):
        with diagnostic.native_stderr(path):
            os.write(2, b"native backend failure\n")
            raise RuntimeError("expected")
    assert path.read_bytes() == b"native backend failure\n"
