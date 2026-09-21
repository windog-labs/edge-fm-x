import pytest

torch = pytest.importorskip("torch")

from vlaforge.deployment.aoti_export import normalize_scalar_overloads


class Arithmetic(torch.nn.Module):
    def forward(self, value):
        return ((value - 1.0) * 0.5 + 0.125) / 2.0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_scalar_tensor_overloads_remain_bitwise_equal_and_source_is_immutable(dtype):
    value = torch.tensor([0.0, -0.0, 0.1, -0.2, 120.0], dtype=dtype)
    original = torch.export.export(Arithmetic(), (value,))
    before = str(original.graph)
    normalized, rewrites = normalize_scalar_overloads(original)
    assert len(rewrites) == 4
    assert str(original.graph) == before
    assert all(row["after"].endswith(".Scalar") for row in rewrites)
    expected, actual = original.module()(value), normalized.module()(value)
    assert torch.equal(
        expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
    )
    normalized.validate()
    again, repeated = normalize_scalar_overloads(normalized)
    assert not repeated
    assert str(again.graph) == str(normalized.graph)


def test_tensor_operands_are_not_changed():
    class Add(torch.nn.Module):
        def forward(self, left, right):
            return left + right

    value = torch.ones(2)
    original = torch.export.export(Add(), (value, value))
    normalized, rewrites = normalize_scalar_overloads(original)
    assert not rewrites
    assert str(original.graph) == str(normalized.graph)


def test_integral_scalar_keeps_integer_dispatch_semantics():
    from vlaforge.deployment.aoti_export import normalize_dispatch_graph

    class Integer(torch.nn.Module):
        def forward(self, values):
            return (values + 1).clamp(max=31)

    values = torch.tensor([-4, 0, 20, 40], dtype=torch.int64)
    program = torch.export.export(Integer(), (values,))
    rewrites = normalize_dispatch_graph(program.graph)
    assert any(item.get("kind") == "native_lowering_integral_scalar" for item in rewrites)
    program.graph_module.recompile()
    assert torch.equal(program.module()(values), Integer()(values))
    assert program.module()(values).dtype == torch.int64


def test_rounding_mode_keyword_and_graph_serialization_are_preserved(tmp_path):
    class Rounded(torch.nn.Module):
        def forward(self, value):
            return torch.div(value, 2.0, rounding_mode="floor")

    value = torch.tensor([-3.5, 2.5, 1.1])
    original = torch.export.export(Rounded(), (value,))
    normalized, rewrites = normalize_scalar_overloads(original)
    assert rewrites[0]["after"] == "aten.div.Scalar_mode"
    path = tmp_path / "operator.pt2"
    torch.export.save(normalized, path)
    loaded = torch.export.load(path)
    assert torch.equal(loaded.module()(value), original.module()(value))


def test_backend_pass_is_applied_after_retracing_and_records_actual_rewrites():
    from vlaforge.deployment.aoti_export import prepare_backend_options
    from vlaforge.deployment.aoti_profile import aoti_configs

    configs = aoti_configs("aten-preserving")
    options, audit = prepare_backend_options(configs)
    program = torch.export.export(Arithmetic(), (torch.ones(2),))
    callback = options["post_grad_custom_post_pass"]
    assert callback.uuid() is None
    assert "post_grad_custom_post_pass" not in configs
    assert audit["passes"][0]["source_sha256"]
    callback(program.graph)
    assert sum("before" in item for item in audit["rewrites"]) == 4
    program.graph_module.recompile()
    assert torch.equal(program.module()(torch.ones(2)), Arithmetic()(torch.ones(2)))
    with pytest.raises(ValueError, match="post-grad"):
        prepare_backend_options({**configs, "use_post_grad_passes": False})


def test_proxy_defaults_and_nonfinite_predicates_are_preserved():
    from vlaforge.deployment.aoti_export import normalize_dispatch_graph

    class Predicates(torch.nn.Module):
        def forward(self, values, weight, bias):
            return torch.addmm(bias, values, weight), torch.isfinite(values)

    values = torch.tensor([[0.0, float("inf"), float("-inf"), float("nan")]])
    weight, bias = torch.ones(4, 2), torch.ones(2)
    original = torch.export.export(Predicates(), (values, weight, bias))
    program = original.run_decompositions({})
    before = program.module()(values, weight, bias)
    rewrites = normalize_dispatch_graph(program.graph)
    program.graph_module.recompile()
    after = program.module()(values, weight, bias)
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(after[1], before[1])
    node = next(
        node
        for node in program.graph.nodes
        if node.target == torch.ops.aten.addmm.default
    )
    assert node.kwargs["alpha"] == node.kwargs["beta"] == 1
    assert any(row.get("kind") == "explicit_schema_defaults" for row in rewrites)


def test_explicit_infinity_comparison_uses_native_lowering_without_changing_literal():
    from vlaforge.deployment.aoti_export import normalize_dispatch_graph

    class Compare(torch.nn.Module):
        def forward(self, values):
            return values != float("inf")

    values = torch.tensor([0.0, float("inf"), float("-inf"), float("nan")])
    program = torch.export.export(Compare(), (values,))
    rewrites = normalize_dispatch_graph(program.graph)
    program.graph_module.recompile()
    assert torch.equal(program.module()(values), Compare()(values))
    assert any(
        row.get("kind") == "native_lowering_nonfinite_literal" for row in rewrites
    )


def test_pre_aot_copy_preserves_scalar_shape_source_metadata_and_weight_storage():
    from vlaforge.deployment.aoti_export import prepare_backend_program
    from vlaforge.deployment.aoti_profile import aoti_configs

    class Factory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, unused):
            return torch.ones([], dtype=torch.float32) + self.weight

    value = torch.randn(1, 50, 32)
    original = torch.export.export(Factory(), (value,))
    node = next(n for n in original.graph.nodes if n.target == torch.ops.aten.ones.default)
    node.meta["custom"] = {"caller_annotation": {"kept": True}}
    before = str(original.graph)
    prepared, audit = prepare_backend_program(original, aoti_configs("aten-preserving"))
    assert prepared is not original
    assert str(original.graph) == before == str(prepared.graph)
    assert node.meta["custom"] == {"caller_annotation": {"kept": True}}
    prepared_node = next(n for n in prepared.graph.nodes if n.name == node.name)
    assert prepared_node.meta["custom"]["compile_with_inductor"] == {}
    prepared_node.meta["custom"]["caller_annotation"]["kept"] = False
    assert node.meta["custom"]["caller_annotation"]["kept"]
    assert prepared.state_dict["weight"] is original.state_dict["weight"]
    assert prepared.state_dict["weight"].data_ptr() == original.state_dict["weight"].data_ptr()
    actual = prepared.module()(value)
    assert actual.shape == torch.Size([])
    assert actual.dtype == torch.float32
    assert torch.equal(actual, original.module()(value))
    assert audit["rewrites"][0]["arguments"] == ["size"]
    assert audit["passes"][0]["stage"] == "pre_aot_exported_program"
    prepared.validate()


@pytest.mark.parametrize("profile", ["default", "conservative", "eager-numerics"])
def test_non_proxy_profiles_do_not_prepare_or_clone_program(profile):
    from vlaforge.deployment.aoti_export import prepare_backend_program
    from vlaforge.deployment.aoti_profile import aoti_configs

    program = object()
    prepared, audit = prepare_backend_program(program, aoti_configs(profile))
    assert prepared is program
    assert audit == {"passes": [], "rewrites": []}


def test_two_stage_ledger_binds_source_and_records_decomposed_scalar_factory():
    import hashlib
    from pathlib import Path
    from vlaforge.deployment import aoti_export
    from vlaforge.deployment.aoti_profile import aoti_configs

    class Factory(torch.nn.Module):
        def forward(self, unused):
            return torch.ones([], dtype=torch.float32)

    configs = aoti_configs("aten-preserving")
    original = torch.export.export(Factory(), (torch.randn(2),))
    prepared, pre = aoti_export.prepare_backend_program(original, configs)
    options, post = aoti_export.prepare_backend_options(configs)
    decomposed = prepared.run_decompositions()
    options["post_grad_custom_post_pass"](decomposed.graph)
    assert any(row["target"] == "aten.ones.default" for row in pre["rewrites"])
    assert any(row.get("target") == "aten.full.default" for row in post["rewrites"])
    source_sha = hashlib.sha256(Path(aoti_export.__file__).read_bytes()).hexdigest()
    assert pre["passes"][0]["source_sha256"] == post["passes"][0]["source_sha256"] == source_sha
    assert post["passes"][0]["stage"] == "post_grad_custom_post_pass"
    assert options["post_grad_custom_post_pass"].uuid() is None
    with pytest.raises(ValueError, match="selective decomposition"):
        aoti_export.prepare_backend_program(original, {**configs, "selective_decompose": False})


@pytest.mark.parametrize("dimensions,marked", [(None, False), ([], True), ([0], False)])
def test_optional_empty_int_list_is_not_replaced_with_none(dimensions, marked):
    from vlaforge.deployment.aoti_export import mark_empty_proxy_lists

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    node = graph.call_function(torch.ops.aten.sum.dim_IntList, (value, dimensions))
    graph.output(node)
    before = (node.target, node.args, node.kwargs)
    changes = mark_empty_proxy_lists(graph)
    assert bool(changes) == marked
    assert (node.target, node.args, node.kwargs) == before


def test_dynamic_empty_lists_preserve_existing_native_annotation():
    from vlaforge.deployment.aoti_export import mark_empty_proxy_lists

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    reshape = graph.call_function(torch.ops.aten.view.default, (value, []))
    cat = graph.call_function(torch.ops.aten.cat.default, ([],))
    graph.output((reshape, cat))
    reshape.meta["custom"] = {"compile_with_inductor": {"existing": True}}
    changes = mark_empty_proxy_lists(graph)
    assert [row["arguments"] for row in changes] == [["size"], ["tensors"]]
    assert reshape.meta["custom"]["compile_with_inductor"] == {"existing": True}


def test_omitted_empty_schema_default_is_marked_before_decomposition():
    from vlaforge.deployment.aoti_export import mark_empty_proxy_lists

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    node = graph.call_function(torch.ops.aten.roll.default, (value, [1]))
    graph.output(node)
    changes = mark_empty_proxy_lists(graph)
    assert changes[0]["arguments"] == ["dims"]
    assert node.args == (value, [1])
    assert not node.kwargs
