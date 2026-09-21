import copy

import pytest
import torch
from vlaforge.analysis.unused_state import prune_unused_state, save_pruned_state


class Partial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.unused = torch.nn.Parameter(torch.ones(12))
        self.register_buffer("unused_buffer", torch.ones(5))

    def forward(self, x, unused_user_input):
        return x @ self.used.T


def program(model=None):
    return torch.export.export(
        model or Partial(), (torch.ones(4, 3), torch.ones(7)), strict=True
    )


def test_unused_lifted_weights_removed_without_user_ports_or_compute_pruning(tmp_path):
    source = program()
    original_nodes = [(node.name, node.op, node.target) for node in source.graph.nodes]
    result = prune_unused_state(source, source_artifact_sha256="a" * 64)
    assert set(result.ledger["removed_state"]) == {"unused", "unused_buffer"}
    assert set(result.program.state_dict) == {"used"}
    assert set(source.state_dict) == {"used", "unused", "unused_buffer"}
    assert [
        (node.name, node.op, node.target) for node in source.graph.nodes
    ] == original_nodes
    assert any(node.name == "unused_user_input" for node in result.program.graph.nodes)
    assert result.program.state_dict["used"] is source.state_dict["used"]
    assert result.ledger["source_unique_storage_bytes"] == (6 + 12 + 5) * 4
    assert result.ledger["result_unique_storage_bytes"] == 6 * 4
    assert result.ledger["computation_nodes_removed"] == 0
    assert not result.ledger["runtime_peak_reduction_verified"]
    assert not result.ledger["full_model_output_verified"]
    saved = save_pruned_state(result, tmp_path / "pruned")
    reloaded = torch.export.load(tmp_path / "pruned/program.pt2")
    for value in (1.0, 2.0, -3.0):
        args = (torch.full((4, 3), value), torch.ones(7))
        assert torch.equal(source.module()(*args), result.program.module()(*args))
        assert torch.equal(source.module()(*args), reloaded.module()(*args))
    assert saved["pre_and_post_serialization_state_verified"]


def test_aliased_storage_is_not_falsely_counted_as_freed():
    model = Partial()
    model.unused = torch.nn.Parameter(model.used.view(-1))
    source = program(model)
    result = prune_unused_state(source, source_artifact_sha256="a" * 64)
    assert result.ledger["source_unique_storage_bytes"] == (6 + 5) * 4
    assert result.ledger["result_unique_storage_bytes"] == 6 * 4
    assert result.ledger["removed_state"]["unused"]["logical_bytes"] == 24


def test_nested_graph_parameters_remain_if_passed_to_either_branch():
    class Conditional(Partial):
        def forward(self, x, unused_user_input):
            return torch.cond(
                x.sum() > 0,
                lambda a, w: a @ w.T,
                lambda a, w: -(a @ w.T),
                (x, self.used),
            )

    source = program(Conditional())
    result = prune_unused_state(source, source_artifact_sha256="b" * 64)
    assert "used" in result.program.state_dict
    assert len(result.ledger["source_graph_text_sha256"]) == 3
    for sign in (-1.0, 1.0):
        args = (torch.full((4, 3), sign), torch.ones(7))
        assert torch.equal(source.module()(*args), result.program.module()(*args))


def test_hidden_state_or_input_mutation_is_rejected():
    class Mutation(Partial):
        def forward(self, x, unused_user_input):
            self.used.add_(1)
            return x @ self.used.T

    with torch.no_grad():
        source = program(Mutation())
    with pytest.raises(ValueError, match="pure"):
        prune_unused_state(source, source_artifact_sha256="c" * 64)


def test_unused_user_tensor_and_dead_computation_are_not_optimized_away():
    class Dead(Partial):
        def forward(self, x, unused_user_input):
            unused_user_input.sin()
            return x @ self.used.T

    source = program(Dead())
    original = [
        (node.op, str(node.target))
        for node in source.graph.nodes
        if node.op != "placeholder"
    ]
    result = prune_unused_state(source, source_artifact_sha256="d" * 64)
    assert [
        (node.op, str(node.target))
        for node in result.program.graph.nodes
        if node.op != "placeholder"
    ] == original


@pytest.mark.parametrize("mutation", ["state", "graph", "extra_state"])
def test_save_does_not_publish_a_stale_state_ledger(tmp_path, mutation):
    result = prune_unused_state(program(), source_artifact_sha256="e" * 64)
    if mutation == "state":
        with torch.no_grad():
            result.program.state_dict["used"].add_(1)
    elif mutation == "graph":
        node = next(
            node for node in result.program.graph.nodes if node.op == "call_function"
        )
        node.name += "_changed"
    else:
        result.program.state_dict["extra"] = torch.ones(3)
    with pytest.raises(ValueError, match="changed"):
        save_pruned_state(result, tmp_path / "pruned")
    assert not (tmp_path / "pruned").exists()


def test_save_never_overwrites_an_existing_archive(tmp_path):
    result = prune_unused_state(program(), source_artifact_sha256="a" * 64)
    save_pruned_state(result, tmp_path / "pruned")
    with pytest.raises(FileExistsError):
        save_pruned_state(result, tmp_path / "pruned")


def test_explicit_bad_source_hash_is_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        prune_unused_state(program(), source_artifact_sha256="guessed")


def test_explicit_unused_parameter_placeholder_and_its_signature_are_removed(tmp_path):
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    source = program()
    graph = copy.deepcopy(source.graph)
    first = next(iter(graph.nodes))
    with graph.inserting_before(first):
        unused = graph.placeholder("p_explicit_unused")
        unused.meta["val"] = first.meta["val"].fake_mode.from_tensor(
            source.state_dict["unused"]
        )
    signature = copy.deepcopy(source.graph_signature)
    signature.input_specs.insert(
        0, InputSpec(InputKind.PARAMETER, TensorArgument(unused.name), "unused")
    )
    lifted = torch.export.ExportedProgram(
        root=source.graph_module,
        graph=graph,
        graph_signature=signature,
        state_dict=dict(source.state_dict),
        range_constraints=copy.deepcopy(source.range_constraints),
        module_call_graph=copy.deepcopy(source.module_call_graph),
        example_inputs=source.example_inputs,
        constants=dict(source.constants),
        verifiers=source.verifiers,
    )
    lifted.validate()
    result = prune_unused_state(lifted, source_artifact_sha256="a" * 64)
    assert result.ledger["removed_placeholders"] == [
        {"name": "p_explicit_unused", "target": "unused", "kind": "PARAMETER"}
    ]
    assert all(node.name != unused.name for node in result.program.graph.nodes)
    assert all(
        spec.arg.name != unused.name
        for spec in result.program.graph_signature.input_specs
    )
    save_pruned_state(result, tmp_path / "pruned")
    loaded = torch.export.load(tmp_path / "pruned/program.pt2")
    assert torch.equal(
        lifted.module()(*lifted.example_inputs[0]),
        loaded.module()(*lifted.example_inputs[0]),
    )


def test_nonpersistent_constants_stay_bound_and_cannot_change_before_save(tmp_path):
    class Constant(Partial):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.ones(2), persistent=False)

        def forward(self, x, unused_user_input):
            return x @ self.used.T + self.offset

    source = program(Constant())
    result = prune_unused_state(source, source_artifact_sha256="b" * 64)
    assert "offset" in result.ledger["retained_constants"]
    result.program.constants["offset"].add_(1)
    with pytest.raises(ValueError, match="retained state changed"):
        save_pruned_state(result, tmp_path / "pruned")


def test_unused_tied_parameter_name_cannot_remove_a_live_alias():
    model = Partial()
    model.unused = model.used
    source = program(model)
    result = prune_unused_state(source, source_artifact_sha256="c" * 64)
    args = (torch.ones(4, 3), torch.ones(7))
    assert torch.equal(source.module()(*args), result.program.module()(*args))
    assert result.ledger["result_unique_storage_bytes"] == 24


@pytest.mark.parametrize(
    "mutation", ["parameter_target", "output", "call_spec", "ledger"]
)
def test_metadata_only_changes_cannot_reuse_pruning_evidence(tmp_path, mutation):
    result = prune_unused_state(program(), source_artifact_sha256="f" * 64)
    before = str(result.program.graph)
    if mutation == "parameter_target":
        result.program.graph_signature.input_specs[0].target = "different_weight"
    elif mutation == "output":
        result.program.graph_signature.output_specs[0].arg.name = "x"
    elif mutation == "call_spec":
        from torch.utils import _pytree

        result.program.module_call_graph[0].signature.out_spec = _pytree.tree_structure(
            (0,)
        )
    else:
        result.ledger["full_model_output_verified"] = True
    assert str(result.program.graph) == before
    with pytest.raises(ValueError, match="changed"):
        save_pruned_state(result, tmp_path / "pruned")
    assert not (tmp_path / "pruned").exists()


def test_singleton_nonunit_stride_state_keeps_layout_but_hashes_value(tmp_path):
    source = program()
    source.state_dict["singleton"] = torch.empty_strided((1,), (1600,)).fill_(2.5)
    result = prune_unused_state(source, source_artifact_sha256="a" * 64)
    record = result.ledger["removed_state"]["singleton"]
    assert record["stride"] == [1600] and record["logical_bytes"] == 4
    save_pruned_state(result, tmp_path / "pruned")
