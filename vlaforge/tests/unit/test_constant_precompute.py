import importlib.util
import json
from pathlib import Path

import pytest
import torch
from vlaforge.analysis.constant_precompute import (
    ImmutableSnapshot,
    declare_immutable_snapshot,
    file_sha256,
    graph_sha256,
    precompute_constants,
    save_precompute_bundle,
    tensor_record,
)


class FixedPositions(torch.nn.Module):
    def __init__(self, *, persistent=True, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(24, dtype=dtype).reshape(6, 4))
        self.register_buffer("ids", torch.tensor([4, 0, 2]), persistent=persistent)

    def forward(self, image, other):
        positions = torch.nn.functional.embedding(self.ids.unsqueeze(0).expand(1, -1), self.weight)
        return image + positions, other * positions


def exported(module=None):
    module = FixedPositions() if module is None else module
    dtype = module.weight.dtype
    return torch.export.export(module, (torch.ones(1, 3, 4, dtype=dtype), torch.ones(1, 3, 4, dtype=dtype)))


def snapshot(program, targets=("weight", "ids")):
    return declare_immutable_snapshot(program, targets, snapshot_id="unit-test-only",
                                      source_artifact_sha256="a" * 64)


def fold(program):
    return precompute_constants(program, output_nodes=("embedding",), snapshot=snapshot(program))


def raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def test_real_cpu_export_three_variants_preserve_ports_state_and_bits(tmp_path):
    program = exported()
    before = graph_sha256(program)
    weights = {name: value.clone() for name, value in program.state_dict.items()}
    result = fold(program)
    assert result.control is not program and result.control.graph is not program.graph
    assert graph_sha256(program) == graph_sha256(result.control) == before
    assert graph_sha256(result.folded) != before
    assert result.folded.call_spec == program.call_spec
    assert result.folded.graph_signature.user_inputs == program.graph_signature.user_inputs
    assert result.folded.graph_signature.user_outputs == program.graph_signature.user_outputs
    for name, value in weights.items():
        assert torch.equal(raw(program.state_dict[name]), raw(value))
        assert program.state_dict[name].data_ptr() == result.control.state_dict[name].data_ptr()
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        inputs = tuple(torch.randn(1, 3, 4, generator=generator) for _ in range(2))
        outputs = [variant.module()(*inputs) for variant in (program, result.control, result.folded)]
        for actual in outputs[1:]:
            assert all(torch.equal(raw(left), raw(right)) for left, right in zip(outputs[0], actual, strict=True))
    ledger = save_precompute_bundle(result, tmp_path / "bundle")
    for label in ("control", "folded"):
        path = tmp_path / "bundle" / ledger["exported_artifacts"][label]["path"]
        assert file_sha256(path) == ledger["exported_artifacts"][label]["sha256"]
        loaded = torch.export.load(path)
        assert all(torch.equal(raw(left), raw(right)) for left, right in zip(
            program.module()(*program.example_inputs[0]), loaded.module()(*program.example_inputs[0]), strict=True))
    with pytest.raises(FileExistsError):
        save_precompute_bundle(result, tmp_path / "bundle")
    assert set(ledger["folds"][0]["dependencies"]) == {"weight", "ids"}
    assert ledger["new_constant_bytes"] == 48
    assert ledger["kernel_candidate_selected"] is False
    assert ledger["compiled_artifact_verified"] is False
    assert ledger["full_model_verified"] is False
    json.dumps(ledger, allow_nan=False)


def test_bfloat16_never_casts_raw_bytes_or_changes_dtype():
    program = exported(FixedPositions(dtype=torch.bfloat16))
    result = fold(program)
    record = result.ledger["folds"][0]["output"]
    assert record["dtype"] == "torch.bfloat16" and record["bytes"] == 24
    assert result.folded.state_dict["_vlaforge_precomputed_0"].dtype == torch.bfloat16
    for expected, actual in zip(program.module()(*program.example_inputs[0]),
                                result.folded.module()(*program.example_inputs[0]), strict=True):
        assert torch.equal(raw(expected), raw(actual))


def test_snapshot_is_explicit_and_tamper_fails_closed():
    program = exported()
    proof = snapshot(program)
    with pytest.raises(TypeError):
        proof.tensor_sha256["weight"] = "b" * 64
    with torch.no_grad():
        program.state_dict["weight"].add_(1)
    with pytest.raises(ValueError, match="snapshot changed"):
        precompute_constants(program, output_nodes=("embedding",), snapshot=proof)
    with pytest.raises(ValueError, match="lifted parameter"):
        snapshot(program, ("image",))
    with pytest.raises(ValueError, match="nonempty and unique"):
        snapshot(program, ("weight", "weight"))
    with pytest.raises(ValueError, match="SHA256"):
        ImmutableSnapshot("test", "bad", "a" * 64, {"weight": "b" * 64})


def test_graph_tamper_is_not_repaired_by_matching_tensor_values():
    program = exported()
    proof = snapshot(program)
    node = next(node for node in program.graph.nodes if node.name == "expand")
    node.args = (node.args[0], [1, 3])
    with pytest.raises(ValueError, match="source graph differs"):
        precompute_constants(program, output_nodes=("embedding",), snapshot=proof)


def test_dynamic_user_input_leaf_is_never_inferred_constant():
    program = exported()
    with pytest.raises(ValueError, match="user input leaf"):
        precompute_constants(program, output_nodes=("add",), snapshot=snapshot(program))
    with pytest.raises(ValueError, match="unapproved state"):
        precompute_constants(program, output_nodes=("embedding",), snapshot=snapshot(program, ("weight",)))


def test_nonpersistent_and_unlifted_constants_are_not_snapshot_state():
    program = exported(FixedPositions(persistent=False))
    with pytest.raises(ValueError, match="lifted parameter or persistent buffer"):
        snapshot(program)
    with pytest.raises(ValueError, match="unapproved state"):
        precompute_constants(program, output_nodes=("embedding",), snapshot=snapshot(program, ("weight",)))


def test_view_output_and_direct_external_escape_are_rejected():
    program = exported()
    with pytest.raises(ValueError, match="aliases its dependencies"):
        precompute_constants(program, output_nodes=("expand",), snapshot=snapshot(program))

    class Escape(FixedPositions):
        def forward(self, image, other):
            return torch.nn.functional.embedding(self.ids, self.weight).view(1, 3, 4), image + other

    escaped = exported(Escape())
    with pytest.raises(ValueError, match="external alias escape"):
        fold(escaped)


def test_downstream_mutation_of_previously_fresh_output_is_rejected():
    class Mutates(FixedPositions):
        def forward(self, image, other):
            positions = torch.nn.functional.embedding(self.ids, self.weight)
            positions.view(1, 3, 4).add_(image)
            return positions + other

    program = exported(Mutates())
    with pytest.raises(ValueError, match="mutated through"):
        fold(program)


def test_schema_hidden_unsafe_view_and_state_alias_escape_are_rejected():
    class UnsafeView(FixedPositions):
        def forward(self, image, other):
            value = torch.nn.functional.embedding(self.ids, self.weight)
            return torch.ops.aten._unsafe_view.default(value, [1, 3, 4]), image + other

    with pytest.raises(ValueError, match="alias-unsafe"):
        fold(exported(UnsafeView()))

    class StateEscape(FixedPositions):
        def forward(self, image, other):
            values = super().forward(image, other)
            return *values, self.weight.view(6, 4)

    with pytest.raises(ValueError, match="external alias escape"):
        fold(exported(StateEscape()))


def test_hidden_mutation_and_rng_are_rejected_before_execution():
    class HiddenWrite(FixedPositions):
        def forward(self, image, other):
            self.ids.add_(0)
            return super().forward(image, other)

    with pytest.raises(ValueError, match="effect audit failed"):
        fold(exported(HiddenWrite()))

    class Random(FixedPositions):
        def forward(self, image, other):
            return super().forward(image + torch.rand_like(image), other)

    with pytest.raises(ValueError, match="effect audit failed"):
        fold(exported(Random()))


def test_unknown_python_call_and_non_allowlisted_aten_are_rejected():
    program = exported()
    node = next(node for node in program.graph.nodes if node.name == "embedding")
    node.target = lambda *args: args[0]
    with pytest.raises(ValueError, match="unknown call"):
        fold(program)

    class UnknownPure(FixedPositions):
        def forward(self, image, other):
            return image + torch.linalg.vector_norm(self.weight), other

    program = exported(UnknownPure())
    name = next(node.name for node in program.graph.nodes if "linalg_vector_norm" in str(node.target))
    with pytest.raises(ValueError, match="pure precompute allowlist"):
        precompute_constants(program, output_nodes=(name,), snapshot=snapshot(program, ("weight",)))


def test_nonfinite_dependency_or_result_is_rejected():
    program = exported()
    with torch.no_grad():
        program.state_dict["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        fold(program)

    class Overflow(FixedPositions):
        def forward(self, image, other):
            constant = self.weight * 3.0e38
            return image + constant[:3], other

    program = exported(Overflow())
    with pytest.raises(ValueError, match="nonfinite"):
        precompute_constants(program, output_nodes=("mul",), snapshot=snapshot(program, ("weight",)))


def test_selected_nodes_are_nonempty_unique_and_known():
    program = exported()
    for names in ((), ("embedding", "embedding")):
        with pytest.raises(ValueError, match="nonempty and unique"):
            precompute_constants(program, output_nodes=names, snapshot=snapshot(program))
    for names in (("absent",), ("image",)):
        with pytest.raises(ValueError, match="not an operator"):
            precompute_constants(program, output_nodes=names, snapshot=snapshot(program))


def test_shared_dependencies_and_two_uses_do_not_alias_new_constants():
    class Twice(FixedPositions):
        def forward(self, image, other):
            first = torch.nn.functional.embedding(self.ids, self.weight)
            second = torch.nn.functional.embedding(self.ids, self.weight)
            return image + first, other + second

    program = exported(Twice())
    result = precompute_constants(program, output_nodes=("embedding", "embedding_1"), snapshot=snapshot(program))
    left = result.folded.state_dict["_vlaforge_precomputed_0"]
    right = result.folded.state_dict["_vlaforge_precomputed_1"]
    assert left.data_ptr() != right.data_ptr()
    assert torch.equal(raw(left), raw(right))
    assert len(result.ledger["folds"]) == 2


def test_metadata_is_hashed_and_control_custom_metadata_is_independent():
    program = exported()
    node = next(node for node in program.graph.nodes if node.name == "embedding")
    node.meta["custom"] = {"audit": [1]}
    result = fold(program)
    control_node = next(node for node in result.control.graph.nodes if node.name == "embedding")
    control_node.meta["custom"]["audit"].append(2)
    assert node.meta["custom"] == {"audit": [1]}
    one = tensor_record(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    two = tensor_record(torch.arange(4, dtype=torch.float32).reshape(2, 2).t())
    assert one["sha256"] != two["sha256"]


def test_generic_arithmetic_does_not_depend_on_embedding_or_model_names():
    class Trigonometry(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.arange(12, dtype=torch.float32).reshape(1, 3, 4))

        def forward(self, values):
            return values + self.table.sin()

    program = torch.export.export(Trigonometry(), (torch.randn(1, 3, 4),))
    result = precompute_constants(program, output_nodes=("sin",), snapshot=snapshot(program, ("table",)))
    assert torch.equal(raw(program.module()(*program.example_inputs[0])), raw(result.folded.module()(*program.example_inputs[0])))


class LiteralFrequencies(torch.nn.Module):
    def forward(self, values):
        frequencies = torch.exp(torch.arange(0, 128, dtype=torch.float32, device="cpu") * -9.210340371976184 / 128)
        return values * frequencies.to(dtype=torch.float64)


def test_literal_only_precompute_preserves_factory_device_math_and_changed_inputs(tmp_path):
    program = torch.export.export(LiteralFrequencies(), (torch.ones(2, 128, dtype=torch.float64),))
    node = next(node for node in program.graph.nodes if str(node.target).startswith("aten.to."))
    proof = declare_immutable_snapshot(program, (), literal_only=True,
        snapshot_id="explicit-literal-only", source_artifact_sha256="a" * 64)
    result = precompute_constants(program, output_nodes=(node.name,), snapshot=proof)
    assert not result.ledger["folds"][0]["dependencies"]
    assert result.ledger["new_constant_bytes"] == 128 * 8
    assert result.folded.call_spec == program.call_spec
    assert node.name in result.ledger["removed_nodes"]
    # Metadata assertions are retained even when they keep a CPU producer live.
    def assertions(ep):
        return [node.name for node in ep.graph.nodes if "_assert_tensor_metadata" in str(node.target)]
    assert assertions(result.folded) == assertions(program)
    for value in (0.0, 1.0, -2.5):
        inputs = torch.full((2, 128), value, dtype=torch.float64)
        assert torch.equal(raw(program.module()(inputs)), raw(result.folded.module()(inputs)))
    saved = save_precompute_bundle(result, tmp_path / "literal")
    loaded = torch.export.load(tmp_path / "literal" / saved["exported_artifacts"]["folded"]["path"])
    inputs = torch.arange(256, dtype=torch.float64).reshape(2, 128)
    assert torch.equal(raw(program.module()(inputs)), raw(loaded.module()(inputs)))


def test_literal_only_snapshot_never_approves_state_or_example_inputs():
    program = exported()
    proof = declare_immutable_snapshot(program, (), literal_only=True,
        snapshot_id="literal-no-state", source_artifact_sha256="a" * 64)
    with pytest.raises(ValueError, match="unapproved state or user input"):
        precompute_constants(program, output_nodes=("embedding",), snapshot=proof)
    with pytest.raises(ValueError, match="unapproved state or user input"):
        precompute_constants(program, output_nodes=("add",), snapshot=proof)
    with pytest.raises(ValueError, match="literal-only"):
        declare_immutable_snapshot(program, ("weight",), literal_only=True,
            snapshot_id="mixed", source_artifact_sha256="a" * 64)
    with pytest.raises(ValueError, match="nonempty and unique"):
        declare_immutable_snapshot(program, (), snapshot_id="implicit-empty", source_artifact_sha256="a" * 64)


def test_literal_factory_requires_explicit_dtype_and_device():
    program = torch.export.export(LiteralFrequencies(), (torch.ones(2, 128, dtype=torch.float64),))
    factory = next(node for node in program.graph.nodes if str(node.target).startswith("aten.arange."))
    factory.kwargs = {key: value for key, value in factory.kwargs.items() if key != "device"}
    proof = declare_immutable_snapshot(program, (), literal_only=True,
        snapshot_id="ambient-device-rejected", source_artifact_sha256="a" * 64)
    node = next(node for node in program.graph.nodes if str(node.target).startswith("aten.to."))
    with pytest.raises(ValueError, match="explicit dtype and device"):
        precompute_constants(program, output_nodes=(node.name,), snapshot=proof)


def test_literal_only_cli_requires_explicit_mode_and_preserves_storage(tmp_path):
    program = torch.export.export(LiteralFrequencies(), (torch.ones(2, 128, dtype=torch.float64),))
    path = tmp_path / "frequency.pt2"
    torch.export.save(program, path)
    node = next(node for node in program.graph.nodes if str(node.target).startswith("aten.to."))
    tool = inspection_tool()
    assert tool.main(["precompute", "--program", str(path), "--node", node.name,
        "--output", str(tmp_path / "run"), "--snapshot-id", "literal-cli", "--literal-only",
        "--expected-source-sha256", file_sha256(path)]) == 0
    ledger = json.loads((tmp_path / "run/exports/ledger.json").read_text())
    assert ledger["snapshot"]["tensor_sha256"] == {}
    assert ledger["folds"][0]["dependencies"] == {}
    assert ledger["compiled_artifact_verified"] is False


def test_same_device_dtype_conversion_does_not_qualify_as_fresh():
    class Alias(torch.nn.Module):
        def forward(self, values):
            constant = (torch.arange(0, 128, dtype=torch.float32, device="cpu") / 128).exp()
            return values * torch.ops.aten.to.dtype(constant, torch.float32, False, False)

    program = torch.export.export(Alias(), (torch.ones(128),))
    node = next(node for node in program.graph.nodes if str(node.target).startswith("aten.to."))
    proof = declare_immutable_snapshot(program, (), literal_only=True,
        snapshot_id="alias-rejected", source_artifact_sha256="a" * 64)
    with pytest.raises(ValueError, match="aliases its dependencies"):
        precompute_constants(program, output_nodes=(node.name,), snapshot=proof)


def inspection_tool():
    path = Path(__file__).parents[2] / "tools" / "precompute_exported_constants.py"
    spec = importlib.util.spec_from_file_location("precompute_tool_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_inspection_has_explicit_unverified_state(tmp_path):
    program = exported()
    path = tmp_path / "program.pt2"
    torch.export.save(program, path)
    tool = inspection_tool()
    record = tool.inspect_archive(path, ("embedding",))
    assert set(record["candidates"][0]["dependencies"]) == {"weight", "ids"}
    assert record["source_artifact_sha256"] == file_sha256(path)
    assert record["tensor_storage_loaded"] is False
    assert record["dependency_tensor_hashes_verified"] is False
    assert record["effect_and_alias_audit_completed"] is False
    with pytest.raises(ValueError, match="user input"):
        tool.inspect_archive(path, ("add",))


def test_cli_requires_explicit_identity_before_tensor_load_and_retains_failure(tmp_path, monkeypatch):
    path = tmp_path / "program.pt2"
    torch.export.save(exported(), path)
    tool = inspection_tool()
    monkeypatch.setattr(torch.export, "load", lambda *args, **kwargs: pytest.fail("must not load tensors"))
    output = tmp_path / "missing-contract"
    with pytest.raises(ValueError, match="explicit immutable"):
        tool.main(["precompute", "--program", str(path), "--node", "embedding", "--output", str(output)])
    assert (output / "inspection.json").is_file()
    assert json.loads((output / "failure.json").read_text())["full_model_verified"] is False
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        tool.main(["inspect", "--program", str(path), "--node", "embedding", "--output", str(tmp_path / "bad-sha"),
                   "--expected-source-sha256", "b" * 64])


def test_cli_cpu_end_to_end_produces_control_and_folded_without_overwrite(tmp_path):
    path = tmp_path / "program.pt2"
    torch.export.save(exported(), path)
    tool = inspection_tool()
    output = tmp_path / "run"
    args = ["precompute", "--program", str(path), "--node", "embedding", "--output", str(output),
            "--expected-source-sha256", file_sha256(path), "--snapshot-id", "cpu-test",
            "--immutable-state-target", "weight", "--immutable-state-target", "ids"]
    assert tool.main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "new_exported_programs_written"
    assert report["ledger"]["compiled_artifact_verified"] is False
    assert (output / "exports" / "control.pt2").is_file()
    assert (output / "exports" / "folded.pt2").is_file()
    with pytest.raises(FileExistsError):
        tool.main(args)


@pytest.mark.parametrize("change", ("state", "constant", "graph"))
def test_save_rejects_changes_after_precompute(tmp_path, change):
    result = fold(exported())
    with torch.inference_mode():
        if change == "state":
            result.folded.state_dict["weight"].add_(1)
        elif change == "constant":
            result.folded.state_dict["_vlaforge_precomputed_0"].add_(1)
        else:
            node = next(node for node in result.folded.graph.nodes if node.name == "add")
            node.target = torch.ops.aten.sub.Tensor
    with pytest.raises(ValueError, match="changed before/during export"):
        save_precompute_bundle(result, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_save_does_not_publish_ledger_after_state_changed_during_serialization(tmp_path, monkeypatch):
    result = fold(exported())
    original = torch.export.save

    def tamper(program, path):
        original(program, path)
        if Path(path).name == "folded.pt2":
            with torch.no_grad():
                result.control.state_dict["weight"].add_(1)

    monkeypatch.setattr(torch.export, "save", tamper)
    with pytest.raises(ValueError, match="immutable state changed"):
        save_precompute_bundle(result, tmp_path / "changed")
    assert (tmp_path / "changed" / "folded.pt2").is_file()
    assert not (tmp_path / "changed" / "ledger.json").exists()
