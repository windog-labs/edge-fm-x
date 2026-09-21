import copy

import pytest

torch = pytest.importorskip("torch")

from vlaforge.analysis.constant_precompute import graph_sha256
from vlaforge.analysis.tensor_slice import extract_tensor_slice


class Solver(torch.nn.Module):
    def forward(self, sample, velocity, step):
        return sample + step * velocity, step + 1


def solver():
    args = (torch.randn(1, 5, 6), torch.randn(1, 5, 6), torch.tensor([-0.1]))
    return torch.export.export(Solver(), args), args


def extract(program, inputs=("sample", "velocity", "step"), outputs=("add", "add_1")):
    return extract_tensor_slice(program, inputs=inputs, outputs=outputs, source_artifact_sha256="a" * 64)


def test_exact_solver_and_export_roundtrip(tmp_path):
    program, args = solver()
    before = graph_sha256(program)
    result = extract(program)
    result.validate_inputs(args)
    expected = program.module()(*args)
    actual = result.module(*args)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    archive = tmp_path / "slice.pt2"
    torch.export.save(torch.export.export(result.module, args, strict=True), archive)
    reloaded = torch.export.load(archive).module()(*args)
    assert all(torch.equal(a, b) for a, b in zip(expected, reloaded, strict=True))
    assert graph_sha256(program) == before
    assert [item["node"] for item in result.ledger["nodes"]] == ["mul", "add", "add_1"]
    assert not result.ledger["source_executed"]
    assert not result.ledger["source_output_verified"]
    assert not result.ledger["selected_for_deployment"]


def test_explicit_intermediate_boundary_and_order():
    program, args = solver()
    result = extract(program, inputs=("mul", "sample"), outputs=("add",))
    assert torch.equal(result.module(args[2] * args[1], args[0])[0], Solver()(*args)[0])
    assert result.input_names == ("mul", "sample")
    assert [item["node"] for item in result.ledger["nodes"]] == ["add"]


def test_linear_state_must_be_explicit_and_is_not_copied():
    module = torch.nn.Linear(8, 4)
    value = torch.randn(3, 8)
    program = torch.export.export(module, (value,))
    inputs = tuple(node.name for node in program.graph.nodes if node.op == "placeholder")
    result = extract(program, inputs=inputs, outputs=("linear",))
    assert not dict(result.module.named_parameters())
    assert not dict(result.module.named_buffers())
    assert torch.equal(result.module(module.weight, module.bias, value)[0], module(value))
    with pytest.raises(ValueError, match="escapes explicit inputs"):
        extract(program, inputs=("input",), outputs=("linear",))


@pytest.mark.parametrize("inputs,outputs", [
    ((), ("add",)), (("sample",), ()), (("sample", "sample"), ("add",)),
    (("sample",), ("add", "add")), ("sample", ("add",)),
    (("missing",), ("add",)), (("sample",), ("sample",)),
    ((1,), ("add",)),
])
def test_invalid_boundaries(inputs, outputs):
    program, _ = solver()
    with pytest.raises(ValueError):
        extract(program, inputs, outputs)


def test_unused_and_missing_boundary_rejected():
    program, _ = solver()
    with pytest.raises(ValueError, match="unused"):
        extract(program, outputs=("add_1",))
    with pytest.raises(ValueError, match="escapes"):
        extract(program, inputs=("sample", "step"), outputs=("add",))


@pytest.mark.parametrize("digest", [None, "x" * 64, "A" * 64, "a" * 63])
def test_invalid_source_digest(digest):
    program, _ = solver()
    with pytest.raises(ValueError, match="SHA256"):
        extract_tensor_slice(program, inputs=("step",), outputs=("add_1",), source_artifact_sha256=digest)


def test_dynamic_profile_rejected():
    class Dynamic(torch.nn.Module):
        def forward(self, value):
            return value.sin()
    program = torch.export.export(Dynamic(), (torch.randn(4),), dynamic_shapes=({0: torch.export.Dim("n")},))
    with pytest.raises(ValueError, match="static ExportedProgram"):
        extract(program, inputs=("value",), outputs=("sin",))


def test_implicit_random_rejected():
    class Random(torch.nn.Module):
        def forward(self, value):
            return value + torch.rand_like(value)
    program = torch.export.export(Random(), (torch.ones(3),))
    with pytest.raises(ValueError, match="effect audit|implicit RNG"):
        extract(program, inputs=("value",), outputs=("add",))


def test_metadata_is_not_rewritten():
    program, _ = solver()
    node = next(node for node in program.graph.nodes if node.name == "mul")
    node.meta["custom"] = {"notes": [1]}
    before = copy.deepcopy(node.meta["custom"])
    result = extract(program)
    copied = next(node for node in result.module.graph.nodes if node.name == "mul")
    copied.meta["custom"]["notes"].append(2)
    assert node.meta["custom"] == before


def test_missing_metadata_rejected():
    program, _ = solver()
    next(node for node in program.graph.nodes if node.name == "mul").meta.pop("val")
    with pytest.raises(ValueError, match="metadata"):
        extract(program)


def test_local_write_outside_dataflow_is_not_silently_dropped():
    class LocalWrite(torch.nn.Module):
        def forward(self, value):
            copied = value.clone()
            copied.add_(1)
            return copied.sin()
    program = torch.export.export(LocalWrite(), (torch.randn(3),))
    # Emulate an unfunctionalized graph whose read names the original alias.
    clone = next(node for node in program.graph.nodes if node.name == "clone")
    sine = next(node for node in program.graph.nodes if node.name == "sin")
    sine.args = (clone,)
    program.graph_module.recompile()
    with pytest.raises(ValueError, match="mutation aliases"):
        extract(program, inputs=("value",), outputs=("sin",))
    with pytest.raises(ValueError, match="mutation aliases"):
        extract(program, inputs=("clone",), outputs=("sin",))


def test_unrelated_local_workspace_mutation_is_allowed():
    class IndependentWrite(torch.nn.Module):
        def forward(self, value):
            copied = value.clone()
            copied.add_(1)
            return value.sin(), copied
    value = torch.randn(3)
    program = torch.export.export(IndependentWrite(), (value,))
    result = extract(program, inputs=("value",), outputs=("sin",))
    assert torch.equal(result.module(value)[0], value.sin())


def test_uninitialized_storage_rejected():
    class Empty(torch.nn.Module):
        def forward(self, value):
            return value + torch.empty_like(value)
    program = torch.export.export(Empty(), (torch.ones(3),))
    with pytest.raises(ValueError, match="uninitialized"):
        extract(program, inputs=("value",), outputs=("add",))


@pytest.mark.parametrize("change", ["dtype", "shape", "stride", "offset", "arity", "literal"])
def test_actual_input_profile_validation(change):
    program, original = solver()
    result = extract(program)
    args = list(original)
    if change == "dtype":
        args[0] = args[0].double()
    elif change == "shape":
        args[0] = torch.ones(1, 2, 6)
    elif change == "stride":
        args[0] = torch.ones(1, 6, 5).transpose(1, 2)
    elif change == "offset":
        args[0] = torch.ones(60)[1:31].reshape(1, 5, 6)
    elif change == "arity":
        args.pop()
    else:
        args[0] = 1.0
    with pytest.raises(ValueError, match="slice input"):
        result.validate_inputs(tuple(args))
