import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vlaforge.analysis.terminal_tail import partition_terminal_tail
from vlaforge.frontend import InvocationBuilder, InvocationProgram, tensor_region
from vlaforge.interpreter import InputStamp, Interpreter, TensorView
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.serializer import module_digest, module_from_data, module_to_data
from vlaforge.ir.types import TensorType

SPEC = importlib.util.spec_from_file_location(
    "terminal_partition_ir_tool", Path(__file__).resolve().parents[2] / "tools/terminal_partition_ir.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def fixture(formula, *, duplicate=False):
    class Formula(torch.nn.Module):
        def forward(self, value, other):
            if formula == "sine-product":
                return value.sin() * other
            return value.square() - other
    values = (torch.randn(3), torch.randn(3))
    ep = torch.export.export(Formula(), values)
    output = list(ep.graph.nodes)[-1].args[0][0]
    frontier = output.args[0].name
    partition = partition_terminal_tail(ep, inputs=(frontier, "other"), outputs=(output.name,), source_artifact_sha256="a" * 64)
    payload, predicate = TensorType((3,), "f32"), TensorType((1,), "bool")
    full = tensor_region("target", inputs=(Value("value", payload), Value("other", payload)), outputs=(payload,))(ep.module())
    @tensor_region("guard", inputs=(Value("value", payload),), outputs=(predicate,))
    def guard(value):
        return (torch.isfinite(value).all().reshape(1),)
    builder = InvocationBuilder("two_step", inputs=(InputPort("value", payload), InputPort("other", payload)),
                                outputs=(OutputPort("result", payload),))
    value, other = builder.input("value"), builder.input("other")
    def step(_index, current):
        return builder.call(full, current, other)
    (output_value,) = builder.iterate((value,), step, steps=2)
    if duplicate:
        (output_value,) = builder.call(full, output_value, other)
    (accepted,) = builder.call(guard, output_value)
    invocation = builder.finish({"result": output_value}, accepted=accepted)
    return ep, partition, invocation, values


def execute(module, callables, values):
    module = module_from_data(module_to_data(module))
    program = InvocationProgram(module, callables)
    session = Interpreter(module, regions=callables, validators=program.validators)
    for port, value in zip(module.inputs, values, strict=True):
        session.bind_input(port.name, TensorView(value, tuple(value.shape), port.payload.dtype), InputStamp(revision=1))
    session.run(module.invocations[0].name)
    return session.read_output("result")


@pytest.mark.parametrize("formula", ["sine-product", "square-difference"])
def test_real_two_step_ir_wiring_preserves_outer_program(formula):
    ep, partition, program, values = fixture(formula)
    original = module_digest(program.module)
    modified, report = TOOL.wire_terminal_partition(program.module, target="target", partition=partition)
    baseline = execute(program.module, program.regions, values)
    callables = {name: value for name, value in program.regions.items() if name != "target"}
    callables[report["prefix_region"]] = TOOL.interpreter_tensor_callable(partition.prefix.module(), len(partition.frontier_names))
    callables[report["tail_region"]] = TOOL.interpreter_tensor_callable(partition.tail.module, len(partition.tail.output_names))
    actual = execute(modified, callables, values)
    expected = ep.module()(ep.module()(*values), values[1])
    assert torch.equal(actual, baseline) and torch.equal(actual, expected)
    assert module_digest(program.module) == original
    assert report["target_replacements"] == 1
    assert report["original_nonselected_ir_fields_exact"]
    assert report["tail_operands"][1] == report["prefix_operands"][1]
    assert not report["old_artifact_contracts_inherited"]
    assert not report["compiled_artifact_verified"]
    assert modified.inputs == program.module.inputs and modified.outputs == program.module.outputs
    assert modified.states == program.module.states


def test_duplicate_call_rejected():
    _, partition, program, _ = fixture("sine-product", duplicate=True)
    with pytest.raises(ValueError, match="exactly one"):
        TOOL.wire_terminal_partition(program.module, target="target", partition=partition)


@pytest.mark.parametrize("metadata", [{"memoize": True}, {"artifact_sha256": "a" * 64}])
def test_cached_or_prior_artifact_metadata_rejected(metadata):
    _, partition, program, _ = fixture("sine-product")
    module = replace(program.module, regions=tuple(replace(region, metadata=metadata) if region.name == "target" else region
                                                   for region in program.module.regions))
    with pytest.raises(ValueError, match="uncached"):
        TOOL.wire_terminal_partition(module, target="target", partition=partition)


def test_forged_route_rejected():
    _, partition, program, _ = fixture("sine-product")
    partition.ledger["tail_input_routes"][0]["index"] = 99
    with pytest.raises(ValueError, match="ledger changed"):
        TOOL.wire_terminal_partition(program.module, target="target", partition=partition)
