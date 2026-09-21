"""Synthetic transaction/predicate contract tests, never real model evidence."""

from dataclasses import replace
import pytest

from vlaforge.adapters.shared.output_stage import OutputStageInput, attach_checked_output_stage
from vlaforge.frontend import InvocationBuilder, InvocationProgram, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, TensorRegion, Value
from vlaforge.ir.types import TensorType


def example():
    action, native, boolean = TensorType((1, 3, 4), "f32"), TensorType((1, 3, 2), "f32"), TensorType((1,), "bool")
    builder = InvocationBuilder("checked_output_fixture", inputs=(InputPort("source", action), InputPort("incoming", boolean)),
        outputs=(OutputPort("source_action", action, group="actions"),))
    value, accepted = builder.input("source"), builder.input("incoming")
    program = builder.finish({"source_action": value}, accepted=accepted)
    region = TensorRegion("output", (Value("source", action), Value("accepted", boolean)), (native, boolean))
    return program.module, region


def joined(module, region, **kwargs):
    return attach_checked_output_stage(module, region,
        inputs=(OutputStageInput("source-output", "source_action"), OutputStageInput("acceptance")),
        source_output="source_action", output_name="native_action", **kwargs)


def test_original_acceptance_is_an_explicit_region_operand():
    module, region = example()
    result = joined(module, region)
    old_guard = module.invocations[0].body.operations[-5].operands[0]
    new_call = result.invocations[0].body.operations[-7]
    assert new_call.opcode == "vla.invoke" and new_call.operands[-1] == old_guard
    assert result.outputs[0] == module.outputs[0]


@pytest.mark.parametrize("kind,name", [("unknown", None), ("acceptance", "guessed"), ("module-input", None), ("source-output", "")])
def test_ambiguous_input_sources_fail_closed(kind, name):
    with pytest.raises(ValueError):
        OutputStageInput(kind, name)


def test_missing_incoming_guard_cannot_use_old_unsafe_interface():
    module, region = example()
    with pytest.raises(ValueError, match="incoming acceptance"):
        attach_checked_output_stage(module, replace(region, inputs=region.inputs[:1]),
            inputs=(OutputStageInput("source-output", "source_action"),), source_output="source_action", output_name="native_action")


@pytest.mark.parametrize("case", ["original-rejection", "discarded-nan"])
def test_original_rejection_or_nan_in_discarded_values_prevents_both_publications(case):
    torch = pytest.importorskip("torch")
    from vlaforge.interpreter import Interpreter, TensorView
    from vlaforge.interpreter.executor import InterpreterError

    module, region = example()
    module = joined(module, region)

    def output(source, incoming):
        native = source[..., :2].contiguous()
        return native, (incoming & torch.isfinite(source).all() & torch.isfinite(native).all()).reshape(1)

    interpreter = Interpreter(module, regions={"output": output}, validators=InvocationProgram(module, {}).validators)
    source = torch.zeros(1, 3, 4)
    incoming = torch.tensor([False if case == "original-rejection" else True])
    if case == "discarded-nan":
        source[0, 0, 3] = float("nan")
    for name, value in (("source", source), ("incoming", incoming)):
        port = module.input(name)
        interpreter.bind_input(name, TensorView(value, tuple(value.shape), port.payload.dtype, port.payload.layout, port.device, port.alignment))
    with pytest.raises(InterpreterError, match="failed validation"):
        interpreter.run()
    for name in ("source_action", "native_action"):
        with pytest.raises(InterpreterError, match="committed"):
            interpreter.read_output(name)
