from dataclasses import replace

import pytest

from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.interpreter import InputBinding, Interpreter
from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import InputPort, OutputPort, TensorRegion, Value
from vlaforge.ir.serializer import canonical_json, io_schema_digest
from vlaforge.ir.types import ScalarType
from vlaforge.transforms import split_region_invocation


def fixture():
    integer, boolean = ScalarType("i64"), ScalarType("bool")
    builder = InvocationBuilder("split-nested", inputs=(
        InputPort("a", integer), InputPort("b", integer), InputPort("accepted", boolean)),
        outputs=(OutputPort("x", integer), OutputPort("y", integer)))

    @tensor_region("step", inputs=(Value("a", integer), Value("b", integer)), outputs=(integer, integer))
    def step(a, b):
        raise AssertionError("declaration only")

    x, y = builder.iterate((builder.input("a"), builder.input("b")),
        lambda _i, a, b: builder.iterate((a, b), lambda _j, x, y: builder.call(step, x, y), steps=2), steps=2)
    module = builder.finish({"x": x, "y": y}, accepted=builder.input("accepted")).module
    first = TensorRegion("first", module.region("step").inputs, (integer, integer))
    second = TensorRegion("second", (Value("a", integer), Value("sum", integer), Value("difference", integer)),
                          (integer, integer))
    return module, first, second


def execute(module, regions):
    result = Interpreter(module, regions=regions, validators={"vlaforge_predicate_true": lambda value: value is True}).run(
        inputs={"a": InputBinding(2), "b": InputBinding(3), "accepted": InputBinding(True)})
    return tuple(item.value for item in result.committed_outputs.outputs)


def test_composition_rewrites_nested_call_and_preserves_external_ssa_contract():
    module, first, second = fixture()
    before = canonical_json(module)
    transformed = split_region_invocation(module, "step", first=first, second=second,
                                          second_inputs=(("input", 0), ("first", 0), ("first", 1)))
    assert execute(module, {"step": lambda a, b: (2 * a + b, a - b)}) == execute(transformed, {
        "first": lambda a, b: (a + b, a - b), "second": lambda a, total, difference: (a + total, difference)})
    assert canonical_json(module) == before
    assert transformed.inputs == module.inputs and transformed.outputs == module.outputs
    assert io_schema_digest(transformed) == io_schema_digest(module)


@pytest.mark.parametrize("route", [("input", True), ("unknown", 0), ("first", 2), ("input", -1)])
def test_route_is_not_implicitly_cast_or_guessed(route):
    module, first, second = fixture()
    with pytest.raises(ValueError, match="route"):
        split_region_invocation(module, "step", first=first, second=second,
                                second_inputs=(route, ("first", 0), ("first", 1)))


def test_effectful_or_incompatible_interfaces_are_rejected():
    module, first, second = fixture()
    with pytest.raises(ValueError, match="pure"):
        split_region_invocation(module, "step", first=replace(first, effects=(Effect.RANDOM,)), second=second,
                                second_inputs=(("input", 0), ("first", 0), ("first", 1)))
    with pytest.raises(ValueError, match="compose"):
        split_region_invocation(module, "step", first=replace(first, inputs=tuple(reversed(first.inputs))), second=second,
                                second_inputs=(("input", 0), ("first", 0), ("first", 1)))
