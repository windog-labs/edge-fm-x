from dataclasses import replace

import pytest

from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.interpreter import InputBinding, Interpreter
from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import InputPort, OutputPort, TensorRegion, Value
from vlaforge.ir.serializer import canonical_json, io_schema_digest
from vlaforge.ir.types import ScalarType
from vlaforge.transforms.region_chain import compose_region_invocation


def fixture(*, twice=False):
    integer = ScalarType("i64")
    boolean = ScalarType("bool")
    builder = InvocationBuilder("typed-chain", inputs=(InputPort("a", integer),
        InputPort("b", integer), InputPort("valid", boolean)),
        outputs=(OutputPort("x", integer), OutputPort("y", integer)))

    @tensor_region("step", inputs=(Value("a", integer), Value("b", integer)),
                   outputs=(integer, integer))
    def step(a, b):
        raise AssertionError("declaration only")

    def inner(_j, x, y):
        if twice:
            x, y = builder.call(step, x, y)
        return builder.call(step, x, y)
    x, y = builder.iterate((builder.input("a"), builder.input("b")),
        lambda _i, a, b: builder.iterate((a, b), inner, steps=2), steps=2)
    module = builder.finish({"x": x, "y": y}, accepted=builder.input("valid")).module
    stages = (
        TensorRegion("control", (Value("a", integer), Value("b", integer)), (integer, integer)),
        TensorRegion("middle", (Value("a", integer),), (integer,)),
        TensorRegion("last", (Value("total", integer), Value("middle", integer)), (integer,)))
    routes = ((("input", 0), ("input", 1)), (("input", 0),),
              (("stage", 0, 0), ("stage", 1, 0)))
    outputs = ((2, 0), (0, 1))
    return module, stages, routes, outputs


def compose(module, stages, routes, outputs):
    return compose_region_invocation(module, "step", stages=stages,
                                      stage_inputs=routes, output_routes=outputs)


def execute(module, regions):
    result = Interpreter(module, regions=regions,
        validators={"vlaforge_predicate_true": lambda value: value is True}).run(inputs={
            "a": InputBinding(2), "b": InputBinding(3), "valid": InputBinding(True)})
    return tuple(o.value for o in result.committed_outputs.outputs)


def controls(module):
    def walk(block):
        for op in block.operations:
            if op.opcode != "vla.invoke":
                yield op.opcode, op.results, op.operands, dict(op.attributes), op.location
            for child in op.regions:
                yield from walk(child)
    return [list(walk(i.body)) for i in module.invocations]


def test_three_stage_cross_live_routes_and_distributed_results_inside_nested_loops():
    module, stages, routes, outputs = fixture()
    before = canonical_json(module)
    transformed = compose(module, stages, routes, outputs)
    actual = execute(transformed, {"control": lambda a, b: (a + b, a - b),
        "middle": lambda a: a, "last": lambda total, a: total + a})
    assert actual == execute(module, {"step": lambda a, b: (2 * a + b, a - b)})
    assert canonical_json(module) == before
    assert controls(transformed) == controls(module)
    assert transformed.inputs == module.inputs and transformed.outputs == module.outputs
    assert transformed.states == module.states
    assert io_schema_digest(transformed) == io_schema_digest(module)


@pytest.mark.parametrize("route", [("input", True), ("input", -1), ("stage", 2, 0),
    ("stage", 1, 0), ("stage", -1, 0), ("stage", 0, True), ("stage", 0, 2),
    ("input", 0, 0), ("unknown", 0), ["input", 0]])
def test_invalid_future_or_implicitly_cast_input_routes_are_rejected(route):
    module, stages, routes, outputs = fixture()
    bad = (routes[0], (route,), routes[2])
    with pytest.raises(ValueError, match="route"):
        compose(module, stages, bad, outputs)


@pytest.mark.parametrize("outputs", [((2, 0),), ((2, 0), (2, 0)),
    ((True, 0), (0, 1)), ((3, 0), (0, 1)), ((2, -1), (0, 1))])
def test_incomplete_duplicate_or_invalid_output_routes_are_rejected(outputs):
    module, stages, routes, _ = fixture()
    with pytest.raises(ValueError, match="route"):
        compose(module, stages, routes, outputs)


def test_effectful_cached_colliding_or_mismatched_stages_are_rejected():
    module, stages, routes, outputs = fixture()
    for changed in (replace(stages[0], effects=(Effect.RANDOM,)),
                    replace(stages[0], metadata={"memoize": True})):
        with pytest.raises(ValueError, match="pure uncached"):
            compose(module, (changed, *stages[1:]), routes, outputs)
    with pytest.raises(ValueError, match="collision"):
        compose(module, (replace(stages[0], name="step"), *stages[1:]), routes, outputs)
    with pytest.raises(ValueError, match="type mismatch"):
        compose(module, (replace(stages[0], inputs=(Value("a", ScalarType("bool")),
            stages[0].inputs[1])), *stages[1:]), routes, outputs)


def test_exactly_one_call_site_is_required():
    module, stages, routes, outputs = fixture()
    with pytest.raises(ValueError, match="unknown"):
        compose_region_invocation(module, "missing", stages=stages,
            stage_inputs=routes, output_routes=outputs)
    # An unused but declared Region has no call site and cannot be expanded.
    unused = TensorRegion("unused", module.region("step").inputs, module.region("step").outputs)
    with pytest.raises(ValueError, match="exactly one"):
        compose_region_invocation(replace(module, regions=module.regions + (unused,)), "unused",
            stages=stages, stage_inputs=routes, output_routes=outputs)
    multiple, stages, routes, outputs = fixture(twice=True)
    with pytest.raises(ValueError, match="exactly one"):
        compose(multiple, stages, routes, outputs)


def test_empty_chain_and_route_count_mismatch_are_rejected():
    module, stages, routes, outputs = fixture()
    for selected, bindings in (((), ()), (stages, routes[:-1]), (stages, ((), *routes[1:]))):
        with pytest.raises(ValueError, match="route"):
            compose(module, selected, bindings, outputs)


def test_mixed_typed_early_predicate_does_not_short_circuit_remaining_computation():
    integer, boolean = ScalarType("i64"), ScalarType("bool")
    builder = InvocationBuilder("false-data-predicate", inputs=(InputPort("x", integer),
        InputPort("flag", boolean), InputPort("commit", boolean)), outputs=(
        OutputPort("action", integer), OutputPort("counter", integer), OutputPort("flag", boolean)))

    @tensor_region("original", inputs=(Value("x", integer), Value("flag", boolean)),
                   outputs=(integer, integer, boolean))
    def original(x, flag):
        raise AssertionError("declaration only")

    action, counter, flag = builder.call(original, builder.input("x"), builder.input("flag"))
    source = builder.finish({"action": action, "counter": counter, "flag": flag},
                            accepted=builder.input("commit")).module
    first = TensorRegion("control", source.region("original").inputs, (integer, boolean))
    second = TensorRegion("compute", (Value("x", integer), Value("counter", integer)), (integer,))
    composed = compose_region_invocation(source, "original", stages=(first, second),
        stage_inputs=((("input", 0), ("input", 1)), (("input", 0), ("stage", 0, 0))),
        output_routes=((1, 0), (0, 0), (0, 1)))
    calls = []
    def control(x, flag):
        calls.append("control")
        return x + 1, flag
    def compute(x, counter):
        calls.append("compute")
        return x + counter
    result = Interpreter(composed, regions={"control": control, "compute": compute},
        validators={"vlaforge_predicate_true": lambda value: value is True}).run(inputs={
            "x": InputBinding(7), "flag": InputBinding(False), "commit": InputBinding(True)})
    assert tuple(o.value for o in result.committed_outputs.outputs) == (15, 8, False)
    assert calls == ["control", "compute"]
    with pytest.raises(ValueError, match="type mismatch"):
        compose_region_invocation(source, "original", stages=(first, second),
            stage_inputs=((("input", 0), ("input", 1)), (("input", 0), ("stage", 0, 0))),
            output_routes=((1, 0), (0, 1), (0, 0)))
