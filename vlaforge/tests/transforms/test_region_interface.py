from dataclasses import replace

import pytest

from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.interpreter import InputBinding, Interpreter
from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.serializer import canonical_json, io_schema_digest
from vlaforge.ir.types import ScalarType
from vlaforge.transforms import configure_exact_cache, project_region_inputs, reorder_region_interface


def example():
    integer, boolean = ScalarType("i64"), ScalarType("bool")
    builder = InvocationBuilder("artifact_order", inputs=(
        InputPort("x", integer), InputPort("y", integer), InputPort("accepted", boolean),
    ), outputs=(OutputPort("first", integer), OutputPort("second", integer)))

    @tensor_region("pair", inputs=(Value("a", integer), Value("b", integer)), outputs=(integer, integer))
    def pair(a, b):
        raise AssertionError("declaration only")

    @tensor_region("tail", inputs=(Value("a", integer),), outputs=(integer,))
    def tail(a):
        raise AssertionError("declaration only")

    first, second = builder.call(pair, builder.input("x"), builder.input("y"))

    def outer(_index, a, b):
        return builder.iterate((a, b), lambda _inner, x, y: builder.call(pair, x, y), steps=2)

    first, second = builder.iterate((first, second), outer, steps=2)
    (last,) = builder.call(tail, first)
    return builder.finish({"first": last, "second": second}, accepted=builder.input("accepted")).module


def run(module, implementation):
    runtime = Interpreter(module, regions={"pair": implementation, "tail": lambda value: value + 7},
                          validators={"vlaforge_predicate_true": lambda value: value is True})
    return tuple(output.value for output in runtime.run(inputs={
        "x": InputBinding(2), "y": InputBinding(3), "accepted": InputBinding(True),
    }).committed_outputs.outputs)


def test_actual_nested_execution_preserved_for_equal_typed_reordered_ports():
    module = example()
    before = canonical_json(module)
    transformed = reorder_region_interface(module, "pair", input_order=(1, 0), output_order=(1, 0))
    baseline = lambda a, b: (10 * a + b, 100 * a + 3 * b)
    reordered = lambda b, a: (100 * a + 3 * b, 10 * a + b)
    assert run(transformed, reordered) == run(module, baseline)
    assert run(transformed, baseline) != run(module, baseline)
    assert canonical_json(module) == before
    assert transformed.inputs == module.inputs and transformed.outputs == module.outputs
    assert io_schema_digest(transformed) == io_schema_digest(module)
    assert transformed.region("tail") == module.region("tail")


def test_inverse_order_recovers_identical_ir_and_identity_is_noop():
    module = example()
    transformed = reorder_region_interface(module, "pair", input_order=(1, 0), output_order=(1, 0))
    assert reorder_region_interface(transformed, "pair", input_order=(1, 0), output_order=(1, 0)) == module
    assert reorder_region_interface(module, "pair") == module


def test_mixed_typed_ports_preserve_output_values_and_external_contract():
    integer, boolean = ScalarType("i64"), ScalarType("bool")
    builder = InvocationBuilder("mixed_order", inputs=(
        InputPort("x", integer), InputPort("accepted", boolean),
    ), outputs=(OutputPort("number", integer), OutputPort("flag", boolean)))

    @tensor_region("mixed", inputs=(Value("value", integer), Value("flag", boolean)),
                   outputs=(integer, boolean))
    def mixed(value, flag):
        raise AssertionError("declaration only")

    number, flag = builder.call(mixed, builder.input("x"), builder.input("accepted"))
    module = builder.finish({"number": number, "flag": flag}, accepted=builder.input("accepted")).module
    transformed = reorder_region_interface(module, "mixed", input_order=(1, 0), output_order=(1, 0))

    def execute(selected, implementation):
        runtime = Interpreter(selected, regions={"mixed": implementation},
                              validators={"vlaforge_predicate_true": lambda value: value is True})
        return tuple(output.value for output in runtime.run(inputs={
            "x": InputBinding(2), "accepted": InputBinding(True),
        }).committed_outputs.outputs)

    assert execute(module, lambda value, flag: (value + 3, not flag)) == (5, False)
    assert execute(transformed, lambda flag, value: (not flag, value + 3)) == (5, False)
    assert transformed.inputs == module.inputs and transformed.outputs == module.outputs
    assert io_schema_digest(transformed) == io_schema_digest(module)


@pytest.mark.parametrize("order", [(0,), (0, 0), (0, 2), (-1, 0), (True, False), ("0", "1")])
@pytest.mark.parametrize("field", ["input_order", "output_order"])
def test_invalid_mapping_is_not_partial_or_silently_cast(order, field):
    module = example()
    before = canonical_json(module)
    with pytest.raises(ValueError, match="permutation"):
        reorder_region_interface(module, "pair", **{field: order})
    assert canonical_json(module) == before


def test_unknown_and_effectful_region_are_rejected():
    module = example()
    with pytest.raises(ValueError, match="unknown"):
        reorder_region_interface(module, "absent")
    effectful = replace(module, regions=tuple(
        replace(region, effects=(Effect.RANDOM,)) if region.name == "pair" else region
        for region in module.regions))
    with pytest.raises(ValueError, match="pure"):
        reorder_region_interface(effectful, "pair")


def test_artifact_input_projection_preserves_nested_results_and_external_contract():
    module = example()
    before = canonical_json(module)
    projected = project_region_inputs(module, "pair", keep_indices=(0,))
    assert run(module, lambda a, ignored: (2 * a, a + 1)) == run(projected, lambda a: (2 * a, a + 1))
    assert projected.region("pair").inputs == module.region("pair").inputs[:1]
    assert projected.inputs == module.inputs and projected.outputs == module.outputs
    assert io_schema_digest(projected) == io_schema_digest(module)
    assert canonical_json(module) == before


def test_zero_input_artifact_and_reordered_projection_are_explicit():
    module = example()
    projected = project_region_inputs(module, "pair", keep_indices=())
    assert run(module, lambda a, b: (4, 5)) == run(projected, lambda: (4, 5))
    assert projected.region("pair").inputs == ()
    reordered = project_region_inputs(module, "pair", keep_indices=(1, 0))
    assert run(reordered, lambda b, a: (a * 10 + b, a - b)) == run(module, lambda a, b: (a * 10 + b, a - b))


@pytest.mark.parametrize("indices", [(0, 0), (2,), (-1,), (True,), ("0",), ([0],)])
def test_input_projection_never_guesses_or_coerces_indices(indices):
    with pytest.raises(ValueError, match="integer indices"):
        project_region_inputs(example(), "pair", keep_indices=indices)


def test_input_projection_rejects_unknown_or_effectful_regions():
    module = example()
    with pytest.raises(ValueError, match="unknown"):
        project_region_inputs(module, "missing", keep_indices=())
    module = replace(module, regions=tuple(replace(region, effects=(Effect.RANDOM,))
        if region.name == "pair" else region for region in module.regions))
    with pytest.raises(ValueError, match="pure"):
        project_region_inputs(module, "pair", keep_indices=())


def test_input_projection_requires_cache_contract_to_be_disabled_first():
    module = example()
    module = replace(module, regions=tuple(replace(region, metadata={**region.metadata, "memoize": True})
        if region.name == "pair" else region for region in module.regions))
    cached = configure_exact_cache(module, enabled=True)
    with pytest.raises(ValueError, match="pure uncached"):
        project_region_inputs(cached, "pair", keep_indices=(0,))
    uncached = configure_exact_cache(cached, enabled=False)
    assert len(project_region_inputs(uncached, "pair", keep_indices=(0,)).region("pair").inputs) == 1
