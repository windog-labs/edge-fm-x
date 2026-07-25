from dataclasses import replace

import pytest

from vlaforge.adapters import (
    DRIVING_FIXTURES,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.analysis import verify
from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import Operation


def rules(module):
    return {
        diagnostic.rule
        for diagnostic in verify(module, raise_on_error=False)
    }


def _replace_body(module, operations):
    invocation = module.invocations[0]
    return replace(
        module,
        invocations=(
            replace(
                invocation,
                body=replace(
                    invocation.body,
                    operations=tuple(operations),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "factory",
    [build_smolvla_fixture, build_openvla_fixture, *DRIVING_FIXTURES],
)
def test_valid_fixture_has_no_diagnostics(factory):
    assert verify(factory().module, raise_on_error=False) == ()


def test_read_before_definition_has_context():
    module = build_smolvla_fixture().module
    invocation = module.invocations[0]
    operations = list(invocation.body.operations)
    broken = replace(operations[2], operands=("missing_value",))
    operations[2] = broken
    diagnostics = verify(
        _replace_body(module, operations),
        raise_on_error=False,
    )
    diagnostic = next(
        item
        for item in diagnostics
        if item.rule == "ssa.undefined"
    )
    assert diagnostic.program == module.name
    assert diagnostic.invocation == invocation.name
    assert diagnostic.op == broken.opcode
    assert diagnostic.value == "missing_value"


def test_structured_region_cannot_read_parent_result():
    module = build_openvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    loop_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.opcode == "vla.for"
    )
    loop = operations[loop_index]
    body = loop.regions[0]
    invoke = replace(
        body.operations[0],
        operands=("context", loop.results[0].name, "token_step"),
    )
    operations[loop_index] = replace(
        loop,
        regions=(
            replace(
                body,
                operations=(invoke,) + body.operations[1:],
            ),
        ),
    )
    assert "ssa.undefined" in rules(_replace_body(module, operations))


def test_for_loop_rejects_zero_step():
    module = build_openvla_fixture().module
    operations = tuple(
        operation.with_attributes(step=0)
        if operation.opcode == "vla.for"
        else operation
        for operation in module.invocations[0].body.operations
    )
    assert "control.for_bound" in rules(_replace_body(module, operations))


def test_if_rejects_wrong_branch_yield_types():
    module = build_smolvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    if_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.opcode == "vla.if"
    )
    operation = operations[if_index]
    branch = operation.regions[0]
    broken_yield = replace(
        branch.operations[-1],
        operands=branch.operations[-1].operands[:-1],
    )
    operations[if_index] = replace(
        operation,
        regions=(
            replace(
                branch,
                operations=branch.operations[:-1] + (broken_yield,),
            ),
            operation.regions[1],
        ),
    )
    assert "control.if_yield" in rules(_replace_body(module, operations))


def test_unknown_input_port_is_rejected():
    module = build_openvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    operations[0] = operations[0].with_attributes(input="unknown")
    assert "input.unknown" in rules(_replace_body(module, operations))


def test_pending_state_cannot_be_returned_as_output():
    module = build_smolvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    operations[-1] = replace(
        operations[-1],
        operands=("queue_pending",),
    )
    assert "output.return_type" in rules(_replace_body(module, operations))


def test_validator_must_dominate_commit():
    module = build_smolvla_fixture().module
    operations = tuple(
        replace(
            operation,
            operands=(
                operation.operands[0],
                operation.operands[1],
                "queue_empty",
            ),
        )
        if operation.opcode == "vla.txn.commit"
        else operation
        for operation in module.invocations[0].body.operations
    )
    assert "commit.validator_dominance" in rules(
        _replace_body(module, operations)
    )


def test_success_path_requires_one_commit():
    module = build_smolvla_fixture().module
    operations = tuple(
        operation
        for operation in module.invocations[0].body.operations
        if operation.opcode != "vla.txn.commit"
    )
    assert "commit.zero" in rules(_replace_body(module, operations))


def test_success_path_rejects_double_commit():
    module = build_smolvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    commit_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.opcode == "vla.txn.commit"
    )
    second = replace(
        operations[commit_index],
        results=(
            replace(
                operations[commit_index].results[0],
                name="committed_twice",
            ),
        ),
    )
    operations.insert(commit_index + 1, second)
    assert "commit.double" in rules(_replace_body(module, operations))


def test_tensor_region_cannot_hide_rng_or_mutation():
    module = build_smolvla_fixture().module
    region = replace(module.regions[0], effects=(Effect.RANDOM,))
    assert "region.effect" in rules(
        replace(module, regions=(region,) + module.regions[1:])
    )


def test_output_group_membership_is_verified():
    module = build_openvla_fixture().module
    output = replace(module.outputs[0], group="another_group")
    assert "output.group_membership" in rules(
        replace(module, outputs=(output,))
    )


def test_unknown_extension_opcode_is_rejected():
    module = build_openvla_fixture().module
    operations = list(module.invocations[0].body.operations)
    operations.insert(2, Operation("vendor.unverified"))
    assert "op.unknown" in rules(_replace_body(module, operations))
