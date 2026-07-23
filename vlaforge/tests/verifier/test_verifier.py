from dataclasses import replace

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.ir import ops
from vlaforge.ir.attrs import Effect, EpochExpr, FreshnessConstraint
from vlaforge.ir.program import Block, Operation, Value
from vlaforge.ir.types import TensorType
from vlaforge.validation import mutation


def rules(module):
    return {diagnostic.rule for diagnostic in verify(module, raise_on_error=False)}


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_valid_fixture_has_no_diagnostics(factory):
    assert verify(factory().module, raise_on_error=False) == ()


def test_read_before_definition_has_context():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    broken = replace(
        policy.body.operations[2],
        operands=("missing_value",),
    )
    body = replace(
        policy.body,
        operations=policy.body.operations[:2]
        + (broken,)
        + policy.body.operations[3:],
    )
    diagnostics = verify(
        replace(module, policies=(replace(policy, body=body),)),
        raise_on_error=False,
    )
    diagnostic = next(
        item for item in diagnostics if item.rule == "ssa.read_before_definition"
    )
    assert diagnostic.program == module.name
    assert diagnostic.policy == policy.name
    assert diagnostic.op == broken.opcode
    assert diagnostic.version == "missing_value"


def test_structured_region_cannot_read_its_parent_result():
    module = build_openvla_fixture().module
    policy = module.policies[0]
    operations = list(policy.body.operations)
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
    broken = replace(
        module,
        policies=(
            replace(
                policy,
                body=replace(policy.body, operations=tuple(operations)),
            ),
        ),
    )
    assert "ssa.read_before_definition" in rules(broken)


def test_for_loop_rejects_zero_step():
    module = build_openvla_fixture().module
    policy = module.policies[0]
    operations = tuple(
        operation.with_attributes(step=0)
        if operation.opcode == "vla.for"
        else operation
        for operation in policy.body.operations
    )
    broken = replace(
        module,
        policies=(
            replace(policy, body=replace(policy.body, operations=operations)),
        ),
    )
    assert "control.for_step" in rules(broken)


def test_if_rejects_wrong_branch_yield_types():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = list(policy.body.operations)
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
    broken = replace(
        module,
        policies=(
            replace(
                policy,
                body=replace(policy.body, operations=tuple(operations)),
            ),
        ),
    )
    assert "control.if_yield" in rules(broken)


def test_wrong_state_version_clock():
    assert "state.wrong_version_clock" in rules(
        mutation.wrong_epoch(build_smolvla_fixture().module)
    )


def test_retention_must_satisfy_freshness():
    module = build_smolvla_fixture().module
    state = replace(
        module.states[0],
        retention=2,
        freshness=FreshnessConstraint(max_versions=2),
    )
    assert "state.retention" in rules(
        replace(module, states=(state,) + module.states[1:])
    )


def test_statically_stale_state_epoch_is_rejected():
    module = build_smolvla_fixture().module
    state = replace(
        module.states[0],
        freshness=FreshnessConstraint(max_versions=1),
    )
    policy = module.policies[0]
    operations = tuple(
        operation.with_attributes(
            epoch=EpochExpr(
                "previous",
                state.version_clock,
                offset=-2,
            ).to_dict()
        )
        if operation.opcode == "vla.state.read"
        and operation.attributes["state"] == state.name
        else operation
        for operation in policy.body.operations
    )
    broken = replace(
        module,
        states=(state,) + module.states[1:],
        policies=(
            replace(policy, body=replace(policy.body, operations=operations)),
        ),
    )
    diagnostics = verify(broken, raise_on_error=False)
    diagnostic = next(item for item in diagnostics if item.rule == "state.stale_epoch")
    assert diagnostic.state == "action_queue"
    assert diagnostic.epoch == "control"
    assert diagnostic.version == "offset:-2"


def test_double_write_in_one_transaction():
    assert "state.double_write" in rules(
        mutation.duplicate_stage_write(build_smolvla_fixture().module)
    )


def test_pending_state_cannot_escape():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = list(policy.body.operations)
    operations[-1] = ops.return_values("queue_pending")
    broken = replace(
        module,
        policies=(replace(policy, body=replace(policy.body, operations=tuple(operations))),),
    )
    assert "state.pending_escape" in rules(broken)


def test_required_future_must_be_awaited():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = tuple(
        operation.with_attributes(required_futures=["vision_future"])
        if operation.opcode == "vla.txn.commit"
        else operation
        for operation in policy.body.operations
    )
    broken = replace(
        module,
        policies=(replace(policy, body=replace(policy.body, operations=operations)),),
    )
    assert "commit.future_not_awaited" in rules(broken)


def test_validator_must_dominate_commit():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = tuple(
        replace(
            operation,
            operands=(
                operation.operands[0],
                operation.operands[1],
                "tick",
            ),
        )
        if operation.opcode == "vla.txn.commit"
        else operation
        for operation in policy.body.operations
    )
    diagnostics = rules(
        replace(
            module,
            policies=(replace(policy, body=replace(policy.body, operations=operations)),),
        )
    )
    assert "commit.condition_type" in diagnostics
    assert "commit.validator_dominance" in diagnostics


def test_action_cannot_publish_before_commit():
    assert "action.publish_before_commit" in rules(
        mutation.publish_before_commit(build_smolvla_fixture().module)
    )


def test_success_path_requires_one_commit():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = tuple(
        operation
        for operation in policy.body.operations
        if operation.opcode not in {"vla.txn.commit", "vla.action.publish"}
    )
    broken = replace(
        module,
        policies=(replace(policy, body=replace(policy.body, operations=operations)),),
    )
    assert "commit.zero" in rules(broken)


def test_success_path_rejects_double_commit():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = list(policy.body.operations)
    commit_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.opcode == "vla.txn.commit"
    )
    second = replace(
        operations[commit_index],
        results=(replace(operations[commit_index].results[0], name="committed_twice"),),
    )
    operations.insert(commit_index + 1, second)
    broken = replace(
        module,
        policies=(
            replace(policy, body=replace(policy.body, operations=tuple(operations))),
        ),
    )
    assert "commit.double" in rules(broken)


def test_authoritative_state_cannot_be_overwritten_inplace():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    operations = tuple(
        operation.with_attributes(inplace=True)
        if operation.opcode == "vla.state.stage_write"
        and operation.attributes["state"] == "action_queue"
        else operation
        for operation in policy.body.operations
    )
    broken = replace(
        module,
        policies=(replace(policy, body=replace(policy.body, operations=operations)),),
    )
    assert "state.authoritative_inplace" in rules(broken)


def test_async_effect_race_is_rejected():
    module = build_smolvla_fixture().module
    policy = module.policies[0]
    payload = TensorType((2,), "f32")
    async_body = Block.of((ops.yield_values("image_value"),))
    first = ops.async_execute(
        "future_a",
        payload,
        async_body,
        writes=("action_queue",),
    )
    second = ops.async_execute(
        "future_b",
        payload,
        async_body,
        reads=("action_queue",),
    )
    operations = list(policy.body.operations)
    operations[2:2] = [first, second]
    broken = replace(
        module,
        policies=(
            replace(policy, body=replace(policy.body, operations=tuple(operations))),
        ),
    )
    assert "async.state_race" in rules(broken)


def test_tensor_region_cannot_hide_rng_or_mutation():
    module = build_smolvla_fixture().module
    region = replace(module.regions[0], effects=(Effect.RANDOM,))
    diagnostics = rules(replace(module, regions=(region,) + module.regions[1:]))
    assert "region.hidden_effect" in diagnostics


def test_delete_dependency_mutation_is_detected():
    assert "region.input_types" in rules(
        mutation.delete_dependency(build_smolvla_fixture().module)
    )
