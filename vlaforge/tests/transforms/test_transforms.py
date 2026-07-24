from dataclasses import replace

import pytest

from vlaforge.adapters import build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.interpreter import Interpreter
from vlaforge.ir.program import Block
from vlaforge.transforms import (
    MemoizationSynthesisError,
    canonicalize,
    physicalize_state,
    optimize_whole_program,
    synthesize_epoch_memoization,
    temporal_loop_invariant_code_motion,
)
from vlaforge.validation import compare_traces


def walk(block):
    for operation in block.operations:
        yield operation
        for region in operation.regions:
            yield from walk(region)


def run(fixture, module):
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    for item in fixture.ticks:
        runtime.run_tick("act", item.tick, item.inputs)
    return runtime.trace


def test_epoch_memoization_uses_observation_epoch():
    fixture = build_smolvla_fixture()
    transformed = synthesize_epoch_memoization(fixture.module)
    encode = next(
        operation
        for operation in walk(transformed.policies[0].body)
        if operation.opcode == "vla.invoke"
        and operation.attributes["region"] == "encode_observation"
    )
    assert encode.attributes["memoize_semantics"] == "epoch_state_signature"
    assert encode.attributes["memoize_key"] == ["observation_epoch"]
    assert encode.attributes["memoize_dependencies"] == [
        {
            "kind": "epoch",
            "value": "observation_epoch",
            "subject": "image",
            "max_age_ns": 50_000_000,
            "max_versions": None,
        }
    ]
    assert (
        encode.attributes["memoize_invalidation"]
        == "epoch_or_state_version_or_episode_change"
    )
    assert verify(transformed, raise_on_error=False) == ()


def test_epoch_memoization_rejects_region_without_vla_version_key():
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(region, metadata={"memoize": True})
        if region.name == "queue_zero"
        else region
        for region in fixture.module.regions
    )
    with pytest.raises(
        MemoizationSynthesisError,
        match="memoize.missing_epoch_or_state",
    ):
        synthesize_epoch_memoization(replace(fixture.module, regions=regions))


def test_epoch_memoization_uses_state_logical_version():
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(
            region,
            metadata={"memoize": True},
        )
        if region.name == "sample_noise"
        else region
        for region in fixture.module.regions
    )
    transformed = synthesize_epoch_memoization(
        replace(fixture.module, regions=regions)
    )
    sample_noise = next(
        operation
        for operation in walk(transformed.policies[0].body)
        if operation.attributes.get("region") == "sample_noise"
    )
    assert sample_noise.attributes["memoize_key"] == ["rng_snapshot"]
    assert sample_noise.attributes["state_version_signature"] == [
        "rng:%rng_snapshot"
    ]


def test_transforms_preserve_reference_trace():
    fixture = build_smolvla_fixture()
    transformed = physicalize_state(
        synthesize_epoch_memoization(canonicalize(fixture.module)),
        max_in_flight=2,
        consumer_lag=1,
    )
    report = compare_traces(
        run(fixture, fixture.module),
        run(fixture, transformed),
    )
    assert report.equal, report.format()
    plan = transformed.metadata["physical_state_plan"]
    assert plan["rng"]["capacity"] >= 4


def test_temporal_licm_moves_epoch_proven_prefix_and_preserves_state_action():
    fixture = build_smolvla_fixture()
    body_operations = list(fixture.module.policies[0].body.operations)
    branch_index = next(
        index
        for index, operation in enumerate(body_operations)
        if operation.opcode == "vla.if"
    )
    branch = body_operations[branch_index]
    refill_operations = list(branch.regions[0].operations)
    prefix_index = next(
        index
        for index, operation in enumerate(refill_operations)
        if operation.attributes.get("region") == "encode_observation"
    )
    prefix = refill_operations.pop(prefix_index)
    loop_index = next(
        index
        for index, operation in enumerate(refill_operations)
        if operation.opcode == "vla.for"
    )
    loop = refill_operations[loop_index]
    loop_body = loop.regions[0]
    refill_operations[loop_index] = replace(
        loop,
        regions=(
            replace(
                loop_body,
                operations=(prefix, *loop_body.operations),
            ),
        ),
    )
    body_operations[branch_index] = replace(
        branch,
        regions=(
            Block.of(refill_operations),
            branch.regions[1],
        ),
    )
    nested_prefix_module = replace(
        fixture.module,
        policies=(
            replace(
                fixture.module.policies[0],
                body=Block.of(body_operations),
            ),
        ),
    )
    memoized = synthesize_epoch_memoization(nested_prefix_module)
    result = temporal_loop_invariant_code_motion(memoized)

    assert [item.region for item in result.moved] == [
        "encode_observation"
    ]
    moved_prefix = next(
        operation
        for operation in walk(result.module.policies[0].body)
        if operation.attributes.get("region") == "encode_observation"
    )
    assert moved_prefix.attributes["temporal_licm"] == "moved_to_preheader"
    assert verify(result.module, raise_on_error=False) == ()

    before = run(fixture, nested_prefix_module)
    after = run(fixture, result.module)
    assert _state_action_events(before) == _state_action_events(after)
    assert sum(event.kind == "region" for event in before.events) > sum(
        event.kind == "region" for event in after.events
    )


def test_temporal_licm_rejects_loop_carried_dependency():
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(
            region,
            metadata={"memoize": True, "loop_invariant": True},
        )
        if region.name == "solver_step"
        else region
        for region in fixture.module.regions
    )
    # The solver's loop-carried tensor and induction value have no stable
    # epoch/version signature, so cache-key synthesis itself forbids motion.
    with pytest.raises(
        MemoizationSynthesisError,
        match="memoize.missing_epoch_or_state",
    ):
        synthesize_epoch_memoization(
            replace(fixture.module, regions=regions)
        )


def test_real_models_have_preheader_licm_certificates():
    from vlaforge.adapters import (
        build_real_openvla_action_program,
        build_real_smolvla_action_program,
    )

    modules = (
        build_real_smolvla_action_program(
            chunk_size=50,
            max_action_dim=32,
            output_action_dim=6,
            num_steps=10,
        ),
        build_real_openvla_action_program(action_dim=7),
    )
    for module in modules:
        memoized = synthesize_epoch_memoization(module)
        result = temporal_loop_invariant_code_motion(memoized)
        assert len(result.prehoisted) == 1
        assert result.prehoisted[0].region in {
            "prepare_prefix",
            "generate_action_tokens_prefill",
        }
        assert verify(result.module, raise_on_error=False) == ()


def test_whole_program_pipeline_is_deterministic_for_real_models():
    from vlaforge.adapters import (
        build_real_openvla_action_program,
        build_real_smolvla_action_program,
    )

    modules = (
        build_real_smolvla_action_program(
            chunk_size=50,
            max_action_dim=32,
            output_action_dim=6,
            num_steps=10,
        ),
        build_real_openvla_action_program(action_dim=7),
    )
    for module in modules:
        first = optimize_whole_program(module)
        second = optimize_whole_program(module)
        assert first.optimized_plan.digest() == second.optimized_plan.digest()
        assert first.optimized_arena_bytes < first.baseline_arena_bytes


def _state_action_events(trace):
    return [
        (
            event.kind,
            event.policy,
            event.tick,
            event.op,
            event.data,
        )
        for event in trace.events
        if event.kind
        in {
            "state_read",
            "state_stage",
            "action_pending",
            "transaction_commit",
            "action_publish",
        }
    ]
