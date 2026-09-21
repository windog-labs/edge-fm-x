from __future__ import annotations

from dataclasses import replace

import pytest
from vlaforge.analysis import verify
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir import ops
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import ScalarType, TensorType
from vlaforge.plan import PlanModule, lower_to_plan, physicalize_plan, verify_plan
from vlaforge.plan.replay import (
    ReplayArtifactFacts,
    analyze_replay,
    analyze_replay_memory,
    analyze_replay_structure,
)


def program(shape=(2, 3), *, replay="prefer", device="cuda:0", host_index=False):
    tensor = TensorType(shape, "f32")
    position = TensorType((1,), "i64")
    flag = TensorType((1,), "bool")
    builder = InvocationBuilder(
        "generic_iterative",
        inputs=(
            InputPort("seed", tensor, device=device),
            InputPort("position", position, device=device),
            InputPort("condition", tensor, device=device),
            InputPort("accepted", flag),
        ),
        outputs=(OutputPort("result", tensor, group="result", device=device),),
    )
    seed, pos, condition, accepted = (
        builder.input(name) for name in ("seed", "position", "condition", "accepted")
    )

    @tensor_region(
        "transform",
        inputs=(Value("x", tensor), Value("condition", tensor)),
        outputs=(tensor,),
    )
    def transform(x, condition):
        return x

    @tensor_region(
        "update",
        inputs=(
            Value("x", tensor),
            Value("position", ScalarType("index") if host_index else position),
        ),
        outputs=(tensor, position),
    )
    def update(x, position):
        return x, position

    def step(index, value, position):
        (changed,) = builder.call(transform, value, condition)
        return builder.call(update, changed, index if host_index else position)

    result, _ = builder.iterate((seed, pos), step, steps=4, replay=replay)
    return builder.finish({"result": result}, accepted=accepted)


def loop(plan):
    return next(task for task in plan.tasks if task.opcode == "vla.for")


@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 8)])
@pytest.mark.parametrize("mode", ["prefer", "required"])
def test_replay_has_dedicated_reset_storage_in_the_existing_plan(shape, mode):
    source = program(shape, replay=mode)
    compilation = compile_module(source.module, default_device="cuda:0")
    plan = compilation.plan
    task = loop(plan)
    assert task.attributes["replay"] == mode
    assert not task.attributes["replay_reasons"]
    assert len(task.attributes["replay_seeds"]) == 2
    assert len(task.attributes["replay_staging"]) == 1
    assert len(task.attributes["carry_scratch"]) == 2
    assert analyze_replay_structure(plan, task, source.module).candidate
    decision = analyze_replay_memory(plan, task, source.module)
    assert decision.candidate and decision.device == "cuda:0"
    elements = 1
    for size in shape:
        elements *= size
    assert decision.staging_bytes == 2 * elements * 4 + 8
    assert verify_plan(plan, raise_on_error=False) == ()
    assert PlanModule.from_dict(plan.to_dict()) == plan
    physical = {
        index: item
        for item in plan.arena.physical_buffers
        for index in item.logical_buffers
    }
    end = max(plan.block(task.blocks[0]).tasks)
    for index in (*task.attributes["replay_seeds"], *task.attributes["replay_staging"]):
        assert physical[index].first_task <= task.id
        assert physical[index].last_task >= end
        for other in plan.arena.physical_buffers:
            if (
                other.first_task <= end
                and other.last_task >= task.id
                and other.id != physical[index].id
            ):
                assert (
                    physical[index].offset + physical[index].size_bytes <= other.offset
                    or other.offset + other.size_bytes <= physical[index].offset
                )


def test_default_off_keeps_original_ir_plan_and_memory():
    source = program(replay="off")
    ir_loop = next(
        op
        for op in source.module.invocations[0].body.operations
        if op.opcode == "vla.for"
    )
    assert "replay" not in ir_loop.attributes
    lowered = lower_to_plan(source.module)
    assert not any(key.startswith("replay") for key in loop(lowered).attributes)
    assert not any(
        buffer.source and buffer.source.startswith("replay:")
        for buffer in lowered.buffers
    )
    rebuilt = ops.for_loop_values(
        ir_loop.results,
        ir_loop.operands,
        ir_loop.regions[0].arguments[0],
        ir_loop.regions[0].arguments[1:],
        ir_loop.regions[0],
        lower=0,
        upper=4,
    )
    assert rebuilt == ir_loop


def test_host_induction_is_rejected_before_reserving_replay_storage():
    source = program(host_index=True)
    plan = lower_to_plan(source.module)
    task = loop(plan)
    assert "replay.host_induction" in task.attributes["replay_reasons"]
    assert "replay_seeds" not in task.attributes
    assert not analyze_replay_structure(plan, task, source.module).candidate


@pytest.mark.parametrize(
    "arena,source_device,reason",
    [
        ("cpu", "cpu", "replay.requires_cuda"),
        ("cuda:0", "cpu", "replay.source_device_mismatch"),
        ("cuda:1", "cuda:0", "replay.source_device_mismatch"),
    ],
)
def test_memory_analysis_requires_matching_cuda_sources(arena, source_device, reason):
    source = program(device=source_device)
    plan = physicalize_plan(
        lower_to_plan(source.module), default_device=arena, reuse_temporaries=True
    )
    decision = analyze_replay_memory(plan, loop(plan), source.module)
    assert reason in decision.reasons and not decision.candidate


@pytest.mark.parametrize(
    "mutation", ["alias", "unknown", "truncated", "livein", "reason", "lifetime"]
)
def test_plan_verifier_recomputes_replay_storage_contract(mutation):
    source = program()
    plan = compile_module(source.module, default_device="cuda:0").plan
    task = loop(plan)
    attrs = dict(task.attributes)
    if mutation == "alias":
        attrs["replay_seeds"] = list(task.inputs)
    elif mutation == "unknown":
        attrs["replay_seeds"] = [999999, 999998]
    elif mutation == "truncated":
        attrs["replay_seeds"] = attrs["replay_seeds"][:1]
    elif mutation == "livein":
        attrs["replay_liveins"] = []
    elif mutation == "reason":
        attrs["replay_reasons"] = ["forged rejection with allocated buffers"]
    else:
        seed = attrs["replay_seeds"][0]
        arena = replace(
            plan.arena,
            physical_buffers=tuple(
                replace(item, last_task=task.id)
                if seed in item.logical_buffers
                else item
                for item in plan.arena.physical_buffers
            ),
        )
        plan = replace(plan, arena=arena)
    plan = replace(
        plan,
        tasks=tuple(
            replace(item, attributes=attrs) if item.id == task.id else item
            for item in plan.tasks
        ),
    )
    assert any(
        item.rule == "replay.storage_contract"
        for item in verify_plan(plan, raise_on_error=False)
    )


def test_builder_and_ir_reject_unknown_replay_modes():
    with pytest.raises(ValueError, match="replay"):
        program(replay="automatic_magic")
    source = program()
    invocation = source.module.invocations[0]
    original = next(op for op in invocation.body.operations if op.opcode == "vla.for")
    mutated = replace(original, attributes={**original.attributes, "replay": "bad"})
    invocation = replace(
        invocation,
        body=replace(
            invocation.body,
            operations=tuple(
                mutated if op is original else op for op in invocation.body.operations
            ),
        ),
    )
    module = replace(source.module, invocations=(invocation,))
    assert any(
        item.rule == "control.for_replay"
        for item in verify(module, raise_on_error=False)
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"effect_audit": None}, "replay.missing_effect_audit"),
        (
            {"supports_external_cuda_graph": False},
            "replay.backend_graph_provider_unavailable",
        ),
        ({"graph_provider": None}, "replay.backend_graph_provider_unavailable"),
        ({"device": "cuda:1"}, "replay.artifact_device_mismatch"),
        ({"residency": "invocation"}, "replay.requires_session_residency"),
    ],
)
def test_backend_and_effect_evidence_are_independent_requirements(change, reason):
    from vlaforge.deployment.contract import EffectAudit

    source = program()
    plan = compile_module(source.module, default_device="cuda:0").plan
    good = ReplayArtifactFacts("cuda:0", "session", True, EffectAudit(), "provider")
    facts = {region.name: good for region in source.module.regions}
    assert analyze_replay(plan, loop(plan), source.module, facts).candidate
    facts["update"] = replace(good, **change)
    assert reason in analyze_replay(plan, loop(plan), source.module, facts).reasons


@pytest.mark.parametrize(
    "effect",
    ["hidden_rng", "explicit_rng", "hidden_mutation", "external_io", "lifted_states"],
)
def test_pure_declared_region_does_not_override_artifact_effect_audit(effect):
    from vlaforge.deployment.contract import EffectAudit

    source = program()
    plan = compile_module(source.module, default_device="cuda:0").plan
    audit = EffectAudit(**{effect: ("state",) if effect == "lifted_states" else True})
    facts = {
        region.name: ReplayArtifactFacts("cuda:0", "session", True, audit, "provider")
        for region in source.module.regions
    }
    assert (
        "replay.unsupported_artifact_effects"
        in analyze_replay(plan, loop(plan), source.module, facts).reasons
    )
