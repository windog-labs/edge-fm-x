from __future__ import annotations

from collections import Counter

from vlaforge.adapters.openvla_artifact import (
    FIXED_CACHE,
    OPENVLA_L4_SUPPORT_REGIONS,
    build_compiled_openvla_program,
)
from vlaforge.adapters.openvla_partitioned import (
    decode_chunk_names,
    prefill_chunk_names,
)
from vlaforge.analysis import verify


def test_openvla_artifact_schedule_is_static_and_stateless() -> None:
    module = build_compiled_openvla_program()

    assert module.schema_version == "0.2"
    assert module.states == ()
    assert [item.name for item in module.inputs] == [
        "image",
        "instruction_tokens",
        "instruction_mask",
    ]
    assert [item.name for item in module.outputs] == ["action"]
    assert module.outputs[0].device == "cuda:0"
    assert len(module.regions) == 38
    assert set(OPENVLA_L4_SUPPORT_REGIONS).issubset(
        region.name for region in module.regions
    )
    assert verify(module, raise_on_error=False) == ()


def test_openvla_artifact_schedule_unrolls_only_physical_decode() -> None:
    module = build_compiled_openvla_program()
    invokes = [
        operation
        for operation in module.invocations[0].body.operations
        if operation.opcode == "vla.invoke"
    ]
    counts = Counter(
        str(operation.attributes["region"]) for operation in invokes
    )

    assert all(counts[name] == 1 for name in prefill_chunk_names())
    assert all(counts[name] == 6 for name in decode_chunk_names())
    assert counts["decode_token_embedding"] == 6
    assert counts["token_logits_head"] == 7
    assert counts["detokenize_action"] == 1
    assert not any(
        operation.opcode in {"vla.clock", "vla.tick", "vla.publish"}
        for operation in module.invocations[0].body.operations
    )
    assert module.invocations[0].metadata["core_op_delta"] == 0


def test_openvla_fixed_kv_stays_invocation_local_ssa() -> None:
    module = build_compiled_openvla_program()
    support = module.region("prepare_decode_state")

    assert len(support.inputs) == 65
    assert len(support.outputs) == 77
    assert support.outputs[1:65] == tuple(FIXED_CACHE for _ in range(64))
    assert module.invocations[0].metadata["derived_cache"] == "fixed_kv"
    assert module.invocations[0].metadata["persistent_state"] == "none"
