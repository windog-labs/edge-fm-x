from __future__ import annotations

from vlaforge.adapters.openvla_partitioned import (
    OPENVLA_ACTION_DIM,
    OPENVLA_CHUNK_SIZE,
    OPENVLA_HEADS,
    OPENVLA_HEAD_DIM,
    OPENVLA_HIDDEN_SIZE,
    OPENVLA_LAYER_COUNT,
    OPENVLA_MAX_CACHE_LENGTH,
    OPENVLA_PREFIX_LENGTH,
    _decode_chunk_region,
    _prefill_chunk_region,
    artifact_region_names,
    decode_chunk_names,
    prefill_chunk_names,
)


def test_openvla_partition_contract_is_complete_and_unique() -> None:
    prefill = prefill_chunk_names()
    decode = decode_chunk_names()
    regions = artifact_region_names()

    assert OPENVLA_LAYER_COUNT % OPENVLA_CHUNK_SIZE == 0
    assert len(prefill) == OPENVLA_LAYER_COUNT // OPENVLA_CHUNK_SIZE
    assert len(decode) == OPENVLA_LAYER_COUNT // OPENVLA_CHUNK_SIZE
    assert len(regions) == 36
    assert len(set(regions)) == len(regions)
    assert regions == (
        "prepare_multimodal_prefix",
        *prefill,
        "decode_token_embedding",
        *decode,
        "token_logits_head",
        "detokenize_action",
    )


def test_openvla_fixed_kv_profile_covers_bounded_decode() -> None:
    assert (
        OPENVLA_PREFIX_LENGTH + OPENVLA_ACTION_DIM - 1
        == OPENVLA_MAX_CACHE_LENGTH
    )

    prefill = _prefill_chunk_region(prefill_chunk_names()[0])
    decode = _decode_chunk_region(decode_chunk_names()[0])
    prefill_cache = prefill.outputs[1]
    decode_cache = decode.outputs[1]

    assert prefill.metadata == {"logical_stage": "prefill_layer_chunk"}
    assert prefill.outputs[0].shape == (
        1,
        OPENVLA_PREFIX_LENGTH,
        OPENVLA_HIDDEN_SIZE,
    )
    assert prefill_cache.shape == (
        1,
        OPENVLA_HEADS,
        OPENVLA_PREFIX_LENGTH,
        OPENVLA_HEAD_DIM,
    )
    assert decode.metadata == {
        "logical_stage": "decode_layer_chunk",
        "maximum_cache_length": OPENVLA_MAX_CACHE_LENGTH,
    }
    assert decode.inputs[1].type.shape == (1, 1)
    assert decode.inputs[2].type.shape == (1,)
    assert decode_cache.shape == (
        1,
        OPENVLA_HEADS,
        OPENVLA_MAX_CACHE_LENGTH,
        OPENVLA_HEAD_DIM,
    )
    assert decode.outputs[1:] == tuple(
        item.type for item in decode.inputs[3:]
    )


def test_openvla_partition_is_backend_refinement_not_new_core_ir() -> None:
    logical_stages = {
        _prefill_chunk_region(prefill_chunk_names()[0]).metadata[
            "logical_stage"
        ],
        _decode_chunk_region(decode_chunk_names()[0]).metadata[
            "logical_stage"
        ],
    }

    assert logical_stages == {
        "prefill_layer_chunk",
        "decode_layer_chunk",
    }
    assert all(name.startswith(("prefill_layers_", "decode_layers_")) for name in (
        prefill_chunk_names()[0],
        decode_chunk_names()[0],
    ))
