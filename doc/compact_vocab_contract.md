# EdgeFM Compact Vocab Contract

This documents the runtime contract for `edgefm.compact_vocab.v1`. The feature is explicit opt-in and does not change default FP16/BF16 generate behavior.

## Runtime Config

Enable compact vocab in the engine config:

```json
{
  "compact_vocab": {
    "enabled": true,
    "mapping_path": "compact_vocab.json",
    "reject_unknown_input_ids": true
  }
}
```

`mapping_path` is resolved relative to the engine config directory unless it is absolute. Version 1 requires `reject_unknown_input_ids=true`; there is no unknown-token fallback.

## Artifact Files

A compact-vocab checkpoint must include:

- A pre-pruned embedding / tied `lm_head` checkpoint whose `config.json` has `vocab_size == compact_vocab_size`.
- The original tokenizer files, so external callers keep using original tokenizer ids.
- A TensorRT-Edge-LLM style `vocab_map.safetensors` containing `vocab_map == new_to_old` as an `int32` tensor.
- A mapping file with this shape:

```json
{
  "format": "edgefm.compact_vocab.v1",
  "original_vocab_size": 151936,
  "compact_vocab_size": 98304,
  "old_to_new": [0, -1, 1],
  "new_to_old": [0, 2],
  "special_token_ids": [151643, 151645]
}
```

`old_to_new` length must equal `original_vocab_size`; each kept original id maps to a compact id, and each pruned original id is `-1`. `new_to_old` length must equal `compact_vocab_size` and must be the exact inverse for kept ids.

The packaging helper accepts the same reduced-vocab map shape used by TensorRT-Edge-LLM: compact/reduced ids index a `vocab_map` tensor whose values are original tokenizer ids. EdgeFM materializes the inverse map because runtime requests arrive in original tokenizer id space.

## Runtime Semantics

- Request `token_ids`, request stop tokens, KV-prefix token ids, EOS ids, and config stop tokens are remapped from original ids to compact ids before model execution.
- Generated compact ids are restored through `new_to_old` before writing `Response.token_ids()`.
- External response semantics stay unchanged: callers always see original tokenizer ids.
- If a request token, request stop token, config stop token, EOS id, special token, or prefix token was pruned, the runtime raises a clear error instead of silently remapping it.
- Custom embedding placeholder ids must remap to a contiguous compact range.

## Validation

Tooling:

```bash
python3 scripts/compact_vocab/compact_vocab_artifact.py \
  --input-model-dir <hf-model-dir> \
  --output-dir <compact-model-dir> \
  --vocab-map-safetensors <trt-edge-llm-style-vocab-map.safetensors> \
  --original-vocab-size <original-vocab-size> \
  --special-token-id <eos-or-special-id> \
  --json

python3 scripts/compact_vocab/compact_vocab_artifact.py \
  --validate-artifact-dir <compact-model-dir> \
  --json
```

Targeted coverage:

- `tests/cpp/test_compact_vocab.cpp` validates mapping shape, inverse checks, special/stop/EOS preservation, non-identity remap/restore, and pruned request stop-token rejection.
- `tests/engine/test_qwen2_generate.py::test_generate_compact_vocab_non_identity_remaps_request_response_and_stop` verifies non-identity request remap, response restore, and request stop-token remap through real `generate()`.
- `tests/scripts/test_compact_vocab_artifact.py` validates mapping construction, checkpoint cropping, `vocab_map.safetensors` emission, and artifact validator failures.

Current smoke: Qwen2.5-0.5B real checkpoint packaging with 64 kept base ids plus BOS/EOS produced a valid `66`-token artifact and validated the cropped vocab tensor shape.
