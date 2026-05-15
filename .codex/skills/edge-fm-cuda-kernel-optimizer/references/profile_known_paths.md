# Edge-FM Profile Known Paths

Use this catalog before calling a hotspot "new".

The goal is not to prove the hotspot is fully solved already. The goal is to
avoid skipping over an existing Edge-FM path that should have been the first
suspect.

## 1. Attention Hotspots

### Decode attention

- First suspect:
  - existing FlashInfer decode tuned path
- Relevant implementation:
  - `src/operators/attention_op.cu`
  - `src/layers/attention.cu`
- First tuning entrypoint:
  - `scripts/tune/tune_qwen_attention_decode.py`
- Common retained record shape:
  - `impl_id=flashinfer_attention_decode_sm80_tuned`

### Prefill attention

- First suspect:
  - existing FlashInfer prefill path or wrong prefill/decode split
- Relevant implementation:
  - `src/operators/attention_op.cu`
  - `src/layers/attention.cu`
- First tuning entrypoint:
  - `scripts/tune/tune_qwen_attention_prefill.py`

## 2. Linear Hotspots

### `fused_qkv` / `attention_output` / `mlp_down`

- First suspect:
  - operator-table record mismatch or stale tactic selection
- Relevant implementation:
  - `src/layers/linear.cu`
  - `src/operators/linear_impl.cu`
- First tuning entrypoint:
  - `scripts/tune/tune_qwen_cublaslt.py`
  - `scripts/tune/retune_qwen_operator_tables.py`

### What to check first

- `hw_profile` is correct for the current GPU
- the selected layer role matches the hotspot layer
- the right family table is being loaded:
  - `operator_impl_table_llm.json`
  - `operator_impl_table_vlm.json`

## 3. Fused Gate-Up / SwiGLU Hotspots

- First suspect:
  - decode fused gate-up fast path did not fire
- Relevant implementation:
  - `src/operators/fused_gate_up_activation_op.cu`
  - `src/layers/activation.cu`
- First tuning entrypoint:
  - `scripts/tune/tune_qwen_decode_swiglu.py`
  - `scripts/tune/tune_fused_gate_up_cublaslt.py`

## 4. Norm Hotspots

- First suspect:
  - norm implementation mismatch before inventing a new fused norm path
- Relevant implementation:
  - `src/operators/norm_op.cu`
  - `src/layers/layernorm.cu`
- First check:
  - confirm whether the run should already be on `flashinfer_norm`

## 5. Benchmark And Validation Loop

After a hotspot is tied to a known path:

1. run the matching microbench or tuning script
2. update the relevant operator table
3. materialize platform configs if needed
4. rerun the end-to-end benchmark or prepared-case profile

Primary files:

- `scripts/operator_table/materialize_platform_configs.py`
- `scripts/operator_table/validate_operator_tables.py`
- `tests/engine/test_qwen2_generate.py`
