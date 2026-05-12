# 3060 QKV/OProj TensorRT Bridge Review Gate

## Why This Is The Next Candidate

After the optional TensorRT MLP bridge with `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`,
the `GateUp-FP16 + QKV/OProj linear` full 18-case matrix is the current best
low-risk bridge:

- Improves `15/18` cases versus GateUp-FP16
- Reaches or beats TRT in `3/18` cases
- Largest remaining gap: `3B / 2048x32`, EdgeFM `953.29 ms` versus TRT `927.69 ms` (`+25.60 ms`)
- The residual long-prefill gap is still dominated by prefill work, not decode

The `3B / 2048x1` graph-off residual profile still explains why this bridge is
worth keeping on the active path:

- QKV: `+16.39 ms`
- OProj: `+10.54 ms`
- FlashInfer prefill attention: `+11.29 ms`
- remaining bridge casts: `+11.88 ms`, mostly DownProj BF16->FP16 cast

`EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is now a blocked diagnostic path, not an
active optimization route. The warmed `3B / 2048x32` rerun still ended in
`act_and_mul_kernel launch failed: out of memory`, so it should remain out of
the active queue until there is a new memory policy.

## Probe Evidence

Temporary probe:

- `.tmp_codex/probes/profile_trt_linear_subengine.py`
- summary: `.tmp_codex/bench/20260511_trt_linear_qkv_oproj_3b_m2048_summary.json`

The probe builds runtime-weight TensorRT linear engines with EdgeFM resident
weight layout `[out_features, in_features]`:

```text
input BF16 [M, K] x weight BF16/FP16 [N, K]^T -> output BF16 [M, N]
```

For `3B / m=2048`:

| Path | QKV+OProj across 36 layers |
| --- | ---: |
| Current EdgeFM profile | `57.95 ms` |
| TRT-Edge-LLM reference | `31.01 ms` |
| TensorRT linear probe, BF16 weights | `39.37 ms` |
| TensorRT linear probe, FP16 weights | `35.24 ms` |

The inspector reports TensorRT `sm80_xmma_gemm_f16f16_*` plus Myelin cast/FcCast
style tactics. This is consistent with the user's request to check Myelin/XMMA,
but the implementation must reach those tactics through TensorRT engines rather
than a fake source-level `xmma` operator id.

## Prototype Evidence

The first prototype is implemented as a compile/runtime-gated, prefill-only
bridge. It remains default-off and uses generated `.engine` files from
`.tmp_codex/bench` during experiments.

Important correctness finding:

- Qwen2.5 Q/K/V projections have BF16 bias tensors.
- The first no-bias QKV TensorRT engine omitted this bias and failed token
  correctness at generated token 0.
- OProj has no bias, which is why OProj-only was correct.
- The bridge now distinguishes bias-aware and no-bias engines; QKV binds the
  BF16 bias tensor, while OProj uses the no-bias engine.

Raw root-cause artifact:

- `.tmp_codex/bench/3060_20260511_qkv_bias_operator_diagnostic.json`

Target and regression slices with `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`,
`EDGE_FM_PREFILL_TRT_LINEAR=1`, `EDGE_FM_TRT_LINEAR_ROLES=both`, and BF16
linear weights:

| Case | Correctness | EdgeFM mean | GateUp-FP16 mean | TRT mean | Gain vs GateUp | Gap vs TRT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3B / 2048x32` | pass | `948.91 ms` | `970.55 ms` | `927.69 ms` | `21.64 ms` (`2.23%`) | `+21.23 ms` |
| `3B / 2048x64` | covered by same `m=2048` graph path | `1597.05 ms` | `1621.74 ms` | `1584.36 ms` | `24.69 ms` (`1.52%`) | `+12.68 ms` |
| `1.5B / 2048x32` | pass | `482.96 ms` | `491.52 ms` | `474.23 ms` | `8.56 ms` (`1.74%`) | `+8.73 ms` |

Raw summaries:

- `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x32_summary.json`
- `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x64_summary.json`
- `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_1p5b_2048x32_summary.json`

Full matrix evidence:

- `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_full_llm_matrix_runs3_summary.json`
- `18/18` cases completed
- `3/18` cases beat TRT: `0.5B / 512x32`, `0.5B / 512x64`, `3B / 512x64`
- top remaining gaps: `3B / 2048x32` `+25.60 ms`, `3B / 2048x64` `+19.38 ms`, `0.5B / 2048x64` `+18.01 ms`, `3B / 1024x32` `+13.21 ms`

Conclusion: the prototype clears the first target and key regression-slice
performance gates, and it now has full-matrix evidence. It is still an
experimental/default-off candidate until same-run TRT comparison, packaging,
and final cleanup finish.

## Proposed Prototype

Implement a small optional Qwen2.5 prefill-only bridge:

- compile gate: reuse `BUILD_TRT_MLP_BRIDGE=ON` for the current experimental
  TensorRT bridge build
- runtime gate: `EDGE_FM_PREFILL_TRT_LINEAR=1`
- role gate: `EDGE_FM_TRT_LINEAR_ROLES=qkv,oproj,both`, default `both`
- engine directory: `EDGE_FM_TRT_LINEAR_ENGINE_DIR`, falling back to
  `EDGE_FM_TRT_MLP_ENGINE_DIR` for convenience during experiments
- precision mode: BF16 activation input/output, FP16 compute inside TensorRT
- weight modes:
  - default `EDGE_FM_TRT_LINEAR_FP16_WEIGHTS=none`
  - optional `qkv`, `oproj`, or `both` only after a memory probe

The first prototype should only support the native EdgeFM prefill boundary:

- QKV input: normalized hidden tensor `[seq_len, hidden]`
- QKV output: existing fused QKV buffer `[seq_len, q_dim+k_dim+v_dim]`
- OProj input: attention output `[seq_len, hidden]`
- OProj output: existing hidden states buffer `[seq_len, hidden]`

The bridge must preserve existing Q/K/V split and KV-cache copy behavior. It is
not a license to rewrite the attention path.

## Fallback And Safety

The native path remains the source of truth unless every bridge condition holds:

- runtime env is enabled
- stage is prefill
- input/output/weight tensors are GPU BF16 2D tensors
- the matching TensorRT engine exists
- engine tensor names, shapes, and dtypes match the expected EdgeFM layout
- TensorRT `setTensorAddress` and `enqueueV3` both succeed

Any failure returns `false` and Qwen2.5 calls the existing `LinearLayer`
implementation. Missing engines and IO mismatches should log once per shape.

## Correctness Gate

Before any performance claim:

- default/native build must still pass the existing generate and operator gates
- bridge build must pass
  `test_generate_token_alignment_prefill_cuda_graph_first_request`
- the target slice must match
  `transformers.generate(do_sample=False, eos_token_id=None)` for all generated
  tokens before benchmark numbers are accepted

## Performance Gate

First target:

- `Qwen2.5-3B / prefill=2048 / decode=32`
- compare against the latest GateUp-FP16 bridge target and latest official TRT
  baseline
- headline numbers remain CUDA graph `EdgeFM` versus `TRT-Edge-LLM`; graph-off
  and probe numbers are attribution only

Acceptance for this prototype:

- target CUDA graph median improves by at least `1%` versus the current
  GateUp-FP16 bridge without introducing correctness failures
- memory headroom remains materially safer than the high-memory MLP `both` mode
- no generated `.engine` or `.tmp_codex` artifact is committed

## Open Risks

- Python ABI split still prevents a same-process bridge EdgeFM + TRT comparison.
  Until fixed, bridge results compare against the latest official TRT baseline.
- BF16-weight QKV/OProj mode does not intentionally create persistent FP16
  weight copies, but 3B still needs a real memory probe for TensorRT engine,
  context, and workspace overhead before full-matrix/default decisions.
- TensorRT enqueue inside CUDA graph capture worked for the MLP bridge, but QKV
  and OProj still need full-matrix coverage across all official shapes.
- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is a blocked diagnostic path, not an
  active optimization route.
- This bridge is an experiment, not a default path. Default-on requires a later
  packaging/reproducibility review.
