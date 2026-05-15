# Owner A Phase 3 Decision Gates

This round keeps the default runtime conservative. The existing full-logits greedy path remains default. `lm_head_top1` is now available as an explicit default-off experiment, while DeepGEMM remains probe-only until artifact, hardware, and benchmark evidence justify a runtime binding.

## Benchmark Matrix

Use the single-case profiler for each row in the matrix:

```bash
EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 \
EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 \
python3 scripts/profile/profile_edgefm_generate_case.py \
  --model-path <model-dir> \
  --prefill-len 2048 \
  --decode-len 32 \
  --use-cuda-graph \
  --lm-head-top1 \
  --runs 3 \
  --json
```

Matrix:

- Models: `0.5b`, `1.5b`, `3b`.
- Prefill lengths: `512`, `1024`, `2048`.
- Decode lengths: `32`, `64`.
- CUDA graph: off and on.

The JSON output keeps `edgefm.generate_profile.v1` stable and adds `owner_a_decode_breakdown`, which reports decode model plus `lm_head`, sampler, finalize, graph replay timing percentages, and `lm_head_top1` activation/step counts.

## Greedy Sampler

The greedy sampler now performs argmax directly over Float32 logits and no longer writes a full temporary `batch_size * vocab_size` scaled-logits buffer. This is correctness-preserving because positive temperature scaling does not change argmax in the `temperature < 1e-6` greedy path.

Smoke profile after the change, Qwen2.5-1.5B, `prefill_len=6`, `decode_len=4`, CUDA graph off:

- `runs=3`, `warmup=1`: `prefill_sampler_ms ~= 0.016`, `decode_sampler_ms ~= 0.048`.
- `runs=1`, `warmup=0`: `prefill_sampler_ms ~= 0.106`, `decode_sampler_ms ~= 0.047`.

## `lm_head_top1`

- Default runtime: full logits.
- Current status: implemented, experimental, default-off.
- Enable flag: `runtime.lm_head_top1.enabled=true`, or profile with `--lm-head-top1`.
- Acceptance gate for making it default: exact token alignment plus at least `1%` end-to-end CUDA graph improvement on the target slice.
- Fallback rule: unsupported dtype, shape, stage, graph state, quant path, or sampling mode must use the existing full logits path.

### Probe Note: 2026-05-14

On RTX 3060 / `cuda_sm86`, standalone Qwen2.5-1.5B decode `lm_head` (`m=1`, BF16, shape `1536 -> 151936`) measured about `1.374ms` median using the existing cublasLt route. Running the 8 heuristic cublasLt candidates did not find a better tactic; `algo_0` was effectively tied with baseline, while split-K and other candidates regressed.

This is why full logits stays default. The current experimental route exists to gather end-to-end evidence; it must prove it beats the current full-logits route, not just avoid the logits write, because the matrix-vector multiply remains the dominant work.

Implementation smoke after wiring the default-off route, Qwen2.5-0.5B, `prefill_len=6`, `decode_len=4`, `runs=1`:

- CUDA graph off: default `decode_ms ~= 13.744`, top1 `decode_ms ~= 13.949`; top1 skipped sampler but did not improve end-to-end decode.
- CUDA graph on: default `decode_ms ~= 10.569`, top1 `decode_ms ~= 10.542`; this is below the `>=1%` decision threshold and not a decision-grade benchmark.

## DeepGEMM

Probe command:

```bash
python3 scripts/profile/owner_a_phase3_decision_gates.py --json
```

DeepGEMM remains disabled by default. Eligible scope is limited to prefill dense linear or future FP8/W8A8 paths; it is not a first-choice `m=1` decode path. Unsupported source, import, hardware, shape, or build state must fallback without changing FP16/BF16 behavior.

Official DeepSeek DeepGEMM requirements are SM90 or SM100, C++20, PyTorch 2.1+, and CUDA 12.3+ for SM90 / 12.9+ for SM100. The official API is JIT-compiled and expects callers to handle input transposition, FP8 casting, scaling factors, and layout transforms outside the GEMM call.

Local integration status:

- This workspace does not have the standalone `deep_gemm` Python package installed.
- It does have `flashinfer.deep_gemm` and FlashInfer's vendored DeepGEMM-derived source.
- The installed FlashInfer adapter exposes SM100-oriented FP8 grouped GEMM entry points, not a drop-in BF16/FP16 dense linear replacement for the current Qwen path.
- edge-fm's public `DType` has no explicit Float8 enum yet, and current Qwen weights load as FP16/BF16. A real runtime binding needs a future FP8/W8A8 artifact contract with weight tensors, activation/weight scales, scale layouts, and conversion ownership.
- Probe note on RTX 3060 / `cuda_sm86`: source/import are present, but hardware support is false. No runtime binding should be planned from this machine's result.

Recommended integration path, when an SM90/SM100 machine and FP8/W8A8 artifacts are available:

1. Keep `linear -> cublasLt` as the default.
2. Add a default-off `linear` impl candidate named `deepgemm` only after the FP8 artifact contract exists.
3. Restrict support to prefill large-M dense linear roles first: `fused_qkv`, `attention_output`, `fused_gate_up`, `mlp_down`.
4. Do not use it for `m=1` decode or `lm_head` in this round.
5. Make `supports(ctx)` reject unsupported dtype, missing scales, missing layout transform, unsupported SM, CUDA graph incompatibility, and unavailable JIT/runtime artifacts.

## Prefix KV

Prefix KV should be tracked as implemented, not deferred.

Current contract:

- `kvcache.requests[].prefix_token_ids` is parsed by `KVManager`; compact vocab mode remaps these ids before storing the slot metadata.
- KV remains continuous per request and per layer. Read pointers stay at the slot base, while write pointers are offset by `prefix_size`.
- `Scheduler::create_context()` validates that request `token_ids` start with the configured prefix.
- `StandardEngine::warmup()` pre-fills the prefix into the configured KV slot and can use that state to capture the decode CUDA graph.
- Request-time prefill skips the prefix tokens and only writes the non-prefix suffix, so decode continues from one contiguous KV buffer.

Remaining limits:

- No paged attention or non-contiguous KV layout.
- No semantic or approximate prefix cache lookup beyond the configured request slot.
- No INT8 KV format or scale-buffer contract in this path.

## Deferred Work

Continuous INT8 KV is still deferred. Owner A should keep future continuous-buffer interface expectations stable, but engine-side INT8 KV should wait for Owner C attention/kernel support and a concrete K/V scale contract.
