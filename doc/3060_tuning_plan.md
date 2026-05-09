# 3060 LLM Runtime and Operator Tuning Plan

## Goal

Close the full 18-case RTX 3060 LLM matrix for:

- `Qwen2.5-{0.5B,1.5B,3B}`
- `prefill={512,1024,2048}`
- `decode={32,64}`

Official comparisons are only `EdgeFM(cuda graph)` versus `TRT-Edge-LLM`.
`graph-off` and `nsys` are attribution tools only.

The standing tuning rules live in [3060_tuning_rules.md](./3060_tuning_rules.md).

## Current Baseline

- Latest official raw artifact: `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off.json`
- Latest summary artifact: `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off_summary.json`
- Latest accepted target slice: `.tmp_codex/bench/3060_20260509_3b_2048x32_prefill_swiglu_default_off.json`
- Latest accepted regression slice: `.tmp_codex/bench/3060_20260509_prefill_swiglu_default_off_regression_0p5_1p5_2048x32.json`
- Latest full-matrix largest gap after default-off: `3B / 2048x32`, EdgeFM mean `1125.25 ms` versus TRT mean `927.69 ms`, mean gap `+197.57 ms`, prefill gap `+201.15 ms`
- Previous full-matrix largest gap before default-off: `3B / 2048x32`, EdgeFM mean `1161.92 ms` versus TRT mean `928.48 ms`, mean gap `+233.44 ms`
- Current diagnosis: the remaining gap is still prefill dominated, especially long-prefill 3B and 1.5B cases
- Decode is already close to TRT or better for the large models, so the work queue should stay prefill-first
- Formal prefill SwiGLU diagnostic helper: `scripts/tune/profile_prefill_swiglu_kernels.py`
- Latest accepted optimization: prefill SwiGLU fusion is now default-off because the existing TensorRT-LLM/CUTLASS fused MoE reuse path regressed 3060 end-to-end prefill; use `EDGE_FM_PREFILL_SWIGLU_FUSION=1` only for explicit diagnostics.
- Latest validation gate: `2026-05-09 17:40 +0800`; cleanup before commit passed diff check, Python compile, build/install, operator table validation, operator gate, and generate alignment. Logs: `.tmp_codex/bench/3060_20260509_final_build_install.log`, `.tmp_codex/bench/3060_20260509_final_validate_operator_tables.log`, `.tmp_codex/bench/3060_20260509_final_operator_gate.log`, `.tmp_codex/bench/3060_20260509_final_generate_alignment.log`
- Latest post-default-off profile: `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping.nsys-rep`, triage `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping_triage.md`, role summary `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping_role_summary.json`
- Latest profiling conclusion: the rejected `fused_moe` prefill SwiGLU hotspot disappeared. `3B / 2048x32` graph-off prefill is now dominated by existing linear work: GateUp `242.41 ms` (`50.48%`), DownProj `125.26 ms` (`26.09%`), QKV `32.16 ms`, attention `32.72 ms`, OProj `23.69 ms`.
- `3B / 2048x64` graph-off mapping confirms the same long-prefill pattern: GateUp `243.82 ms` (`50.62%`) and DownProj `124.74 ms` (`25.90%`) dominate prefill. Raw: `.tmp_codex/nsys/3060_3b_2048x64_default_off_mapping_role_summary.json`
- TRT-Edge-LLM `3B / 2048x1` nsys reverse attribution is now available: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.nsys-rep`, summary `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_kernel_summary.md`
- TRT prefill kernel result: total kernel time `285.46 ms`, GEMM `240.70 ms` (`84.32%`), Gate+Up `136.50 ms`, DownProj `71.36 ms`, QKV `16.96 ms`, OProj `14.06 ms`, AttentionPlugin `22.90 ms`, SwiGLU `14.53 ms`
- Current inference from TRT trace: TRT is still using per-layer Gate+Up, SwiGLU, and DownProj structure, but its TensorRT XMMA/Myelin GEMM paths are much faster than EdgeFM's current linear choices for the same long-prefill roles.
- Latest rejected low-risk route: 3B `m=2048` cublasLt and FlashInfer attention table sweeps produced only sub-1% official CUDA graph movement. No table change is accepted from that sweep.
- Cleanup note: unaccepted extra CUTLASS tile heuristics in `src/operators/linear_impl.cu` were removed before commit; the file is back to the original three stable CUTLASS configs.

## Accepted Work So Far

- `src/engine/engine_factory.cpp`, `src/models/qwen2_5/qwen2_5.cpp`, `src/utils/device/weight_loader.*`
  - shared checkpoint paths now reuse GPU weights instead of duplicating them and OOMing 3B on 3060
- `src/operators/attention_op.cu`
  - FP16 decode support is kept on the tuned FlashInfer decode path
  - FP16 prefill still uses the generic prefill path
- `scripts/tune/tune_qwen_cublaslt.py`
  - `--dtype bf16|fp16` is available for aligned shape-signature tuning
- `tests/operators/_test_utils.py`, `tests/operators/test_prefill_linear.py`
  - operator tests now resolve the active hw profile instead of hardcoding stale `cuda_sm80` assumptions
- `src/operators/activation_op.cu`
  - tail cleanup is accepted and should remain neutral for current Qwen hidden sizes
- `examples/config/platform/3060/operator_impl_table*.json`
  - accepted 0.5B and 3B table records remain in the mainline 3060 tables
  - `1p5b_fused_qkv_m2048` keeps `algo_index=4`; `algo_index=6` is rejected on 3060 because it regressed the operator gate
- `tests/engine/test_qwen2_generate.py`
  - LLM benchmark releases the Transformers model before allocating EdgeFM/TRT runtimes; this is required for 3B to run on 3060 12GB

## Mainline Candidate

`cutlass_prefill_swiglu` / `try_forward_prefill_swiglu_fused` remains in the codebase only as an explicit diagnostic candidate.

It is default-off because the first 3060 end-to-end validation showed a regression when enabled:

- 3B prefill SwiGLU microbench, current fused versus two-stage:
  - `512`: fused `2.085 ms`, two-stage `1.832 ms`, delta `+0.253 ms`
  - `1024`: fused `4.144 ms`, two-stage `3.795 ms`, delta `+0.349 ms`
  - `2048`: fused `8.239 ms`, two-stage `7.177 ms`, delta `+1.062 ms`
- Official target case after default-off:
  - `3B / 2048x32`: EdgeFM mean improved by `42.08 ms` versus the 14:34 baseline; prefill gap improved by `37.07 ms`
- Full matrix after default-off:
  - 18/18 cases completed
  - EdgeFM remains faster than or equal to TRT in `1/18` mean cases and `1/18` median cases
  - EdgeFM mean improved in all 18 cases versus the 14:34 baseline, with improvements from `0.45 ms` to `36.67 ms`
- Regression slice after default-off:
  - `0.5B / 2048x32`: EdgeFM mean improved by `7.53 ms`
  - `1.5B / 2048x32`: EdgeFM mean improved by `22.04 ms`

Do not re-enable it by default unless a new existing-kernel configuration proves correctness and at least `>=1%` CUDA graph end-to-end improvement without regression.

## Next Work Queue

1. Inspect whether current EdgeFM can select or wrap an existing third-party GEMM path closer to TRT's `sm80_xmma_gemm_*_trt` behavior for `3B m=2048` Gate+Up and DownProj. Keep this bounded to operator-level selection/probing first.
2. Review [3060_fused_mlp_review.md](./3060_fused_mlp_review.md) before any larger fused MLP / plugin-style prefill path changes. This remains the gate for engine/layer/operator boundary changes.
3. If review approves a larger path, prototype it behind an env/table switch and reuse existing third-party kernels first; do not write a new kernel from scratch.
4. Continue small table/runtime checks only when a profile shows a specific existing path mismatch. Keep only changes that improve the target case by at least `1%` and do not create meaningful regressions elsewhere.
5. Optional attribution follow-up: collect `1.5B / 2048x32` graph-off mapping trace to confirm the same GateUp/DownProj-dominated profile before investing implementation time.

## Acceptance Criteria

- Correctness gate passes for every accepted change.
- `scripts/operator_table/validate_operator_tables.py` passes after table edits.
- The official 18-case matrix is eventually `EdgeFM(cuda graph) <= TRT-Edge-LLM` everywhere, or the residual gap is documented with raw artifacts and a measured explanation.
- Failed experiments are reverted quickly and moved to rejected/obsolete notes instead of staying in the current status section.
