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
- Benchmark alignment check: EdgeFM `prefill_ms` includes prefill prepare/token copy, but this does not explain the gap. EdgeFM graph-off `3B / 2048x32` memcpy total is `0.623 ms`; EdgeFM graph-on formal trace shows `0.044 ms` memcpy outside CUDA graph; TRT explicit synchronize probe changes wall time by only about `0.6 ms`. Raw: `.tmp_codex/bench/3060_20260509_trt_3b_2048x32_sync_probe.json`
- `3B / 2048x64` graph-off mapping confirms the same long-prefill pattern: GateUp `243.82 ms` (`50.62%`) and DownProj `124.74 ms` (`25.90%`) dominate prefill. Raw: `.tmp_codex/nsys/3060_3b_2048x64_default_off_mapping_role_summary.json`
- TRT-Edge-LLM `3B / 2048x1` nsys reverse attribution is now available: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.nsys-rep`, summary `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_kernel_summary.md`
- TRT prefill kernel result: total kernel time `285.46 ms`, GEMM `240.70 ms` (`84.32%`), Gate+Up `136.50 ms`, DownProj `71.36 ms`, QKV `16.96 ms`, OProj `14.06 ms`, AttentionPlugin `22.90 ms`, SwiGLU `14.53 ms`
- Current inference from TRT trace: TRT is still using per-layer Gate+Up, SwiGLU, and DownProj structure, but its TensorRT XMMA/Myelin GEMM paths are much faster than EdgeFM's current linear choices for the same long-prefill roles.
- Myelin/XMMA direct reuse check: `third_party/TensorRT-Edge-LLM` does not expose a public/source-visible BF16/FP16 dense Myelin/XMMA GEMM launcher for `LinearImpl`. Do not add a fake `myelin` or `xmma` impl id. The only valid way to test those tactics is a reviewed TensorRT-backed subgraph/engine bridge or an isolated `.tmp_codex` feasibility probe.
- Myelin/XMMA subengine probe: the isolated TensorRT `GateUp -> SwiGLU -> DownProj` engine reproduced the internal tactics. BF16 selected `sm80_xmma_gemm_bf16bf16_*` plus `__myl_SiluMul_*` but measured `11.33 ms` median per layer, slower than the current EdgeFM MLP estimate `10.62 ms`. FP16 selected the same `sm80_xmma_gemm_f16f16_*` family seen in TRT-Edge-LLM and measured `6.26 ms` median per layer, matching TRT's inferred `6.18 ms`. Raw: `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_bf16_kernel_summary.md`, `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_fp16_kernel_summary.md`
- CUTLASS prepacked-B layout probe: classic `third_party/cutlass` `device::Gemm` with `RowMajor x RowMajor` B layout improved the best FP16 GateUp/DownProj probes slightly but still did not approach TRT FP16 XMMA. Best GateUp `6.666 ms` versus TensorRT FP16 `3.856 ms`; best DownProj `3.440 ms` versus TensorRT FP16 `1.984 ms`. Raw: `.tmp_codex/bench/3060_20260510_cutlass_layout_fp16_sweep_summary.json`
- Source-visible third-party search status: no direct 3060 dense FP16 XMMA-equivalent runner has been found in the vendored tree. `trtllm_gemm_runner` is FP8/E2M1-to-BF16 only; `tgv_gemm` is SM100/UMMA/TMA-oriented and not an SM86 path; TensorRT-LLM CUTLASS FP16/BF16 code here is MoE/grouped, fused-MoE, or quantized GEMM. The only current build-integrated fused-MoE helper is the rejected `cutlass_prefill_swiglu` diagnostic path.
- Latest cuTile MatMul probe: `2026-05-10 11:28 +0800`, `third_party/cutile-python/samples/MatMul.py` was run with the pip-provided TileIR compiler enabled via `nvidia-cuda-tileiras`. Best persistent `64x128x64` results were FP16 `GateUp 9.87 ms` / `DownProj 4.80 ms` and BF16 `GateUp 9.75 ms` / `DownProj 4.79 ms`, which is slower than current cublasLt/BF16 and far behind TRT FP16 XMMA. Raw artifacts: `.tmp_codex/bench/3060_20260510_cutile_matmul_fp16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_bf16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_summary.json`
- Latest rejected low-risk route: 3B `m=2048` cublasLt and FlashInfer attention table sweeps produced only sub-1% official CUDA graph movement. No table change is accepted from that sweep.
- Latest post-commit diagnostics: existing `LinearCutlassImpl` configs do not beat the current 3060 cublasLt table for `3B / BF16 / m=2048`; a temporary FP16 EdgeFM checkpoint also did not close the `3B / 2048x32` gap. Raw artifacts: `.tmp_codex/bench/3060_20260509_probe_3b_m2048_cutlass_vs_cublaslt.json`, `.tmp_codex/bench/3060_20260509_3b_2048x32_fp16_edgefm_probe.json`
- Latest cublasLt layout probe: native row-major descriptors were neutral/slightly slower for `3B / m=2048` GateUp/DownProj, and tested cublasLt packed weight transform/layout combinations returned unsupported. Raw artifacts: `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_gateup_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_downproj_bf16.jsonl`
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

1. Move the FP16 TensorRT-backed MLP bridge to an explicit review artifact before any production implementation. The review must cover precision policy, generation correctness, weight duplication/refit, RTX 3060 12GB memory pressure, CUDA graph compatibility, and fallback.
2. Keep source-visible third-party search open only for materially different runners. Do not spend more time on small classic CUTLASS `device::Gemm` layout variants, TGV SM100 paths, FP8/E2M1 `trtllm_gemm_runner`, cuTile dense MatMul sweeps, or the already-rejected fused-MoE prefill path unless a new tactic source appears.
3. Keep BF16 TensorRT MLP bridge out of the active implementation queue; the isolated BF16 subengine did not beat EdgeFM's current MLP estimate.
4. Exclude already-tested dead ends from the active queue: the current three `LinearCutlassImpl` configs, temporary FP16 checkpoint conversion, FP16-only cublasLt table tuning, cublasLt native row-major descriptors, and the attempted cublasLt packed weight layouts are not sufficient to close the prefill gap.
5. Do not spend the next optimization round on removing EdgeFM token/KV copy overhead unless a new profile contradicts the current `sub-1 ms` memcpy evidence.
6. Review [3060_fused_mlp_review.md](./3060_fused_mlp_review.md) before any larger fused MLP / plugin-style prefill path changes. This remains the gate for engine/layer/operator boundary changes.
7. Continue small table/runtime checks only when a profile shows a specific existing path mismatch. Keep only changes that improve the target case by at least `1%` and do not create meaningful regressions elsewhere.
8. Optional attribution follow-up: collect `1.5B / 2048x32` graph-off mapping trace to confirm the same GateUp/DownProj-dominated profile before investing implementation time.

## Acceptance Criteria

- Correctness gate passes for every accepted change.
- `scripts/operator_table/validate_operator_tables.py` passes after table edits.
- The official 18-case matrix is eventually `EdgeFM(cuda graph) <= TRT-Edge-LLM` everywhere, or the residual gap is documented with raw artifacts and a measured explanation.
- Failed experiments are reverted quickly and moved to rejected/obsolete notes instead of staying in the current status section.
