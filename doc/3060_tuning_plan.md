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
- Latest bridge full-matrix result: `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_full_llm_matrix_runs3_summary.json`
- Latest bridge conclusion: GateUp-FP16 MLP bridge plus bias-aware BF16 TensorRT QKV/OProj linear bridge is the current best low-risk bridge. It improves `15/18` cases versus GateUp-FP16, beats the latest official TRT baseline in `3/18` cases, and leaves the largest gap at `3B / 2048x32` with `+25.60 ms`.
- Latest memory blocker: `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is memory-blocked on 3B and must stay out of the active queue until there is a new memory policy.
- Latest bridge residual profile: `2026-05-12 09:33 +0800`, graph-off mapping on the current GateUp-FP16 + QKV/OProj bridge still shows prefill dominated by `gateup_matmul_edgefm_layout_myl0_3` (`43.3%`), `__myl_FcCast_myl0_8` (`22.5%`), and FlashInfer prefill attention (`10.8%`). The analyzer action table points first to the linear impl path, then to prefill attention. Raw triage: `.tmp_codex/nsys/3060_20260512_qkv_oproj_bridge_3b_2048x1_mapping_triage.md`.
- Latest attention prefill tuning probe: `2026-05-12 10:07 +0800`, the isolated 3B / 2048 attention win from `prefill_cta_tile_q=128` did **not** survive end-to-end. A paired bridge run on the same environment measured `3B / 2048x32` at EdgeFM `947.754 ms` with the current `prefill_cta_tile_q=64` table entry and `948.438 ms` with `128` (`+0.684 ms`, `+0.07%`). The current attention table therefore stays at `64`; keep `128` only as a rejected diagnostic result. Raw: `.tmp_codex/bench/3060_20260512_attention_tile128_bridge_paired_edgefm_3b_2048x32.json`; isolated probe raw: `.tmp_codex/bench/3060_20260512_attention_prefill_3b_2048_tune.json`.
- Latest full-matrix largest gap after default-off: `3B / 2048x32`, EdgeFM mean `1125.25 ms` versus TRT mean `927.69 ms`, mean gap `+197.57 ms`, prefill gap `+201.15 ms`
- Previous full-matrix largest gap before default-off: `3B / 2048x32`, EdgeFM mean `1161.92 ms` versus TRT mean `928.48 ms`, mean gap `+233.44 ms`
- Current diagnosis: the remaining gap is still prefill dominated, especially long-prefill 3B and 1.5B cases
- Current best bridge diagnosis: after GateUp-FP16 + QKV/OProj linear, the residual gap is concentrated in 3B long-prefill prefill work, especially GateUp/cast overhead and the remaining attention path.
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
- Strategy update for 3060: do not keep spending cycles on more CUTLASS retuning or a new handwritten CUTLASS-style kernel as the primary route. The source-visible search has already converged: TRT is getting a meaningful part of its advantage from compiler-generated closed Myelin/XMMA/FcCast tactics that are not directly callable from EdgeFM operator impls. The active route is therefore reviewed TensorRT subgraph/subengine bridging for selected prefill modules, kept compile/runtime/default-off until correctness, memory, and regression gates clear.
- Myelin/XMMA direct reuse check: `third_party/TensorRT-Edge-LLM` does not expose a public/source-visible BF16/FP16 dense Myelin/XMMA GEMM launcher for `LinearImpl`. Do not add a fake `myelin` or `xmma` impl id. The only valid way to test those tactics is a reviewed TensorRT-backed subgraph/engine bridge or an isolated `.tmp_codex` feasibility probe.
- Myelin/XMMA subengine probe: the isolated TensorRT `GateUp -> SwiGLU -> DownProj` engine reproduced the internal tactics. BF16 selected `sm80_xmma_gemm_bf16bf16_*` plus `__myl_SiluMul_*` but measured `11.33 ms` median per layer, slower than the current EdgeFM MLP estimate `10.62 ms`. FP16 selected the same `sm80_xmma_gemm_f16f16_*` family seen in TRT-Edge-LLM and measured `6.26 ms` median per layer, matching TRT's inferred `6.18 ms`. Raw: `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_bf16_kernel_summary.md`, `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_fp16_kernel_summary.md`
- Latest TRT FP16 MLP feasibility: `2026-05-11`, actual Qwen2.5-3B layer weights work in a memory-safe runtime-weight TensorRT engine. One `67 KB` engine with `gateup_weight` and `down_weight` as runtime inputs selected `sm80_xmma_gemm_f16f16_*` plus Myelin activation, validated against torch FP16 reference, and the fresh rerun measured layer 0 at `6.06 ms` median and layer 35 with the same engine at `6.26 ms` median. This matches TRT-Edge-LLM's inferred `6.18 ms/layer` and avoids the `~130 MB/layer` serialized constant-weight engine duplication. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_rerun_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_rerun_verify.json`
- Latest production-layout TRT MLP feasibility: `2026-05-11`, a more realistic TensorRT runtime-weight engine binds EdgeFM-shaped BF16 weights directly (`GateUp [up,gate,H]`, DownProj `[H,I]`), casts activation/weights to FP16 inside TensorRT, uses transposed MatMul operands, runs Myelin activation, and casts output back to BF16. Inspector confirms FP16 XMMA, not BF16 XMMA. Medians: 3B `m=2048` layer 0 `7.02 ms` and layer 35 reuse `7.18 ms`, 3B `m=1024` `4.12 ms`, 3B `m=512` `2.47 ms`, 1.5B `m=2048` `4.29 ms`, 0.5B `m=2048` `1.57 ms`. This is slower than pure FP16 runtime weights but avoids persistent FP16 weight copies and remains materially faster than current 3B `m=2048` EdgeFM MLP estimate `10.62 ms/layer`. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_bf16_edgefm_layout_fp16_compute_reuse_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_1p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_0p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`
- CUTLASS prepacked-B layout probe: classic `third_party/cutlass` `device::Gemm` with `RowMajor x RowMajor` B layout improved the best FP16 GateUp/DownProj probes slightly but still did not approach TRT FP16 XMMA. Best GateUp `6.666 ms` versus TensorRT FP16 `3.856 ms`; best DownProj `3.440 ms` versus TensorRT FP16 `1.984 ms`. Raw: `.tmp_codex/bench/3060_20260510_cutlass_layout_fp16_sweep_summary.json`
- Source-visible third-party search status: no direct 3060 dense FP16 XMMA-equivalent runner has been found in the vendored tree. `trtllm_gemm_runner` is FP8/E2M1-to-BF16 only; `tgv_gemm` is SM100/UMMA/TMA-oriented and not an SM86 path; TensorRT-LLM CUTLASS FP16/BF16 code here is MoE/grouped, fused-MoE, or quantized GEMM. The only current build-integrated fused-MoE helper is the rejected `cutlass_prefill_swiglu` diagnostic path.
- Latest cuTile MatMul probe: `2026-05-10 11:28 +0800`, `third_party/cutile-python/samples/MatMul.py` was run with the pip-provided TileIR compiler enabled via `nvidia-cuda-tileiras`. Best persistent `64x128x64` results were FP16 `GateUp 9.87 ms` / `DownProj 4.80 ms` and BF16 `GateUp 9.75 ms` / `DownProj 4.79 ms`, which is slower than current cublasLt/BF16 and far behind TRT FP16 XMMA. Raw artifacts: `.tmp_codex/bench/3060_20260510_cutile_matmul_fp16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_bf16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_summary.json`
- Latest experimental bridge matrix: `2026-05-11 11:43 +0800`, an env/compile-gated TensorRT prefill MLP bridge can bind EdgeFM-owned BF16 GateUp/DownProj weights at runtime, run TensorRT internal FP16 XMMA/Myelin compute, and return BF16 output. It also exposed and fixed an independent prefill CUDA graph first-capture bug: after capture, the first request must replay the captured graph before `generate()` advances. Correctness passes for the first-request prefill graph regression and `3B / 2048x32` bridge+CUDA graph versus `transformers.generate(do_sample=False)`. The bridge now has engine coverage for all 9 official MLP shapes (`0.5B/1.5B/3B x prefill 512/1024/2048`). The experimental 18-case EdgeFM-only matrix improved all cases versus native, with total mean improvement `662.46 ms`, but only `2/18` cases beat the latest official TRT baseline. Largest remaining gaps: `3B / 2048x32` bridge `986.44 ms` versus TRT `927.69 ms` (`+58.75 ms`), `3B / 2048x64` `+53.65 ms`, `3B / 1024x32` `+46.05 ms`. Clean raw: `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_clean.json`; summary: `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_summary.json`; correctness: `.tmp_codex/bench/3060_20260511_1117_3b_2048x32_trt_mlp_bridge_transformers_generate_compare_after_prefill_graph_fix.json`.
- Latest bridge residual profile: `2026-05-11 11:50 +0800`, `3B / 2048x1` graph-off nsys shows the bridge MLP GEMM/activation core is already aligned with TRT: bridge GateUp+SwiGLU+DownProj without casts is `216.92 ms`, versus TRT inferred `222.39 ms`. Bridge prefill kernel total is still `350.41 ms` versus TRT `285.46 ms`, delta `+64.95 ms`. The largest extra item is TensorRT internal BF16 weight casts `29.86 ms` plus activation/output casts `3.77 ms`; remaining non-MLP deltas are QKV `+16.08 ms`, OProj `+10.04 ms`, and prefill attention `+10.40 ms`. Summary: `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_residual_summary.json`; triage: `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_mapping_triage.md`.
- Latest FP16 bridge weight-input diagnostic: `2026-05-11 12:03 +0800`, isolated TensorRT `3B / m=2048` EdgeFM-layout MLP probes show BF16 weight-input bridge `7.02 ms/layer`, GateUp FP16 input `6.60 ms/layer`, DownProj FP16 input `6.90 ms/layer`, and both GateUp+DownProj FP16 inputs `6.25 ms/layer`. Pure FP16 runtime weights remain `6.06 ms/layer`. The `both` nsys trace confirms the large BF16 weight-cast kernels are gone, leaving input/output/activation casts and two `sm80_xmma_gemm_f16f16_*` GEMMs. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_both_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_gateup_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_down_verify.json`, `.tmp_codex/nsys/trt_mlp_3b_m2048_fp16weights_both_probe.sqlite`.
- Latest GateUp-FP16 bridge target result: `2026-05-11 12:36 +0800`, the compile/runtime-gated bridge can bind persistent FP16 GateUp copies for `3B / m=2048` using a TensorRT BF16->FP16 cast subengine. `3B / 2048x32` CUDA graph matches `transformers.generate(do_sample=False, eos_token_id=None)` for all `32` generated tokens. Warmed performance: EdgeFM mean `964.20 ms`, median `964.34 ms`, prefill avg `334.83 ms`, decode avg `629.12 ms`. This is `-22.24 ms` versus the BF16 bridge and `-161.06 ms` versus native EdgeFM, but remains `+36.51 ms` slower than the latest TRT baseline `927.69 ms`. Raw: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_runs3_clean.json`; correctness: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_transformers_compare_clean.json`.
- Latest validation after GateUp-FP16 WIP: `git diff --check`, default build/install, bridge build/install, operator table validation, default generate alignment, default `test_prefill_linear.py`, default `test_fused_gate_up_activation.py`, and bridge-build prefill-graph first-request regression passed on `2026-05-11 12:36 +0800`. `test_attention_decode.py` is currently all skipped in this environment, so it did not provide a decode-attention regression signal. The build-path helper now preserves explicit `EDGE_FM_BUILD_DIR` priority; this prevents bridge gates from silently importing the default native build.
- Latest GateUp-FP16 full-matrix result: `2026-05-11 12:54 +0800`, all official MLP shapes now have GateUp-FP16 TensorRT engines and the 18-case EdgeFM-only CUDA graph matrix completed without missing-engine fallback. GateUp-FP16 improved all `18/18` cases versus the BF16 bridge, with total mean improvement `191.57 ms` and mean per-case improvement `10.64 ms`. It reaches or beats the latest official TRT baseline in `3/18` mean cases and `3/18` median cases, so it is still not sufficient for the final target. Raw: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_clean.json`; summary: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_summary.json`.
  - Largest remaining gaps: `3B / 2048x32` GateUp-FP16 `970.55 ms` versus TRT `927.69 ms` (`+42.86 ms`), `3B / 2048x64` `+37.37 ms`, `3B / 1024x32` `+23.43 ms`, `0.5B / 2048x64` `+18.09 ms`, and `3B / 1024x64` `+17.86 ms`.
  - Cases currently at or faster than TRT: `0.5B / 512x32`, `0.5B / 512x64`, and `3B / 512x64`.
- Latest GateUp-FP16 residual profile: `2026-05-11 13:18 +0800`, `3B / 2048x1` graph-off mapping shows the bridge MLP core is now essentially at TRT parity: GateUp+SwiGLU+DownProj without casts is `222.54 ms` versus TRT `222.39 ms`. The remaining graph-off stage delta is `+50.62 ms`: DownProj BF16->FP16 weight cast `9.93 ms`, activation/input/output casts `1.95 ms`, QKV `+16.39 ms`, OProj `+10.54 ms`, FlashInfer prefill attention `+11.29 ms`, and norm `+0.31 ms`. Raw: `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_residual_summary.json`.
- Latest GateUp+DownProj-FP16 target slice: `2026-05-11 13:24 +0800`, `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` passed `3B / 2048x32` CUDA graph token correctness against `transformers.generate(do_sample=False, eos_token_id=None)` and measured EdgeFM mean `952.10 ms`, median `952.18 ms`, prefill avg `322.12 ms`. This improves `3B / 2048x32` by `18.45 ms` versus GateUp-FP16 and leaves a `+24.41 ms` gap versus latest TRT `927.69 ms`. Memory probe leaves only `161.6 MiB` free after first generate and timed runs, so this is not safe to expand/default-enable for 3B without a new memory policy. Raw: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_summary.json`; memory: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_memory_probe_clean.json`.
- Latest QKV/OProj TensorRT linear probe: `2026-05-11 13:35 +0800`, isolated runtime-weight TensorRT linear engines for `3B / m=2048` EdgeFM `[out,in]` weight layout show a plausible next `~18.6-22.7 ms` gain versus current EdgeFM QKV+OProj. BF16-weight inputs estimate QKV+OProj `39.37 ms` across 36 layers; FP16-weight inputs estimate `35.24 ms`; current EdgeFM profile is `57.95 ms`; TRT reference is `31.01 ms`. This needs a review-gated QKV/OProj bridge design before production code. Raw: `.tmp_codex/bench/20260511_trt_linear_qkv_oproj_3b_m2048_summary.json`.
- Latest QKV/OProj bridge target and regression slice: `2026-05-11 14:05 +0800`, the QKV/OProj bridge is now bias-aware for QKV. The no-bias QKV engine failed because Qwen2.5 Q/K/V projections have BF16 bias; OProj has no bias. With GateUp-FP16 MLP bridge plus BF16 TensorRT QKV/OProj linear bridge, correctness passed for `3B / 2048x32` and `1.5B / 2048x32`. Performance slices are all positive versus GateUp-FP16: `3B / 2048x32` `970.55 -> 948.91 ms` (`-21.64 ms`, remaining TRT gap `+21.23 ms`), `3B / 2048x64` `1621.74 -> 1597.05 ms` (`-24.69 ms`, gap `+12.68 ms`), and `1.5B / 2048x32` `491.52 -> 482.96 ms` (`-8.56 ms`, gap `+8.73 ms`). The path now has full-matrix evidence from the later 18-case run, but it remains compile/runtime/default-off pending same-run TRT comparison and packaging/cleanup. Raw summaries: `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x32_summary.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x64_summary.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_1p5b_2048x32_summary.json`.
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
- `src/engine/tasks/token_generation/cuda/standard_engine.cpp`, `tests/engine/test_qwen2_generate.py`
  - prefill CUDA graph first-capture now immediately replays the captured graph before returning to `generate()`
  - a regression test verifies the first request without a warmed prefix returns a valid prefill sample token

## Experimental Candidate

`BUILD_TRT_MLP_BRIDGE=ON` + `EDGE_FM_PREFILL_TRT_MLP=1` is a measurable prefill MLP candidate, not a default path.

- It uses one runtime-weight TensorRT engine per MLP shape and binds EdgeFM resident BF16 weights directly.
- It keeps the native `linear + activation + linear` fallback untouched when disabled, when an engine is missing, or when IO validation fails.
- It improves all 18 experimental EdgeFM-only CUDA graph cases versus the latest native full-matrix baseline.
- It reaches or beats the latest official TRT baseline in only `2/18` mean cases (`0.5B / 512x32` and `0.5B / 512x64`).
- It still leaves the largest gaps on 3B long prefill: `3B / 2048x32` `+58.75 ms`, `3B / 2048x64` `+53.65 ms`, `3B / 1024x32` `+46.05 ms`.
- Do not default-enable or commit generated TensorRT engines. The next gates are a reproducible engine builder, full optional-path correctness gate coverage, same-run TRT comparison from a compatible Python/ABI build, and residual-gap profiling.

`EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup` is the current sharper sub-candidate.

- It keeps the same bridge default-off gates and creates persistent FP16 GateUp copies through a TensorRT BF16->FP16 cast subengine; no handwritten conversion kernel is used.
- It now has engine coverage for all official MLP shapes: `0.5B/1.5B/3B x m=512/1024/2048`.
- It improves all `18/18` full-matrix cases versus the BF16 bridge, with mean per-case improvement `10.64 ms`.
- It improves `3B / 2048x32` from the BF16 bridge `986.44 ms` to `970.55 ms` in the full-matrix rerun, with prefill avg `339.82 ms`.
- It still trails the latest TRT baseline by `+42.86 ms` on `3B / 2048x32` and adds about `3.02 GiB` of persistent weight memory for 3B.
- Keep it experimental/default-off until more shape engines, regression slices, and full optional-path correctness coverage are available.

`EDGE_FM_PREFILL_TRT_LINEAR=1` with `EDGE_FM_TRT_LINEAR_ROLES=both` is the current residual prefill bridge candidate.

- It uses runtime-weight TensorRT linear engines for QKV and OProj only in prefill, preserving native Q/K/V split, KV-cache copy, attention, decode, and fallback behavior.
- QKV must use a bias-aware engine because Qwen2.5 Q/K/V projections carry BF16 bias. Missing that bias was the root cause of the earlier QKV token mismatch.
- With GateUp-FP16 underneath, the full 18-case matrix improves `15/18` cases versus GateUp-FP16 and reaches or beats TRT in `3/18` mean and median cases.
- The largest remaining gap in that matrix is `3B / 2048x32` at `+25.60 ms`.
- Keep it default-off until any future cleanup is proven against the full matrix and a same-process TRT comparison is available.

`EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is now a blocked diagnostic, not an active optimization route.

- It produced the best warmed single-slice improvement on `3B / 2048x32`, but the rerun still hit `act_and_mul_kernel launch failed: out of memory`.
- Do not expand or default-enable it for 3B without a new memory policy.

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

1. Keep the QKV/OProj linear bridge as the active low-risk prefill path and decide whether the next cleanup is operator-table-style or still bridge-level, based on the smallest review surface that preserves the current gains.
2. Profile the remaining 3B long-prefill gap after GateUp-FP16 + QKV/OProj linear. The current `prefill_cta_tile_q=128` attention sweep is rejected end-to-end, so revisit attention only if a new TRT-Edge-LLM plugin reuse route or a new FlashInfer config is profile-justified.
3. Do not spend time on `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` until there is a new memory policy. It is a blocked diagnostic path, not an active optimization route.
4. Keep BF16 TensorRT MLP bridge out of the active implementation queue; the isolated BF16 subengine did not beat EdgeFM's current MLP estimate.
5. Convert the TensorRT bridge paths from prototype to a clean optional feature only if they can build/generate all required engines reproducibly and pass the full correctness gate. Keep them compile/runtime-gated and default-off.
6. Fix the Python/ABI split so bridge-enabled EdgeFM and `edge_fm_trt` can run in the same benchmark process, or produce an equivalent official 3-way wrapper. Current bridge matrices compare against the latest full-matrix TRT baseline, not same-run TRT.
7. Keep source-visible third-party search open only for materially different runners. Do not spend more time on small classic CUTLASS `device::Gemm` layout variants, TGV SM100 paths, FP8/E2M1 `trtllm_gemm_runner`, cuTile dense MatMul sweeps, or the already-rejected fused-MoE prefill path unless a new tactic source appears.
8. Exclude already-tested dead ends from the active queue: the current three `LinearCutlassImpl` configs, temporary FP16 checkpoint conversion, FP16-only cublasLt table tuning, cublasLt native row-major descriptors, and the attempted cublasLt packed weight layouts are not sufficient to close the prefill gap.
9. Do not spend the next optimization round on removing EdgeFM token/KV copy overhead unless a new profile contradicts the current `sub-1 ms` memcpy evidence.
10. Review [3060_fused_mlp_review.md](./3060_fused_mlp_review.md) before any larger fused MLP / plugin-style prefill path changes. This remains the gate for engine/layer/operator boundary changes.
11. Continue small table/runtime checks only when a profile shows a specific existing path mismatch. Keep only changes that improve the target case by at least `1%` and do not create meaningful regressions elsewhere. The current `prefill_cta_tile_q=128` attention candidate does not meet that bar and stays rejected.

## Acceptance Criteria

- Correctness gate passes for every accepted change.
- `scripts/operator_table/validate_operator_tables.py` passes after table edits.
- The official 18-case matrix is eventually `EdgeFM(cuda graph) <= TRT-Edge-LLM` everywhere, or the residual gap is documented with raw artifacts and a measured explanation.
- Failed experiments are reverted quickly and moved to rejected/obsolete notes instead of staying in the current status section.
