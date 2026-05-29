# CODA Kernels RTX 3060 Evaluation

## Scope

Evaluate `HanGuo97/coda-kernels` as a third-party dependency and as a possible
Edge-FM optimization path for RTX 3060 Qwen3.5 and Qwen2.5 inference.

## Third-Party State

- Vendored as a git submodule at `third_party/coda-kernels`.
- Pinned commit: `acc7dd250263020af94de8e25af2c669b7da551a` (`acc7dd2`, `main`).
- URL: `https://github.com/HanGuo97/coda-kernels.git`.
- Local integration status: research/reference only. It is not wired into
  CMake, EdgeFM runtime dispatch, or any default operator table.

## Technical Fit

CODA/Rapier expresses Transformer work as GEMM-plus-epilogue programs on top of
CUTLASS CuTeDSL. This is conceptually relevant to Edge-FM paths such as fused
GEMM epilogues, residual/RMSNorm fusion, SwiGLU fusion, and cross-entropy or
lm-head style reductions.

The current upstream implementation is not a direct RTX 3060 candidate:

- The README states that CODA targets NVIDIA Hopper/H100.
- `rapier/gemm/gemm_quack.py` hard-codes `arch = 90`.
- The GEMM mainloop uses Hopper-specific WGMMA, TMA, warpgroup, persistent
  scheduling, and `cutlass.utils.hopper_helpers`.
- The local evaluation GPU is NVIDIA GeForce RTX 3060, compute capability 8.6
  (`sm_86`), so Hopper-only kernels cannot be compiled or run as-is.
- Import smoke shows additional missing runtime dependencies:
  `quack`, `fla`, and `hilt` are required by the useful GEMM/high-level paths.
- The submodule has no top-level `LICENSE` file. Several files cite derived
  sources or SPDX snippets, but production vendoring needs upstream license
  clarification before any default-path integration.

## Existing RTX 3060 Baseline Surface

The current 3060 operator tables already route the relevant Qwen paths through
local tuned implementations:

- Qwen3.5 (`operator_impl_table.json`):
  - decode linear: `cublasLt`
  - decode fused gate/up activation: `edgefm_decode_swiglu_warp`
- Qwen2.5 LLM (`operator_impl_table_llm.json`):
  - decode attention: `flashinfer_attention_decode_sm80_tuned`
  - decode linear: `cublasLt`
  - prefill attention: FlashInfer prerotate and TRT context FMHA plugin entries
  - prefill linear/MLP: `cublasLt`, `cutile`,
    `cutlass_prefill_linear_source_op`, and `cutlass_prefill_mlp_source_op`

Because CODA is sm90-oriented, there is no valid paired coda-vs-baseline
candidate for RTX 3060 without first porting the mainloop from WGMMA/TMA to an
sm80/sm86-compatible MMA/cp.async design.

## Skill Recommendation

Do not create a standalone CODA skill yet.

Recommended handling:

- Keep CODA notes inside `edge-fm-cuda-kernel-optimizer` as a research reference
  for GEMM-plus-epilogue fusion ideas.
- Cross-link from `cutlass-skill` only as a CuTeDSL/Hopper reference, not as a
  general Edge-FM implementation workflow.
- Do not merge CODA into default Edge-FM operator workflows until one of these
  is true:
  - Edge-FM has Hopper/H100 as an active target, or
  - CODA adds maintained sm80/sm86 kernels, or
  - an explicit porting project creates a local sm86-compatible CODA-style
    kernel with correctness and benchmark evidence.

Suggested trigger text if it is documented in an existing skill:

- "Use CODA/Rapier as a reference when designing Hopper GEMM-plus-epilogue
  kernels with CuTeDSL, WGMMA, TMA, and Epilogue Visitor Tree style fusion. Do
  not select it for RTX 3060/sm86 runtime candidates without an sm86 port."

## Decision

Rejected for direct RTX 3060 integration.

The submodule is useful as a reference library, but it should not be connected
to Edge-FM runtime dispatch or operator tables for Qwen3.5/Qwen2.5 on RTX 3060.
The expected effort is a porting project, not a normal integration task.

## RTX 3060 Baseline And Smoke Results

All runs used `build-3060`, `EDGE_FM_CONFIG_DIR=examples/config/platform/3060`,
CUDA graph enabled, `lm_head_top1` enabled, one timed run, and the existing
3060 operator tables. JSON artifacts are under `.tmp_codex/coda_eval/`.

| Case | Shape | Table | avg ms | prefill ms | decode ms | decode step ms | Result |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen3.5 0.8B baseline | p128/d32 | `operator_impl_table.json` | 189.636 | 20.618 | 168.877 | 5.448 | pass, 32 tokens |
| Qwen2.5 0.5B baseline | p512/d32 | `operator_impl_table_llm.json` | 127.436 | 13.182 | 114.127 | 3.682 | pass, 32 tokens |
| Qwen2.5 1.5B smoke | p256/d16 | `operator_impl_table_llm.json` | 364.254 | 191.019 | 173.075 | 11.538 | pass, 16 tokens |
| Qwen2.5 3B smoke | p128/d8 | `operator_impl_table_llm.json` | 386.139 | 220.452 | 165.544 | 23.649 | pass, 8 tokens |

There is no CODA candidate column for RTX 3060 because CODA cannot run on
`sm_86` as currently vendored. A paired comparison would require an sm86 port.

## Verification Log

- `git submodule status third_party/coda-kernels`: pinned at `acc7dd2`.
- `python3 .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py`
  captured RTX 3060 / `sm_86` environment under `.tmp_codex/coda_eval/env_3060.json`.
- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`:
  passed.
- `PYTHONPATH=third_party/coda-kernels python3 ...` import smoke:
  - `rapier`: OK
  - `rapier.gemm.gemm_interface`: failed, missing `quack`
  - `models.ops`: failed, missing `fla`
  - `kernels.gens.gpt`: failed, missing `quack`
- `cmake --build build-3060 --target edge_fm_python -j$(nproc)`:
  passed.
- Baseline/smoke profile artifacts:
  - `.tmp_codex/coda_eval/qwen3_5_0p8b_p128_d32_baseline.json`
  - `.tmp_codex/coda_eval/qwen2_5_0p5b_p512_d32_baseline.json`
  - `.tmp_codex/coda_eval/qwen2_5_1p5b_p256_d16_smoke.json`
  - `.tmp_codex/coda_eval/qwen2_5_3b_p128_d8_smoke.json`

## Follow-Up

If CODA is revisited, use an H100/Hopper target first. For RTX 3060, the more
realistic next step is to borrow the fusion ideas and implement native Edge-FM
sm86 kernels behind operator table flags, then benchmark against the current
`cublasLt`, FlashInfer, CUTLASS, CuTile, and `edgefm_decode_swiglu_warp` paths.
