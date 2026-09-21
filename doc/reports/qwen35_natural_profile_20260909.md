# Qwen3.5 Natural-Image Profile

Both official Python baselines completed formal sampling, remote independent
audit and local re-audit of the complete retrieved evidence. Their CDF PNGs were
inspected. They remain a separate official-Python baseline from the later
accepted VLAForge C++ deployment in
`qwen35_native_formal_20260910.md`.

## Audited Formal Results

Each model completed five independent processes with 128 warmups and 1024
measured calls per process: 5120 measured calls and 5760 complete token outputs.
Every output, including warmups, exactly matches its same-backend reference.
No warmup or pilot measurement enters the table.

| Model | TTFT Mean ms | TTFT p99 ms | Input-to-Tokens Mean ms | Input-to-Tokens p99 ms | Decode tokens/s | Peak Allocated bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 152.937227 | 164.270234 | 558.793883 | 595.405434 | 36.958862 | 1805916160 |
| Qwen3.5-2B | 156.301746 | 171.681141 | 545.668130 | 592.475612 | 38.524127 | 4532012032 |

Both TTFT and total latency start with encoded image bytes in host memory and
include preprocessing and H2D. Completion means the first CPU token for TTFT,
or all 16 CPU tokens for total latency. Checkpoint/processor initialization,
image file IO, validation/log IO and text detokenization are excluded. Decode
throughput is the sum of the remaining 15 tokens per call divided by the sum of
completion-minus-TTFT intervals. Peak memory is the Torch allocated-memory
counter, not a measurement of all process/device memory. Diagnostic vision and
forward CUDA timings are retained separately, outside formal samples.

These are fixed-image/prompt latency profiles. Repeatability within the official
backend does not establish optimized-deployment parity or general task quality.
The two runs use different H20 GPU UUIDs; their small latency differences do not
establish a model-size performance ranking.

The complete local archives are under
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`:
`qwen-formal-0p8/` and `qwen-formal-2b/`. The corresponding
`qwen-formal-0p8-figures/` and `qwen-formal-2b-figures/` contain latency tables,
four raw CDF CSVs, PNG/PDF figures and source-bound reports. The renderer reran
the complete audit locally and matched all worker results to the remote audit.

| Model | Remote Audit SHA256 | Final Report SHA256 |
|---|---|---|
| 0.8B | `a5a551c9fdb44458600ac3f55651caca20c22837c26036e925bd3c1aa73a1d86` | `e236ba9fceab1a3f949f44f3f3a0bf48cb49ceb3f5af63f72653e6c82095e206` |
| 2B | `2719dbf3b53fa48d5696b990a10a70e3da5356cadd3e670d8f8f6c4797c15436` | `69f0a94da5edf7ee195aa5bac86d4bbec8262fff81792e85c99af83c2090d0d9` |

## Input and Pilot

The profile uses a lossless PNG of `images/cam_high`, episode 0 frame 0 from
`physical-intelligence/aloha_pen_uncap_diverse`, revision
`e82d8b40b8ac66c0b40273dd80a077dfc40b732e`. Sixteen frames were extracted and
pixel-compared with their hash-bound original NPZ tensors; only frame 0 is used
by this fixed benchmark profile. This is not a sixteen-image formal evaluation.
Input-pack manifest SHA256:
`d55062ab967e551a0d805e436bd0df9580b0621cadce8f9c4f7decce73ad3f9e`.

Prompt: `Describe the visible objects and the robot interaction in one short sentence.`
Both processors produce 327 input tokens, including 300 image tokens, from the
640x480 RGB image. The output profile is 16 greedy tokens, with the documented
minimum/maximum length settings. The full image, vision, prefill and decode
paths execute. Optional accelerated FLA/causal-conv1d kernels remain unavailable;
the official Torch implementation is used.

| Model | Pilot Worker PID | GPU | TTFT Mean ms | Input-to-Tokens Mean ms |
|---|---:|---|---:|---:|
| 0.8B | 4179601 | H20-2 GPU1 | 141.421 | 523.125 |
| 2B | 4179685 | H20-2 GPU2 | 149.256 | 521.720 |

Each row is only 2 warmups + 2 measured calls. Full tokens agree with the
untimed same-backend reference; both pilots passed independent ownership/raw
output/timing checks. Pilot audit SHAs are respectively
`0c209f3b00a9412fbf0f06a4b72a78db0fb67cd618a4fa217053d97cd3f071ca`
and `bd013c8fe1cb9e8666ed2e1e848c97dc35afe023b6ef0134bd7feaaf53b9b26f`.
These few measurements do not rank the models or establish stable performance.

## Formal Sampling

- 0.8B controller 4184329, supervisor 4184345, GPU UUID
  `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`.
- 2B controller 4184340, supervisor 4184346, GPU UUID
  `GPU-3596876b-e679-6289-f784-c8d98294ed23`.
- Each uses five sequential independent workers, 128 warmups and 1024 measured
  calls per worker. Raw tokens include warmups, and phase-labeled CSVs retain
  every measured call. No results from the earlier architecture-diagram smoke
  enter these campaigns.
- Each worker uses the registered CUDA ownership handshake. All ten workers
  have terminal successful reports and passed independent audits.

The NAS root is
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/qwen-natural-profile-20260909/`.
Terminal state is recorded in `formal-{0p8,2b}-controller.json`,
`formal-{0p8,2b}/campaign.json` and per-worker reports. Do not relaunch either
completed campaign. The existing `audit_multimodal_baseline.py` and
`artifacts/recovery-audit-20260909/report_multimodal_formal.py` produced the
audits, tables and CDFs above without rerunning GPU inference.

## Historical Native Feasibility

The accepted fixed-profile native route is documented in
`qwen35_native_formal_20260910.md`. The attempts below remain historical
rejections and are not the final native result.

The updated per-model failures, source identities and independent 2B complete-token
rechecks are in `qwen35_native_blockers_20260910.md`. The older attempts in this
section used 0.8B; they do not by themselves establish a 2B export failure.

All feasibility attempts below are separate from formal baseline timing:

1. `qwen-native-feasibility-20260909`: tracing failed because the generation
   API received a traced Tensor where an integer length was required.
2. `...-002`: explicit absolute profile length advanced tracing to an actual
   PyTorch JIT alias-analysis assertion for `aten::full` with a boolean scalar.
3. `...-003`: the generic boolean-fill compatibility helper produced a saved
   TorchScript candidate; complete tokens match on two real frames after reload.
   Candidate SHA256:
   `5d6c84eac32d3ac4e5585918225b66753b589aff85fb562568535fd54f21dea9`.
   The graph contains no `prim::PythonOp`, but this alone is not a deployment
   certificate or complete effect analysis.
4. `...-004`: prompt, masks, image grid and output length were explicitly fixed;
   only pixels remain a dynamic argument. The frozen candidate also matches two
   real-frame token references. Numerical JIT freezing optimizations were disabled.

The initial TS-to-ExportedProgram conversion was rejected by a data-dependent
convolution-size guard. Fixed conditioning advanced it to a cache-copy size
equation (`Eq(u98 + 327, u219)`), which was also correctly rejected. A separate
`qwen-export-feasibility-20260909-003` used static input axes instead of the
converter's automatic dynamic-axis policy, but it was still rejected at the same
cache-copy size equation. It did not produce an accepted ExportedProgram. Explicit
cache/state/profile adaptation remains necessary; the original guards are intact.

No missing effect, numerical-provider or C++ gate is bypassed. These are
restricted-profile research artifacts, not proof of arbitrary prompt/resolution
support, an accepted VLAForge bundle, or no-Python performance.

## Shared Compatibility Helper

`trace_boolean_fill_compatible` normalizes Python boolean fill scalars to integer
zero/one while preserving explicit or inferred tensor dtype. It uses a scoped
TorchFunctionMode and does not patch the shared Torch installation or default
Region compilation path. Its ledger grants no numerical/effect certificate.

The JIT assertion was reproduced in a minimal CPU example. The helper and normal
export tests passed 33 cases; the separate opt-in suite passed 10 cases including
an actual LibTorch C++ consumer with invalid Python environment variables.
Full CPU regression: 2190 passed / 74 skipped / zero failed. The C++ consumer was
an opt-in skip there and is covered by its explicit successful run.
