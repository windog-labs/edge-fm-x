# Qwen3.5 EdgeFM-Only Optimization Plan

- Created: `2026-05-21T17:13:55+08:00`
- Scope: Qwen3.5 text-only greedy generation for `examples/qwen3.5-0.8b/qwen3.5-0.8b` and `examples/qwen3.5-2b/qwen3.5-2b`
- Tooling: Edge-FM CUDA Optimizer workflow, NSYS for attribution, NCU for roofline/counter evidence
- Constraint: no large runtime rewrite; optimize model/operator kernels and narrow model-local wiring only
- Stage status: local ceiling/blocker phase completed at `2026-05-22T14:46:14+08:00`. Iter97 is accepted; do not start another automatic long-loop unless the user explicitly resumes optimization.

## Goal

For each material Qwen3.5 hotspot operator, reach at least `95%` of a documented, operator-specific ceiling, or record a concrete, profile-backed blocker explaining why the remaining gap requires a larger algorithm/runtime change.

`95%` does **not** mean raw device peak for every kernel. Raw SM/DRAM Speed-of-Light is only a valid target when the kernel has enough parallelism, enough work per launch, and a roofline model that matches the algorithm. For Qwen3.5, several important kernels are recurrence-bound, tiny-grid-bound, or graph-replay launch-chain-bound; forcing them to hit 95% of hardware peak would be a misleading target.

Ceiling definition:

- GEMM/linear-like kernels: best measured library tactic for the same shape/dtype on the same GPU (`cuBLAS`, `cuBLASLt`, FlashInfer, or a validated CUTLASS/CUDA repro). Hardware tensor-core peak is a diagnostic reference, not the pass/fail target.
- Memory/elementwise kernels: measured bandwidth roofline only when the launch has enough bytes and occupancy to saturate memory. For tiny kernels, use a shape-identical standalone/replay ceiling and require end-to-end improvement.
- Recurrent/scan-like kernels such as GatedDeltaNet: recurrence-aware, token-safe ceiling based on unavoidable token-order dependency plus available parallelism across heads/value/state tiles. NCU Speed-of-Light alone is not sufficient because this class has sequential dependence; the plan must state the effective ceiling formula before claiming `95%`.
- CUDA graph replay chains: optimize by reducing graph replay time and accepted matrix latency. Individual sub-kernels with microsecond runtime may have low raw SOL by construction; do not treat their raw SOL as a failure if standalone alternatives are flat or regress full generate.

Target status labels:

- `at_ceiling`: measured implementation is within `95%` of the documented operator-specific ceiling.
- `plateaued`: repeated token-safe candidates fail to improve end-to-end latency; further gains need a new algorithm or standalone long-loop search.
- `blocked`: the next plausible improvement requires a runtime/scheduler redesign, a TRT/plugin port, or a correctness contract change.
- `open`: a material hotspot still has a local, token-safe optimization candidate not yet tested.

Current stop condition:

- P0 GatedDeltaNet prefill: Iter97 is the best accepted local current-layout implementation. Remaining raw SOL gap is recurrence/barrier/low-wave limited and requires a new standalone algorithm/layout search.
- LMHead greedy top1: accepted default path is memory-roofline limited by prior NCU (`DRAM 97.46%`, occupancy `94.50%`).
- Decode `from_ab`, fixed RMSNorm, attention, and add: local candidates are either accepted, flat, token-risky, or cross-model rejected.
- Further meaningful gains are out of this phase and belong to a larger Humanize + KernelPilot/standalone algorithm search.

## Source-Of-Truth Policy

- `CURRENT.md`: latest status only, no stale hypotheses.
- `state.json`: machine-readable current facts; update after each accepted or rejected iteration.
- `iteration-ledger.md`: append-only factual ledger: change, correctness, benchmark, accept/reject.
- `OPTIMIZATION_PLAN.md`: this plan. Update only when the loop policy or target matrix changes.
- Old failed hypotheses should be either deleted from `CURRENT.md` or moved into the ledger as rejected attempts.

## Baseline Matrix

Maintain EdgeFM graph-on/off timing for:

| Model | Prefill | Decode |
|---|---:|---:|
| 0.8B | 128 | 128 |
| 0.8B | 512 | 128 |
| 0.8B | 1024 | 128 |
| 2B | 128 | 128 |
| 2B | 512 | 128 |
| 2B | 1024 | 128 |

TRT comparison is currently blocked by missing TensorRT-Edge-LLM Qwen3.5 exporter/runtime/plugin support. Keep optimizing EdgeFM-only and mark TRT as blocked until that separate port exists.

## Correctness Gate

Every accepted optimization must pass:

1. Target operator/layer tests.
2. Qwen3.5 fresh dump token alignment for 0.8B and 2B.
3. Graph/non-graph sequence consistency.
4. If code touches shared runtime/operator paths, run the relevant Qwen2/Qwen2.5-VL regression.

Minimum commands:

```bash
cmake --build build --target edge_fm_python -j$(nproc)
EDGE_FM_BUILD_DIR=build pytest -q tests/operators/test_qwen3_5_runtime_ops.py -q
EDGE_FM_BUILD_DIR=build EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 pytest -q tests/engine/test_qwen3_5_generate.py -s
```

Shared-path regression when needed:

```bash
EDGE_FM_BUILD_DIR=build pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s
```

## Optimization Loop

For each loop:

1. Pick the top remaining hotspot from NSYS graph-on formal trace.
2. Collect or reuse graph-off mapping trace for source attribution.
3. Collect NCU for the exact hotspot kernel and extract:
   - duration
   - SM throughput
   - memory throughput
   - achieved/theoretical occupancy
   - active threads per warp
   - dominant stalls
   - roofline/SOL rules
4. Define the operator-specific ceiling before coding, and state why raw hardware peak is or is not a valid target.
5. Select one method per Edge-FM CUDA Optimizer axis:
   - compute
   - memory
   - latency
6. Make a narrow kernel/operator change.
7. Rebuild.
8. Run correctness gates.
9. Benchmark affected cases first, then the full matrix when the candidate is accepted.
10. Accept only if correctness passes and the affected benchmark improves or moves the operator closer to its documented ceiling. For tiny/recurrent kernels, standalone NCU wins are not enough; the affected full-generate benchmark must not regress.
11. Update `CURRENT.md`, `state.json`, and `iteration-ledger.md`.
12. Send a cc-connect notification at each meaningful milestone.

Reject/revert policy:

- Correctness failure: fix once if the cause is local and obvious; otherwise reject.
- Performance regression: reject unless the change enables a required later optimization and is explicitly recorded.
- Larger runtime redesign required: record blocker and pause for user confirmation.

## Hotspot Order

### P0: GatedDeltaNet `gated_delta_sequence_kernel`

Current status:

- Original NSYS: `99.7%` of prefill GPU time on 0.8B p128/d32.
- Original NCU: `275.57 ms`, SM throughput `5.96%`, achieved occupancy `2.08%`, active threads/warp `1.00`.
- Iter3 NCU: `3.45 ms`, SM throughput `27.82%`, achieved occupancy `19.06%`, active threads/warp `28.42`.
- Current accepted prefill path is `gated_delta_sequence_precomputed_kernel<bf16,true,16>` with the Iter39 token-boundary barrier and Iter59 tile16/thread256 layout.
- Iter66 ceiling refresh: production tile16 NCU and barrier-safe standalone tile16 NCU both measured `1.85 ms`; this is the current token-safe recurrence ceiling for the accepted layout.
- Iter95 tile24 improved standalone event/NCU slightly but regressed full Qwen3.5 generate, especially 2B long-prefill, so standalone-only gains are not accepted.

Next work:

- Treat the accepted tile16/thread256 layout as `at_ceiling` for its current recurrence-safe design.
- Continue only with a new standalone token-stable algorithm/layout that can plausibly transfer to full generate. Do not repeat simple retile, q/k+decay precompute fusion, paired q/k precompute, explicit FMA, or reduction-order changes already rejected in the ledger.
- Continue improving occupancy and reducing sync/stall overhead only when the candidate preserves token recurrence semantics and has full-generate evidence.
- Candidate directions:
  - standalone search for a new state layout that keeps per-token math order but improves coalescing or lowers barriers
  - multi-layer or batched-linear-attention scheduling only as a documented future runtime/scheduler blocker, not a silent refactor
  - KernelPilot/Humanize long-loop repro for the recurrence kernel if local candidates remain exhausted

Acceptance:

- Keep token alignment exactly matching Transformers.
- For the current layout, `>=95%` is already satisfied by the Iter66 apples-to-apples ceiling. A new P0 production transfer must improve affected full-generate benchmarks, not just standalone NCU.

### P1: Decode Linear-Attention One-Step Chain

After P0 stabilizes, profile graph-on decode with enough generated tokens to expose replay contents.

Candidate kernels:

- depthwise conv state update
- compute g/beta
- one-step recurrent update
- gated RMSNorm
- output projection

Candidate directions:

- fuse decode conv/state update where correctness permits
- fuse `compute_g_beta` with recurrent step for decode
- specialize one-token path separately from prefill sequence path
- use CUDA graph replay after each change to confirm launch overhead is not hiding kernel gains

Acceptance:

- Raw hardware peak is not a valid target for tiny-grid one-token recurrent kernels. Use a shape-identical standalone/replay ceiling and require 0.8B/2B graph-on generate improvement.
- Current `from_ab` path is `plateaued` after Iter91/94 unless a new standalone candidate beats the column-fused 128-thread schedule and transfers to full generate.

### P2: Dense Linear / LM Head

Use operator-table known path first.

Candidate work:

- validate current `hw_profile=3060` and selected records
- retune cuBLASLt records for Qwen3.5 shapes
- benchmark `lm_head_top1` gate only if token alignment and end-to-end graph-on speedup justify it

Acceptance:

- target cuBLAS/cuBLASLt tactic should be within `95%` of best measured library tactic for the exact shape/dtype.
- For default greedy Qwen3.5, `lm_head_top1` is the accepted path. Its stage1 NCU is already memory-roofline (`DRAM 97.46%`, achieved occupancy `94.50%`), so further work requires an algorithmic top1 replacement that reduces required bytes rather than launch-shape retuning.

### P3: Full-Attention Layers

Only 6 layers use full attention, but p512/p1024 may expose them after GatedDeltaNet improves.

Candidate work:

- validate FlashInfer/operator table selection
- tune prefill attention path for `num_q=8`, `num_kv=2`, `head_dim=256`
- measure whether Qwen3.5 head_dim=256 needs a dedicated path

### P4: Small Elementwise/Norm/State Kernels

Optimize only after major kernels stop dominating.

Candidate work:

- fuse adjacent Qwen3.5-only elementwise ops
- optimize RMSNorm/gated RMSNorm only if NSYS shows material share
- keep changes model-local under `src/operators/qwen3_5`

## Notification Policy

Use:

```bash
$HOME/.local/bin/codex-notify-wechat '【Codex/edge-fm-x】<status>: <result>; <next>'
```

Send notifications for:

- baseline/profile completion
- each accepted/rejected optimization
- correctness failure/blocker
- full matrix benchmark completion
- final target achievement or explicit plateau

The local wrapper may fall back from WeChat to Feishu; this satisfies the stage-notification requirement on this machine.

## Stop / Escalation Criteria

Continue the loop until:

- all material hotspot operators reach `>=95%` of their documented operator-specific ceiling; or
- remaining gaps require a larger runtime/algorithm redesign; or
- two consecutive accepted iterations improve the full matrix geomean by `<1%` and NCU shows no clear next local kernel change.

If plateau is reached, escalate to a standalone repro / Humanize + KernelPilot loop only for the specific kernel with a stable reference and benchmark.
