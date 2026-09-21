# Goal: EdgeFM J6M SmolVLA-Only HBM Performance

Status: active goal prompt. This supersedes the model-coverage scope in
`doc/edgefm_j6m_goal_prompt_v2.md` for the current execution round. Historical
plans for other models remain documented, but they are not current completion
criteria. The 2026-09-14 user correction prioritizes HBM timing and limits
quantization-fidelity work; it supersedes the former strict-fidelity
completion requirement.

Worktree:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x`

Paper:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/ICRA_2027_EdgeFM.pdf`

## Objective

Measure the existing SmolVLA HBM artifacts on real `j6m-2`, prioritizing
stage latency, throughput, CPU/BPU profile, and reproducible timing evidence.
Deliver the stage measurements first, then extend to resident full-chain
timing. An unfinished resident runner does not block stage delivery. Reuse the existing
same-platform operator ablation. Limit fidelity work to a small recorded
sanity check; do not iterate quantization to satisfy the former strict gate.
Do not run or report Qwen3.5, RDT, pi0, pi0.5, or CogACT in this round.

Other measured host or board work may be reused as reference evidence, but
must never be presented as a J6M result for another model. H20/H100 traces are
references only.

## Fixed Scope

- Hardware: `j6m-2` only.
- `j6m-1`: deferred until its runtime is complete.
- Orin: deferred because hardware is unavailable.
- Models: SmolVLA only.
- Qwen3.5 0.8B/2B, RDT-1B, pi0, pi0.5, and CogACT: `deferred`, explicitly not
  run and not claimed.
- Precision: FP32 may be used only as a numerical reference. FP32 is never a
  deployment candidate or a latency result.
- Reuse compiled FP16-body / INT16-linear artifacts. Do not start calibration
  sweeps, per-layer precision searches, or recompilation driven solely by
  numerical-error thresholds.
- Record fidelity separately from performance. Failure of the former strict
  numerical gate does not block timing of an otherwise executable HBM. Still
  check input/output contracts, runtime success, and finite outputs.

## Existing Accepted Baseline

Reuse stage-060:

`artifacts/j6m-feasibility-20260911/smolvla-stage-060-horizon-nash-m-fp16-linear-int16`

The accepted stage-060 result is one stateful SmolVLA solver step:

- FP16 body with INT16 `Conv` and `MatMul`;
- `skip_fuse_to_hzswish`;
- mean latency `90.088 ms` on `j6m-2`;
- five-process CDF from `128` warmup plus `1024` measured calls per process;
- board action cosine `0.9999994983908165` and max-abs `0.0036740303` for the
  tested complete-step output.

This is not full SmolVLA E2E. The 90 ms number is a single solver-step
measurement and must not be multiplied or presented as the full action-chunk
latency without measurement.

## Current Progress Snapshot

As of 2026-09-11, a file-orchestrated SmolVLA chain pilot exists:

`artifacts/j6m-feasibility-20260911/smolvla-stage-062-full-chain-20260911/board-chain/sample-000000/full-chain-v1`

The `prefix -> 10 step -> finish` execution sequence ran on `j6m-2` with real
captured inputs. The final action cosine is `0.999903171999195`, but the
default strict gate does not pass:

- maximum absolute error: `0.050837501883506775`;
- mean absolute error: `0.01609863852150738`;
- Prefix cache minimum cosine against the H20 trace:
  `0.994239330291748`;
- Prefix cache maximum absolute error: `1.310546875`.

This remains a `pilot` with the recorded fidelity failure. Under the
2026-09-14 scope, measure Prefix performance without making further
quantization tuning a prerequisite. A generic ONNX lowering pass replaces `Tanh` with
`Sigmoid`, decomposes last-axis `LayerNormalization`, and rewrites statically
identifiable `ScatterND` blocks to `Slice + Concat`. The V2 Prefix HBM is
already measured on `j6m-2` at `1405.032 ms`; its remaining external-CPU
`ScatterND` work is the dominant cost. A V5 lowering removes all 63
`ScatterND` nodes and has passed host parity and H20 `batch=1` HMCT
quantization. The V5 HBM compile has passed: `454521544` bytes, SHA256
`b05a1c4d62d29b37e29f7a347f7003440b73cbdbe6d76b44d67c1491adf30993`; board
offline disassembly reduces Prefix from `232` nodes (`157` CPU, `75` BPU) to
`64` (`44` CPU, `20` BPU) and removes both `ScatterND` and `Pow`.

The network blocker was resolved on 2026-09-14. V5 board `model_info`,
one-sample inference, and five-frame profiled `perf` have completed. The
same-condition smoke comparison is V5 `555.395 ms` versus V2 `1406.642 ms`;
the actual profile confirms removal of the external-CPU `ScatterND` work.
These are smoke results, not the final five-process CDF. The limited V5
Prefix check has finite outputs and an exact mask, with minimum cache cosine
`0.9940398655074391` and maximum absolute error `1.4026820659637451`.
Record this error without further quantization tuning. No replacement-chain
fidelity run is required for the current stage-performance deliverable.

Current sampling and queue state is tracked in
`doc/edgefm_j6m_goal_status.md` and
`artifacts/j6m-performance-20260914/FINALIZE.md`. Verify live process identity
and remote campaign records before resuming; historical PIDs or snapshots
do not authorize a duplicate campaign. Reuse completed, protocol-valid
individual runs and preserve evidence for excluded contended runs.

## Subsequent E2E Extension

Implement and verify the real SmolVLA path in this order:

1. `prefix`: two images, two masks, state, tokens, and token mask to padding
   mask and 32 flattened KV-cache tensors (five dimensions x 64 values each).
2. `initialize`: noise `[1, 50, 32]` to step index `[1]`.
3. `step`: ten dependent stateful solver updates using the stage-060 contract.
4. `finish`: final sample `[1, 50, 32]` to action chunk `[1, 50, 6]` and
   done flag.

The initial verifier may orchestrate stages through files on `j6m-2`. A
resident multi-model runner is required before the reported E2E latency is
called production-formal.

The existing chain manifest has three HBM models: prefix, step, and finish.
Initialization uses the captured noise input and a constant initial step
index; a fourth HBM is not required. Preserve these initialization semantics
in the resident runner and document the measured boundaries.

## General Design Constraints

- Keep stage export, calibration, lowering, compilation, and board execution
  generic through adapters and descriptors.
- Model-specific behavior belongs in model adapters, contracts, and
  quantization recipes, not in one-off graph rewrites in the compiler.
- Operator recipe overrides such as FP16 body plus INT16 linear operators are
  generic compiler controls. Reuse the existing recipe in this round; this
  design capability does not authorize a new quantization search.
- Do not repair or depend on the old broken `.pt2e` archives. Export from the
  validated TorchScript traces or another reproducible source.
- Use TorchScript/ONNX parity checks with real captured inputs before
  compiling each new stage.
- Use calibration and held-out inputs from separate samples.
- Record every compiler fallback and the first unsupported operator, dtype,
  shape, state path, or memory boundary.

## Acceptance Gates

The current performance deliverable is complete when the following evidence
exists. These criteria do not certify lossless quantization or task quality.

1. Existing runnable Prefix, Step, and Finish artifacts have real stage
   timing evidence on `j6m-2`; any ineligible artifact has an explicit boundary
   report. Compare V5 and V2 Prefix using the same inputs and configuration.
2. For the selected existing HBM, one captured sample has an output
   sanity check and comparison with the existing reference.
   Record cosine, max-abs, and mean-abs without a quantization-tuning loop.
3. Stage measurements execute actual HBMs with fixed captured inputs and
   documented contracts. Resident full-chain timing remains a subsequent
   extension, not a prerequisite for the stage performance deliverable.
4. Fidelity status remains explicit. The historical thresholds
   `max_abs <= 0.05` and `mean_abs <= 0.01` are reported if evaluated, but
   passing them is not a prerequisite for the performance deliverable.
5. Stage `model_info`, `infer`, `perf`, and profile evidence exists. Record
   model load separately and document the data-copy and synchronization
   boundary. If the tool does not expose separate timings, report that gap
   explicitly rather than inventing zero cost or substituting command wall time.
6. Five independent processes with 128 warmup and 1024 measured calls produce
   a CDF for each eligible stage, running processes sequentially. Full-chain
   CDF belongs to the subsequent resident-runner extension.
7. Peak resident memory, workspace memory, BPU/VPU placement, CPU fallback, and
   board metadata are recorded.
8. Reuse and cite the existing same-board operator ablation, recording its
   actual coverage. Do not expand quantization experiments to fill gaps.
9. Limit accuracy claims to the exact verified sample count. Eight-sample
   board fidelity expansion is deferred for the current performance focus.

## Required Evidence

Store the following under
`artifacts/j6m-feasibility-20260911/smolvla-stage-062-full-chain-20260911/`
or a later numbered stage directory:

- source HEAD and dirty-patch hash;
- checkpoint, captured-input, calibration, and held-out hashes;
- TorchScript/ONNX exports and parity reports;
- HBM hashes and sizes for prefix, step, and finish, plus initialization inputs
  and constant-state metadata;
- compiler placement/fallback reports;
- exact board commands, PIDs, logs, exit codes, and board metadata;
- raw latency samples, summary JSON, CDF CSV, PNG, and plot source;
- limited output-check report, with actual errors, stage scope, and sample
  count, and the final evidence index.

## Final Reporting Rules

- Report SmolVLA results only.
- Label timing scope as `stage` or `resident full-chain`; keep file-chain runs
  labeled `pilot` until resident timing exists. Report performance completion
  and fidelity status separately. Do not use an unqualified `E2E passed` or
  `lossless` claim when fidelity has not passed.
- Keep `j6m-1`, Orin, Qwen, RDT, pi0, pi0.5, and CogACT as `deferred`.
- Never substitute host results for board results.
- Never call stage-060 a full SmolVLA deployment.
