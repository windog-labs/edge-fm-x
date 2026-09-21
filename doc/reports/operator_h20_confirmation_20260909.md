# Registered H20 Operator Confirmation

The continuation adds strict ownership evidence to repeated operator timing.
This is separate from prior exploratory measurements, full-model speedups and
the concurrently running pi0.5 formal campaign.

Latest closure: all six families, remote/local audits, unified CSV and inspected
PNG/PDF are complete. See `h20_operator_table_20260909.md`. The initial failed
attempts described below remain excluded and the accepted controllers have exited.

## Monitoring Fix

The older repeat driver treated a missing `/proc/<pid>` as an owned process.
NVML PIDs can be outside the container namespace, so that rule could accept an
unidentified owner. A regression reproduced this behavior before the fix.

The v2 driver now uses the existing GPU register/reset/register handshake,
records container and NVML PIDs separately, verifies the physical UUID and
sets `CUDA_VISIBLE_DEVICES` itself. The worker registers before importing
backend modules or Torch. A monitor exception terminates the owned worker and
stops subsequent launches. Missing `/proc` no longer grants ownership.

The same public driver accepts a fixed candidate recipe, qualifies it once and
then alternates ATen/candidate order across five independent process pairs.
Already compiled candidates can be reused with the existing full provenance
and numerical checks. Source files and target-reference identities are checked
throughout the campaign. Qualification timings are excluded from the repeated
summary. No model-name dispatch was added.

`audit_operator_repeats.py` re-reads CPU-mapped complete output tensors, including
signed-zero bits, checks source/target reference identities, rehashes reports,
checks UUID/PID/handshake observations, rejects overlapping workers and recomputes
the summary from raw graph-batch times. Legacy v1 records do not receive the new
ownership guarantee retroactively.

Validation: 46 focused monitor/worker tests passed. After adding the auditor,
the complete CPU suite passed **2180 tests**, with 73 explicit skips and zero
failures. Hardware and unavailable-dependency skips are not new model results.
Full-suite XML SHA256:
`f13781b13cc40cbbae8508fa6155920a7e9df179c9dc4aec2cdfad7c7e4eb878`.

## Fixed Candidate Plan

| Family | Actual Captured Node | Candidate Recipe |
|---|---|---|
| VLA solver/update | `solver_update` | Inductor ATen |
| RoPE frequencies | `wrap_with_set_grad_enabled` | Inductor ATen, existing package reuse |
| Attention | `scaled_dot_product_attention` | Inductor ATen |
| GEMM/Linear | `linear` | Inductor autotune |
| LayerNorm | `layer_norm` | Inductor ATen preserving |
| Embedding | `embedding` | Inductor ATen |

The plan uses existing real exported examples, not randomly generated shapes.
LayerNorm represents the normalization family; this is not separate RMSNorm
coverage. Transported workloads explicitly regenerate the target eager reference.
Each candidate must preserve its complete target reference before timing.
These are graph-batched device measurements, not individual-call latency CDFs.

## Preserved Attempts

`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/operator-h20-confirmation-20260909/`
started on x8-1 GPU4 (`GPU-d1d90db4-423d-4393-a8c1-8b8ced20338e`), controller
3039079. The solver candidate qualified, and four repeated workers completed.
Worker 3042735 / NVML 1975247 then observed foreign NVML PID 1973128. The monitor
terminated only its own worker and the group stopped. This is incomplete, not a
completed five-pair result; remaining families were not launched.

The first x8-2 GPU0 recovery under
`runs/operator-h20-solver-recovery-20260909/` also stopped on foreign ownership
during qualification. It produced no completed repeated measurements.

The next recovery, controller **4169563**, uses x8-2 GPU6
(`GPU-2b9df5d3-1016-2167-2320-89139d3db269`) under
`runs/operator-h20-solver-recovery-20260909-002/`. It completed all ten repeated
workers and passed both remote and local independent audits. It reused the qualified package
and the unchanged frozen source; old incomplete repetitions are not mixed into
the new pairs. Inspect `repeats/campaign.json`, `repeats/monitor-*/monitor.json`
and the live process before taking any next step. Independent audit is required
before publishing a completed comparison. No kernel is selected for deployment.

## Solver Result

ATen and Inductor each completed five independent processes with registered GPU
ownership and complete tensor equality. Qualification timing is excluded.

| Lane | Mean of Five Process Medians, us |
|---|---:|
| ATen | 4.289051 |
| Inductor ATen | 2.203624 |

The observed reduction is 48.6221%, or 2.085427 us for this captured solver
workload. This is an operator device-time result; full-model integration and
stable E2E benefit are not established.

- Candidate package SHA256: `00b05a4b86d579b917c5f1c9fdfa23c7a1eef95b9fb7178684be3a60dddcca6a`.
- Remote full tensor/source/owner audit SHA256:
  `d7c215f1506dbda78fa6c9f66934a78904bc2661e440527140ecb5df0ce32b5a`.
- Local complete tensor/timing/owner recheck SHA256:
  `7dae2d7e059b7dfbb306f646e8f380f0cfc66d5d6e7c8369c3df0a0e2354f05f`.
- Local evidence: current-worktree `artifacts/recovery-audit-20260909/solver-h20-confirmed/`.

The remaining five families started separately on GPU6 under
`runs/operator-h20-remaining-20260909/`, controller **4175354**. It uses the same
frozen worker implementation and original fixed recipe plan, excluding the already
completed solver. Follow its live PID, `controller.json`, `launch.log` and each
family's monitor records. All five groups subsequently finished and passed
independent audits; they are now included with the solver in the six-family table.

All paths above share the base
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/`. The pi0.5 campaign uses its own
unchanged source snapshot and GPU1. Separate GPUs do not imply exclusive host
CPU, memory or NAS resources; retain this limitation in performance claims.
