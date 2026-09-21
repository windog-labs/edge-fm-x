# EdgeFM J6M Goal Status

Last updated: 2026-09-14 20:20 CST (formal-r4 and Finish campaigns audited)

## Latest Pause and Resume Request

At the user's request, V2 process 3 was stopped with SIGTERM at 16:42:18.
The complete argv and start ticks of PID 1288936 were verified before the
signal. It exited with code -15 after 427 raw calls; the entire partial run
is excluded. The controller and Finish queue also exited, and at 16:42:46
the board had no BPU clients and ratio 0. `user-pause-r3-20260914.json`
records the interruption. This is not a model failure.

The user subsequently authorized resumption. At 16:53:46, another evaluation
was still running: Stage1 replay PID 1307863 and its controller PID 1306518,
with BPU ratio 52. No SmolVLA campaign was restarted then and no foreign
process was signaled. The user subsequently requested an explicit wait and
later confirmed that the board was available. At 17:22:49, the board had
ratio 0 and no BPU clients or existing benchmark controllers.

`formal-r4` completed after reusing seven complete runs from terminal
`formal-r3` and rerunning interrupted V2 process 3 from the beginning.
`user-resume-r4-20260914.json` records the idle snapshot, source hash,
commands and verified process identities. Finish queue `finish-queue-r4.json`
also completed both its smoke and formal phases. Earlier resume details below
are historical, not current aliveness.

## User Resume

The user explicitly requested resumption after their own job finished,
superseding the remaining wait in the earlier ten-minute pause.
`user-pause-20260914.json` is historical; `user-resume-20260914.json`
records the new authorization, source hashes and launched process identities.
The board was idle, with no registered BPU clients, before the new campaign.

At the pause check, controller PID 1171137, queue PID 1172179 and V2 process 2
PID 1246364 were already absent; no `hrt_model_exec` process remained.
No stop signals were sent. The controller's exit status is unknown, its
`campaign.json` is stale at `running`, and the queue recorded failure because
the dependency ended without completed evidence. Preserve this evidence;
do not infer success or blame a particular cause for the termination.
V5's five completed processes and V2 process 1 are reused by `formal-r3`.
The incomplete V2 process 2 is excluded; its replacement has completed. The original
`formal-r2` files remain untouched. `formal-r2-terminal-snapshot` records
the observed terminal condition while retaining `campaign-unreconciled.json`;
its failed status is not an inferred model failure or a fabricated exit code.

Source HEAD: `592fc320d57398571dcb4ab686d5fb55fe9bac86`

## Overall

Status: the complete existing-HBM stage timing/CDF campaign is audited on
`j6m-2`. Prefix V2/V5, Step-060 and Finish output crop each have five
independent processes and 5120 measured samples. Full-chain resident timing
is a later extension and remains outside this stage delivery.

The Horizon toolchain and `j6m-2` runtime were usable in the recorded runs. SmolVLA stage-060
passes real J6M execution, five-process latency measurement, complete board
fidelity, and a same-platform operator ablation. A file-orchestrated
`prefix -> 10 step -> finish` pilot has also executed on `j6m-2`, but it does
not pass the strict full-action fidelity gate. RDT, pi0, pi0.5, CogACT, and
Qwen were not executed on J6M in this campaign and are not completion criteria in the current round.

User correction on 2026-09-14: prioritize timing of the existing SmolVLA
HBMs. Keep fidelity work to one captured-sample sanity/comparison run for the
selected artifact, record actual errors, and do not run quantization searches
or recompile solely to meet error thresholds. The historical strict-fidelity
failure does not block performance measurement; it remains a limitation on
accuracy claims. FP32 deployment remains excluded.

The evidence root is `artifacts/j6m-performance-20260914/`. Fresh five-frame
profiles give V5 Prefix `555.395 ms` versus V2 `1406.642 ms`; the final
five-process formal means are `553.930 ms` versus `1402.998 ms` (2.532809x).
External-CPU `InplaceScatterND` is absent in V5; V2 has 63 such nodes totaling
`812.164 ms` in the profiled smoke run.

The original formal campaign (`formal/`, PID 1137694) stopped after V5 run 2
because another perception replay registered on the BPU. Run 2's after
snapshot identifies foreign PID 1160177, so that entire run is retained but
excluded from the clean-board dataset. The original Finish queue also stopped.
No foreign processes were stopped. The board was verified idle again at
14:39:10 CST.

The completed campaign is `formal-r4/`, initially controlled by PID `1341072`.
It contains five independent processes per stage, each with 128 warmup and
1024 measured calls. The whole campaign audit passed against
`formal-r4/campaign.json`; no duplicate campaign should be started.

V5 now has all five processes, each with 1152 raw calls and 1024 measured
samples. The 5120-sample mean is 553.929668 ms, P50 553.805 ms, P95 555.397 ms,
P99 557.017 ms, maximum 570.170 ms, and inverse-mean throughput 1.805283 stage
calls/s. Peak process-lifetime RSS is 33.75 MiB, separate from the HBM plan.
The maximum sample is retained. Processes 2-5 each have 640 resource checks
without observed foreign clients; reused process 1 has before/after snapshots
only, and this limitation is retained in the audit report.

`final-prefix-v5-summary/` contains the strict five-process stage audit,
measured CSV, CDF CSV and immutable campaign/source snapshots.
`final-prefix-v5-cdf/` contains the visually inspected PNG/PDF.
`evidence-index-prefix-v5.md` retains the V5-specific evidence, and
`evidence-index-final.md` indexes the complete multi-stage delivery.
The updated auditor verifies smoke raw logs, same-board identity, exact
measurement commands and resource-monitor consistency. Its four related
test files pass 22 tests and six subtests; no active producer was changed.

The final V2 formal mean is `1402.997920 ms` with P50 `1402.430 ms`, P95
`1408.200 ms`, P99 `1412.990 ms`, maximum `1431.410 ms`, and peak RSS
`35.8125 MiB`. The final Step-060 mean is `89.920599 ms`, P99 `95.632 ms`,
and peak RSS `34.0 MiB`. Finish output crop mean is `0.261185 ms`, P99
`0.373 ms`, and peak RSS `27.0625 MiB`. All four rows use 5120 measured
samples and are covered by the whole-campaign audits.

The Finish queue completed smoke and formal sampling only after the stage
campaign succeeded. Its dependency and output evidence are retained in
`finish-queue-r4.json`, `final-finish-summary/` and `finish-smoke-output-check.json`.

The current governing goal prompt:

`doc/edgefm_j6m_goal_prompt_smolvla_only.md`

`v2` remains the historical model-coverage feasibility plan. The SmolVLA-only
prompt supersedes its P1/P2 scope for this round: Qwen, RDT, pi0, pi0.5, and
CogACT are deferred and are not completion criteria.

The current feasibility report:

`doc/reports/j6m_feasibility_assessment_20260911.md`

## Closed

| Item | Status | Evidence |
|---|---|---|
| Reuse existing Conda Horizon packages | passed | `vlaforge-openvla`; HBDK 4.5.5, HMCT 2.5.6 |
| F32 operator probe compilation | passed for non-F64 probes | `artifacts/j6m-feasibility-20260911/run-006/report.json` |
| F64 state/output probe | unsupported | same run, `f64_state` convert crash |
| INT8 calibration and HBM compilation | passed for non-F64 probes | `artifacts/j6m-feasibility-20260911/run-007-quant/report.json` |
| J6M load/infer/perf smoke | passed on `j6m-2` | `board-j6m-2/board_perf_smoke_quant.log` |
| Operator support summary | passed | `summary-001/operator-support-matrix.{json,md}` |
| Goal prompt v2 | published | `doc/edgefm_j6m_goal_prompt_v2.md` |
| SmolVLA stage-060 compile | passed | FP16 body, INT16 Conv/MatMul, 108642048-byte HBM |
| SmolVLA stage-060 board run | passed on `j6m-2` | 90.088 ms mean, 11.062 steps/s, profile evidence |
| SmolVLA five-process CDF | passed | 5 x 128 warmup + 1024 measured; 5120 samples |
| SmolVLA stage-060 step-output fidelity | passed for one board sample | cosine `0.9999994983908165`, max-abs `0.0036740303`; not full-chain action fidelity |
| SmolVLA operator ablation | passed | `summary-001/operator-support-matrix.md` |
| SmolVLA INT16 Softmax/Where boundary | failed as expected | cosine `0.3523425661`; not compiled |
| SmolVLA full-chain-v1 file pilot | pilot executed; strict fidelity failed | `.../full-chain-v1/chain-report.json`; cosine `0.9999031720`, max-abs `0.0508375019`, mean-abs `0.0160986385` |
| Generic Tanh/LayerNorm lowering | passed host parity | `.../prefix-horizon-lowering-v2/ort-parity.json`; minimum cache cosine `0.9999999999976662` |
| VPU-friendly `Sqrt/Div` LayerNorm variant | passed host parity | `.../prefix-horizon-lowering-v3/ort-parity.json`; minimum cache cosine `0.9999999999975593`, max-abs `2.9027462e-5` |
| Lowered-v2 Prefix board profile | passed on `j6m-2` | `1405.032 ms` prefix latency; BPU `186.637 ms`, external CPU `1215.067 ms`; `ScatterND` fallback `811.092 ms` across 63 nodes |
| Generic static-Scatter lowering | passed unit, parity, and remote quantization | V5 has 0 remaining `ScatterND`; H20 HMCT `batch=1` output SHA256 `262d842a591b29d359ff26cf069e54252296cb6e0bd6184650b1b29d0285c6de` |
| Lowered-v5 Prefix compile | passed | `454521544`-byte HBM, SHA256 `b05a1c4d62d29b37e29f7a347f7003440b73cbdbe6d76b44d67c1491adf30993`; 5091.3 s compile |
| Lowered-v5 HBM placement | disassembly and 2026-09-14 board profile agree | `64` nodes (`44` CPU, `20` BPU) versus V2 `232` (`157` CPU, `75` BPU); `ScatterND` and `Pow` removed |
| V5/V2 Prefix board smoke | measured on 2026-09-14 | `artifacts/j6m-performance-20260914/smoke-comparison.json`; 555.395 / 1406.642 ms |
| V5 limited output check | one sample checked; no lossless claim | All 33 outputs finite, mask exact, minimum cache cosine 0.9940398655; no quantization retry |
| Step-060 fresh board smoke | measured on 2026-09-14 | 91.050 ms; prior formal CDF retained |
| V5 formal timing and CDF | measured; five processes audited | `final-prefix-v5-summary/`; 5120 samples, mean 553.929668 ms, P99 557.017 ms; PNG/PDF inspected |
| Step smoke output evidence reuse | finite; single-step comparison only | `step-output-check.json`; both output hashes exactly match the prior stage-060 board dumps |
| Finish output-crop check and timing | passed | `finish-smoke-output-check.json` and `final-finish-summary/`; exact one-sample crop plus 5120 measured calls |
| V5 environment snapshot summary | ten before/after snapshots summarized | `prefix-v5-environment.json`; BPU 41.225-42.695 C; temperature scale checked against vendor status tool; not continuous telemetry |

## Current Delivery and Subsequent Work

| Item | Status | Next gate |
|---|---|---|
| V2/Step formal stage CDF | passed | `final-stage-summary/report.json`; each stage has 5120 samples and a complete CDF |
| Finish output-crop timing | passed | `final-finish-summary/report.json`; 5120 samples, separately labeled output handling |
| Stage performance delivery | passed | final stage/Finish summaries, raw logs, memory and CDF evidence under `artifacts/j6m-performance-20260914/` |
| SmolVLA full J6M E2E timing | subsequent extension; file-chain pilot only | Does not block stage delivery; preserve historical fidelity limitations |
| SmolVLA resident E2E runner | not implemented | Keep prefix, step, and finish HBMs resident; initialize from noise and constant step index; measure latency/CDF without file or SSH overhead |
| Qwen3.5 0.8B/2B J6M | deferred | User restricted current round to SmolVLA |
| RDT J6M | deferred | User restricted current round to SmolVLA |
| pi0/pi0.5 J6M | deferred | User restricted current round to SmolVLA |
| CogACT J6M | deferred | User restricted current round to SmolVLA |
| SmolVLA multi-sample board fidelity | deferred for performance focus | Preserve existing evidence and report the exact verified sample count |

## Board State

| Board | Serial | Status |
|---|---|---|
| j6m-1 | `0e1941183392371c` | runtime incomplete: `libbpu.so.2` missing |
| j6m-2 | `0e190a0d3392371c` | reachable 2026-09-14; UCP 3.14.7, HBRT 4.9.7, BPU lib 2.2.6; idle BPU before campaign; existing onboard services preserved |

## Next Action

The stage campaign and Finish queue are complete. The final audit commands,
statistics and CDFs are retained under `artifacts/j6m-performance-20260914/`.
Finish is a one-node Slice artifact with F32 input/output representation, not
a floating neural-network forward pass; its timing is reported separately as
output handling. Use `formal-r4`, not the historical `formal`, `formal-r2` or
`formal-r3` campaigns, when tracing the accepted raw evidence.
The producer now rejects
registered foreign BPU clients even at zero utilization, monitors clients
during each run, and stops only its own child when contention appears. The
consumer rejects contended snapshots. All excluded raw data remains under
the original campaign and is indexed by `interference-audit.json`.
No calibration or HBM recompilation is planned.

`input-provenance.json` verifies all seven Prefix and 35 Step input hashes
against the original data handoff/materialization. The fixed sample is
sample 0 from `lerobot/svla_so100_pickplace`, in the original calibration
subset; it is not a held-out accuracy evaluation. `FINALIZE.md` under the
new evidence root records exact collection and finalization commands.

The ready-to-run V5 chain manifest is:

`.../board-chain/sample-000000/chain-manifest-lowered-v5.json`

The V5 Prefix HBM is:

`.../horizon-nash-m/prefix-lowered-v5-b1/smolvla-fresh-prefix-horizon-lowered-v5-fp16-linear-int16.hbm`

The first local V5 HMCT attempt was OOM-killed at about 29.5 GiB RSS on the
31 GiB workstation. The authoritative V5 quantization completed on
`zzm-h20-x8-2` with `--calibration-batch-size 1`. Reuse its compiled HBM;
additional quantization and graph-variant searches are deferred in this
performance-focused round. Do not present the existing `full-chain-v1` pilot
or stage-060 as resident full-chain timing.

At `2026-09-11 22:10 CST`, both `10.1.200.209` and `10.1.200.208` were
unreachable from the workstation and from both H20 hosts. This is an external
board connectivity blocker, not a model result.
The connectivity blocker above is historical and was resolved by the
2026-09-14 live board check.
