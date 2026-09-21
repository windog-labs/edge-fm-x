# SmolVLA J6M Existing-HBM Performance

Updated: 2026-09-14 20:20 CST. Status: the complete existing-HBM stage
campaign is audited on `j6m-2`: Prefix V2/V5, Solver Step-060, and Finish
output crop each have five-process timing/CDF evidence. This report does not
claim full-model E2E performance.

Evidence: `artifacts/j6m-performance-20260914/`.

## V5 Formal Stage Result

Five independent sequential processes each produced 1152 raw calls, with
128 warmups excluded and 1024 measured calls retained. No individual slow
samples were removed. The same-board audit checked all smoke command logs,
all five raw process logs, frame indices, exact commands, resource evidence
and non-overlapping formal process windows.

| Stage | Samples | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Stage calls/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix V5 | 5120 | 553.930 | 553.805 | 555.397 | 557.017 | 570.170 | 0.908 | 1.805283 |

These are `infer` per-call timer measurements, not SSH wall time or full
action-chunk latency. Peak process-lifetime RSS across the five processes is
33.75 MiB. Their separately logged model-load times are 839.774, 842.100,
834.736, 832.074 and 837.162 ms. The static HBM plan is reported separately
below; neither metric alone represents total device allocation.

The ten before/after process snapshots have BPU temperatures of
41.225-42.695 C, CPU policy frequency readings of 2000000 kHz with the
`performance` governor, and BPU frequency readings of 1500000000 Hz.
`prefix-v5-environment.json` retains raw values, converted ranges and every
snapshot hash. The board-specific temperature divisor of 1000000 was
cross-checked against `/usr/hobot/bin/hrut_somstatus`; raw readings before
and after that tool call, its output and executable hash are retained in
`thermal-unit-evidence.json`. This conversion is not assumed for other
boards. These ranges describe ten snapshots, not continuous telemetry or
proof that frequency never changed between snapshots.

`final-prefix-v5-summary/` holds the measured CSV, exact CDF, audited summary
and immutable source snapshots. `final-prefix-v5-cdf/` holds the PNG/PDF;
the PNG was visually inspected including its complete tail. The longest
570.170 ms call remains in process 5's raw log and the CDF.

Process 1 is reused with before/after resource snapshots only. Processes 2-5
each have 640 one-second resource observations without observed foreign BPU
clients. This does not prove the absence of activity between poll points.
The previously contended process remains separately preserved and excluded
as a whole. Performance is measured; fidelity remains a limited sample check
and is not certified lossless. The complete campaign is indexed by
`evidence-index-final.md`.

## Final V2 and Step Results

The audited `formal-r4` campaign contains five independent sequential
processes for each stage, with 128 warmups and 1024 measured calls per
process. No individual latency samples were removed.

| Stage | Samples | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Stage calls/s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix V2 | 5120 | 1402.998 | 1402.430 | 1408.200 | 1412.990 | 1431.410 | 2.797 | 0.712759 | 35.8125 |
| Prefix V5 | 5120 | 553.930 | 553.805 | 555.397 | 557.017 | 570.170 | 0.908 | 1.805283 | 33.7500 |
| Solver Step-060 | 5120 | 89.921 | 89.834 | 91.249 | 95.632 | 96.615 | 1.164 | 11.120922 | 34.0000 |

The formal V2/V5 mean-latency ratio is `2.532809x` in favor of V5. These
throughputs are stage calls/s; Step calls are solver steps, and neither is
an action-chunk rate. The final audited statistics, raw samples and exact-rank
CDF CSVs are under `artifacts/j6m-performance-20260914/final-stage-summary/`.

The final report is based on `formal-r4`; it verifies complete raw logs,
sequential frame indices, exact commands, board identity, resource evidence,
and non-overlapping process windows. The interrupted partial runs from
earlier pauses remain preserved and excluded as whole processes.

## Same-Condition Smoke Results

Each row uses five profiled `hrt_model_exec perf` frames, one thread and
default core selection. Both Prefix versions use the same seven captured
input files; SHA256 values are in `smoke/campaign.json`. Runs are sequential.

| Stage | Mean ms | Calls/s reported by perf | BPU ms | External CPU ms | Minimum HBM plan MiB |
|---|---:|---:|---:|---:|---:|
| Prefix V2 | 1406.642 | 0.711 | 186.942 | 1216.335 | 560.451 |
| Prefix V5 | 555.395 | 1.800 | 165.712 | 388.700 | 543.660 |
| Solver Step-060 | 91.050 | 10.945 steps/s | 31.659 | 57.498 | 119.401 |

V5 is 2.533x faster than V2 in this short profiled comparison. V2 has 63
external-CPU `InplaceScatterND` nodes totaling 812.164 ms; V5 has zero. This
confirms the intended placement change on the actual board. The comparison
includes the existing V2/V5 compiler differences and is not an isolated
single-operator causal ablation. The formal CDF results for both Prefix
versions are reported above.

## Memory and Timing Boundaries

The HBM memory figures are static compiler plans, not process RSS. Perf
process lifetime peak RSS from Linux `wait4` was 35.625 MiB (V2), 34.0625 MiB
(V5), and 34.0625 MiB (Step). This includes process creation; driver-managed
model buffers are not accounted for by treating RSS as total model memory.
Sampled `/proc` VmHWM values and original KiB counts are also retained.

The same perf processes reported model-to-DDR load times of 860.903 ms (V2),
834.492 ms (V5), and 493.322 ms (Step). These load times are separate from
the reported frame latency. The tool does not expose separate input-copy,
output-copy, and synchronization timings, so no independent values or zero
costs are claimed. Command wall time includes additional overhead and is
not used as model latency.

The timer boundary was checked against the actual board executable, retained
as `hrt_model_exec.aarch64` with SHA256
`b4cd2252a31f38ca43ffaa1a6ad2ce9ad7705ca5a73a8267a4f90402564f7e5e`.
Both `infer` and `perf` start timing before `hbDNNInferV2` and stop after
`hbUCPWaitTaskDone`, including task construction, submission and waiting.
Input file loading, output metadata handling/dumping, and task release are
outside the per-inference timer. Internal runtime copies may still occur
inside the measured APIs; they are not independently timed by this tool.
Exact symbol addresses and disassembly evidence are indexed in
`timing-boundary-audit.json`.

The live process maps confirm DNN/UCP/HBRT are loaded from `/app/lib`, and
BPU from `/usr/hobot/lib/libbpu.so.2.2.6`, rather than assuming the adjacent
`hrt_model_exec/aarch64/lib` copies are in use. No runtime libraries or
search paths were changed.

## Limited Output Check

Fixed inputs trace to the existing SmolVLA-Base handoff for
`lerobot/svla_so100_pickplace`, dataset revision
`728583b5eaf9e739a7f119e2def466fa1d552402`, sample 0 / dataset index 0.
The handoff records checkpoint SHA256
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
and upstream revision `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.
All seven Prefix inputs match the handoff bytes, local sample, original
calibration archive, and board input hashes. The 35 Step inputs match the
existing stage-039 materialization, including its documented BF16-to-F32
cache representation conversion. `input-provenance.json` retains all paths,
hashes, shapes and dtypes.

Sample 0 is in the historical calibration subset, not the held-out subset.
These are checkpoint-processed tensors under the recovered SO100 statistics
profile; raw camera transport, physical action-unit calibration, and robot
task success are outside the measurement boundary. No held-out quality claim
is made from this fixed-input timing sample.

One captured V5 Prefix sample was compared with the existing H20 reference.
All 33 output tensors are finite and the padding mask is exact. Minimum KV
cache cosine is 0.9940398655074391; maximum cache absolute error is
1.4026820659637451. The reference mask is explicitly converted to bool to
match the documented BOOL8 output contract. No quantization retry is planned.
This check does not certify final-action fidelity or lossless quantization.

`prefix-v5-output-check.json` contains every output hash, dtype, and metric.

The two existing Step smoke dumps have identical SHA256 hashes to the
previously validated stage-060 board outputs. `step-output-check.json`
links those bytes and the original reference arrays, verifies finite values
and the exact discrete step index, and records the numerical errors for
this single solver update. No additional board inference is needed for
this reuse; it is not a full-action quality result.

## Finish Output Crop

Finish is measured separately as a one-node `Slice` output-crop HBM with F32
I/O representation and no floating-point arithmetic in its source graph.
Five sequential processes produced 5120 calls: mean `0.261185 ms`, P50
`0.254 ms`, P95 `0.304 ms`, P99 `0.373 ms`, maximum `1.154 ms`, standard
deviation `0.0350 ms`, and inverse-mean throughput `3828.710 calls/s`.
Peak process RSS is `27.0625 MiB`. This row describes output handling and
must not be presented as a neural-network FP32 deployment result.

The fresh smoke output check used one captured input. It is finite, has shape
`[1,50,6]`, exactly matches the pinned ONNX Slice crop, and has zero maximum
and mean absolute error. This is crop correctness only; it does not establish
full action quality. `final-finish-summary/` and `final-finish-crop-cdf/`
contain the audit, statistics, raw samples and CDF evidence.

## Formal Sampling and Finish

The completed `formal-r4` campaign covers V5 Prefix, V2 Prefix, then Step,
sequentially. Each stage has five independent processes with 128 warmup and
1024 measured calls per process. The original controller (PID 1137694) stopped after a foreign
perception replay began using the BPU during V5 run 2. That entire run is
preserved but excluded based on the foreign owner in its after snapshot,
not based on its latency values. No individual outliers are removed.

After the replay ended and the board was verified idle, `formal-r2` resumed
with controller PID 1171137, reusing clean V5 run 1 and replacing the contended
run 2. The original failure and all raw logs remain in `formal/`.
`interference-audit.json` records the foreign PID, evidence hashes, exclusion
reason and resumption. The final audit uses the completed `formal-r4` campaign.

During the later user-requested pause, the controller, its current child
and the Finish queue were found already absent. The original controller
exit status was not captured; its stale `running` JSON is not completion
evidence. The user then explicitly authorized early resumption. With the
board verified idle, `formal-r3` launched as controller PID 1261254, reusing
the five completed V5 processes and V2 process 1. Only incomplete V2 process 2
is restarted before continuing the remaining work. No stop signals were sent
and no other user's processes were terminated.

`formal-r2-terminal-snapshot/` retains the raw data and original
`campaign-unreconciled.json`, plus a terminal observation with unknown exit
code; the original `formal-r2` files are unchanged. The new campaign rejects
the incomplete process as a whole. `user-pause-20260914.json` and
`user-resume-20260914.json` document authorization and provenance. The earlier
V5 figure and V2 process-1 statistics remain valid and are not remeasured.

The later explicit pause stopped V2 process 3 with a recorded SIGTERM exit,
unlike the earlier unexplained process disappearance. The complete terminal
`formal-r3` evidence is retained. After an intervening Stage1 board evaluation
and another explicit user resume, the board was verified idle at 17:22:49.
`formal-r4` started as controller PID 1341072, reusing all five V5 runs and
both completed V2 runs. Its rejected-reuse record identifies the interrupted
V2 run 3. `user-pause-r3-20260914.json` and `user-resume-r4-20260914.json`
preserve this distinct interruption and resumption. No foreign tasks were
terminated, and no producer or HBM recipe changed.

Earlier V5 partial summaries remain archived in `interim-prefix-v5-run-1/`,
`interim-prefix-v5-first-two/` and `interim-prefix-v5-first-three/`. The
two-process figure in `interim-prefix-v5-cdf/` is explicitly labeled interim.
The completed five-process V5 result above supersedes those partial figures;
no underlying raw data was removed. `plotting-environment.json` records the
reused Python 3.10 modules and cached pure-Python `packaging` dependency.
No environment rebuild or package upgrade was needed; `FINALIZE.md` gives
the exact working plot commands.

Finish is an existing 7464-byte HBM containing one Slice, with F32 input and
output representation and no floating-point arithmetic in its source ONNX.
It crops `[1,50,32]` to `[1,50,6]`; its five-process timing is reported above
as output handling, not FP32 neural deployment. The completed dependency
queue and process identities are recorded in `finish-queue-r4.json`.

The existing 2026-09-11 Finish output was checked offline against the exact
ONNX source pinned by `compile-report-action.json`, using ONNX's reference
evaluator rather than reconstructing Slice semantics manually. Its 300
output elements are finite and exactly match the source crop (max-abs and
mean-abs zero). The remote queued input and HBM hashes were rechecked on
2026-09-14 and match the archived pilot. `finish-existing-output-check.json`,
`finish-smoke-output-check.json` and `finish-board-input-hashes-20260914.txt`
retain the historical and fresh crop evidence. This is output-crop
correctness, not full-action fidelity.

The compiled action-only Finish HBM has one output. The original export's
separate finite/done predicate is not an HBM output in this artifact; finite
input/output checks remain host-side evidence. A future resident runner
must preserve the complete source contract rather than assuming a second
HBM output or equating crop correctness with complete-model fidelity.

The existing operator probe ablation is reused from
`doc/reports/smolvla_j6m_stage060_20260911.md`. It uses synthetic fixed-shape
probes and a separate INT8 recipe, with historical FP32 comparison data.
No new FP32 deployment or quantization search was run in this campaign.
The reused HBM hashes and raw profiler latencies were checked for six
families (12 rows) in `reused-operator-audit.json`. The historical protocol
uses 200/200 frames for Attention, but 100 FP32 versus 200 INT8 frames for
the other listed families. These remain historical exploratory comparisons,
not matched five-process measurements. GEMM, LayerNorm, and Embedding are
one combined probe, not separately measured operator rows.

## Reproducibility

- Board: `j6m-2`, `root@10.1.200.209`, serial `0e190a0d3392371c`.
- Runtime: UCP 3.14.7, HBRT 4.9.7, BPU library 2.2.6; HBM compiler 4.5.5.
- Existing onboard services remain active; BPU ratio was zero before launch.
- Every command has before/after process, temperature, frequency and memory snapshots.
- Source HEAD: `592fc320d57398571dcb4ab686d5fb55fe9bac86`.
- `tracked-worktree.patch` records tracked changes only; its SHA256 is
  `13f9ccd6d1786efb2da646614d4ed1b83cc3f13510da05b6ac1e413bf50658c6`.
- The untracked generic runner is independently hashed in each campaign;
  `vlaforge/tools/benchmark_horizon_hbm.py` has six passing focused tests,
  including zero-ratio foreign clients, interruption, and clean-run reuse.
- `campaign-manifest.json` pins model hashes, input paths and sample counts.
- `smoke/` contains raw model_info, infer, perf, profile, dumps, and process evidence.
- `summarize_smoke.py` regenerates `smoke-comparison.json` and the limited output check.
- `vlaforge/tools/summarize_horizon_campaign.py` audits completed campaigns or
  explicitly selected completed stages before generating statistics and CDF CSV.
  Stage-only output retains the actual overall campaign state. The four related
  test files pass 22 tests and six subtests, including smoke-log integrity,
  board identity, incomplete-stage rejection and contradictory resource metadata.
- `run_finish_after_formal.py` records and verifies the dependency process identity
  before starting the queued output-crop campaign.
- `audit_reused_operator_ablation.py` checks the six historical operator families
  without rerunning board or quantization work.
- `audit_input_provenance.py` verifies all 42 Prefix/Step input files against
  the existing handoff/materialization and recorded board hashes.
- `finish-disassembly.json` and `step-disassembly.json` retain compiler memory evidence.
- Full-chain resident timing, final-action validation, and a full-chain CDF remain subsequent work.
