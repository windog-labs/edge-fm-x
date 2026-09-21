# Agent Operator Selection Ledger

This is an evidence ledger, not a claim that an optimized kernel has been
selected for deployment. All rows retain the original artifact and report
hashes; no failed candidate is converted to a pass.

## Fresh H100 Integration (2026-09-08)

An independent campaign on `zzm-h100-x8` GPU1 is stored at
`artifacts/edgefm-vla-goal/20260906-044509/solver-integration-h100x8-20260908-001`.
It uses a new same-host official SmolVLA reference for 16 real episodes, not
RTX outputs as a numeric oracle. Generic terminal partitioning records 160
actual solver calls, including real index strides, and verifies both original
and split complete IR against all official normalized/native outputs.

| Recipe | Graph-batch median us | Complete step validation |
|---|---:|---|
| ATen | 3.897455 | 160 calls / 320 tensors exact |
| Inductor-ATen | 2.255696 | 160 calls / 320 tensors exact |
| Inductor-ATen preserving | 3.766021 | 160 calls / 320 tensors exact |

There is one independent microbenchmark process per recipe and a fixed budget
of two compiled candidates. These are not per-call latency CDFs. Initial
all-case validation was rejected by the nested numerical-context guard;
fresh independent validation processes reuse the measured packages, pass all
160 cases, and preserve the initial failed reports.

The candidate being integrated is the exact measured Inductor package with SHA
`73c3d5f22c7cfdb50e7b7bc4b9d973c482afc7bcc1480466c6842a9cfe5799f9`;
microbenchmark report SHA
`4e6708128692a8c7acb3148937f1ca0c7db84c44207985dc58efa295a669e8e8`.
The intended comparison is original model, partitioned TS control, and the
same partition with that exact AOTI package. Only partitioned-control versus
candidate isolates this kernel; original versus control measures partition
overhead. No speedup or selection is recorded yet.

At 09:08 UTC, native integration was still pending. Initial direct loading
failed because Torch 2.10's package loader accessed an unimported `codecache`;
the next attempt imported Torch before GPU ownership registration and was
correctly rejected. A fresh `integration-003` imports the dependency only after
registration and reuses the unchanged original package. Both failures and
their monitored logs remain intact. Public source snapshots are not replaced
in any existing run directory.

By 09:20 UTC, all three complete C++ required-replay pilots and the independent
audit passed: 144 calls / 288 full F32 normalized/native tensors byte-exact,
each worker reports `REPLAY_FINAL,11,1,10,48,0`, with numerical-provider
enforcement and no Python libraries in process maps. The exact measured
package is in the candidate bundle and absent from the two controls. The
generated Session verifies artifact SHA/size before loading; maps show the
package's AOTI wrapper, rather than a second TS tail. Temporary loader files
are cleaned at exit, so their post-exit bytes are not separately archived.
Remote pilot audit SHA:
`8380c0a0ddbdfd8b4df6815da6b594da0de1b0dcc3a5d7a6e47c468add399694`.
Local retrieval independently rechecked 288 tensors / 86400 values, all three
lanes' references, maps, counters and telemetry, and the downloaded exact
candidate package. It contains 2971 files / 368820832 bytes; report SHA:
`a157bfd242f3156fb4ef8f620dee464ab5f5bc82d3d3205de5097255c644690c`.
The SSH pilot launcher returned a transport error after completion; a fresh
read confirmed the remote controller passed and both controller/auditor PIDs
had exited. No native pilot was duplicated.

Formal paired comparison started at 09:20 UTC with controller PID3835514,
rotating original/partitioned/candidate order across five blocks. Each lane
retains the same pilot-verified bundle, GPU and 128+1024 protocol. Formal
measurement, independent audit and selection are not yet complete. The
analysis contract explicitly records that it was declared after sampling
began, before inspecting formal latency values; confidence uses exhaustive
paired block bootstrap and a one-sided sign-flip check, not 5120 independent
per-call observations. No threshold may be relaxed after seeing the result.

The standalone AOTI loader also now imports its required codecache dependency
inside the public helper. A new regression first reproduced the missing
initialization; 66 operator-tool/selection tests then passed. CPU regression016
passed 2098 tests with 83 explicit skips. The remote campaign retains its
original frozen source plus the explicitly recorded driver initialization fix.

### Formal Outcome At 09:45 UTC

All 15 paired native workers, per-lane aggregation, remote independent audit,
local raw recheck, table and visual CDF/tail inspection have completed. Each
lane contributes 5120 measured calls and 11520 full F32 tensors. Across all
lanes, 17280 calls / 34560 tensors / 10368000 values are byte-exact, including
warmups. Required reports `REPLAY_FINAL,11,1,10,1152,0` in every process.

| Lane | Mean ms | p99 ms | Chunks/s |
|---|---:|---:|---:|
| Original TS model | 44.875762 | 59.283026 | 22.283744 |
| Partitioned TS control | 44.090836 | 55.724240 | 22.680450 |
| Same AOTI solver package | 43.748341 | 55.082885 | 22.858010 |

Candidate versus partitioned control has a 0.776794% point reduction, but the
paired 95% bootstrap interval for the latency difference is
`[-0.740279, 1.587530] ms`; one-sided sign-flip probability is 0.28125.
The confidence gate fails. The public v2 decision is correctly **rejected**,
with `correctness_passed=true` and `end_to_end_integrated=true`. This is now a
genuine same-candidate same-platform integrated result, unlike the invalid
historical H20/RTX pair below, but not a verified stable speedup. Original to
partitioned is a separate control change; its point difference is not credited
to the candidate kernel. No outliers or process blocks were discarded.

- Remote formal audit SHA: `a1a6189a28ebe776871890ff8429667894ca3a9cb540e924830ce0d411c87021`.
- Local original/control/candidate audits: `af50e6944da47c67d888b650cb210e0417358afeb348c57bbb8e2319a25c9142`, `b0c65c702dd230468e87288533c17dcf0f480bc3612659470516b3eadba1c2e5`, `54a3855d92ed07c32bd54aa44194d9e2cb0317c245b2e119f2330d2c033670a0`.
- Local visual closeout SHA: `fadc9f0435395e9ace0bf03013d4b94c457ec4f51d1c22233a8e010c928f5775`.
- Raw evidence: `formal-retrieval-001/evidence/integration-003/prepared-*/runs/`.
- Decision, table, paired statistics, PNG/PDF CDF: `formal-analysis-001/`.

These paths are relative to the fresh H100 campaign root above. All native
workers and the remote formal auditor exited. Next G4 work is a new operator
candidate with complete numerical proof and a stronger expected E2E effect,
plus remaining same-platform family coverage and skill reuse. No candidate
has been promoted merely because an experiment ran successfully.

## Fixed Protocol

- Platform: H20, LibTorch 2.10.0+cu128, CUDA 12.8, actual captured CUDA
  operator inputs with recorded shape, dtype, stride, layout and storage
  offset.
- Families covered by actual examples: Attention (`scaled_dot_product_attention`),
  GEMM/Linear, LayerNorm, Embedding. RoPE has a separate RTX repeated workload.
  VLA-specific solver/update candidates are recorded separately.
- Correctness gate: complete output structure and byte identity against the
  captured eager reference. A graph-batch mean is not a per-call latency CDF.
- Search budget: the H20 microbenchmark campaign attempted 16 candidate
  configurations; the solver campaign evaluated three candidates with one
  independent process and 160 complete solver steps per candidate.

## Evidence

| Family | Baseline / candidate evidence | Correctness | Timing evidence | Selection |
|---|---|---|---|---|
| Attention | `operator-profiles/h20-microbench-v2/scaled_dot_product_attention-{aten,inductor-aten,inductor-autotune,inductor-aten-preserving}` | All measured candidates byte-exact | 20 ms CUDA graph batches; raw samples retained | Not selected; no complete-model E2E result |
| GEMM | `operator-profiles/h20-microbench-v2/linear-{aten,inductor-aten,inductor-autotune}` | Measured candidates byte-exact; preserving candidate retained as failed | Raw graph-batch means and compile time retained | Not selected; no complete-model E2E result |
| LayerNorm | `operator-profiles/h20-microbench-v2/layer_norm-{aten,inductor-aten-preserving}` | Measured candidates byte-exact; regular/autotune candidates rejected by numeric gate | Raw graph-batch means retained | Not selected |
| Embedding | `operator-profiles/h20-embedding-repeats-v1/` | Five independent ATen and Inductor runs byte-exact | Mean per operation 0.0021265963 vs 0.0018667301 ms on the recorded workload | Not selected; no complete-model E2E gain |
| RoPE | `operator-profiles/rtx-rope-repeats-v1/` | Five independent processes per lane byte-exact | ATen 32.067544 us vs Inductor 7.854182 us median means | Not selected; RTX workload, no integrated model result |
| VLA solver/update | `operator-profiles/solver-candidates-v1/summary/report.json` | 160 complete steps and both outputs byte-exact for all three candidates | ATen 3.068011, Inductor 1.726649, preserving 2.942838 us | Not selected; full-model integration pending |

The actual extracted graph manifest is
`operator-profiles/actual-prefix-operator-examples-attempt2/report.json`.
It contains four real operator signatures and input packs; it is not a
fixture. The static inventory remains explicit that graph node multiplicity
is not a measured launch count.

## Current Decision

`selected_for_deployment=false` for every candidate. The new H100 solver now
passes capture, compilation, correctness and actual same-package complete-model
remeasurement, but fails the performance-confidence gate. Other families still
lack matched full-model results. Continue with a new candidate and the same
input/numerical contract; a microbenchmark speedup or uncertain positive point
estimate must not be promoted to a paper-level E2E gain.

## Model-Independent Selection Gate (2026-09-08)

The public selection contract is now
`vlaforge/deployment/operator_selection.py`, schema
`vlaforge.operator_selection/2`. It accepts an immutable operator report and an
immutable complete-model E2E report, hashes both, and requires capture/compile,
full-output correctness, measured E2E integration, and an explicit E2E speedup
threshold. If the E2E report contains `confidence_gate_passed=false`, selection
is rejected even when the point estimate is positive. No model name or GPU name
is used for dispatch; `skill_key` carries the operator signature and `hardware`
is provenance.

Correction after raw-evidence review (2026-09-08 08:00 UTC):
`operator-profiles/operator-selection-v1/decision.json` is an **ineligible
evidence pair**, not a real integrated negative result. Its H20 Embedding AOTI
candidate (`aten.embedding.default`, BF16 weight `1024x768`, indices `1x1024`)
was paired with an RTX 3060 static-precompute campaign. Moreover, its 0.214856%
point estimate and `[-0.3720797, 0.6559971]` ms interval describe original versus
an unchanged clone-control, not the folded variant and not the measured AOTI
kernel. The source and control graph hashes are identical. The static pass
has a separate control-to-folded estimate of -0.277006%, with interval
`[-1.1905255, 0.5683290]` ms; that remains a different optimization mechanism.

The historical decision and all raw measurements are unchanged. The candidate
microbenchmark still has actual byte-exact output and a 13.920929% point
speedup, but there is **no verified same-candidate same-platform E2E integration**.
The independent `operator-selection-v1/provenance-audit-001.json` rehashed all
120 raw E2E evidence files, including 15 GPU telemetry files, and pinned the
decision, candidate report, comparison and precompute ledger. Audit SHA:
`1a94c630c256432056a7e9f707768b69e56c898f7ee49bd62b5af934d0246911`.

The earlier v1 API and operator benchmark tests passed (`47 passed` in
`operator-selection-tests-003.xml`), but they do not establish candidate-to-model
or platform binding. The v1 API accepted caller-supplied integration flags and
speedups, so it was not a sufficient deployment acceptance gate.

The v2 fix now requires the measured artifact digest in the candidate bundle
and absent from the baseline, identical GPU types and measurement/input/output/
numerical contracts, distinct bundle and execution-audit identities, and a
reported speedup consistent with both lane means. Confidence evidence is
required explicitly; zero gain cannot select a candidate. The Interface and
its limits are described in `vlaforge/spec/operator-selection.md`. This is a
metadata consistency check, not authentication of measurements: actual bundle
payload and execution audits still must be performed by the report producer.

A regression first reproduced unbound flags being accepted. All 29 focused
tests now pass. `operator-selection-v1/binding-real-rejection-002.json` also
checks the actual historical pair and a negative control that changes only its
confidence flag: v1 would select the negative control, while v2 rejects both
with `end_to_end_integrated=false`. Its SHA is
`5ffdb2ff493387e9ddacc4b833559fe64cc056c2782793ea2cc78c5f6fd1f126`.
These are gate regression results, not new model experiments. There is still
no verified selected-kernel E2E benefit, and all historical files are retained.
