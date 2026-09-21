# Recorded SmolVLA and RDT Host Pipelines

These adapters use the same `validation/host_pipeline.py` timer, complete typed
output checks and `NativeTensorSession` C ABI as the completed OpenPI campaigns.
There are no model-name branches in the timer, audit driver or core runtime.
Source snapshots are separate from all completed campaigns; old data and failed
attempts remain intact. Formal acceptance requires five independent processes
per configuration, 128 warmups and 1024 measured calls each, remote/local complete
audits and inspected CDFs. A passed pilot does not meet that requirement.

## SmolVLA Formal Results

Campaign-002 has now completed all ten formal workers, independent remote audit,
local rechecking of every complete output and inspected CDF/tail figures.
Each configuration has five processes, 128 warmups and 1024 measured calls per
process. All 10240 measured latencies and 23040 complete output tensors including
warmups are retained. No outlier was removed and no pilot enters the table.

| Configuration | Mean ms | p99 ms | Fresh Chunks/s | Preprocess + H2D Mean ms | Inference + Postprocess + D2H Mean ms |
|---|---:|---:|---:|---:|---:|
| Official PyTorch | 206.161639 | 220.579781 | 4.850563 | 2.379354 | 203.782285 |
| Session required | 50.726747 | 53.682128 | 19.713466 | 2.216963 | 48.509785 |

Every normalized/native F32 action chunk matches both frozen references byte
for byte: MSE and max absolute error are zero. Minimum computed cosine is
0.9999999999999998 for normalized output and 1.0 for native output. Each native
worker starts at zero counters and records 1152 ten-step replays with zero
ordinary fallback. The Python host is explicit; file decoding, initialization,
robot transport and validation/logging remain outside this original-input timing.
Both distributions contain visible process groups, which are retained; exclusive
CPU/NAS ownership was not established. These data do not measure task success.

Complete local archive:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-formal/smolvla/`.
It includes `audit-formal.json`, `local-formal-audit.json`, all worker/monitor
records and complete output bytes, plus latency CSVs, exact CDF CSVs, PNG/PDF
and `figures/report.json`. Remote identity is the campaign-002 protocol below.

Remote formal audit SHA256:
`e54dd74393db232f712b1ec8c3e93623979f9d1566d51f27326a4f2869bcb8a3`.
Local complete-data audit SHA256:
`28f6a675b68dec41a9325e6db2b20fbf101256098d755fa8c0eb9ad261959acd`.
Figure report SHA256:
`ef83d20c2cef6402064861d1be29b2d8b7e44fe2bf92f4a5bed731fc2f7f5936`.

## SmolVLA Provenance

The input package contains 16 distinct recorded SO100 observations from 16
episodes of `lerobot/svla_so100_pickplace`, revision
`728583b5eaf9e739a7f119e2def466fa1d552402`. All nine pinned source files were
verified. Dataset-native state, two decoded uint8 camera arrays, task text and
the original saved noise are retained. Pixel uint8/float roundtrips were checked
exactly during extraction on H20-2. Shared NAS transfer through the existing
`huoshan-private` channel was file transport only. The redundant slow direct
transfer was stopped after verifying the complete alternate copy; its partial
directory remains excluded.

`adapters/smolvla_host_pipeline.py` invokes the pinned official preprocessing,
model and action postprocessor, including the explicitly recovered SO100 state
statistics. The input tool and host adapter share `prepare_smolvla_statistics`;
no statistics were fitted or guessed. Before measurement, all eight processed
ports for all 16 observations must match the frozen model-tensor inputs exactly.
Native action output remains the official numerical scale, without claiming
physical robot calibration.

Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/raw-smolvla-20260910-001/`.
The `raw-inputs-001/` package, `shared-001/` build and 303-file `source/` snapshot
are frozen. Source-provenance SHA256:
`e990f201cfe124b5b203ce530c11c8a865898e2586165292284fad2b90668cfa`.
Shared library SHA256:
`2426a41fd929cd19c9a65d837b3a177ce048977a75639a0eda96ad97d7713e77`.
The original required bundle is unchanged.

Two distinct GPU pilots have passed remote independent audits and local
complete-data rechecks. Each has two independent workers, each with 16 warmups
and 32 measured calls: 96 calls and 192 complete normalized/native output tensors
byte exact. Native workers record 48 ten-step replays with zero ordinary fallback.

| Campaign | Host / GPU UUID | Pilot Controller | Official / Native Worker PID | Formal State |
|---|---|---:|---|---|
| campaign-001 | H20-1 / GPU-6e8d3a5a-7894-c59a-0f3b-200f15c6a7df | 3199641 | 3199669 / 3200724 | Controller 3202649 failed before any worker: a foreign compute owner appeared |
| campaign-002 | H20-2 / GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5 | 111807 | 111822 / 112543 | Controller 113987 finished successfully; 10/10 formal workers, remote/local audits and CDF complete |

No foreign process was stopped. The first formal failure and owner preflight
are preserved in `campaign-001/`. The second campaign has a new GPU-bound protocol
and its own complete pilot; neither its protocol nor its results replace the first.

| Campaign | Protocol SHA256 | Remote Pilot Audit SHA256 | Local Pilot Audit SHA256 |
|---|---|---|---|
| campaign-001 | `367fb7b838785e2be6462a898708dd198ae98318246d07eb9ab406186b996fab` | `06b7c41e6fe2c5ee38a877a4a269db472aa29c0e1e44ffa7874e00f117a27cb0` | `5ce26c6c3d2e64ad379afc6e641d05aa1bf220b193f44eaa81f776c619d247e8` |
| campaign-002 | `ed36bed169991b55c5c04fa093d2e77e5ecca4363ccdb22740c9db1255dc1010` | `89ba4f9fda5944cde3533396b617f3592e6f927ca6436deef43833edea0698ea` | `e9b2722ecea879f7f2a77e1f23993ed8eeb9e0abffd83d61aa894699054b46cd` |

Local full pilot archives are
`artifacts/recovery-audit-20260909/raw-host-generic-pilots/smolvla/` and
`artifacts/recovery-audit-20260909/raw-host-generic-pilots/smolvla-002/` in the
current worktree. No pilot timing is promoted to a formal result.

## RDT

All ten formal workers are now complete, followed by independent remote audit,
local complete-output rechecking and inspected CDF/tail figures. Controller
119430 ended successfully and must not be relaunched. Each configuration has
five independent processes, 128 warmups and 1024 measured calls per process.

| Configuration | Mean ms | p99 ms | Fresh Chunks/s | Preprocess + H2D Mean ms | Inference + Postprocess + D2H Mean ms |
|---|---:|---:|---:|---:|---:|
| Official PyTorch | 239.685959 | 301.828993 | 4.172126 | 56.220232 | 183.465727 |
| Session required | 221.804152 | 230.311815 | 4.508482 | 56.396278 | 165.407873 |

All 10240 measured latencies and 23040 complete BF16 output tensors including
warmups are retained. Both unified and native actions match both references
byte for byte: MSE and max absolute error are zero, with minimum cosine
0.9999999999999998 and 0.9999999999999999 respectively. Every required worker
starts at zero counters and completes 1152 five-step replays with zero ordinary
fallback. The official distribution has a visible long tail; no outliers were
removed. Exclusive host CPU/NAS ownership is not established.

The complete local archive is
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-formal/rdt/`.
Its `figures/` contains recomputed latency/CDF CSVs, PNG/PDF, full-output metrics
and `report.json`. The source protocol and GPU identity are recorded below.

| Evidence | SHA256 |
|---|---|
| Remote formal audit | `6ada5d23045fc71e39327a11c6fb4fc3079e09737a48b0a85722f3151e1b9526` |
| Local complete-data formal audit | `dfa64a6cab14147a70fc16132d7540bd78c35804c55d6850d3448cd6a89bee82` |
| Figure report | `959b3240ca4958ffa2e99b70f5b14b62f774af610bab0cb0b6b5880880a233a6` |

`adapters/rdt/rdt_host_pipeline.py` retains the actual six history-major camera views,
native AgileX state, instruction, saved BF16 noise and generator state. It uses
the official tokenizer, SigLIP preprocessing and state transform. Original
OpenCV BGR arrays remain unchanged when passed to PIL, matching the archived
profile. This is not a calibrated RGB claim. All 16 processed input sets and
generator-state-to-noise checks have passed on H20-2 GPU0.

The official path calls the original T5 encoder, SigLIP tower, RDT sampler and
AgileX action transform under the separately named Torch 2.10 environment.
Preprocessing is performed by the qualified adapter; this is not the untouched
Torch 2.1 wrapper environment. The sampler keeps its original random draw and
five-step DPMSolver scheduler. Its saved generator state is restored inside the
timed preparation segment. The Session consumes the same explicit noise tensor.
Both return the complete BF16 unified and native actions before the wrapper's
final lossless FP32 cast. No scheduler equation or random tensor is substituted.

Both pilot workers passed, followed by independent remote and local complete-data
audits: 96 calls and 192 complete BF16 output tensors are byte exact against both
frozen references. The native worker records 48 replays of the five-step loop,
with zero ordinary fallback. Pilot worker PIDs are official 113228 (NVML 3008111)
and native 115289 (NVML 3038908). Formal controller 119430 and first worker119581
were subsequently observed live; the accepted terminal formal results are above.

Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/raw-rdt-20260910-001/`.
Campaign `campaign-001/`, controller PID 113167, H20-2 GPU0 UUID
`GPU-ae321531-504c-f416-e693-94fa02779130`.
Protocol SHA256:
`c67ec05c167af36530ca14a11f326f89cbcf65503c9edeca377aa5a89d75c5a4`.
304-file source-provenance SHA256:
`23faeff2db211b6f8ce5b863972a4ce4657faa04cb23de04341be7ebdfbfbdb0`.
Shared library SHA256:
`f72b47d26acedae6d324ba54214066c9c1d02f0fe83a6a07eeff530bc59142b4`.
The old model-tensor data/reference packages and required bundle remain unchanged.
Remote pilot audit SHA256:
`04ddddcb32f1eb989e9816005a4e7b21168ee077e07d44cea0a0941f3f14b0dc`.
Local complete-data pilot audit SHA256:
`2dacc85d4638d4a8e0a6544c933265dde9b4fe9c721d6808a83dda1347417f50`.
The complete local pilot archive is
`artifacts/recovery-audit-20260909/raw-host-generic-pilots/rdt/` in the current
worktree. Both original shared-library build receipts are also retained locally
under `smol-host-shared-001/` and `rdt-host-shared-001/` in the recovery root.

## Validation and Remaining Work

SmolVLA focused tests: 69 passed. Latest full CPU regression after adding RDT:
2249 passed, 74 skipped, zero failures. The local RDT/native Session focused set
has 51 passed and 7 dependency skips; the RDT environment on H20-2 then ran its
CPU adapter/input/reference/scheduler tests with 53 passed and one CUDA-only test
skipped. No GPU benchmark was counted as a CPU test.

Latest full CPU XML SHA256:
`89eb5f7118db7d75a8500bee5a60f984a7a4b91d8cc292e1213681b14040dd75`.
Remote dependency-complete CPU XML SHA256:
`0ac80b76ba40035b93008781ea9c239014db360c5dd48b3c7cf2a8a9119ebcc7`.
Both are retained under `artifacts/recovery-audit-20260909/` in the current worktree.

Final update: CogACT original-input formal sampling, remote/local complete-output
audits and CDF checks are complete. The final five-model table and complete quality
summary are in `h20_vla_experiment_delivery_20260910.md`; final CPU regression is
2259 passed and 74 skipped, documented in `h20_final_validation_20260910.md`.
The earlier Qwen native export blocker was superseded by the fixed-profile
native result in `qwen35_native_formal_20260910.md`; the rejected cache-copy
routes remain in `qwen35_native_blockers_20260910.md`.
CogACT's original Meta configuration and nominal 3B label remain unverified.
Orin/J6M/BPU execution is deferred. This Python host workflow does not grant a
standalone no-Python deployment certificate.

Historical CogACT preparation inventory (superseded by the audited formal run in
`raw_cogact_host_pipeline_20260910.md`): the actual 16-frame raw source is at
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/cogact/real-observations-spaced-16/`.
Its official dependency roots are under `cogact/source-20260906-1030/` in the same
NAS base; the original reference command in `cogact/reference-candidate-001/report.json`
names all three source checkouts, weights and tokenizer lock. This inventory is
not a passed raw-input pipeline or a new numerical reference.
