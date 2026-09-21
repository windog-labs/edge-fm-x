# CogACT Public-Protocol Replay Campaign

Update at 2026-09-08 12:22 UTC: the separate
`cogact-numerical-h20-20260908-002` candidate now has fresh numerical-provider
bindings, full-IR verification and three-policy native pilot verification
(144 calls, 720 complete output tensors), independently rechecked locally.
See [the numerical compile report](torchscript_numerical_compile_20260908.md)
for the preserved failed pipeline receipt and successful isolated test recovery.
That candidate still needs its own formal timing campaign. The run 004 formal
results below remain legacy, unbound evidence and do not inherit the new
provider guarantee. Original configuration, model-scale, external RNG tape and
board limitations remain unchanged.

Status at 2026-09-08 09:58 UTC: run 004 has completed all fifteen formal
workers, aggregation, remote independent audit, local complete-output recheck,
table and visual CDF/tail inspection. All related processes exited. The
configuration/RNG/provider/board limitations below remain unaccepted, and the
available-hardware Goal and full paper acceptance are not complete. Earlier
run001/run002 failures and run003 pilot evidence remain preserved.

Remote root:
`zzm-h20-x8-2:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-004/`.
Local campaign/scripts:
`artifacts/edgefm-vla-goal/20260906-044509/cogact-replay-20260908/`.

## Original Source and Boundary

Local HEAD remains `592fc320d57398571dcb4ab686d5fb55fe9bac86`, with existing
worktree changes preserved. The 292 implementation files were copied from the
current local source into an isolated snapshot and verified before transfer.
Rsync dry-run and actual transfer used no deletion and created 579 files,
26,013,698 bytes; `rsync-001.json` records the completed transfer.

The source bundle is the existing verified four-Region TorchScript candidate,
manifest SHA256
`8a033197d6d95b20f88c4b92e79e14731f66d7be0392dedda9010c6bcd9e980f`.
The public `build_session_variants.py --torchscript-shared-context` generates
off / batch-only / required bundles from the same archives. It explicitly
changes only the execution profile/capability and declaration IDs, without
changing effect, numerical or complete input/output contracts. Fresh output
validation is required; a successful build does not certify capture or replay.

Built manifest SHAs (off / batch-only / required):
`5b394365b02dbf3af19f214f0f4d607a5ca64e927fad4fb178c4b50dc87e1fc7`,
`acdcf7a037485a6e43a271f884c373afd61307ce22f3ece1176d4a7bdf8c5145`,
`162972fc58227c461ef0528f91e64240b6777dd880613cfc52bf1e862aa4449c`.
Derived three-policy protocol SHA:
`ded4759575178384c584849c8a966a5f18cfcb7558405c8511bba67f1cbab69a`.

The new protocol SHA256 is
`b6790ce395ad7e8444de7f983fc5e84f9d3e68d185bd7e35b35b078b9d19ddae`.
It selects actual Fractal episode-0 frames
`0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,114`, retaining original inputs,
noise tape and references from the independently reverified 115-frame data pack.
Spaced-frame manifest SHA:
`85251486df36a62651e2f12ab916318c3560daf63b40de419b3329250b45eae1`.
Fresh source audit SHA:
`1a4f1ac1905df8c6846309777ff4ffa41a2503ceb07c4e110140301391fbae18`.
The known historical audit hash mismatch is retained and not used as proof.

All five ABI outputs are required: raw F32 `[16,7]`, normalized F32 `[16,7]`,
primary native F64 `[16,7]`, U8 `[16]` RNG receipt and I64 `[1]` draw count.
Integer outputs use exact byte checks, not floating cosine/MSE conversion.

Formal settings are 128 warmups + 1024 measured calls in five processes per
policy, balanced over the 16 frames. The public resident model-tensor boundary
includes online vision, text prefix, full N10 CFG/DDIM and native output
conversion, excluding sensor preprocessing, transfers, tape production and robot
transport. Pilot settings are 16 + 32 and never contribute to formal CDFs.

## Limits and Next Commands

This remains the pinned **public-dependency configuration candidate**. Original
Meta configuration is unavailable/unverified; installed model scale is about
7.63B elements, not the nominal 3B label. The external producer supplies the
11-draw tape; autonomous C++ PRNG is not implemented by these bundles.
Original legacy TorchScript contracts lack numerical-provider bindings. They
are retained as such, not retrospectively promoted to v2 enforcement. New
same-precision output agreement would not remove any of these limitations.

`run_after_build.py` checks a successful terminal build before prepare, pilot,
independent five-port audit and real-evidence negative tests. Only after all
pilot gates pass does it launch formal sampling, report and independent formal
audit. Every stage has create-only receipts/logs; a failed stage prevents later
launches. Do not repeat it when its controller is alive or receipts exist.

```sh
ROOT=/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-001
PY=/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/envs/cogact-cu128-py311/bin/python
# Original pipeline failed and exited; inspect it, do not relaunch it:
"$PY" -m json.tool "$ROOT/pipeline-controller.json"
```

No vendor baseline, Orin/BPU, low-bit quality or full-paper acceptance is
inherited. Data-only board packs are not runnable board engines.

## Failure and Recovery

The required worker PID 3699674 exited with code 10 and
`graph capture abort failed`; GPU ownership and monitoring were clean. The
unchanged native runner reproduced this on its first call in
`debug/single-call-001` (PID 3700873, exit 10), so it was not a long-soak or
external-owner failure. The real saved step reproducer narrowed it to CPU
`arange -> mul -> div -> exp -> to(cuda)` inside the captured timestep embedding.
`debug/step-ts-002` preserves the complete CUDA capture error. The first smaller
probe incorrectly relocated CPU constants; its separate device-mismatch failure
is retained and was not treated as the original bug.

The public constant-precompute module now permits explicit literal-only
declarations and audited fresh device conversions, without approving state or
example inputs as constants. Only a fixed 128-value / 512-byte frequency vector
is precomputed. Actual timestep, scheduler, CFG, saved noise and all four carry
ports are unchanged. Metadata assertions are retained. A separate generic
runtime diagnostic fix preserves both primary capture and abort errors while
keeping the poisoned-context prohibition; its C++ regression first failed and
then passed. All 11 CPU native tests and 2079 CPU Python tests pass, with 83
explicit skips.

Recovery root: `/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-002`.
Its new source snapshot includes these changes; no source in run 001 or the
still-running pi05 formal campaign was overwritten.
`fold-step-002/report.json` records fresh capture/compile, original/control/folded
N10 complete-carry byte agreement and three GPU graph replays. New step archive
SHA `43670c897cd8b7e71580eb39f30b8498eb84f8f45506eb9cb023e3a44c4ec624`;
capture-status-only SHA `4941275b03805e17a3e471374b95025f3e2873018e32fdf70a63b3977c3cfcf1`;
precompute ledger SHA `c4324a6832a1987c03ee915412f04af91b54bf975d7fe176d77c4115ce2d8028`.
The first recovery attempt hit the exported module's unsupported `eval()`;
that failure remains, and a normal FX module with unchanged graph was used for
fresh recapture in attempt 002.

`direct-001/report.json` verifies the complete updated model on all 16 selected
frames: 80 full outputs match both original reference sets byte-for-byte, with
global RNG unchanged. It produces **fresh** same-artifact references for the
new compiled Region instead of relabeling old references. C++ build, native
replay pilot and independent audit remain separate gates.

The subsequent run-002 base build (controller PID 3709383, child 3709384)
exited 1 with `KeyError: region_name`. The script had archived
`CaptureOutcome.report`, an unsupported-status report, in `fresh-capture.json`
instead of the full `CaptureOutcome.evidence`. The saved export and real-step
execution are valid, but that status file is not a complete capture certificate.
No base bundle was created; `base.log` and `base-controller.json` remain failed.
Run 003 uses `save_exported_region` plus explicit evidence-field checks and
repeats capture/compile and the complete reference chain. It does not fabricate
the missing certificate or relabel the previous references.

## Verified Native Recovery

Run 003 is `/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-003`.
Its source snapshot contains the same 292 implementation hashes as current local
code and completed CPU regression 014. The new script uses
`save_exported_region`, retaining both the support-status report and the complete
graph/I/O/effect/numerical capture evidence. Step archive SHA256:
`9fcc48396a17c3f59ceb77d5181e2ffe0e6a887560f3941f7e2ae391611d0dfc`;
detailed capture SHA:
`8e1c4e5d4b67c406f93389016c98e77f9502fda26a0534f10673cb774cd7bac8`.
All N10 original/control/folded outputs and three real CUDA graph replays pass.

GPU0 gained an external NVML owner (PID 3699192) before the full-reference
launch. `direct-001/preflight-owners.json` records the refusal; no worker was
launched and no external process was stopped. The corrected reference and
native pilot use idle H20 GPU1, UUID
`GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`. `direct-002/` independently recomputes
all 16 full-model observations with all 80 original output tensors byte-exact,
producing new same-artifact references and preserving global RNG state.

Base bundle SHA:
`f85019ed4b356e83ecb2aff1a3e8de67509b80258b652d8777da598baf0f0d6c`.
Required bundle SHA:
`66d82b6035744f87ef8ff1bb5d31d8b927ba9d2e87d6aa56728f68ee85142a45`.
`transformation-lineage-audit.json` SHA
`9058e1215fcc8ce5e4f9011e64d718dd3b8ac44cc155987da18778179f376d40`
verifies the unchanged semantic IR and original inputs, all four contracts,
the single changed Region, literal-only 512-byte ledger, and freshly generated
complete references. No legacy contract is promoted to numerical-provider v2.

Native PID 3719342 exits 0 after 16 warmup + 32 measured calls, covering every
frame three times. All 240 full tensors / 16944 values match both original eager
and fresh same-artifact references byte-for-byte. Three floating outputs have
MSE/max-abs 0 and cosine minimum 0.9999999999999998. The two integer ports have
exact-byte checks only, with no artificial floating metrics. Actual maps contain
neither `libpython` nor `libtorch_python`; ownership and telemetry are clean.
The runtime reports `REPLAY_FINAL,10,1,10,48,0`. Pilot mean is 76.918972 ms,
but this is not a formal speedup comparison against the older unmodified step.

Independent pilot audit SHA:
`981f98dce0442563c3972c7f3971f4cab5c6fe3e3e046ce14310478ec7926458`.
All 12 real-evidence tests pass: complete pilot, corrupt/truncated/missing F64,
U8 and I64 outputs, failed worker and missing replay counter. XML SHA:
`15aa88ae25b266672a7dc57befefeeb26f605e4e8ace936160c2582f92fd7874`.
Run-002 failed logs/status reports have also been retrieved under local
`failed-recovery-002-evidence/`, with their failed receipt/log hashes verified.

Local recovery evidence: `pilot-recovery-003-local-001/`, containing 507 copied
files / 21,337,998 bytes, excluding full model payloads and runtime DSOs. Those
large payloads were verified remotely, not claimed as a locally runnable bundle.
Local raw audit SHA:
`e0ae324fa447fdced9264999042434e40a5b33913bb1e468ae4783bad77be345`.
It independently checks all 240 complete tensors / 16944 values, timings,
transferred references, frozen source, maps and telemetry bindings.

The first strict local metric-JSON recheck failed despite all raw bytes matching:
63 cosine/norm fields differed by exactly one ULP on the two NumPy-1.26.4 hosts.
`local-metric-drift-001.json` preserves every difference and the original strict
verifier SHA. CPU reduction dispatch is the inferred cause; the exact hardware
kernel difference was not independently isolated. An explicit optional
`--cross-host-metrics` mode keeps all actual/direct/eager bytes exact and all
MSE/RMSE/max-abs checks unchanged. Only finite cosine/norm metadata may differ
by up to four ULP, and both values are recorded. The default mode remains strict;
neither remote fidelity records nor model quality gates were edited. Fourteen
local real-data tests additionally reject damaged F64/U8/I64 bytes, nonzero
MSE/max-abs and excessive cosine/norm changes. XML SHA:
`9f02986d58ceceebccc28b4f7219d3ef728d91d0c3820ee42a701acba3d603e5`.

All build/prepare/native/audit/test/retrieval processes have exited. At
2026-09-08 07:12 UTC the H20-2 compute-owner query was empty; this is only a
point-in-time observation, not a reservation. The Goal remains active.

## Formal Run 004 Completed

Updated 2026-09-08 09:58 UTC. The proposed in-place run003 build above was not
used. A new isolated `cogact-replay-20260908-004` directory was prepared instead,
with the same verified new-step base bundle and fresh direct references. It
contains a frozen local implementation, standard `variants/` and `prepared/`
paths, and the unchanged generic mixed-output worker auditor. The older run003
pilot and run001/run002 failures remain untouched.

All three builds, three fresh pilots, independent pilot audit and 12 actual
evidence tests have passed. The single pipeline PID3724808 launched formal
controller PID3728739, child PID3728742, at 08:00 UTC. All 15 workers and
aggregation passed, followed by independent auditor PID3756699. The final
local recheck and visual closeout passed by 09:58 UTC. GPU binding is H20-2 GPU1, UUID
`GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`. No formal stage is duplicated.

The local source and transfer receipts are under
`cogact-replay-20260908/formal-004/` and `formal-004-rsync-001.json`.
The initial local setup failed on a generated `__pycache__` path; its partial
snapshot and script were retained separately before the clean snapshot and
309-file transfer. No partially copied source launched a remote GPU job.

Recovery is inspection first, not restarting the pipeline:

```sh
ssh zzm-h20-x8-2 cat /xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-004/pipeline-controller.json
ssh zzm-h20-x8-2 tail -n 20 /xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-replay-20260908-004/formal.log
ssh zzm-h20-x8-2 ps -p 3724808,3728739,3728742 -o pid,ppid,etime,args
```

The source new-step base bundle SHA is
`f85019ed4b356e83ecb2aff1a3e8de67509b80258b652d8777da598baf0f0d6c`;
the run004 source protocol SHA is
`43918fde19f5b544fffcfd6fe64110579b6e4e1a13568fff7f83580c09cf5e3a`.
No old off/batch timing was mixed into this new-step comparison.

| Policy | Mean ms | p99 ms | Std ms | Fresh chunks/s | Peak sampled MiB |
|---|---:|---:|---:|---:|---:|
| off | 85.576719 | 95.596377 | 3.541414 | 11.685421 | 29433 |
| batch-only | 84.403288 | 94.568688 | 2.917209 | 11.847880 | 29433 |
| required | 75.538978 | 78.687230 | 1.010959 | 13.238199 | 29573 |

Every policy has five independent processes, 128 warmups and 1024 measured
calls per process: 15360 measured timings / 17280 total calls. All 86400 full
output tensors / 6099840 values match both reference sets byte-for-byte.
F32/F64 outputs have MSE/max-abs zero; minimum primary F64 cosine is
0.9999999999999998. U8 RNG receipts and I64 draw counts remain exact bytes,
without floating conversion. Each required process reports
`REPLAY_FINAL,10,1,10,1152,0`: ten captured steps, 1152 replay and no ordinary
fallback. Maps show no Python runtime; telemetry has no foreign GPU owner.

Measured mean reduction is 11.7295225%, with 140 MiB more sampled device
memory. These are model-tensor results, not sensor-to-action or memory-saving
claims. The maximum retained measured latency is 129.142787 ms for off and
98.421972 ms for required. No pilot sample or discarded outlier enters the CDF.

Local `formal-004-local-001/` contains 766 files / 125997264 bytes, excluding
large model payloads and runtime DSOs verified remotely. All raw outputs,
timings, CDF ranks, maps, telemetry and copied source identities were rechecked.
The explicit cross-host metric mode records 22680 individual cosine/norm
metadata differences within four ULP; all actual bytes and zero-error gates
remain exact. This is not a relaxed action-quality threshold.

- Remote independent formal audit SHA: `63d15591ad3b9bce8cb11f38cedb0a23c1d3c10d7f9b340ab6faa25966b0eb9f`.
- Passed pipeline SHA: `631f2b74c37c1cd45cc2f3b62fd93f549234017419516ed53356533d19f1b296`.
- Local raw audit SHA: `4337a597236190824f9c855edf116ef185b6ed334ff28192dc0bf99b8892975b`.
- Visual closeout SHA: `10e9d9e628fef13b6b4429dbdcd3a0b0b743e03b5a0ba0b414780d59e40aa7eb`.
- Raw data: `formal-004-local-001/evidence/prepared/runs/`.
- Complete table, per-output metrics and PNG/PDF: `formal-004-local-001/figures/`.

These paths are relative to the local campaign root above. The original Meta
configuration, actual 7.63B scale, external RNG producer and legacy numerical
provider limitations still apply. `eligible_for_lossless_paper_table` remains
false in the public table; successful observed output checks do not remove
unmet provenance/provider gates. Next work must explicitly resolve those
contracts and add vendor/same-platform comparisons, not rerun this completed
create-only campaign or call it the requested nominal 3B/board result.
