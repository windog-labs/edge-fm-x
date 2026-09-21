# RDT Execution Variants: 2026-09-08 Recovery

Status: all three dual-output CUDA formal policies passed independent remote
and local audits. Recovery preserved the existing controller PID 3607880 and
supervisor PID 3607882 without restarting them. All measurement, report, audit
and retrieval processes have exited. Current hardware stage and full paper
acceptance remain incomplete.

## Formal Result

Each policy used five independent processes, 128 warmups and 1024 measured calls
per process. There are 15360 measured timings and 17280 complete calls including
warmup. All 34560 full BF16 outputs / 157040640 values are byte-exact against both
official and same-artifact references. MSE/max-abs are zero; minimum computed
cosine is 0.9999999999999998. Warmups and pilots are not mixed into the formal CDF.

| Policy | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Fresh chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 176.710522 | 175.820378 | 181.867842 | 185.705522 | 258.497711 | 3.515430 | 5.658973 |
| batch-only | 172.734449 | 172.390374 | 175.765479 | 178.065728 | 204.611051 | 2.203643 | 5.789233 |
| required | 164.397820 | 164.435253 | 166.226521 | 168.415117 | 173.440502 | 1.233310 | 6.082806 |

Required reduces mean latency by **6.967724%** against the same-source ordinary
path. Every required worker records `REPLAY_FINAL,12,1,5,1152,0`; all five solver
steps are captured once, all 1152 calls replay, and there is no ordinary fallback.
This is execution-policy evidence, not an Agent-selected operator result.

Sampled peak device memory is 13675 / 13693 / 14283 MiB for off / batch / required,
respectively, identical across each policy's five workers. Required increases it
by **608 MiB**, not a memory reduction. Sampling includes startup and is not an
allocator peak measurement. All latency outliers remain in the CDF and table.

The formal controller ran for 4463.156 seconds, about 74.4 minutes. Report and
independent NAS file/output audits followed; they did not rerun inference.

- Remote independent audit SHA256:
  `95a17d627aed2fb59c591bda725f1ea74137d7236ec85e7481b817845693b8fe`.
- Local complete raw/timing/telemetry audit SHA256:
  `47d5a91a1c307f1a31a56e8e3edc99c0b04fcce225c745bbfb3c25b696977d9e`.
- Local evidence root: `artifacts/edgefm-vla-goal/20260906-044509/rdt-replay-resume-20260908/dual-003/`.
  `formal-closeout/complete-table.csv` and `report.json` are generated from the
  checked raw results, including max latency and sampled memory. The original
  public table and per-call data remain under `prepared/`.
- `figures/latency-cdf-tail.png` and `.pdf` were generated and the PNG was viewed:
  curves are nonblank and text does not overlap. PNG SHA256:
  `d132d3e5ea55692d37485638a0efd6930774b019a39cb3e14c28fb891332c4dc`.
- `formal-retrieval-001/report.json` verifies 626 copied/retained files, 463764274
  bytes, and every previously retrieved pilot file unchanged. Large weights,
  deployed payloads and runtime libraries remain on NAS; this is not a complete
  local runnable model bundle.

The reference boundary still covers one episode's 16 histories, online T5,
six processed images, five DPMSolver steps and official complete robot output.
No preprocessing/transfer/robot transport timing, vendor baseline, board run,
physical calibration or low-bit acceptance is inherited.

## Source And Recovery

- Local HEAD: `592fc320d57398571dcb4ab686d5fb55fe9bac86`, dirty worktree retained.
- H20-2 recovery preflight found no old RDT build/benchmark worker; all eight
  GPUs reported 0 MiB compute memory. This was a snapshot, not a reservation.
- Original `rdt-replay-20260908-001` failed before GPU execution with
  `replay.backend_execution_context_unavailable`. It selected the synchronous
  TorchScript profile. No success is inferred from its off-only build.
- `rdt-replay-20260908-002` began rebuilding that older, unified-only source.
  The newer dual-output source was then confirmed in the existing records.
  Only this task's verified process group was interrupted before GPU execution;
  its terminal `KeyboardInterrupt` receipt and partial artifacts remain intact.
  No single-output timing will substitute for the full output contract.
- Current run: `zzm-h20-x8-2:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/rdt-replay-20260908-003`.
  It uses the immutable local-source snapshot under sibling run `002/source/vlaforge`.
  All 292 implementation file hashes matched local source before launch.

## Contract

The input is the already verified eight-Region bundle
`runs/rdt-dual-output-003/build/bundle`, SHA256 of `bundle.json`:
`78f93c00e9669da77754f171d4ccc76be61773d03217b2b0d115efd13472f8e8`.
Reference protocol is `runs/rdt-dual-output-formal-002/protocol.json`, SHA256:
`a4246f0614597c7f06b2216a7289b5e2a453d4465cd0a1ac98615a1b7b1cbcd4`.
It retains 16 real observation histories, saved noise, full unified actions
and official robot action conversion, and all eight numerical policy bindings.

The public `build_session_variants.py --torchscript-shared-context` option
explicitly selects the existing CUDA TorchScript ExecutionContext provider.
It preserves archive hashes, effect audits, numerical bindings and I/O, records
the before/after contracts, and creates fresh off/batch-only/required bundles.
It does not edit the source bundle, relocate tensors, certify capture from
capability flags, or add model-name branches to IR/Plan/runtime.

## Verification

- Targeted source tests: 68 passed, 2 explicit CUDA opt-in skips;
  `artifacts/edgefm-vla-goal/20260906-044509/rdt-replay-resume-20260908/tests-002.xml`.
- New full CPU regression: 2059 passed, 83 skipped, 0 failed; 72.76 seconds
  including supervisor overhead. Snapshot unchanged before/after execution.
  `resume-cpu-regression-012/report.json` binds source and raw test logs.
  Its minimal validation environment lacks optional dependencies including
  safetensors, so its skips differ from regression 011; none count as GPU/model passes.
- Source synchronization first reported three unreadable old generated CMake
  files under `out/`. A subsequent source-only sync excluded generated outputs,
  exited 0, and all 292 implementation hashes matched. The failure is retained
  in `rdt-replay-resume-20260908/recovery-audit.json`.
- All three pilots passed: each policy used 16 warmup + 32 measured calls,
  totaling 144 calls and 288 complete BF16 tensors. Unified output is
  `[1,64,128]`; primary robot output is `[1,64,14]`. All 1,308,672 values are
  byte-exact against both official and same-artifact references, with MSE and
  max-abs zero. Minimum cosine is 0.9999999999999998; byte equality, not this
  rounded cosine alone, establishes the measured numerical result.
- Native PIDs were 3606546/off, 3606713/batch-only and 3606876/required.
  Required records `REPLAY_FINAL,12,1,5,48,0`: all five solver steps captured,
  48 replay calls and zero ordinary fallback. Batch-only records 48 ordinary
  calls and no capture/replay. All actual process maps exclude Python.
- `dual-003/independent-pilot-audit.json` SHA256
  `dce9d5e3c24d38c77c878ba3238cc79f8e4b549072fe796cd7ba83c405024a4b`
  rechecks original NPY references, frozen bytes, all eight preserved contracts,
  source hashes, bundle payloads, actual runtime libraries, numerical bootstrap,
  raw outputs, timing summaries, GPU ownership and replay counters.
- Audit negative tests: 6 passed, including truncated/corrupt/missing robot
  output, failed execution and missing runtime replay counter. Their XML SHA256
  is `e1cbd6e40915dbb25811639a7feea05d958630e6eaa0a90c0dc776cca1374269`.
- Raw outputs, serialized references, telemetry, logs and metadata are retrieved
  under `artifacts/edgefm-vla-goal/20260906-044509/rdt-replay-resume-20260908/dual-003`.
  `local-retrieval-audit.json` independently verifies all 288 downloaded tensors
  and the hash-bound converted references. Large models and runtime payloads
  remain on NAS; this local folder is evidence, not a full runnable model bundle.
- Pilot means are 177.610439/off, 172.942881/batch-only and 162.707644/required ms.
  These remain 32 samples per policy from one process and are not used in the
  separate formal result above.
- The build, prepare, pilot, audit and retrieval jobs exited before formal
  execution. A fresh empty-GPU preflight preceded the formal controller launch
  at 2026-09-08 02:43 UTC. Each of fifteen workers still has its own ownership
  guard and immutable-source verification. Formal acceptance was recorded only
  after the subsequent successful report, remote audit and local byte recheck.
- Local data-handoff schema changes made after launch are not synchronized into
  the frozen 292-file runtime snapshot. They cannot alter the running experiment.

## Resume Commands

Check the named controller/runner process and terminal receipt before invoking
the next stage. Every stage refuses an existing stage receipt/output; do not
delete it to force a restart. GPU ownership is checked before each worker.

```sh
ROOT=/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/rdt-replay-20260908-003
PY=/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/envs/rdt-aoti-cu128-py311/bin/python
# Build, prepare, pilot and independent pilot audit have already passed.
# Formal and report stages have completed; do not invoke them again or remove receipts.
# Optional repeat audit only, with a NEW audit output filename:
SOURCE=/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/rdt-replay-20260908-002/source/vlaforge
PYTHONPATH="$SOURCE/python:$SOURCE/tools" $PY "$ROOT/audit_replay.py" \
  --root "$ROOT" --mode formal --output "$ROOT/independent-formal-audit-NEW.json"
```

These are H20 model-tensor resident-input experiments, not Orin/BPU results,
sensor-to-action timing, a matched vendor comparison, low-bit deployment, or
a proof of robot task success.
