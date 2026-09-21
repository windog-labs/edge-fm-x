# pi0.5 H20 Output-Stage Recovery

Status: the complete H20 dual-output formal campaign, independent remote audit,
local five-model recheck and visually inspected CDF have completed. See
`h20_vla_formal_table_20260909.md` for the final table and evidence hashes.

## Formal Campaign

All fifteen formal workers exited 0. Controller 4125122 is terminal and passed.
The independent audit checked 17,280 calls / 34,560 complete normalized/native
tensors; all bytes agree. Off/batch-only/required mean is
180.262594/169.152800/95.344118 ms, p99 is 300.239577/196.051244/113.095929 ms.
All outliers are retained. Formal audit SHA256:
`772c3fd98e2dfc478df6ce7259af16d81ade9e0d97e974e69bff603ff0dd0849`.
The final CSV, per-output metrics, PNG/PDF and data recheck are in current-worktree
`artifacts/recovery-audit-20260909/pi05-formal/`.

The three policy bundles and public benchmark runners were rebuilt from the
same frozen source and artifacts. All three pilots completed 16 warmups plus
32 measured calls. Independent verification checked 144 calls / 288 complete
F32 normalized and F64 native output tensors against both reference sets;
every tensor is byte-exact. Pilot audit SHA256:
`ba1297cc5b9bc0b0d4416cbb58e5d723878a94dd7e033780e93ee3dacefa79b1`.

| Policy | Pilot Worker PID | Pilot Mean ms | Final Replay Counter |
|---|---:|---:|---|
| off | 4119030 | 181.872718 | no replay task |
| batch-only | 4119491 | 180.593017 | `REPLAY_FINAL,13,0,0,0,48` |
| required | 4119755 | 96.303269 | `REPLAY_FINAL,13,1,10,48,0` |

These are pilot means only. Seven real-evidence tests passed, including copies
with altered normalized tails, truncated native outputs, failed exit status,
foreign telemetry owners and missing replay counters. The original pilot
files were not edited. Logs: `pilot-audit-tests.log`. The relevant local
output/variant/benchmark regression also passed 98 tests.

Formal controller PID **4125122**, public benchmark supervisor **4125254** and
first off worker **4125319** were observed live after launch. The formal
protocol is five processes per policy, 128 warmups plus 1024 measured calls
each, rotating policy order, over all sixteen frames. The GPU is H20-2 GPU1,
UUID `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`. Each native worker performs
the ownership registration handshake and is monitored for foreign owners.

Resume by inspecting these exact paths under the NAS root below:

- `campaign-formal.json`: controller receipt, child PID and frozen source map.
- `campaign-formal.stdout.log`: completed process markers.
- `prepared/runs/`: actual per-worker raw samples, complete outputs and telemetry.
- `formal-launch.log`: persistent launcher output.

Do not launch formal, report or audit stages again into these directories.
They are complete, and pilots did not enter the formal aggregates. New expanded
measurements must use a new protocol and directory; inspect existing receipts
before any continuation.

## Full Model Continuation

All four public output-bundle phases completed: `direct`, `build`, native
execution, and `verify`. The unchanged normalized artifacts were materialized
on NAS alongside the new checked output stage. Actual C++ execution used an
invalid Python environment and loaded the bundle's numerical provider.

- Native controller PID 4099502, native worker PID 4099557, NVML PID 280141;
  H20 GPU1 UUID `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`.
- Three complete calls: F32 `[1,50,32]` normalized plus F64 `[1,50,14]`
  native outputs, all bytes equal to both official and direct references.
  This is one warmup plus two diagnostic measurements, not a formal CDF.
- Bundle SHA256: `ee1bfc82c21cee98b712c8c3c8ae1960350f31fcc96dc08cfeb7eac5c1c57d1e`.
- Public verify report SHA256:
  `bd899eae8f9865a1f84b9726e79893d78ef4e33df8e74a375f587951e2c330b5`.
- Local independent raw-byte/maps/ownership audit SHA256:
  `827a005b6fcfc5620d58179018bb9d8c276c3ac1047257f95b1dab0f73343313`.

`series-001/` then reused sixteen archived H20 official reference calls. Each
input binary equals the corresponding official prepared NPZ array. All 32 full
direct outputs match their own official reference bytes; the interpreter
receives a fresh revision for each frame, and explicit-noise calls leave CPU
and CUDA RNG unchanged. The observations are sixteen chronological frames from
one ALOHA episode, not sixteen episodes. The old official reports record H20
device type but not UUID; the current worker's UUID is recorded separately.
No official reference inference was repeated.

Series report SHA256:
`1963acf386d178e5b3c0dcb449678cd5710b726accd8dc376e8ca78aa24b285c`.
Derived protocol SHA256:
`22c8b0016a1d11f43dd41d593cbc2b768cc0bc2fb87712da12382b078a41205f`.
Its inherited model label contains `normalized`; the authoritative output
contract explicitly contains both ports. This is not a single-output protocol.

The next create-only stages use `run_pi05_formal_stage.py`: variants, prepare,
pilot, independent pilot audit, formal, report and independent formal audit.
Each stage records a separate receipt and requires the preceding completed
gate. An existing lock or failed directory must not be deleted to force a rerun.

## Reused Evidence

The existing H20 normalized native Session report remains unchanged:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/pi05-h20-20260908-001/pi05-h20-v2-native-session-20260908/report.json`,
SHA256 `0be86d351e2f02a2005ef945460b82c3d8d4b0cf5686f9e6e57633e9abdb3660`.
Its v2 validated capture SHA is
`b061e7a8421045110a3d1c1b3f352e0dbe56fd5484ebccf4f71686d1d43e34d4`.
No checkpoint conversion or full-model recapture was repeated.

The native source supplied the capture identity and complete numerical policy.
The public `openpi_output_capture` adapter used the recorded official processor
config, preserving the incoming acceptance predicate and checking the complete
normalized action tensor before cropping/scaling.

## Actual H20 Gates

Host `zzm-h20-x8-2`, GPU1 UUID `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`.
Controller PID 4082080, worker PID 4082135, registered NVML PID 148729.
The worker and controller exited 0. Owner registration/reset/registration and
sampled exclusion checks passed; 503 source-file hashes matched before/after.

| Path | Full native output | MSE | Max-abs | Cosine | Acceptance tests |
|---|---|---:|---:|---:|---|
| eager CUDA | F64 `[1,50,14]`, 700 values exact | 0 | 0 | 1 | false and nonfinite rejected |
| saved/reloaded CUDA | F64 `[1,50,14]`, 700 values exact | 0 | 0 | 1 | false and nonfinite rejected |
| AOTI CUDA | F64 `[1,50,14]`, 700 values exact | 0 | 0 | 1 | false and nonfinite rejected |

The original incoming predicate was not dropped. A nonfinite element outside
the final cropped action dimensions was rejected by all three paths. AOTI
payloads were materialized on NAS, retaining the package and manifest lineage;
this run does not require runtime deletion of mapped temporary libraries.

- Processor report SHA256:
  `5de13cf0fd2e7d5aff590d2e5331de86613eb1b7f7da6c4f977038bb24b38b6d`.
- AOTI package SHA256:
  `9b321e065b6cc2757a6d51a10e4e96aae1083f3105dd010ed4dd59cba1673c4d`.
- Local independent byte/payload audit SHA256:
  `92c96af5dfe79d051658c875159fcaaa4c36040eacc5930b47dc593b9d0c200a`.

## Continuation Paths

NAS root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/pi05-dual-recovery-20260909/`.

- `source/vlaforge/`: new frozen source snapshot; post-transfer checksum dry
  run found zero differing files. No old experiment snapshot was replaced.
- `processor-001/processor/`: completed output-stage capture, raw arrays,
  AOTI compiler report and materialized payloads.
- `processor-001/monitor/`: actual ownership handshake and worker logs.
- `dual-001/prepare.json`: passed CPU integrity/IR/Plan/typed renderer check;
  source artifacts verified, but no model execution or C++ build happened here.
- `dual-001/invocation_ir.json` and `runner.cpp`: prepared full dual-output ABI.

Local evidence is under
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`,
including `pi05-capture/`, `pi05-native/`, `pi05-output-processor/`,
`pi05-dual-prepare/` and `pi05-processor-independent-audit.json`.

The processor and all single-frame output-bundle phases above are complete;
do not relaunch them. Continue from the latest terminal campaign receipt and
never treat the three-call native test or sixteen-frame direct test as formal
off/batch/required statistics.
Orin/J6M remain deferred; no physical calibration or robot-task claim is made.
