# RDT Completed-Worker Resources

Checkpoint: 2026-10-22 04:51 UTC. **Only two completed workers in the new
authorized service window are covered. The fifteen-process RDT campaign,
whole-window resource acceptance and service restoration are not complete.**
No target collector, engine, precision, inputs/noise, power mode or sampling
protocol was changed. This is host processing of already archived evidence;
the next replay worker continues without a competing GPU job.

## Observed Envelopes

| Whole Worker | Conventional TRT | Shared Stream |
|---|---:|---:|
| Directory | `round-00-off` | `round-00-batch-only` |
| Wall duration including initialization/warmup (min) | 58.949 | 59.921 |
| Retained resource snapshots | 3177 | 3153 |
| Visible-owner scans | 106 | 106 |
| Admission MemAvailable (KiB) | 39488116 | 38777620 |
| Minimum sampled MemAvailable (GiB) | 23.893898 | 23.025414 |
| Maximum sampled process RSS (MiB) | 6464.949 | 6468.742 |
| Maximum sampled process VmHWM (MiB) | 6806.949 | 6810.781 |
| Sampled process VmSwap (KiB), min/max | 0 / 0 | 0 / 0 |
| Highest valid thermal reading (C) | 60.187 | 60.156 |
| Highest sampled GPU thermal reading (C) | 60.000 | 59.937 |
| Sampled GPU frequency, min/max (MHz) | 1300.5 / 1300.5 | 1300.5 / 1300.5 |
| Maximum inter-snapshot gap (s) | 3.499097 | 3.668708 |
| Gaps above 2 s | 106 | 106 |
| Maximum permission-denied process FD entries | 265 | 263 |

GiB is KiB / 1048576; MiB is KiB / 1024. CPU/GPU shared physical memory is
not summed. A process high-water value is not a private GPU allocation peak,
complete model footprint or sum of weights/workspace/device buffers. These
system availability values include all background activity and cache effects;
their between-worker difference is not a causal leak measurement.

All 6330 snapshots have strictly increasing monotonic timestamps inside their
worker's start/end interval, with the exact recorded sample count. Admission
exceeds the frozen 16 GiB threshold; every retained sampled MemAvailable value
meets the unchanged 2 GiB floor. This verifies sampled values, not availability
between snapshots. The requested one-second cadence is not an observed maximum
gap guarantee. The report retains both measured gaps and boundary coverage:
first-snapshot delay 0.007172/0.003025 s, final-snapshot-to-exit
0.999746/0.719050 s. Board civil time remains wrong; monotonic time is used.

Each window contains initialization, warmup and full requests. The stored CSV
has segment durations but no absolute per-request monotonic timestamp, so
resource samples are not assigned to exact measured requests or classified as
steady-state allocation evidence. No graph/heap event instrumentation, thermal
equilibrium criterion, sub-sample frequency guarantee or energy result follows.

## Coverage Limits

The existing ownership validator checks every visible worker PID/start/argv
and rejects any other visible process except the two existing `gnome-shell`
processes. Permission-denied FD counts remain nonzero; this does not prove
exclusive ownership of the whole device. The two explicitly authorized Java
services remain under the separately monitored guard, not newly controlled by
this offline audit.

All three CV temperature channels report invalid -256 C in every sample:
18990 invalid readings total. They are retained as invalid counts, never averaged
as temperatures. Other sensor names/values and every unreadable file count remain
in the JSON. Direct INA3221 paths are permission-denied; preserved `tegrastats`
text here has no usable power rail measurement. No power/energy aggregation,
reconstructed old samples, replacement sensor values or new daemon client was
introduced during the formal run.

All recorded files in the selected completed directories are rehashed against
their campaign snapshots. Their separate complete-output receipts bind the
same source, boot, frozen protocol, engines and launch request. Those receipts
already check all 4608 complete native/unified action tensors byte-exact. The
resource report references that quality evidence; it is not a new inference or
a replacement for the final full-model auditor.

## Evidence and Reproduction

Root: `artifacts/orin/20261022-01/`.

- `completed-prefix-001/`: only the completed conventional directory and its
  matching campaign/protocol snapshots.
- `completed-prefix-002/`: only the completed shared-stream directory and its
  matching snapshots; an active worker entry in metadata is not copied data.
- `completed-case-audit-001.json` and `completed-case-audit-002.json`: separate
  full-file and full-action checks; both reject formal-campaign acceptance.
- `completed-resource-request-001.json`: explicit selected directories and
  fixed helper, serial, boot and new launch-request bindings.
- `audit_completed_resources.py`: request-driven host-only composer, reusing
  existing `resource_floor` and `resource_summary` validators without changing
  either the original final auditor or the active target code.
- `completed-resource-audit-001.json`: two resource envelopes; formal,
  full-campaign, restoration and new-target-execution flags all false.
- `test_completed_resources.py` / `completed-resource-regression-001.xml`:
  nine passing tests, including real data and private-copy corruption, changed
  owner/floor, wrong boot, duplicate/active selection, missing-series and
  reversed-time checks. Original telemetry hashes remain unchanged.

After synchronizing these scoped results to the board, eight state/server and
nine desktop/tablet/mobile browser tests pass. Receipts under
`artifacts/orin-task-board/` are `completed-resource-node-tests-010.xml` and
`completed-resource-browser-tests-011.xml`; they are UI/evidence checks, not
new model or resource-window acceptance.

Use a fresh output path, preserving existing evidence:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-01/audit_completed_resources.py \
  --request artifacts/orin/20261022-01/completed-resource-request-001.json \
  --output artifacts/orin/20261022-01/NEW-completed-resource-audit.json
```

The terminal fifteen-process resource/data/restoration recipe remains in the
[managed audit runbook](orin_managed_campaign_audit_20261022.md). Current partial
resource envelopes do not fill RDT's main table/CDF or complete any remaining
model. Required formal-model completion remains 1/5.

## First Replay Follow-Up

By 05:44 UTC the new `round-00-required` finished all 128 warmups and 1024
measured calls. Only its completed directory and matching metadata were copied
to `completed-prefix-003`; the following shared worker's active files were not
copied. `completed-case-audit-003.json` checks every archived file hash and all
2304 complete action tensors byte-exact. Actual loop evidence is 1152 replays,
five captured steps and zero ordinary fallback. Three completed workers now
total 6912 independently checked full tensors, still only 3/15 processes.

| Replay Whole-Worker Resource Metric | Observed Value |
|---|---:|
| Worker wall duration (min) | 58.048353 |
| Resource snapshots | 3091 |
| Admission MemAvailable (KiB) | 38204592 |
| Minimum sampled system MemAvailable (GiB) | 21.811531 |
| Samples with process RSS/HWM fields | 3090 / 3091 |
| Maximum observed process RSS (MiB) | 6469.203125 |
| Maximum observed process VmHWM (MiB) | 6811.097656 |
| Highest valid thermal reading (C) | 60.437 |
| Highest sampled GPU thermal reading (C) | 60.406 |
| Maximum inter-snapshot gap (s) | 3.768484 |
| Visible-owner scans / maximum denied FD entries | 104 / 264 |

The **last** readable process-status record has no VmRSS/RssAnon/VmHWM/VmSwap
fields. Its recorded state is `R (running)`, at monotonic 129776154209216 ns;
the case finished at 129778482665376 ns. This is a missing observation, not
zero memory. Neither an exit/teardown cause nor a process-memory/SDK leak is
established from this record. The complete original status text and resource
row remain unchanged. System MemAvailable remains present and checked there.

The first extra envelope composer assumed every readable status had those
optional fields and rejected the replay row. Its original code is frozen as
`audit_completed_resources_v1.py`, matching the first report's source digest;
`completed-resource-audit-002-rejected.log` retains the reproduced exit-1 trace.
There is no accepted `completed-resource-audit-002.json`.

The current composer retains every row and records missing fields, exact
sample/time/state, per-counter observation counts and
`process_memory_fully_observed_at_snapshots=false`. It never fills absent values
with zero or infers their cause. Present but invalid units/values still reject.
No original full-data, sampled system-memory, numerical or count gate is
weakened: process envelope completeness is an explicit additional diagnostic,
not a replacement for the original guards.

`completed-resource-request-002.json` explicitly selects all three completed
archives. `completed-resource-audit-003.json` now checks all **9421** resource
rows and their hashes, including the missing-counter row. Original reports/
sources/logs remain present. `completed-resource-regression-002.xml` has **13
passing** checks, including the real replay record, missing-not-zero handling,
malformed-present-field rejection and all previous bad-floor/owner/hash cases.
The whole-campaign/formal/restoration flags remain false. RDT continues its
remaining twelve processes; no partial main-table/CDF result is issued.

## Fourth Completed Worker

By 06:44 UTC `round-01-batch-only` exited zero with its complete 128+1024 calls.
`completed-prefix-004` contains only that ended directory and matching metadata.
`completed-case-audit-004.json` checks another 2304 full tensors byte-exact and
all recorded hashes, totaling **9216 complete tensors / 4 of 15 workers**.
Its 1152 ordinary loop calls and zero captured replay steps are the expected
shared-stream mode, not a CUDA-graph fallback failure.

The fourth worker has 3173 resource rows over 59.434296 minutes. Minimum sampled
system availability is 21.354137 GiB; observed process RSS/HWM maxima are
6466.253906/6808.183594 MiB and maximum valid temperature is 60.781 C. Maximum
sample gap is 3.734267 s. Its one readable status row without process-memory
counters is disclosed by the existing missing-field logic, with no new code
or zero fill. Cause remains unestablished. Visibility remains permission-limited:
106 owner scans, maximum 263 denied process-FD entries.

Request `completed-resource-request-003.json` and result
`completed-resource-audit-004.json` cover all **12594** retained rows from four
archives, including **two** missing-process-field rows across the third/fourth
workers. The sampled system floor still passes at every retained row. Earlier
reports/sources/rejection logs remain; no rows or inactive original data are
removed. The active next replay worker is not included. Full model-data/CDF,
whole-window resources and actual service restoration still await completion.

## Fifth Completed Worker

By 07:43 UTC `round-01-required` completed all 128 warmups + 1024 measured
requests and exited zero. `completed-prefix-005` contains only this finished
directory and matching metadata. `completed-case-audit-005.json` verifies all
2304 full tensors byte-exact and every archived hash, totaling **11520 checked
tensors / 5 of 15 workers**. Actual replay counters are 1152 replays, five
captured steps and zero ordinary fallback.

This worker contributes 3091 resource rows over 58.135532 minutes. Minimum
sampled system availability is 20.480152 GiB. Observed process RSS/HWM maxima
are 6463.292969/6805.132813 MiB; highest valid thermal reading is 61.187 C.
Maximum inter-sample gap is 3.834254 s. All this worker's readable process
snapshots contain memory fields; there is no new missing-field row. Current-user
visibility remains limited: 104 owner scans, maximum 265 denied FD entries.

`completed-resource-request-004.json` and `completed-resource-audit-005.json`
cover all **15685** retained resource rows from five completed directories.
The prior **two** missing-process-memory observations remain explicit, not
deleted/zero-filled; all sampled system floors still pass. No peak, leak-free,
continuous-availability or power claim follows. The next active conventional
worker is excluded. Total measured 5120 across these modes is not the required
5120 per mode. Full formal data/CDF and actual restoration are still pending.

## Sixth Completed Worker

By 08:42 UTC `round-01-off` completed all 128 warmups + 1024 measured calls and
exited zero. Only that finished directory and matching metadata were archived
to `completed-prefix-006`. `completed-case-audit-006.json` independently checks
another 2304 complete tensors byte-exact and all file hashes. Ordinary-mode
stderr has no unexpected REPLAY records. Cumulative completion is **13824 full
tensors / 6 of 15 workers**, two independent processes per mode, not five.

This worker contributes 3139 resource rows over 59.337852 minutes. Minimum
sampled system availability is 20.104565 GiB; observed RSS/HWM maxima are
6468.996094/6811.164063 MiB and highest valid thermal reading is 61.093 C.
Maximum inter-sample gap is 3.926411 s. No new missing-process-memory row is
present. Current-user ownership remains incomplete: 105 scans, at most 261
permission-denied FD entries.

Request `completed-resource-request-005.json` and result
`completed-resource-audit-006.json` cover all **18824** retained rows across six
finished archives. Sampled system floors pass; the old **two** missing process
observations remain explicit and unchanged. Active replay is excluded. Per-mode
2048 measured requests is not 5120; no partial main table/CDF, full-window
resource acceptance, isolated GPU peak, leak freedom or restoration is claimed.

## Seventh Completed Worker

By 09:40 UTC `round-02-required` completed all 128 warmups + 1024 measured calls
and exited zero. `completed-prefix-007` contains only this ended directory and
matching metadata. `completed-case-audit-007.json` verifies all 2304 additional
full tensors byte-exact and all archived hashes, totaling **16128 tensors / 7
of 15 workers**. Actual counters are 1152 replays, five steps, ordinary zero.

The worker contributes 3001 resource rows over 57.251379 minutes. Minimum
sampled system availability is 19.255554 GiB; observed process RSS/HWM maxima
are 6803.250000/6803.503906 MiB, highest valid thermal reading 61.875 C and
maximum sampling gap 4.093599 s. No new process-memory fields are missing.
Owner visibility remains partial: 101 scans, at most 265 denied FD entries.

`completed-resource-request-006.json` and `completed-resource-audit-007.json`
check all **21825** retained resource rows from seven finished directories.
Sampled system floors pass and the previous **two** missing process observations
remain explicit. Replay currently has three processes/3072 measured; other
modes two/2048. None has five/5120. The active next conventional worker is
excluded; no formal main table/CDF, complete peak accounting, leak freedom,
whole-window resource acceptance or actual restoration is claimed.

## Eighth Completed Worker

By 10:40 UTC `round-02-off` completed all 128 warmups + 1024 measured calls
and exited zero. Only its finished directory and matching metadata were copied
to `completed-prefix-008`. `completed-case-audit-008.json` checks another 2304
complete tensors byte-exact and all file hashes, totaling **18432 tensors / 8
of 15 workers**. Ordinary-mode stderr has no unexpected REPLAY records.

This worker contributes 3140 resource rows over 59.728931 minutes. Minimum
sampled system availability is 19.713200 GiB; observed process RSS/HWM maxima
are 6803.824219/6805.289063 MiB, highest valid thermal reading 61.093 C, and
maximum sampling gap 4.181037 s. There is no new missing-process-memory row.
Owner visibility remains limited: 105 scans, at most 268 denied FD entries.

Request `completed-resource-request-007.json` and result
`completed-resource-audit-008.json` check **24965** retained resource rows from
eight finished directories. All sampled system floors pass; the previous
**two** missing process observations remain explicit. Conventional/replay have
three processes/3072 measured, shared mode two/2048, not five/5120. The active
shared worker is excluded; no formal data/CDF, full resource coverage, isolated
GPU peak, leak freedom or actual restoration is claimed.

## Ninth Completed Worker

By 11:39 UTC `round-02-batch-only` exited zero after all 128 warmups + 1024
measured calls. Only this finished directory and matching metadata were copied
to `completed-prefix-009`. `completed-case-audit-009.json` checks another 2304
complete tensors byte-exact and all archived hashes, totaling **20736 tensors /
9 of 15 workers**. Its 1152 ordinary calls are the intended shared-mode path,
not a replay fallback failure.

The worker contributes 3043 resource rows over 58.603479 minutes. Minimum
sampled system availability is 18.966969 GiB; observed RSS/HWM maxima are
6455.808594/6798.000000 MiB, highest valid thermal reading 61.281 C and maximum
sample gap 4.225057 s. No new process-memory fields are missing. Owner coverage
remains incomplete: 102 scans, at most 263 denied FD entries.

Request `completed-resource-request-008.json` and result
`completed-resource-audit-009.json` check all **28008** retained resource rows
from nine finished directories. All sampled system floors pass; the previous
**two** missing process observations stay explicit. Every mode has three
processes/3072 measured, not five/5120. The next active conventional worker is
excluded; formal data/CDF, full resource coverage and restoration remain pending.

## Tenth Completed Worker

By 12:38 UTC `round-03-off` exited zero after all 128 warmups + 1024 measured
calls. Only this finished directory and matching metadata were copied to
`completed-prefix-010`. `completed-case-audit-010.json` verifies another 2304
complete tensors byte-exact and all archived hashes, totaling **23040 tensors /
10 of 15 workers**. Ordinary-mode stderr has no unexpected REPLAY records.

This worker contributes 3043 resource rows over 59.289331 minutes. Minimum
sampled system availability is 20.595989 GiB; observed RSS/HWM maxima are
6805.437500/6806.957031 MiB, highest valid thermal reading 61.156 C and maximum
sample gap 4.337734 s. No new process-memory fields are missing. Owner coverage
remains incomplete: 102 scans, at most 267 denied FD entries.

Request `completed-resource-request-009.json` and result
`completed-resource-audit-010.json` check all **31051** retained resource rows
from ten finished directories. All sampled system floors pass; the previous
**two** missing process observations stay explicit. Conventional now has four
processes/4096 measured; shared/replay each have three/3072, not five/5120.
The next active shared worker is excluded. These partial checks do not certify
formal data/CDF, full-window resources, private GPU peaks, leak freedom or
actual service restoration. The original collector, engines, inputs/noise,
precision, power mode and full sampling protocol are unchanged.

## Eleventh Completed Worker

By 13:37 UTC `round-03-batch-only` exited zero after all 128 warmups + 1024
measured calls. Only this finished directory and matching metadata were copied
to `completed-prefix-011`. `completed-case-audit-011.json` verifies another
2304 complete tensors byte-exact and all archived hashes, totaling **25344
tensors / 11 of 15 workers**. Ordinary=1152 is the intended shared-mode path.

The worker contributes 3022 resource rows over 58.699843 minutes. Minimum
sampled system availability is 20.084621 GiB; observed RSS/HWM maxima are
6457.652344/6799.734375 MiB, highest valid thermal reading 61.375 C and maximum
sample gap 4.256973 s. It adds no missing process-memory row. Owner coverage
remains permission-limited: 101 scans, at most 267 denied FD entries.

Request `completed-resource-request-010.json` and result
`completed-resource-audit-011.json` check all **34073** retained resource rows
from eleven finished directories. All sampled system floors pass; the previous
**two** missing process observations stay explicit. Conventional/shared modes
each have four processes/4096 measured; replay has three/3072, not five/5120.
The next active replay worker is excluded. Whole-campaign data/CDF, full resource
coverage and actual restoration remain pending; no target work or protocol
change was introduced by the parallel host preparation.

## Twelfth Completed Worker

By 14:35 UTC `round-03-required` exited zero after all 128 warmups + 1024
measured calls. Only this finished directory and matching metadata were copied
to `completed-prefix-012`. `completed-case-audit-012.json` checks another 2304
complete tensors byte-exact and all archived hashes, totaling **27648 tensors /
12 of 15 workers**. Actual replay is 1152 calls, five captured steps, and zero
ordinary fallback.

The worker contributes 2935 resource rows over 57.408864 minutes. Minimum
sampled system availability is 19.623272 GiB; observed RSS/HWM maxima are
6804.425781/6805.625000 MiB, highest valid thermal reading 61.406 C and maximum
sample gap 4.478427 s. No new process-memory fields are missing. Owner coverage
remains permission-limited: 98 scans, at most 263 denied FD entries.

Request `completed-resource-request-011.json` and result
`completed-resource-audit-012.json` cover all **37008** retained resource rows
from twelve finished directories. All sampled system floors pass; the previous
**two** missing process observations stay explicit. Every mode has four
processes/4096 measured, not five/5120. The next active shared worker is excluded.
Formal data/CDF, whole-window resources and actual restoration remain pending;
no target computation, precision, noise or protocol was changed.

## Thirteenth Completed Worker

By 15:35 UTC `round-04-batch-only` exited zero after all 128 warmups + 1024
measured calls. Only this finished directory and matching metadata were copied
to `completed-prefix-013`. `completed-case-audit-013.json` checks another 2304
complete tensors byte-exact and all hashes: **29952 tensors / 13 of 15 workers**.
Ordinary=1152 is the intended shared mode, not replay fallback.

This worker contributes 3106 resource rows over 59.673888 minutes. Minimum
sampled system availability is 19.224525 GiB; observed RSS/HWM maxima are
6464.691406/6806.609375 MiB, maximum valid temperature 60.562 C, maximum sample
gap 4.615084 s. It adds no missing-process-counter observation. Owner visibility
remains limited: 104 scans, at most 264 denied FD entries.

Request `completed-resource-request-012.json` and result
`completed-resource-audit-013.json` cover **40114** retained rows from thirteen
finished directories. All sampled system floors pass; the two old missing
process observations remain. Shared mode now has five processes/5120 measured;
the others have four/4096. The next active replay worker is excluded, and the
complete fifteen-process/CDF/restoration gates remain unchanged.

## Fourteenth Completed Worker

By 16:32 UTC `round-04-required` exited zero after all 128 warmups + 1024
measured calls. Only its finished directory and matching metadata were copied
to `completed-prefix-014`. `completed-case-audit-014.json` independently checks
2304 more complete tensors/all hashes: **32256 tensors / 14 of 15 workers**.
Actual replay is 1152 calls, five captured steps, zero ordinary fallback.

This worker adds 2956 resource rows over 57.331913 minutes. Minimum sampled
system availability is 20.146603 GiB; observed RSS/HWM maxima are
6795.933594/6797.902344 MiB, maximum valid temperature 60.375 C, and maximum
sample gap 4.582249 s. It adds no missing-process observation. Owner visibility
remains limited: 99 scans, at most 265 denied FD entries.

Request `completed-resource-request-013.json` and result
`completed-resource-audit-014.json` cover **43070** retained rows from fourteen
finished directories. Shared/replay have five processes/5120 measured each;
the final conventional worker is active and excluded. The two previous missing
observations and global 4.615084 s maximum gap remain. Sampled system floors
pass, but formal data/CDF, full resources and restoration are still pending.
