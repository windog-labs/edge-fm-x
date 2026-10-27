# RDT Resource Observation

Status: bounded read-only observation, **not a performance audit, model-leak
diagnosis or exclusive-platform attestation**. The formal collector continues
with its frozen protocol. No inference, service stop, power/clock change or
model rebuild was started by this observation.

Latest follow-up at 05:27: MemAvailable is about 1.19 GiB and swap is full;
the current RDT worker still has VmSwap zero. Memory-exhaustion risk is higher.
All three observations and their limits are retained below.

## Source

Captured at 2026-10-22 03:24:48--03:24:50 Asia/Shanghai, on Orin serial
`1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`.
Current worker PID/start ticks: `1726522` / `9008251`, third-round shared stream.
Checkpoint source was `checkpoint-20261021192007187.json`; later read-only
checks confirm the controller and worker remained live and sampling advanced.

Evidence root:
`artifacts/orin/20261021-07/resource-observation-20261022-001/`.
`snapshot.json` binds the checkpoint, capture source, exact SSH read commands,
raw bytes/hashes, retained line bounds and independently sampled observations.
The tool only reads existing telemetry tails, a small process listing and
the two selected existing Java process statuses. It does not run GPU-owner
queries or a competing benchmark on the target.

## Memory Snapshot

| Counter | Earliest Completed Worker's Last Telemetry | Current Shared-Stream Telemetry |
|---|---:|---:|
| MemAvailable | 10417084 KiB, about 9.93 GiB | 3252168 KiB, about 3.10 GiB |
| SwapFree | 7360528 KiB, about 7.02 GiB | 12 KiB, effectively full |
| KernelStack | 437664 KiB | 610112 KiB |
| AnonPages | 33874876 KiB | 40557984 KiB |

Those observations are about 27262 seconds apart. They are different workers,
modes and points in execution, not a matched-phase leak experiment. The change
establishes an environmental resource concern; it does not identify a causal
allocator or prove monotonic growth throughout the campaign.

Current RDT process: VmRSS `4004544 KiB`, VmHWM `6953012 KiB`, VmSwap `0 KiB`.
HWM includes startup and is not an inference-only steady-state peak. Zero
worker VmSwap does not prove absence of driver/UVM allocations, system-pressure
effects or latency interference. No CPU/GPU/shared-memory values are summed.

## Existing Platform Services

| PID | Existing Process | RSS (KiB) | VmSwap (KiB) | Threads in PS Snapshot |
|---|---|---:|---:|---:|
| 4287 | `data-collection.jar` | 13371636 | 14840480 | 18379 |
| 4286 | `xs-ugv-starter-0.0.1-SNAPSHOT.jar` | 12994304 | 14898672 | 18361 |

The current RDT worker has two threads in the same PS snapshot. These services
are not the experiment controller/worker and were not terminated. Their
individual RSS values include shared pages and must not be added as a claimed
physical-memory total. A single per-process snapshot does not establish a
Java leak, a worker leak or the cause/size of a timing change. Service retention
requirements for subsequent experiments were asked asynchronously; the current
RDT campaign's environment is not changed in anticipation of an answer.

## One-Hour Follow-Up

The same read-only tool captured
`artifacts/orin/20261021-07/resource-observation-20261022-002/` at
2026-10-22 04:25:01--04:25:03 Asia/Shanghai. Serial, boot and capture-source hash
match the earlier observation; current worker is the next conventional-mode
process, PID/start ticks `1791145` / `9365687`, not the earlier shared-stream
worker. Checkpoint `checkpoint-20261021202454598.json` was freshly verified
before capture and subsequent polling still matches the live worker.

| Counter (KiB) | 03:24 | 04:25 |
|---|---:|---:|
| System MemAvailable | 3252168 | 2504768 |
| System SwapFree | 12 | 13428 |
| System KernelStack | 610112 | 632896 |
| Current worker VmRSS | 4004544 | 3994900 |
| Current worker VmSwap | 0 | 0 |
| PID 4287 RSS | 13371636 | 13589384 |
| PID 4286 RSS | 12994304 | 13171340 |
| PID 4287 VmSwap | 14840480 | 14857620 |
| PID 4286 VmSwap | 14898672 | 14914812 |

The two selected Java processes' PS thread counts change from 18379/18361
to 19092/19073. Available system memory falls by about 0.71 GiB between these
observations, while swap remains nearly full and each observed RDT worker has
VmSwap zero. Different RDT processes/modes and non-atomic samples still prevent
causal allocation attribution; these data do not by themselves diagnose a
Java/model leak or quantify a latency effect. No service or sampler setting
was changed, and the service-retention question remains pending.

Ownership visibility in the second snapshot has 259 permission-denied entries,
again exposing only the current worker and the same two desktop processes.
This is still not an exclusive-owner attestation. The first snapshot/raw hashes
are retained unchanged. `readonly-resource-regression-r2.xml`: **8 passed**,
extending unit/provenance checks to both actual snapshots and their differing
worker identities. The original six-test XML is preserved.

## 05:27 Follow-Up

Third capture: `resource-observation-20261022-003/`, using the same tool,
serial and boot at 2026-10-22 05:27:04--05:27:06 Asia/Shanghai. Worker
1854657/9720131 is the next shared-stream process; the sampler and existing
services are unchanged. A fresh matching checkpoint preceded the bounded reads.

| Counter (KiB Unless Noted) | 04:25 | 05:27 |
|---|---:|---:|
| System MemAvailable | 2504768 | 1243536 |
| System SwapFree | 13428 | 0 |
| System KernelStack | 632896 | 656288 |
| Current worker VmRSS | 3994900 | 3990200 |
| Current worker VmSwap | 0 | 0 |
| PID 4287 RSS | 13589384 | 14021776 |
| PID 4286 RSS | 13171340 | 13618324 |
| PID 4287 PS thread count | 19092 | 19825 |
| PID 4286 PS thread count | 19073 | 19807 |

These observations establish continuing system pressure and increasing observed
RSS/thread counts in the selected existing services. They still do not prove
the causal allocator, a service/model leak or the timing impact. The RDT samples
come from separate worker processes and are not a same-worker leak test.
Low MemAvailable and full swap create a memory-exhaustion risk even though the
current worker's VmSwap is zero; zero does not certify safe future allocation
or absence of global-pressure effects. No service, memory policy or sampler
setting was altered. Service retention for later experiments remains pending.

The owner sample reports 260 permission-denied entries and the same limited
visibility boundary. `readonly-resource-regression-r3.xml`: **9 passed**,
covering all three frozen snapshots, their raw hashes/units and separate worker
identities. Earlier six/eight-test receipts are unchanged. Resource trajectories
and full outputs still require post-collection audit; this is not acceptance
or rejection of model quality or measured latency.

## Visibility And Reporting

The last retained ownership sample exposes the current worker and two
`gnome-shell` processes. It reports **260 permission-denied process entries**
and explicitly limits coverage to current-user device-FD visibility. An earlier
read in this check reported 263; this coverage count varies over time. No full
GPU exclusivity claim is supported by these observations.

Available telemetry/tegrastats and their original invalid/permission-denied
sensor fields are preserved. Target civil-clock text remains incorrect and is
not used as the experiment clock; host capture times and monotonic telemetry
timestamps are retained.

Formal latency/throughput/CDF interpretation must inspect complete per-process
resource trajectories, ownership limitations and mode/round order. Preserve
all measured rows, including pressure-associated slow rows; do not trim them,
center means or assert that memory pressure has no effect. A complete-action
quality pass alone does not resolve this environmental comparability question.
This observation does not by itself reject or certify the campaign.

## Reproduction And Checks

```bash
PYTHONNOUSERSITE=1 python artifacts/orin/20261021-07/capture_readonly_resources.py \
  --checkpoint artifacts/orin-task-board/checkpoint-20261021192007187.json \
  --extra-pid 4287 --extra-pid 4286 \
  --output artifacts/orin/20261021-07/resource-observation-new
```

This command takes a new read-only observation; its numbers need not reproduce
an older live snapshot. It rejects an existing output directory and never
restarts the model or changes services. Refresh/check live identities before
using an old checkpoint/PID; recorded snapshots are not perpetual liveness.

`readonly-resource-regression.xml`: **6 passed** for strict KiB/zero parsing,
invalid-unit rejection, raw-file hashes, bounded read commands and the recorded
coverage limit. These are host provenance/parser checks, not new model quality
or timing results. See [Goal status](../edgefm_orin_goal_status.md) for the
unchanged five-model scope and current formal progress.

## Board Verification

The new resource description legitimately adds the common memory task to
search results for RDT. An old browser assertion hardcoded three matches;
its failure (one failed, six passed) is preserved as
`artifacts/orin-task-board/browser-tests-resource-warning-failed-001.xml`, with
trace/context under `resource-warning-browser-failed-001/`. Search behavior
and real task content were not changed to force that count.

The workflow test now uses isolated client-side fixtures for summary/remaining
text matches and explicitly includes common tasks. All seven browser tests
pass, including new risk/evidence-page overflow checks and screenshots at
desktop, tablet, 390px and 320px widths. Six Node state/service checks also
pass. These UI checks never overwrite production task state or Orin data and
are not experimental performance/quality evidence.
