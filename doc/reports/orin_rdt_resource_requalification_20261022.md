# RDT Authorized Service Window and Requalification

Checkpoint: 2026-10-22. Fresh pilot qualification has passed and a new formal
campaign is running, **not** a completed RDT formal result. The five-model Goal is
unchanged. Earlier failed/partial RDT evidence and Norm negative results remain.

## Authorization and Scope

The user explicitly authorized checking management/recovery and temporarily
stopping exactly `data-collection.jar` and
`xs-ugv-starter-0.0.1-SNAPSHOT.jar` during RDT formal sampling. The manager is
the existing same-user Supervisor PID 4237, start ticks 27537. Only the two
named programs are controlled through its loopback XML-RPC endpoint. The
desktop user manager, Supervisor itself and other programs are not stopped.
No sudo access, configuration editing or permanent autostart change is used.

Configuration SHA256 remains
`6fe6a951d5bd7bc461afa39bb1e13eb8a2428ede94424d9d3df97c0225a9a6c0`.
JAR SHA256 values remain:

- Starter: `ea54c2c7a19951b7570df6f83f9cdc6664c190e6cfecf4003559081c774d8aa8`.
- Data collection: `2ef6f0e926e69c7c7523b2925a974b1e70e1c45c61de0c6757fe2f2b0ef06748`.

Actual PID/start/argv, parent identity, JARs and configuration are checked
before each controlled stop. The installed manager reports successful manual
stop as **PID 0 / Exited (100)**, not Stopped (0). The first stricter assertion
failed and automatically restored both programs; no other managed program
changed. Its failed/restored lease is retained. A new inspection bound the
newly assigned PIDs before the second attempt. The revised check requires zero
PID and a non-running terminal state (0 or 100); Starting, Running, Stopping
and Fatal are rejected. The second lease confirms both selected programs are
Exited/PID 0 and every other managed program's PID/state is unchanged.

## Resources

| Read-Only Snapshot | System MemAvailable (KiB) | SwapFree (KiB) |
|---|---:|---:|
| Before authorization, 02:19 UTC | 12056156 | 0 |
| Two programs stopped, 02:31 UTC | 45602956 | 29874976 |

These are non-atomic system observations, not additive process RSS, a precise
model footprint or a proven service/model leak. Serial remains 1421524312594,
boot `3c801f5d-6897-4d97-b021-3c52db38a750`; visible GPU users are the two
existing desktop processes, with restricted-process coverage explicitly limited.
Existing SDK/MAXN/clocks are unchanged. Future resources still require monitoring.

## Restoration Ownership

A durable independent guard owns lease 002. It monitors that only the two
authorized programs remain stopped while waiting for a hash-bound qualified
formal activation. If no activation arrives within 3600 seconds, it restores
them. After the formal collector exits, fails or is interrupted, it terminates
only its owned collector process group and restores the two programs through
the same manager. It checks configuration/JAR hashes and actual restarted
commands, and records any restoration failure rather than claiming success.
It does not stop or restart unrelated programs to force equality.

Restoration cannot be guaranteed after an unhandled SIGKILL/kernel or power
failure; the manual manager commands below remain available. Permanent startup
configuration is untouched. Do not restore while the lease-owned inference
worker is still running. The guard's PID/start/argv must be checked when resuming.

```bash
ssh orin /data/LogisticsServiceMaster/start/supervisord-xs \
  -c /data/LogisticsServiceMaster/start/supervisor_Normal.conf \
  ctl -s http://127.0.0.1:9001 start \
  xs-ugv-starter-0.0.1-SNAPSHOT.jar data-collection.jar
```

## New Model Run Gates

Fresh three-mode raw pilots use the exact frozen eleven engines, rounding DSO,
raw observations/noise, horizon 64 and five solver steps. Each mode retains
4 warmup + 16 measured full calls. The native runner still checks both full
same-Orin action outputs; an independent audit must check every output and
actual replay count before formal activation. No Torch, H20 timing, precision
search, new action dimensions or stage-only substitution is introduced.

The new formal campaign starts **all fifteen processes anew**, five per
mode, 128 warmup + 1024 measured each. Old eleven complete processes and the
partial twelfth remain a separate interrupted dataset and are never merged.
New operational guards require >16 GiB admission and stop the owned worker
below 2 GiB sampled system availability; these do not change numerical gates,
sample counts or latency measurements. The parent service guard restores on
collector exit. Pilot estimates do not certify the final main table or CDF.

Fresh pilot independent audit: all 120 complete tensors pass byte equality and
the unchanged same-Orin numerical gate. Real replay evidence is 20 calls, five
captured steps, zero ordinary fallback. Ordinary/shared/replay pilot means are
3007.947554/3082.802890/2986.883486 ms; shared mode is slower and is retained.
These remain short pilots, not formal table values or a causal resource study.

Formal activation occurred at about 02:45 UTC. Collector PID/start ticks are
2258939/11907296; first ordinary worker 2260072/11912672. The board refresh
checked both actual identities, not only saved lock/launch files. The frozen
new request SHA256 is
`fbff4ad030659b5136b4b630df7548fceb1c1497e58f01928fb8e33fbea4d976`.
Pilot-derived pure sampling is 14.524214 hours; initialization, hash checks,
resource variation and audit add time. Keep running/accepted states distinct.

The immutable `service-window-freeze-001.json` preserves the initial window;
the mutable target `rdt-service-lease-002.json` will record later restoration.
Their initial SHA256 is
`50db2e183ed4222f5a9510677104fbd009136a228319638c37c681515a2413de`.
Do not require the mutable post-restoration receipt to equal its initial hash.

## Evidence

Root: `artifacts/orin/20261022-01/`.

| Evidence | File |
|---|---|
| Direct user permission, exact two-program scope | `service_authorization.json` |
| Pre-change read-only environment/process observation | `readiness-001.json` |
| Management, PID/start/argv, unchanged config/JAR hashes | `services-inspection-001.json`, `services-inspection-002.json` |
| First protected failure and automatic restoration | `rdt-service-lease-001.json` |
| Successful exact-program temporary stop | `rdt-service-lease-002.json` |
| Post-stop environment | `readiness-after-services-002.json` |
| Guard launch request | `service-guard-launch-request-001.json` |
| Fresh three-mode pilot launch request | `rdt-pilot-launch-request-001.json` |
| Fake-manager scope/state/restore checks, no real signals | `service-control-regression-002.xml`, 22 passed |
| Fresh pilot complete-output/replay audit | `rdt-pilot-audit-001.json` |
| Immutable window and formal activation/request | `service-window-freeze-001.json`, `formal-request-001.json`, `qualified-formal-activation-001.json` |
| Active guard and first formal worker checkpoints | `service-guard-checkpoint-001.json`, `formal-preflight-checkpoint-002.json` |
| Window/failure/cleanup/fresh model contract regressions | `managed-campaign-regression-004.xml`, 36 passed |

Target guard data:
`/data/datalog/edgefm-orin/20261022-01/service-guard-001/guard.json`.
Target fresh-pilot data:
`/data/datalog/edgefm-orin/20261022-01/rdt-raw-pilots-001/`.
Current facts and subsequent results are synchronized to the
[task board](http://127.0.0.1:4317); this checkpoint is not formal acceptance.
No other target GPU experiment is launched while this collector owns the window.
At 02:53 UTC the first ordinary worker completed all 128 warmups and wrote its
first measured rows. Actual controller/worker identities still match; native
quality flags are not substituted for the eventual complete independent audit.
Six board state/server and nine browser checks pass at this running checkpoint.

At 02:56 UTC, guard 2242355, collector 2258939 and worker 2260072 all match
their actual PID/start/argv. Guard heartbeat age was 6.43 seconds, both selected
programs remain PID0/Exited, and the first worker has 128 warmup + 68 measured
rows written; 0/15 full workers are complete. These live checks are a verified
wait, not formal acceptance. Latest retained guard source is
`service-guard-checkpoint-002.json`; dynamic counters use the board checkpoint.

For a new one-shot read-only progress/identity check, run:

```bash
node tools/orin-task-board/refresh-run.mjs
```

That command reads the active new campaign path from the board. It does not
restart experiments, restore services or initiate background remote polling.
Never relaunch solely because an observation times out. Check the actual guard,
collector and current worker identity before any control action.
