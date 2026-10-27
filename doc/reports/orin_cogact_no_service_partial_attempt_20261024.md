# CogACT No-Interruption Partial Formal Attempt

Checkpoint: 2026-10-23 21:48UTC. **One complete worker is independently verified;
the full campaign fails next-process memory admission and is not accepted.**
All partial data, failed state, telemetry and cleanup evidence remain intact.
No application is stopped/started, no memory threshold is lowered and no
automatic retry or partial-data merge is performed. Formal VLA coverage stays4/5.

## Actual Completed Scope

The first ordinary-source worker completes128warmup+1024measured=1152requests,
with allfive public outputs checked independently:5,760complete tensors,
including warmup, are byte-exact to the unchanged same-Orin reference. Worst
normalizedMSE0/minimumcosine1and native supplementalMSE0. F32/F64outputs are
finite; full U8RNG and I64counter outputs are exact bytes, never float-cast.
Same7.63Bpublic-dependency model,12TRT/four native providers, full prefix and
allten action iterations remain. No nominal3B/source-BF16/robot-equivalence claim.

This is **1/10required independent processes,1024/10240measured**, not the full
two-mode five-round dataset. No Arena process starts. Both the whole-model
and `--require-complete` resource auditors reject full-campaign acceptance.
The completed ordinary process is not silently reused to fill a later attempt.

## Explicit Failure And Resource Evidence

The collector exits with `formal memory admission failed` before creating the
next case or native worker. The unchanged admission is25,165,824KiB(24GiB),
runtime floor4,194,304KiB(4GiB). This is not a numerical/unsupported-operator
failure, a runtime-floor violation or a monitoring timeout.

| Recorded Point | MemAvailable (KiB) |
|---|---:|
| First worker environment admission |26,731,444|
| First memory guard admission |26,718,516|
| Minimum worker telemetry sample |6,825,916|
| Minimum memory-guard sample |6,535,832|
| Environment after first worker |19,523,884|
| Later read-only observations |19,400,432and19,234,916|

The rejected next-admission check does not save its exact failing sample before
raising. Its explicit error and threshold are retained; nearby observations
are not substituted for that missing instant.673worker telemetry samples and
3379guard samples pass their actual floor. Maximum gaps are7.88964688sfor
telemetry and0.253238208sfor guard data. Missing process/sensor values and owner/
power permission gaps are preserved, not filled with zero or treated as coverage.

Sampled workerVmRSSmax is4,219,924KiBandVmHWM4,428,160KiBincluding initialization
and warmup; VmSwap samples are0. GPU thermal samples reach57.531C. These are
observed process/system envelopes, not an isolated total physical GPU peak or
calibrated request energy. CPU/GPU unified memory is not summed. Changes in
global memory/cache or RSS do not prove a model/driver leak or identify a
particular application's responsibility.

The two Java processes show combined RSS around3.6GiBafterward. RSS is not
guaranteed releasable physical memory, and stopping only those two is not
established to restore stable24GiBadmission. The follow-up question asks for
a stable free-resource window or permission to assess an explicit additional
interruption scope; assessment is not permission to execute a stop. No extra
daemon is controlled and the expired six-hour scope is not reused.

## Cleanup And Data Integrity

The collector's guard/tegrastats cleanup is terminal. The owned compute unit
fails with MainPID0and its post hook observes an empty cgroup. Final read-only
closure checks five monitor/collector/guard/native/tegrastats birth identities
terminal and rehashes5,788target paths. Original Java PIDs942765/942805remain
Running under original Supervisor4237; the monitor's read-only observations
preserve births, argv and cwd. No application management RPC is present in this
path. This is not application-health or recovery-of-in-memory-state evidence.

Closure covers this attempt's files, binaries, protocol and output; it does
not claim to re-audit every external SDK/shared-engine payload. The earlier
shared-resource audit remains separately pinned. All partial data are kept.
Eight actual/archive/source-inverse/no-application-control tests pass. The
resource auditor's explicit declared-bound extension has a whole-file inverse
back to the unchanged source35auditor, retaining all missing/sensor/owner checks.

The1024measured ordinary records are available for diagnosis, but no whole-model
formal latency/CDF, paired speedup or five-model completion is published. The
background is Java-on/user-cgroup, not idle/isolated and not identical to older
Java-stopped model campaigns. All subsequent work must revalidate authoritative
resource/process state and use a fresh explicitly frozen attempt, never infer
a restart from a timeout or remove slow/failed rows.

## Evidence

- [Terminal partial-attempt closure](../../artifacts/orin/20261023-109/partial-attempt-closure-001.json).
- [All5760complete output tensors](../../artifacts/orin/20261023-109/partial-model-audit-001.json).
- [Single-worker resource audit](../../artifacts/orin/20261023-109/partial-resource-audit-001.json).
- [Failed next-admission snapshot](../../artifacts/orin/20261023-109/checkpoint-003.json).
- [Empty owned-cgroup post hook](../../artifacts/orin/20261023-109/unit-post-001.json).
- [Eight source/actual-partial checks](../../artifacts/orin/20261023-109/partial-attempt-tests-002.xml).
- [Original full no-interruption plan](orin_cogact_formal_no_application_control_20261024.md).

Recheck the archived first process without target computation:

```sh
env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=artifacts/orin/20261023-109 python3 -B -m pytest -q \
  -p no:cacheprovider artifacts/orin/20261023-109/test_partial_attempt.py
```
