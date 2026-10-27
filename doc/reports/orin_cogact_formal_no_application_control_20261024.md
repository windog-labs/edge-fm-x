# CogACT Formal Collection With Applications Running

Terminal update:[the partial attempt](orin_cogact_no_service_partial_attempt_20261024.md)
ends after the first complete worker because next-process24GiBadmission fails.
All5,760first-worker outputs and resources independently pass;1/10processes is
not accepted as the full cohort. The following running checkpoint is historical.

Checkpoint: 2026-10-23 21:21UTC. **The full formal cohort is running, not
accepted.** At this verified snapshot:0/10complete processes, first ordinary
process609calls (128warmup+481measured). All live worker-declared quality gates
pass so far, but independent complete-output/resource/terminal audits remain.
Required formal model acceptance stays4/5.

## Permission And Route Correction

The user-provided active Goal requests formal native Orin inference. The pending
extra permission concerns temporarily stopping the two Java applications.
Earlier readiness drafts applied that service-specific gate too broadly to a
route that would keep applications running. This run explicitly uses the
already requested **no-application-interruption** route; it does not fabricate
a new human answer, active service lease or formal outage authorization.

Fresh live checks now find enough memory without stopping anything:26,969,636KiB
available at21:00and26,792,068KiBat21:10, above the unchanged24GiBadmission.
Only desktop gnome-shell GPU holders are visible.261permission gaps prevent an
exclusive-owner attestation; they are not hidden. Both original Java processes
are Running. If required protections fail, only this experiment is stopped:
there is no automatic switch to an unapproved application-outage route.

`services_stopped=false`, `service_control="none"` and
`original_application_control_permitted=false` are explicit in the real spec.
The original six-hour build/pilot authority is expired and unused. Separate
permission remains mandatory for any future original-application stop/start.
No new human approval or service recovery is claimed for this run.

## Unchanged Complete Experiment

- Actual approximately7.63Bpublic-dependency CogACT identity, not nominal3B.
- Same twelve TRT engines/four typed providers and exact shared resource pool.
- Fresh complete prefix and allten denoising iterations, unchanged inputs,
  explicit noise/RNG tape, precision profile and normalized-primary policy.
- Two modes:ordinary-source and verified-arena, differing only in Arena layout;
  no full-iteration replay is invented.
- Five independent processes per mode, serial rotated rounds,128warmup+
  1024measured,batch1:11,520calls/10,240measured/57,600complete tensors.
-24GiBstart/4GiBruntime floor. No full-FP32fallback, quantization search,
  trimming, mean centering, reduced count or automatic restart.

The spec SHA256is
`71a58228011cd1b8a9b1c0393f5f292f0caba2e42a1cae51d9e11036d7765cf3`.
Actual target `collector --validate-only` passes before launch, reading real
profile/reference/library/metadata/resource-catalog bytes without inference.
The [qualified source package](orin_cogact_formal_source_readiness_20261024.md)
and earlier actual full pilots remain the quality prerequisites.

## Actual Owned Run

The existing UID1000user manager runs only the new transient unit
`edgefm-cogact-nostop-20261023-109.service`, `Type=exec`, four-hour **compute**
watchdog,30sstop timeout, cgroup cleanup and a post-observation hook. This is
not a four-hour application outage. User manager/slice memory and CPU quotas
are observed as infinity, with user slice TasksMax164405; no resource limits,
SDK, power mode, persistent system unit or unrelated degraded unit are changed.

At21:21UTCthe actual unit/monitor PID1100771/start27194631, collector1100773/
27194644and native worker1100883/27195157have matching live PID births and argv.
The unit is active/running; `Result=success` while active is not completion.
The first process samples a minimum6,535,832KiBavailable so far, above4GiB.
Live flags are declarations, not the final independent tensor audit.

The monitor performs read-only original-manager checks and verifies the original
Java PIDs942765/942805, births, argv and cwd every30seconds. It has no application
stop/start/restore method. If these conditions change, it terminates only its
own collector. The original programs stay outside the compute unit's cgroup.
The post hook only observes own-cgroup cleanup; no application restore is needed
or permitted. The four-hour watchdog is an own-compute safety choice, not an
inferred service-interruption grant.

## Measurement Limits And Remaining Audit

Java-on/background/user-cgroup conditions differ from earlier Java-stopped
campaigns. Keep this declared setting with every row; do not label it isolated
idle-model timing or assume invariant background CPU load. Within this model,
both variants use the same declared setting and interleaved protocol; resource,
temperature/frequency, owner and per-round timing evidence must stay visible.
Neither the original zero-warmup pilots nor H20latencies are speedup denominators.
Sixteen repeated observations are not independent trajectories or robot outcomes.

The run is incomplete until alltenprocesses finish and the new typed auditor
reads every actual output, resource/no-application-control audits pass, complete
CDFs/tables are generated and target/binary/unit closure is independently checked.
Source/launch/snapshot tests do not add formal acceptance. Shared D4and optional
Qwen remain outside this batch's completion claim.

## Evidence And Resume

- [No-service full formal spec](../../artifacts/orin/20261023-109/formal-no-service-spec-001.json).
- [Original-goal execution basis and explicit absence of an outage grant](../../artifacts/orin/20261023-109/no-interruption-basis-001.json).
- [Actual transient compute-unit launch](../../artifacts/orin/20261023-109/launch-receipt-001.json).
- [First verified live snapshot](../../artifacts/orin/20261023-109/checkpoint-002.json).
- [No-application-control path checks](../../artifacts/orin/20261023-109/no-service-path-tests-001.xml).

Use a fresh checkpoint output, never restart based on a monitoring timeout:

```sh
python3 -B artifacts/orin/20261023-109/snapshot_no_service.py \
  --output /ABS/FRESH-live-checkpoint.json
```

Confirm the same unit, PID/start/argv and actual logs on each continuation.
All raw data remains under `/data/datalog/edgefm-orin/20261023-109/formal-001/`.
