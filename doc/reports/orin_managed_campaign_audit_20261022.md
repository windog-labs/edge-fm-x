# Managed Orin Campaign Audit

Status: host audit tooling and frozen bindings checked; **the RDT campaign is
still running, and neither its final data nor its eventual restoration is
certified here**. No running target collector, worker or service guard was
edited/restarted to add these checks. No additional target GPU job was started.

## Separate Evidence Levels

| Level | Required Evidence | Current State |
|---|---|---|
| Live execution | Actual PID/start/argv and advancing CSV, not just JSON status | Board read-only checks match the active collector/worker |
| Frozen bindings | Collector/helper/guard digests, exact activation argv, immutable window, direct authorization, input pilot | `frozen-bindings-audit-002.json` passes; explicitly not completion or liveness |
| Complete model data | All fifteen processes, every 128+1024 row, all full outputs, actual replay, raw timing/resource hashes | Waiting for collection; the existing full auditor is reused, not weakened |
| Resource window | >16 GiB admission, every retained sampled value >=2 GiB, exact source/time bounds | Full-window check waits for terminal files; no continuous-exclusivity claim |
| Restored services | Terminal guard/lease plus a new read-only PID/start/argv, parent, JAR/config/boot observation | Pending; current two services are deliberately stopped for sampling |

An actual attempt to run final qualification on the current checkpoint returns
`pending-no-formal-claim` and creates no accepted result. The read-only restoration
observer returns `pending-no-restoration-claim`, without starting a model or
changing services. A prior rollback receipt, an old PID, or a `restored` word
alone cannot certify the future restoration.

The final composer invokes the original `20261021-07/audit_raw_formal.py`
complete-output auditor, preserving its fifteen-process/count/order, native
output bytes, active-coordinate projections, replay/no-fallback and unfiltered
timing requirements. It then qualifies the managed-window/restoration contract.
An operational pending/rejected result is not relabeled as a model numerical
failure. The original model-data auditor remains independently available;
data and restoration scopes stay explicit.

## Sources and Tests

Tool root: `artifacts/orin/20261022-01/`.

| Item | Source/Evidence |
|---|---|
| Frozen frame, terminal, restore and sampled-memory checks | `managed_window_audit.py` |
| Pure read-only post-restoration observation | `observe_service_restoration.py` |
| Explicit source/archive/identity roles | `managed-audit-template-001.json` |
| Audit specification writer | `prepare_managed_audit_spec.py` |
| Composed complete-data audit | `audit_managed_campaign_v2.py` |
| Real current binding result, no model-data claim | `frozen-bindings-audit-002.json` and `managed-audit-spec-bindings-001.json` |
| Real pending restoration observation | `restoration-observation-pending-001.json` |
| Combined source/control/postprocessing regression | `managed-final-audit-regression-002.xml`, 86 passed |

Tests use actual frozen current bindings, the retained first rollback/next
inspection identities and one existing Norm resource trace. Synthetic terminal
metadata/timestamps are explicitly unit fixtures, not a fabricated complete
RDT experiment. Checks reject changed argv/source, expanded service scope,
reduced protocol, old-process identity, changed JAR/config/parent/boot, bad
memory units/floors/timestamps, and an unfinished campaign. No test signals
real services or runs a target model. The first binding receipt's exact writer
is retained as `audit_managed_campaign_bindings_v1.py`; its bytes independently
match that receipt's recorded source digest.

## Terminal Runbook

Do not execute this terminal-data recipe until the actual collector has exited
and the guard reports completed restoration. Recheck real identities first;
an observation timeout never authorizes duplicate collection. Use fresh output
suffixes for retries, preserving earlier pending/rejected files.

```bash
mkdir -p artifacts/orin/20261022-01/formal-retrieval-001
mkdir -p artifacts/orin/20261022-01/restoration-final-001
rsync -a orin:/data/datalog/edgefm-orin/20261022-01/rdt-raw-formal-001 artifacts/orin/20261022-01/formal-retrieval-001/
scp orin:/data/datalog/edgefm-orin/20261022-01/service-guard-001/guard.json artifacts/orin/20261022-01/restoration-final-001/guard.json
scp orin:/data/datalog/edgefm-orin/20261022-01/rdt-service-lease-002.json artifacts/orin/20261022-01/restoration-final-001/lease.json
python artifacts/orin/20261022-01/observe_service_restoration.py \
  --window /data/datalog/edgefm-orin/20261022-01/service-window-freeze-001.json \
  --guard /data/datalog/edgefm-orin/20261022-01/service-guard-001/guard.json \
  --lease /data/datalog/edgefm-orin/20261022-01/rdt-service-lease-002.json \
  --output artifacts/orin/20261022-01/restoration-final-001/observation.json
python artifacts/orin/20261022-01/prepare_managed_audit_spec.py \
  --template artifacts/orin/20261022-01/managed-audit-template-001.json \
  --capture artifacts/orin/20261022-01/formal-retrieval-001/rdt-raw-formal-001 \
  --guard artifacts/orin/20261022-01/restoration-final-001/guard.json \
  --restored-lease artifacts/orin/20261022-01/restoration-final-001/lease.json \
  --restoration-observation artifacts/orin/20261022-01/restoration-final-001/observation.json \
  --output artifacts/orin/20261022-01/managed-audit-spec-final-001.json
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-01/audit_managed_campaign_v2.py \
  --spec artifacts/orin/20261022-01/managed-audit-spec-final-001.json \
  --phase final --output artifacts/orin/20261022-01/final-audit-001
```

Only an actual successful final audit may produce `model-data-audit.json`,
`all-requests.csv` and the composed `audit.json`. No final directory from that
recipe exists as a completed result at this checkpoint. The plotting input is
the model-data audit, not the metadata-only binding report:

```bash
PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=/home/zhangzimo/miniconda3/lib \
PYTHONPATH=vlaforge/python:/home/zhangzimo/miniconda3/envs/vlaforge-openvla/lib/python3.10/site-packages:/home/zhangzimo/miniconda3/lib/python3.13/site-packages \
/home/zhangzimo/miniconda3/pkgs/python-3.10.20-h17756b0_1/bin/python3.10 \
  artifacts/orin/20261021-07/plot_raw_formal.py \
  --audit artifacts/orin/20261022-01/final-audit-001/model-data-audit.json \
  --csv artifacts/orin/20261022-01/final-audit-001/all-requests.csv \
  --output artifacts/orin/20261022-01/formal-cdf-001 \
  --title 'RDT complete raw-observation E2E on Orin'
```

This retains all 5120 measured rows per mode and full tails. No stage sums,
outlier trimming, process centering, old interrupted rows or pilot statistics
are substituted for that result. A current restored process contract does not
prove application readiness, former in-memory state, absence of later restarts
or exhaustive private-SDK/driver allocation accounting.

## Completed-Worker Check

`audit_completed_cases.py` can independently check explicitly selected finished
workers while the remaining campaign continues. It rejects an active worker
before reading its partial files, checks every recorded file/hash, exact
128+1024 timing rows, complete native/unified outputs and actual replay counters.
It does not emit pooled latency/CDF or accept the whole campaign. Tests use
one old retained full worker and reject one-ULP corruption, a missing row or
unexpected replay; those test inputs are not new campaign results.

Only after the named worker exits zero, archive its immutable directory and
the matching campaign/protocol snapshots to a fresh host prefix. Prefer the
next worker's warmup window for transfer; do not copy live partial output or
delete any rows. Then, for example:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-01/audit_completed_cases.py \
  --template artifacts/orin/20261022-01/managed-audit-template-001.json \
  --capture artifacts/orin/20261022-01/completed-prefix-001 \
  --case round-00-off \
  --output artifacts/orin/20261022-01/completed-case-audit-001.json
```

Its receipt always has `formal_sampling=false` and
`formal_acceptance_claimed=false`. The final fifteen-process and service
restoration gates remain unchanged. Combined completed-case/window/control/
full-output/CDF regression: `managed-and-completed-regression-004.xml`,
91 passed. This is a host tool test count, not experimental acceptance.

Actual use at 03:46 UTC: only the finished new `round-00-off` and matching
campaign/protocol snapshots were transferred during the next worker's warmup.
`completed-case-audit-001.json` independently verifies all 2304 complete tensors
and recorded files, with both formal flags false. The active shared-stream
worker's partial files were not copied. This is bounded early data checking,
not the terminal audit recipe or a new RDT main-table/CDF result.

Board validation at this update: eight state/server tests and nine browser
tests pass. Dynamic live headings are derived from checked counters and cannot
mark a campaign accepted. Coverage tests use isolated progress/pilot fixtures,
not a production-data assumption that formal-model count must stay at 1/5.

The 04:45 UTC shared-stream completion is separately archived in
`completed-prefix-002` and checked by `completed-case-audit-002.json`: another
2304 complete tensors byte-exact, still not formal acceptance. Host-only
`audit_completed_resources.py` composes the existing resource/ownership/floor
checks for these two selected completed archives. Its explicit request,
`completed-resource-audit-001.json`, nine-test receipt and
[resource report](orin_rdt_completed_resources_20261022.md) retain sampling gaps,
invalid/unreadable sensors and permission limits. The terminal recipe above
continues to require all fifteen processes and actual restoration; no partial
resource result is used to bypass it.

By 05:44 UTC the new replay completion is separately archived in
`completed-prefix-003`; `completed-case-audit-003.json` verifies another 2304
complete tensors, 1152 actual five-step replays and zero ordinary fallback.
Only 3/15 new workers are complete. The selected-resource request 002 and
report 003 cover all 9421 retained first-round rows, explicitly disclosing one
readable process-status row with missing memory counters. The first rejected
extra-audit log/source remain; absent fields are never zero-filled, malformed
present fields still reject, and the original full-data/system-memory guards
are unchanged. This does not qualify the active campaign or restore services.
The real current count is still 1/5; the fixture's 2/5 is not a model result.
