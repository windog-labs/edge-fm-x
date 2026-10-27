# RDT Resource Safety Interruption

The five-model Goal remains active and incomplete. This RDT campaign is
**interrupted, not formally accepted**. No model-quality failure is inferred
from the operational stop. No failed/slow samples are deleted or relabeled.

## Action And Evidence

At 2026-10-22 05:54:31--05:54:33 Asia/Shanghai, an immediate recheck observed
60 complete telemetry records spanning 68.902 seconds, all below 512 MiB
MemAvailable with swap exhausted. That window ranged from 292292 to 465340 KiB;
the live recheck was 435292 KiB available and zero SwapFree. This is an
operational safety guard, not a changed numerical or latency acceptance gate.

Only current same-user worker PID/start ticks `1918313` / `10076756` received
SIGTERM, after checking actual serial, boot, UID, parent PID, exact argv, start
ticks, current campaign/worker identity, unchanged 5 x (128 + 1024) protocol and
both sustained/fresh pressure. No other process or service received a signal.
The frozen collector observed worker exit `-15`, retained its failed campaign
record and exited by its existing failure handling. Subsequent read-only
checks find neither controller nor worker live/matched; no restart occurred.

Receipt:
`artifacts/orin/20261021-07/safety-stop-20261022-001/receipt.json`.
It preserves the input checkpoint, exact capture/stop source hash, bounded
pressure policy, the single signal and terminal observations. Safety policy /
identity regression: `safety-stop-regression.xml`, 15 passed without host tests
sending any process signals.

## Retained Data

- Eleven full processes exited 0 after 128 warmups + 1024 measured calls each:
  11264 complete-process measured records, with their original output/telemetry
  files and recorded hashes preserved.
- Worker twelve has 128 warmups and 147 complete written measured rows before
  interruption. Raw files, including any incomplete trailing payload, remain
  untouched; these rows are not called a complete twelfth process.
- Total written measured rows: 11411, not the required 15360. The fixed protocol
  still requires fifteen complete independent workers, five per mode.
- The interrupted campaign and protocol were retrieved into the new folder
  `artifacts/orin/20261021-07/interrupted-retrieval-001/raw-e2e-formal-001/`.
  SCP exited 0 before any dependent archive/output audit. Archive integrity and
  complete-output validation are separate gates, not implied by transfer exit.

The post-stop receipt observes MemAvailable 12730776 KiB. This non-atomic system
difference is not a precise model-memory measurement or allocator attribution.
Existing Java/platform services were not stopped, restarted or reconfigured.

## Bounded Retained Checks

The retrieved campaign SHA256 is
`76546b9bf3dfe0167aae3b0c2caea2639c8910ac5b6af559b2c981f5c7f00ebb`;
protocol SHA256 is
`6cd61893b2672698c18ea05c88217e420e31adca7618a2143d5459974149bf1e`.
Both independently match the now-terminal target files. All recorded file
sizes/hashes match for the eleven complete cases, with exactly 128 warmups and
1024 measured rows per case and timing conservation intact. Their counts by
mode are off 4, shared stream 4, replay 3, **not five per mode**.

The unchanged formal auditor's complete-output primitive rechecks all 25344
full-case native/unified action tensors, including warmups, against the frozen
same-Orin conventional reference. All are finite, byte-exact, MSE/max-abs zero;
active-coordinate projections and actual replay counters are checked without
padding-diluted error claims. Source maps contain no Torch/Python model runtime.

Receipt: `interrupted-retained-checks-001/retained-checks.json` under the task
artifact root. Its schema/status explicitly limit scope to retained evidence,
`formal_sampling=false`, and rejection by the original full-protocol gate.
No formal pooled latency/CDF or causal speedup is generated.

The interrupted case's raw output sizes are exactly the 275 fully written CSV
rows; there is no extra/truncated tensor tail in these files. A separate
supplement checks all 550 corresponding tensors, including its 128 warmups and
147 measured rows, byte-exact with the same reference. No prefix is copied or
trimmed and no raw file changes. Receipt:
`interrupted-partial-payload-checks-001/partial-payload-checks.json`.
The interrupted process remains incomplete and does not become process twelve.

Total checked retained action tensors: **25894**. All recorded measured rows
are preserved, but only 11264 belong to complete processes; 147 are partial.
`interruption-verification-regression.xml`: 25 passed, including 15 pure safety
policy/identity tests and ten interruption/hash/scope checks. Test fixtures do
not signal processes and are not extra board experiments.

## Scope And Next Gates

The full-protocol auditor must reject this failed/incomplete campaign. It must
not be loosened to certify eleven workers or merge partial rows into a formal
fifteen-worker claim. Independently check retained file hashes and the complete
outputs of finished workers, clearly labeling that bounded partial evidence.
No formal RDT CDF/main-table acceleration is published from this interruption.

Resource/owner limitations remain documented in
[resource observations](orin_rdt_resource_observation_20261022.md). The question
of retaining the two existing Java services in later tests is still pending.
Before new target sampling, establish an authorized, adequate resource setting,
review comparability and freeze the actual run plan. Do not silently resume,
discard older rows, reduce samples, change precision/checkpoint/steps or move
to the next required model as a substitute for closing RDT.

The Goal is not marked complete or user-paused. Work continues on evidence
preservation/verification; the blocked hardware collection is explicit in the
task board, while model deployment and the earlier raw pilot retain their
separate, previously verified scope.
