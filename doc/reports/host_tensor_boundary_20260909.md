# H20 Host Tensor Timing

Current closure: all 75 formal workers, five remote/local audits, the complete
15-row summary and five inspected CDFs have passed. See
`h20_host_tensor_formal_table_20260909.md`. The running-job snapshots below are
historical; do not restart those completed campaigns. Original-input E2E remains
separate and pending formal validation.

The shared Session benchmark now supports an explicit host-model-tensor boundary:
per-call H2D, fresh input binding, Session run/completion, and all output D2H.
Raw image preprocessing and robot transport remain outside this boundary.
The resident result archives are retained under their original definitions.

The implementation is model-neutral. It reuses typed input/output declarations,
fixed sample/reference packs, the existing numerical provider and GPU ownership
handshake. No model names or model-specific dispatch were added to the runner,
runtime or timing audit. Primary F32/F64 actions and secondary floating or exact
integer outputs are all copied before the end timestamp. Reference comparisons,
replay telemetry and output logging follow that timestamp.

Every warmup/measured invocation records four adjacent timing segments. The
auditor requires their exact sum to match both `host_call_ns` and the main CSV.
Formal aggregation and campaign summaries require the sidecar hash and reject
missing, changed, reordered or inconsistent timing evidence.

## Verification

- Focused protocol/runner/summary tests: 167 passed. The full generated C++ runner
  was executed in two CPU ABI fixtures with explicit delayed transfer stubs.
  Those tests verify transfer counts, sample cycling, full mixed-dtype output
  storage and segment accounting. They are not CUDA performance measurements.
- Complete CPU regression: 2204 passed, 74 skipped, zero failed. Hardware/optional
  dependency skips remain skips. Earlier failed fixture runs are retained.
- Focused XML SHA256:
  `6a127ba944149690dbd0ab355ec217b8f4e5baf2e2bfcc9e58da722a966742cb`.
- Complete CPU XML SHA256:
  `cdf288c430251da20360aa8a6a73421cfe5a6e87276f5e1c36a4f35c3fa5f131`.

Test artifacts are under
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`.

## H20 Campaign

New isolated root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/host-tensor-20260909-001/`.
The snapshot binds 298 implementation files and source commit
`592fc320d57398571dcb4ab686d5fb55fe9bac86`. The live remote HEAD has advanced to
`83251d56a1a682729ef145535f6c4de23a4ce250`; no merge or worktree reset was made.
The local source-provenance report SHA256 is
`5be8aad5998700587bda3d6751fa3fb73df2777ba55074fc5650efa620f8d4d4`.

The first campaign reuses the completed pi0.5 dual-output protocol, references
and three policy bundles. Its prepare controller PID 24221 and child 24235 have
completed successfully (exit 0). They ran on
`zzm-h20-x8-2`; target UUID is
`GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`. The initial resource check found no
compute owners or allocated GPU memory on H20-2, and about 9.2 PiB free on NAS.
H20-1 had allocated device memory and was not selected. Ownership is checked
again before every worker and monitored continuously.

The pi0.5 pilot controller 27236 and child 27316 completed all three policies.
Remote independent audit and local complete-data recheck passed: 144 calls,
288 complete F32/F64 outputs match both references byte for byte. Per-policy
mean host latency is 161.579535 / 161.166618 / 94.182012 ms for
off / batch-only / required. These are only 32 measured calls per policy and
must not enter the formal table or be compared with the old formal distribution.
Mean H2D is about 0.300-0.305 ms and all-output D2H about 0.025-0.027 ms.

Remote pilot audit SHA256:
`d3ee8818e0b1fcb9b3ed6d83a2156db41aafd66745d5584502e0b1552ec27e1a`.
Local full-output audit SHA256:
`761100c8e7923cb9cf2ab354066a5cd31bbd3caf56805e4b0d9dd8a202d0826e`.
Local archive: `artifacts/recovery-audit-20260909/host-pi05-pilot/`.
Formal controller 31003, child 31032, has started and was verified live; it is
not marked complete. All five models have now passed their complete remote/local
pilot audits and entered formal measurement: 720 pilot calls and 1872 complete
output tensors, all byte exact. CogACT includes five typed output ports. Other GPUs and initial CPU
compilation share this host, so host-wide CPU/NAS exclusivity is not claimed.

| Model | GPU Index | Formal Controller | Formal Supervisor | Status |
|---|---:|---:|---:|---|
| SmolVLA | 0 | 33468 | 33501 | formal running |
| pi0.5 | 1 | 31003 | 31032 | formal running |
| pi0 | 2 | 33483 | 33583 | formal running |
| RDT | 3 | 35605 | 35626 | formal running |
| CogACT | 4 | 39618 | 39647 | formal running |

The GPU UUID, every actual worker PID and sidecar SHA are in each campaign's
protocol/controller/worker records. `status_host_campaigns.py` reads `/proc`
controller liveness and terminal worker reports without restarting any job.
Do not rerun a completed stage in its existing directory.

| Model | Independent Pilot Audit SHA256 | Local Complete-Data Audit SHA256 |
|---|---|---|
| SmolVLA | `39f5a7c9d114cf0966c6da8cbd5b78cf3a57facfcc17340833cd128bc80c0fc8` | `584554d496baf4496fb3f2ecfe863174dbccd2c0755a4cbdb2954263a1369082` |
| pi0 | `f0261109de31407bda43343dc49611fcf1c46e8a9cdbc3911ac65225ac7cfee8` | `be110e4652acd9ad6dcdb3a9ccbfedff79f89fa8f621712fe64cbe80da001a18` |
| RDT | `9899b6040af77211a63d284a2838e0341d6f82a0d2ab1342c12a712f2e1dd12b` | `79c0a8a82d3c3e1bbfecc721ad2b6222d5829f988b9ac6709e234d04c9636a7c` |
| CogACT | `7525fc1c444c5dfa88bf9d25e7ec0dc5485720106a3f31a2f926471e6b91361c` | `d39b49bdbf19a0db1277cc567130761df9ac463f26b2c64cf5ac8624bb8b4087` |

At the archived 22:32 CST observation all five formal controllers and their
supervisors were live. Terminal worker reports passed for 11 of the planned 75
processes: SmolVLA 3/15, pi0 3/15, pi0.5 4/15, RDT 1/15 and CogACT 0/15.
This is an in-progress snapshot, not a completed formal audit. The exact observed
state is in `artifacts/recovery-audit-20260909/host-campaign-status-20260909-2232.json`.

After a formal controller and all fifteen workers complete, the separate
`report` stage recomputes the aggregate, then `audit_host_session_campaign.py`
in formal mode checks it independently. Complete raw inputs/outputs, timing,
`source/` metadata and remote audits must be retrieved before publishing local
tables/CDFs. `audit_retrieved_host_outputs.py` supports both pilot and formal
archives. Building, launching or observing a subset of workers is not completion.

Build/pilot/formal stages, independent audits and raw files have distinct paths.
A completed build does not imply a completed pilot or formal experiment. The
five-model host table and original-input/official-baseline comparison remain
pending until their execution and independent evidence are complete.
