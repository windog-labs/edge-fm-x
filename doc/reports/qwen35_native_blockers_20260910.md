# Qwen3.5 Native Deployment Evidence

> Historical route report: superseded for the fixed H20 profile on 2026-09-10
> by [Qwen3.5 Native H20 Formal Deployment](qwen35_native_formal_20260910.md).
> The failure evidence below is retained because those cache-copy export routes
> were correctly rejected before the accepted explicit-state route was built.

Both official Python image+text baselines have completed five independent
processes, 128 warmups and 1024 measured calls per process, complete-token audits
and inspected CDFs. Those accepted results, including TTFT, decode throughput,
input-to-token latency and memory, are in `qwen35_natural_profile_20260909.md`.
They do not establish VLAForge native deployment or optimized-output parity.

Native deployment remains pending: both sizes now have actual cache-copy export
rejections. The 2B two-image token difference was resolved for the tested profile
by disabling optimization only while executing the saved trace; its subsequent
export still failed. The actual failures below are retained and
must not be replaced by text-only fixtures, a successful first image, or an
unaccepted TorchScript candidate. These are implementation/profile limitations
of the tested route, not proof that either architecture cannot be deployed.

## Observed Results

| Model / Attempt | Observation | Accepted Native Deployment |
|---|---|---|
| 0.8B trace 004 | Saved/reloaded fixed-profile TorchScript tokens match both real images | No; effect and deployment gates remain |
| 0.8B export 003 | Static-axis conversion fails at cache-copy size guard `Eq(u98 + 327, u219)` | No accepted ExportedProgram |
| 2B probe 001 | Formal-reference input/token match; saved trace matches image 0, but differs in 9 of 16 tokens for image 15 | Rejected by complete-token gate |
| 2B probe 002 | Disabling JIT optimization during capture causes `candidate.ts` serialization to fail with a missing `forward` method | No accepted saved candidate |
| 2B probe 003 | Saved-trace execution without JIT optimization matches both images; export fails at `Eq(u194 + 327, u315)` | No accepted ExportedProgram |

All probes use actual image+text processing, vision encoding, prefill and
16-token greedy generation. Fixed prompt, image grid, masks and output length
restrict the tested profile; only pixels are a dynamic trace input. Image
indices 0 and 15 refer to the frozen sixteen-image input manifest. This is not
a formal multi-image performance benchmark. Image-file IO and tracing are not
included in the already completed Python baseline measurements.

## 0.8B Export Failure

H20-2 GPU3 UUID `GPU-c66f42bd-8a1c-423b-e2eb-d0cc72b9fe0b`.
Trace controller/worker: 3147/3149. Export controller/worker: 14479/14481
(export NVML PID 1222199). All are terminal. The retrieved complete token arrays
were independently compared again: both pairs are exact. The trace report hash
matches the export report's bound predecessor. The export stderr contains the
actual `meta_copy_` guard failure; ownership observations identify only its worker.

The separate static-axis conversion retained the existing effect and numerical
gates. It did not produce an accepted Region, C++ bundle or native formal timing.
Changing the automatic dynamic-axis policy alone did not resolve the cache-copy
size equation. Explicit model state/cache contracts still require implementation.

Remote roots under
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/`:
`qwen-native-feasibility-20260909-004/` and
`qwen-export-feasibility-20260909-003/`.
Current-worktree copies under `artifacts/recovery-audit-20260909/` are
`qwen-trace-feasibility-004/` and `qwen-export-feasibility-003/`.
The large trace archive is retained on NAS and was freshly rehashed there.

| Evidence | SHA256 |
|---|---|
| Saved trace archive | `01727950c1c41c83ff4efad2c182977f15403b92362976bdb221332005fb84a8` |
| Trace report | `7cd037a27ab0106f4c57f7caf843a11a3b76547faec89c151a24b1e9666504af` |
| Export failure report | `4590c78b9bcec10ddd88d1489cbc1d66d818f218b56e2797d854e11439546728` |

## 2B Complete-Token Rejection

Probe 001 reverified all thirteen model assets, the frozen source snapshot,
actual loaded runtime versions and recorded image identities. The complete
processor outputs for image 0 match the existing formal input package byte for
byte. Its untimed reference matches all sixteen saved official baseline tokens.
Torch 2.10.0+cu128 and the actually imported Transformers 5.12.1 are recorded;
the full numerical context and imported-source hashes are retained.

After saving/reloading the trace, image 0 remains exact. For image 15 the first
five tokens agree, followed by differences at zero-based indices
`[5,7,8,9,10,11,12,13,15]`. The complete reference and candidate arrays are retained.
No cosine score over integer token IDs substitutes for this exact comparison.
The first failed comparison aborts before ExportedProgram conversion.

An independent CPU audit on H20 rehashed 536 external source/model/input files,
the complete trace archive and every bound output, checked the terminal owner
monitor and recomputed the mismatch positions. A second local audit rehashed the
retrieved full archive and repeated the complete token comparisons. Both audits
report `failure_evidence_verified` and `candidate_accepted=false`; this status
accepts the failure evidence only.

Probe 001 controller/worker: 142389/142395, NVML PID 3466567, H20-2 GPU3 UUID
`GPU-c66f42bd-8a1c-423b-e2eb-d0cc72b9fe0b`.
Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/qwen-native-2b-20260910-001/`.
Complete local archive:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-2b-001/`.

| Evidence | SHA256 |
|---|---|
| Probe 001 protocol | `7c20dd0be3e6f34daffe4ee365ea5335fdf03127fe8d680dd811e23a10d61755` |
| Probe 001 driver | `74e56e244adfef6c64bfaf46820284267afbdf482a1a161c11ae2dce981a6d19` |
| Probe 001 failure report | `bfa0eb61fdfb114865e49f98b8f4392aaf435f7474fe2fdde63ce4372c4e8d85` |
| Remote independent failure audit | `91dc595fcebe19572a701f8b9095a5cc87614c6ac73522446574f3d1d1fec7af` |
| Local complete-data failure audit | `fe4d77282836949fb90de04b1f8784ff9ca9fc9a8904c59aa7c1f4be9d8db880` |

Probe 002 is a separately retained diagnostic, controller/worker
147744/147749. Its broader JIT switch failed while saving the trace, before
the intended saved-trace numerical comparison. It cannot establish whether
execution optimizations caused the first attempt's token difference. Its local
archive is `qwen-native-2b-002/`, failure report SHA256
`4aa0dc99b516859b198341fb67c01443ef7da5fc3736af5ad76548871d83e64f` and
traceback SHA256
`dce12c12c4a2c2fccb2456980ee2f293bf1db73c2a835fbef036d44aa07a90d9`.
Probe 003 kept normal capture and limited the switch to saved-trace execution.
Both complete token arrays now match their independent references, including
the image-0 formal reference. This resolves the observed trace mismatch for the
two tested images under that execution policy; it does not establish arbitrary
input support or the mechanism responsible for the original mismatch.

The subsequent static-axis ExportedProgram conversion failed at `meta_copy_`
with `Eq(u194 + 327, u315)`. The effect audit was not reached and no accepted
ExportedProgram was produced. Controller150106/worker150121 and NVML3566342
ended with failure on the same H20-2 GPU3. Independent remote and local audits
verified the complete saved archive, both exact token pairs, terminal ownership
and the retained export traceback. The remote audit rehashed 537 external files.

The complete local archive is
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-2b-003/`.
Its remote counterpart is
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/qwen-native-2b-20260910-003/`.

| Probe 003 Evidence | SHA256 |
|---|---|
| Protocol | `dbc90711f41f9dc25e28d930317c8bbb396a52d56f7ae24fa59468b960ac634d` |
| Driver | `35516d3775d451b940d2176a7378ae454ca6ed4536ef5499a9e18358e1e34f61` |
| Saved trace | `7feb6b99fbf8def8a246ec7f45373b1d7d0dfdb2966df032965b5f4da7784c08` |
| Failure report | `a33516c96c7c4308c32d6f4b4df5ce72b133c3bd63abf3cc2276641ef2645fbc` |
| Remote independent failure audit | `62a2194a0e9f103828d9e5342fcd746563e852bf90785b50fb9ea2ccee09314f` |
| Local complete-data failure audit | `45fa177930bcd82d57319c914cc4b349b425667c6ae29b30af926542fa444f48` |

## Remaining Acceptance

A deployable candidate still needs complete-output parity, explicit state and
profile guards, an accepted effect-audited Region, numerical provider binding,
C++ bundle execution and independent formal measurement. No gate was relaxed.
The probe and failure-audit helpers are experiment artifacts; no model-name
branch or altered behavior was added to the core Session runtime. Existing
0.8B/2B official baseline data remain unchanged and are usable with their stated
fixed-image Python scope.
