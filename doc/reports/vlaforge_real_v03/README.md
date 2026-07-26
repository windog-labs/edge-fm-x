# VLAForge Host CUDA real-artifact evidence

Date: 2026-07-25

This directory records real-checkpoint compiled-artifact evidence. It is
separate from `vlaforge_real_v02`, which only established eager-to-Invocation
IR L2.

## SmolVLA real L3

The real `SmolVLA-Base` checkpoint was captured as three flat TensorRegions:

- `prepare_prefix`: 4 CUDA inputs, 33 outputs (pad mask plus 16 flattened
  BF16 KV key/value pairs), 2,935 exported graph nodes;
- `solver_step`: pad mask, loop-carried sample, timestep, and 32 KV tensors,
  2,545 exported graph nodes;
- `trim_action_chunk`: one sample input and one action-chunk output.

All three exported programs were compiled to `sm_86` AOTInductor packages and
executed on the RTX 3060. Upstream eager and the exported 10-step pipeline were
bit-exact for the final `[1, 50, 6]` action chunk. The packaged artifacts are
deterministic but not bit-exact to eager:

| Measurement | Result |
|---|---:|
| maximum prefix-output NRMSE | 0.01860094 |
| 10th solver-step NRMSE | 0.01323196 |
| final action maximum absolute error | 0.02784944 |
| final action mean absolute error | 0.00802071 |
| final action NRMSE | 0.01303127 |
| repeated artifact execution | bit-exact |

The acceptance contract was fixed at prefix/solver NRMSE `<= 0.02`, final
maximum absolute error `<= 0.05`, and final mean absolute error `<= 0.01`.
Therefore this is real-model L3 numerical parity, not exact parity.

The recorded single-run timings and 1,796.59 MiB peak CUDA allocation are
pipeline audit metadata, not paper-grade benchmark results. Artifact hashes,
sizes, compile times, per-step errors, versions, and tolerances are in
`smolvla_artifact_l3.json`.

## SmolVLA real L4

The same real prefix/solver/trim packages were combined with five captured
Adapter support Regions (`make_timestep` and four queue operations) in one
verified eight-Region Compile Bundle. The generated no-Python C++ Session ran
on the same RTX 3060 with:

- bit-exact direct-AOTI versus generated-C++ parity for the complete
  `[1, 50, 6]` action chunk;
- one same-revision exact-cache hit and explicit new-revision,
  missing-revision, and episode-reset misses;
- 152 successful primary Session invocations, 304 authoritative state commits,
  and monotonically allocated per-state versions;
- typed C++ and generic C ABI output equality;
- a NaN output-validation failure that aborted the transaction, exposed no
  uncommitted output, and did not increment state versions;
- execution under invalid `PYTHONHOME/PYTHONPATH`, with no `libpython` in
  `ldd`.

The verified static plan assigns 2,314,353 bytes to recomputable derived prefix
cache and 2,464 bytes to authoritative queue/cursor state. Scalar state and
loop-carried Region values use at least 16-byte alignment, which removes the
AOTI implicit aligned-copy fallback found by NCU. These categories
remain semantically separate. Hashes, input fixture identities, bundle digest,
trace counts, and the exact reproduction command are recorded in
`smolvla_artifact_l4.json`.

This upgrades the fixed `SmolVLA-Base` checkpoint to real Host-CUDA L4. It is
not an Orin claim. Paper-facing latency, memory, soak, and profile evidence is
reported separately in `real_cuda_evidence.md`.

## DiffusionDrive real L2

The official NAVSIM checkpoint at Hugging Face revision
`8e3cc29cfdb5aa1a4c0818012f9a250d5153bc71` was validated by exact size and
SHA256, then strictly loaded into upstream Git revision
`9b52ed0ec06b073d82d6f392ab084c7b301c8681` with no missing or unexpected
keys. The four executed upstream source files also match their pinned Git
objects byte-for-byte.

The deployment partition uses five Regions: cached condition encoding,
explicit-noise initialization, timestep construction, a loop-carried denoise
step, and generic multi-output decoding. The original upstream forward was
observed through a read-only hook so all already-computed planner outputs could
be compared. Candidate trajectories, candidate scores, selected trajectory,
BEV semantic map, agent states, and agent labels were bit-exact to the Region
chain. All strict exports replayed with zero error and passed the effect audit.

This proves real-checkpoint frontend/Invocation parity at L2 with zero new core
ops. It does not yet claim compiled-artifact or generated-C++ parity; those are
the L3/L4 steps. The full hashes, shapes, environment, timing, and reproduction
command are in `diffusiondrive_frontend_l2.json`.

## DiffusionDrive real L3

The same five saved exports were compiled for `sm_86` into 248,879,397 bytes
of AOTInductor packages in 48.70 seconds. Saved exported programs remained
bit-exact to eager. The compiled artifact pipeline was deterministic but not
bit-exact to eager: the final selected trajectory had maximum absolute error
`0.00078416`, mean absolute error `0.00011506`, and NRMSE `0.00019730`.
Every Region and named output stayed below the fixed `0.001` NRMSE contract,
and a repeated artifact pipeline was bit-exact.

These timings are single-run audit metadata rather than a paper benchmark.
Hashes, sizes, compile manifests, per-Region metrics, tolerances, and the clean
source revision are recorded in `diffusiondrive_artifact_l3.json`. Generated
no-Python C++ Session parity remains the L4 gate.

## DiffusionDrive real L4

The five fixed-checkpoint AOTInductor packages were assembled into a verified
Compile Bundle at clean repository revision
`2ba35b5e71da0b20f02830682baf2cffaf622f23`. The generated runner executed
with invalid `PYTHONHOME` and `PYTHONPATH`, and `ldd` confirmed that it does not
link `libpython`. Its typed model API and generic C ABI produced identical
results. All six named outputs were byte-exact to the direct AOTI pipeline.

The successful sequence records one exact-cache hit, four misses, five
transaction/output commits, no state commits, and one reset. A NaN failure
injection exposed no uncommitted output, recorded one transaction abort, and
then reused the already committed condition cache when the same revision was
retried successfully. This is a stateless multi-output driving planner path:
there is no action queue and no new core op.

The report `diffusiondrive_artifact_l4.json` records artifact/checkpoint hashes,
the schema digest, bundle and runner hashes, C++ trace summaries, memory-plan
classes, and reproduction command. This is real Host-CUDA L4 on RTX 3060
`sm_86`; it is not an Orin claim. Paper-facing benchmark evidence is reported
separately below.

## Host-CUDA benchmark, soak, and profile

The two real L4 models were measured on the RTX 3060 after ten warmups. Timed
intervals contain one complete model invocation or generated
`ModelSession::Run` through backend synchronization; setup, input upload,
output probing, and reporting are outside the interval.

| Model | eager mean | direct AOTI mean | generated C++ mean | C++ vs eager | C++ overhead vs direct |
|---|---:|---:|---:|---:|---:|
| DiffusionDrive | 19.361 ms | 16.168 ms | 16.304 ms | 1.187x | +0.84% |
| SmolVLA | 112.912 ms | 45.131 ms | 45.194 ms | 2.498x | +0.14% |

Generated C++ and direct AOTI use identical compiled model artifacts. Their
near-equal full-compute times bound VLAForge orchestration overhead; eager
speedups are attributed to upstream AOTI compilation, not to VLAForge-owned
CUDA kernels.

DiffusionDrive exact condition reuse reduced mean latency from 16.304 ms to
2.947 ms (5.533x), with 500/500 cache hits. New and missing revisions produced
500/500 misses and restored full compute. SmolVLA caller-driven action-chunk
consumption averaged 0.689 ms across 500 Runs; its 10 refill points all hit the
same-revision prefix cache, while new/missing revisions missed on every refill.
The queue/cursor remains an Adapter template, not a core-IR assumption.

Both models passed 10,000 consecutive generated C++ Runs with zero transaction
aborts and zero CUDA-memory drift. DiffusionDrive recorded 10,000 exact-cache
hits and 4 KiB RSS drift. SmolVLA recorded 10,000 output commits, 20,000 state
commits, 200 prefix-cache hits, identical pre/post alignment-fix checksum, and
52 KiB RSS drift.

NSYS and NCU profile the no-Python C++ binaries. Their kernel summaries remain
upstream AOTI/cuDNN/CUTLASS/Triton work; no old EdgeFM kernel or custom model
kernel is compiled or claimed. Curated results, raw CSV samples, model-path
reports, profile summaries, hashes, and claim boundaries are in
`real_cuda_evidence.json`, `real_cuda_evidence.md`, and `real_cuda_raw/`.

## Frozen-core held-out generalization

After the real-model and performance work, revision `766e27b` was used as the
core freeze. Octo, GR00T N1.7, and AutoVLA were then audited as two robot and
one driving held-out architecture. The IR, compiler, Plan, codegen,
deployment, runtime, and C++ header Git objects remained unchanged, with
combined fingerprint
`cc2d1b63e2d6cbcd65935b37d69b5f18fae4d2d177c7026a69c6e78f5c80ae6d`.

All pinned upstream source contracts passed. Their deterministic fixtures also
passed verified compilation and exact Semantic/Plan output, state, and trace
parity with zero new core opcodes. This is explicitly L0 source plus L1
executable-fixture evidence, not real checkpoint, artifact, or generated-C++
support. The machine-readable report and reproduction details are in
`../vlaforge_heldout_v01/heldout_audit.json` and `heldout_audit.md`.

AutoVLA was subsequently selected as the real held-out object. Its released
checkpoint backs a three-Region post-attention decoder partition with exact
eager/export/Semantic/Plan trajectory and token outputs, exact trace, one
same-revision hit, two misses, and zero new core ops. This is honest
`L2-partitioned-real-checkpoint-frontend` evidence. A conservative `sm_86`
AOTI attempt retained exact tokens, `1.91e-6` trajectory max-abs error and
bit-exact repeatability, but intermediate Region NRMSE exceeded the
predeclared `1e-3` threshold; it remains L3-candidate. Reports and the complete
scope boundary are in `../vlaforge_autovla_v01/`.

The separate
`../vlaforge_architecture_v01/architecture_surface.{json,md}` audit proves
that the production source and CMake graph contain no physical scheduler,
sensor middleware, publish operation, core action queue, Python runtime
dependency, CUDA kernel source, or edge to old EdgeFM operator code. CUDA is
used only by the optional AOTI backend/runtime memory path for externally
compiled artifacts.

## Final Host-CUDA release gate

The final clean gate passed 215 offline Python tests with nine explicitly
gated real-model tests, one live CUDA AOTI package test, 7/7 CPU CTests, and
8/8 CUDA/AOTI CTests. Both CPU and CUDA installed-package consumers built and
ran.

The gate found and fixed a wheel-only defect: the CLI previously assumed a
Git checkout and did not ship its C++ runtime source. The installed wheel now
contains 24 runtime/CMake/header/backend entries. From a non-Git working
directory it generated and verified an OpenVLA-like Compile Bundle whose
runner executed with invalid `PYTHONHOME/PYTHONPATH` and linked no
`libpython`. Full hashes and test inventories are in
`../vlaforge_release_v01/release_gate.{json,md}`.

## Installed-wheel artifact evaluation

The follow-up artifact evaluation built revision `849a7df` into a wheel,
installed it into a clean venv, removed the source `PYTHONPATH`, and switched
to a non-Git working directory. It imported VLAForge from `site-packages` and
used only the wheel's `share/vlaforge` C++ runtime source.

A real synthetic `sm_86` CUDA AOTI package was compiled and embedded into both
session-resident and invocation-resident Compile Bundles. Both generated C++
Sessions matched the eager reference within `4.35e-9`, ran with invalid Python
environment variables, linked no `libpython`, and rejected schema/ABI,
shape/dtype/device/layout, missing-artifact, and corrupted-artifact cases.
This is a production artifact-substrate test, not real-model evidence.

The accompanying reproducibility manifest hashes all current formal reports
and committed raw summaries, extracts their reproduction commands, and
inventories every `/tmp` reference. Eight real-model checkpoint,
capture/artifact, input and bundle roots require external archival; they
currently total 114.41 GiB. See
`../vlaforge_reproducibility_v01/README.md` and
`reproducibility_manifest.json`.

## OpenVLA-7B real L3

The pinned `openvla/openvla-7b` checkpoint at Hugging Face revision
`47a0ec7fc4ec123775a391911046cf33cf9ed83f` was physically partitioned without
changing the logical Invocation IR. The three logical stages remain bounded
prefill, autoregressive decode, and detokenize; the backend owns 36 physical
Regions:

- multimodal prefix preparation;
- sixteen two-layer prefill chunks;
- token embedding;
- sixteen fixed-KV two-layer decode chunks;
- logits head and action detokenization.

The fixed KV profile covers a 275-token prefix and six decode positions in a
maximum length of 281. Its 64 key/value tensors occupy 147,324,928 bytes and
are loop-carried derived cache, never authoritative Session state.

All active-version normalized exports replayed exactly. The 36 `sm_86`
AOTInductor packages total 28,256,718,272 bytes and compiled in 311.44
seconds; the largest single compile process used 6,549,436 KiB peak host RSS.
Across every artifact output, maximum NRMSE was `0.02688469`, below the fixed
BF16 contract of `0.05`; integer token/mask/position outputs were exact.

Two complete artifact-only pipelines both produced:

`31857, 31864, 31900, 31840, 31860, 31868, 31872`

The runs were bit-exact to each other. Their final actions differed from the
real L2 reference by at most `1.13e-17`. Capture and audit peak CUDA allocated
memory were 2.686 and 1.778 GiB respectively. These are correctness-audit
metadata, not latency benchmark results.

The report `openvla_artifact_l3.json` was produced from clean revision
`7ea773e53c8b24fc96708cd87aa5a4f7d5985b1c`. This is real-checkpoint L3 with
zero new core ops. It does not claim generated no-Python C++ L4; weight-paged
Session artifact residency is a separate gate.

## OpenVLA-7B L4 resource audit

Revision `6c4dc927accc97f74f4cb43607ca06bebf531532` successfully assembled all
38 Regions into a clean-source Compile Bundle. The 36 model packages were
invocation-resident, the two glue packages were session-resident, the generated
runner linked no `libpython`, and the verified static arena was 350,748,288
bytes instead of the 1,215,201,344-byte baseline.

The real C++ execution did not complete and is not L4 evidence. PyTorch
`AOTIModelPackageLoader` extracted a new temporary wrapper shared object on
each repeated decode-package load. Region destruction released CUDA model
residency, but deleted wrapper mappings and their disk backing accumulated in
the process. At the safety stop, the runner had 119 wrapper mappings, 24.48
GiB RSS, 91.93 GiB virtual memory and roughly 82GB of package writes; free
system disk had fallen from 105GiB to 29GiB. Runner CUDA residency remained
only 644MiB, so this was not a device-memory OOM.

The reproducible command, bundle identity, resource samples and claim boundary
are recorded in `openvla_artifact_l4_blocker.json`. OpenVLA therefore remains
real L3. A future L4 retry should use a backend artifact variant that separates
stable verified code/cubin mapping from per-invocation CUDA weight residency;
it must not add a core IR opcode.
