# Orin Qwen Reusable Decode Preparation

## CPU Continuation Checkpoint

2026-10-25 07:30:58 UTC: worker008 succeeded. Reloaded single-token
TorchScript matches all49 eager outputs at all15 steps byte-exact, not only
argmax tokens. The source remains BF16 with the original declared arithmetic.
Archive SHA256: `f121c2c5866f33b19498a1ad2f3ea5f87cc2623d1e57cb87d7869429fc1ca406`.
Six actual SDPA sites were observed with complete49output equivalence.
ONNX export passes structure checks:6633nodes/321initializers/51inputs/49outputs;
its raw BF16 runtime parity is explicitly skipped, not claimed.

Existing generic transforms produce4251deployment nodes and retain all51inputs
and49outputs. Both portable shadows execute all15iterations; all49outputs per
step are byte-exact before/after transforms. All16tokens match the fixed-cache
CPU route, but final logits versus that route retain MSE0.0009795830875004427,
maxabs0.15625/cos0.9999520224328118. This is not target-plugin equivalence,
source-lossless or primary same-Orin quality. The new graph is being staged
for an independently bounded24GiB/4GiB Orin build under the ongoing grant;
no native decoder engine or new target inference is yet claimed.

Evidence:
- `artifacts/orin/20261025-11/reusable-cache-source-006/report.json`
- `artifacts/orin/20261025-11/reusable-cache-worker-008/attempt.json`
- `artifacts/orin/20261025-11/reusable-decode-export-001/export-report.json`
- `artifacts/orin/20261025-11/reusable-decode-native-001/report.json`
- `artifacts/orin/20261025-11/reusable-carrier-parity-001/report.json`

2026-10-25 07:23:45 UTC: worker007 completed with exit0/no guard stop.
All477 original tensor values match after strict source reconstruction.
The archived prefill callee has a trace-reordered ABI: twelve K/V outputs,
logits, RoPE delta, then36 linear states. The original terminal contiguous
return-expression evaluation order establishes an explicit permutation;
no state is assigned by a shape-only guess. Every semantic slot is checked
against the declared state contract and the original cut first-token/delta.

Reconstructed dynamic continuation reproduces all16 tokens and complete
final logits byte-exact against the frozen CPU source. Fixed-capacity342
continuation executes all15 steps with all48 states shape/dtype/finite checked
at every step. Its16 tokens match, but final logits are NOT byte-exact:
MSE0.0008934887472364197, max-abs0.15625, cosine0.999953630789254.
This is source-side numerical difference, not a lossless claim, and does not
pass the MSE<1e-5 condition when applied to this source comparison. It is
not the same-Orin ordinary/optimized primary comparison, which remains open.

At that checkpoint worker008 was tracing and reloading a reusable single-step decoder, requiring
all49 outputs at all15 steps to match the eager fixed-capacity route byte-
exact before export. No Orin inference or formal performance is claimed.

Evidence:
- `artifacts/orin/20261025-11/reusable-cache-source-005/report.json`
- `artifacts/orin/20261025-11/reusable-cache-worker-007/attempt.json`
- `artifacts/orin/20261025-11/validate_reusable_cache_source_validated_attempt007.py`

Checkpoint: 2026-10-25 06:29:46 UTC target terminal audit. The higher-headroom
whole-part retry did not emit an engine. It ended on the unchanged4GiBguard
at minimum4191656KiB,1709.879236864seconds, outerexit-15/native lastbuilding,
15234network layers/zero unresolved policy names. This is not a time limit,
numerical failure or unsupported-model proof. Skipped-tactic warnings remain.

Source graph/profile/weights/builder/plugin/guard/authority were rehashed.
Both owned cgroups are empty. Supervisor4237/configuration/executables/JARs
are unchanged; only the three conditional leases were used and restored:

| Program | Outage Seconds | Restored PID / Start Ticks |
| --- | ---: | --- |
| data-collection | 1733.703965888 | 3498108 / 39149104 |
| starter | 1721.370656736 | 3498074 / 39148985 |
| router_manager | 1719.153879072 | 3498022 / 39148871 |

compotask_service1684984/start30057656was not stopped. Unrelated programs
are unchanged, recovery within the original7200second window. No target
model inference or formal metric was generated. There is no live Orin task
from this phase; Goal remains active, not paused or complete.

## Next Execution Representation

Do not repeat this whole-part source with marginally more admission memory.
Investigate reusable single-token decode with explicit bounded K/V, recurrent
and convolution state. All15decode iterations and16complete tokens remain
required. No smaller horizon, precomputed-vision E2E, alternate model, changed
precision or loosened quality gate is accepted.

The existing adapter has bounded-cache primitives and an exact dynamically
growing reference. Their output and state contracts must be validated before
export or deployment. A state-only feasibility experiment is CPU preparation,
not a target model or performance result. Previously accepted prefix engines
and raw outputs remain; the failed unrolled source and all warnings are kept.

Initial CPU preparation exposed and retained three real boundaries:

- TorchScript includes three nonpersistent rotary tensor attributes absent
  from Python state_dict. Restore only existing same-type/same-shape declared
  tensor attributes; every learned state still loads strictly. No key is dropped.
- The local Transformers vision constructor lacks a helper expected by the
  existing adapter. Do not upgrade/install or fabricate a vision result; use
  the already validated original archived prefill for source-side tests.
- SSA-cut frontier production order is not semantic layer-state order.
  A3Dconvolution tensor incorrectly routed into a4DK/Vslot was rejected.
  Replay the original prefill tuple, verify every semantic slot and derive
  explicit mappings rather than guessing by shapes or model names.

Historical pre-worker007 CPU source-continuation checks were not accepted. An additional
archived lifted-call ABI rejection is retained. The next copied-root-method
probe preserves the actual original call arguments and records returned
contracts before attempting any semantic-state adaptation. Source reconstruction
and cached continuation must compare complete16tokens and all final logits
against the frozen same-CPU reference; all48state tensors at every step must
be finite with declared shapes/dtypes. No export or target result is claimed.

## Raw Observation Recovery

The original16PNGframes are recovered from the shared NAS through the available
4090SSH path after a read-only NAS web check timed out. All PNG hashes match
the old manifest. Original source observation manifests and all16NPZarchives
were independently rehashed. Decoded pixels in the source-declared UINT8CHW
layout match both the original hashes and actual archived arrays. HWC transport
is separately typed/hashed, not a pixel edit or resize. An initial incorrect
HWC hash interpretation was rejected and retained before positive publication.

Dataset: physical-intelligence/aloha_pen_uncap_diverse, pinned revision
e82d8b40b8ac66c0b40273dd80a077dfc40b732e; cam_high,480x640RGB,16frames
from one episode. This is not16independent trajectories or a robot-success result.
All128original frozen validation files were also recovered and rehashed.

The official processor semantic recipe is recorded, not represented as the
original downloaded bytes. CPU comparison now passes all16complete
FLOAT[1200,1536]pixel tensors,byte-exact/MSE0/maxabs0,using the existing
Transformers5.3.0image processor. The first attempt stopped on missing local
sample1files,not a numeric mismatch; all128original files have since been
recovered/rehashed. No recipe,pixel,parameter,tolerance or environment changed.
This validates source input preparation only,not native online preprocessing,
tokenization,target model quality or performance.
Official metadata: [Qwen preprocessor config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/preprocessor_config.json).

## Evidence

- `artifacts/orin/20261025-09/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-10/verified-raw-observations-001.json`
- `artifacts/orin/20261025-10/raw_layout_rejection_001.json`
- `artifacts/orin/20261025-10/official-processor-recipe-001.json`
- `artifacts/orin/20261025-10/processor-pixel-parity-002.json`
- `artifacts/orin/20261025-11/nonpersistent_scope_rejection_001.json`
- `artifacts/orin/20261025-11/vision_compatibility_rejection_002.json`
- `artifacts/orin/20261025-11/frontier_order_rejection_003.json`
- `artifacts/orin/20261025-11/lifted_prefill_rejection_004.json`

Five accepted VLA campaigns and retained paper data are unchanged. Qwen0.8B
and2Bnative/resident/quality/formal requirements remain open, no completion
inferred from source preparation or diagnostics.
