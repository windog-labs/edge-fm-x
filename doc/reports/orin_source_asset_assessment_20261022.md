# Orin Source Asset Assessment

Date: 2026-10-22. **D0 host preparation only, not new model deployment or target
support acceptance.** RDT continues its frozen formal campaign. No other Orin
GPU task, new H20/H100 model execution, export or weight conversion was run.
Current Orin formal-model completion remains 1/5.

## Revalidated Assets

All five original data/reference packs remain under the original repository's
`artifacts/edgefm-vla-goal/20260906-044509/board-handoff/`, not the current
worktree's corresponding path. The public current-worktree `board_handoff`
validator rechecked every file/typed input and complete stored reference pair.
An independent fixed manifest pin is checked before that validation. Private
temporary copies reject a changed output bit and a false board-verification
promotion. Original packs are revalidated afterward; no source bytes changed.

| Model | Pack | Files / Bytes | Samples | Complete Reference Pairs | Actual IR Solver Steps |
|---|---|---:|---:|---:|---:|
| SmolVLA | `smol-dual-data-001` | 315 / 139950196 | 16 | 32 | 10 |
| RDT | `rdt-dual-data-001` | 265 / 88012821 | 16 | 32 | 5 |
| pi0 | `pi0-dual-data-002` | 322 / 64678337 | 16 | 32 | 10 |
| pi0.5 | `pi05-dual-data-001` | 328 / 64715108 | 16 | 32 | 10 |
| CogACT | `cogact-data-002` | 3351 / 283085128 | 115 | 575 | 10 |

Total: **4581 files, 640441590 bytes, 179 stored samples and 703 complete
direct/eager reference pairs**. All reference pairs are byte-exact. This is
not 179 independent trajectories or new inference. These packs contain input,
reference, provenance and invocation-IR declarations, **not checkpoint weights,
portable neural ONNX, target engines or executable target providers**.

The old `board_executed=false`/pending target templates describe the scope of
each historical pack, not a claim that current SmolVLA/RDT Orin work never ran.
They are not rewritten to inherit current hardware or provider acceptance.

## Frozen Identity Pins

| Model | Manifest SHA256 | Inherited Checkpoint Pin |
|---|---|---|
| SmolVLA | `1554ec215a3a02967ff56cec334918892ded6e0e507288689edff0fa6026eede` | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| RDT policy only | `2fbf37103e22a49026fefe59a81fce14fb777dfe6d2ace615eab923a6dde2514` | `fe217f4491ea882b0b52df1cb23ae4e8a11c1328ed29f2ed712e02aad2c02102` |
| pi0 | `5f455275e12602ac36685b6749b51cf3eb2c4d55eb9a270edb3df7f3bd2911c4` | `07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518` |
| pi0.5 | `5257c4c07c0b758eb19c5cbeef4c2aeeb813a211c47ae0e9fbb9813061af6d63` | `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6` |
| CogACT public-dependency profile | `c39053b6e02b351f3790326356b40b7bb522853993db770aef7431ea4b1c6b47` | `1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0` |

The new audit **does not rehash large checkpoint weight payloads**. These pins
and parameter records are inherited from already frozen source evidence; they
do not establish present target weight residency or compiled artifact size.

pi0's conversion record reports **3501372176** parameter elements, pi0.5's
**3616757520**. Each retains 263323648 zeroed unused parameter elements;
they are not silently subtracted/pruned from the checkpoint identity. The
converted storage precision is BF16, source OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`. pi0 uses `pi0_aloha`, 48 text slots,
and no discrete-state input; pi0.5 uses `pi05_aloha`, 200 slots and discrete
state conditioning. Equal action shapes do not make these interchangeable.

CogACT's installed-weight coverage records **7630224071** elements, including
all four source parameter groups. Keep the public dependency/gated-original-
Meta limitation; neither compute omission of unused logits nor the DiT size
makes this a complete 3B checkpoint.

RDT's policy pin is **not a whole-stack pin**. Separately pinned source
components include policy `robotics-diffusion-transformer/rdt-1b` revision
`eb09036cc64ca4945051acbd1bd581d30a1d7711`, text `google/t5-v1_1-xxl` revision
`3db67ab1af984cf10548a73467f0e5bca2aaaeb2`, and vision
`google/siglip-so400m-patch14-384` revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. The matrix retains individual
weight-file hashes and explicitly does not count a whole stack from file sizes.

## Porting Boundaries

| Source Contract | Consequence for the Later Native Target Path |
|---|---|
| All packs use preprocessed model tensors, not decoded raw observations | Reuse the separately pinned H20 raw-profile roots/processors; native online preprocessing is still required for Orin E2E |
| pi0/pi0.5 F64 state `[1,32]`; F32 normalized `[1,50,32]`; F64 native `[1,50,14]` | Preserve types or declare and validate an explicit host/device partition and source-derived conversions; no silent F32 downcast |
| pi0 text `[1,48]`, pi0.5 text `[1,200]`, original I64 IDs | Separate declared shapes/processor contracts; no padding/truncation to force one engine profile |
| CogACT F32 raw/normalized `[16,7]`, F64 native `[16,7]`, U8 RNG state and I64 draw count | Keep every complete port and exact integer comparison; do not compute misleading RNG cosine |
| CogACT explicit initial/ten step noise and twelve RNG snapshots | This qualified data profile is an external-noise-tape contract, not autonomous RNG. Preserve all real noise and receipt semantics |
| Later H20 autonomous CogACT uses LibTorch CUDA RNG | Its `libtorch_rng.h`/ATen runner is not TRT-only and must not be copied to Orin under a no-Python label |
| H20 TorchScript/AOTI/native binaries and abstract invocation IR | These are not aarch64 TRT engines or neural ONNX. Export, target compile and complete-model validation remain real work |

All observed I64 inputs fit I32 **in these finite packs**. The audit performs
no cast. A later conversion needs explicit range/roundtrip guards and complete
target output checks, not an arbitrary-input support claim. Source tensor ABI
or checkpoint storage dtype is not evidence of target compute precision.

Possible native postprocessing/RNG providers must be expressed through common
typed contracts and declarations, not model-name branches in runtime or timing
code. No full-model FP32 fallback, quantization search, shorter horizon/loop,
precomputed encoder feature or omitted output is approved by this assessment.

## Source Versions Are Separate

The CogACT data pack contains all 115 released frames from one episode, while
the qualified H20 raw formal profile uses 16 spaced frames. Its separate
autonomous RNG comparison uses one frozen observation/seed schedule and a
different timing boundary. These are distinct profiles, not one combined
reference set. The data pack's old `source_numerical_provider_enforcement=false`
does not inherit later native numerical-provider or autonomous-RNG guarantees.

The earlier pi0 model card is a four-step deterministic fixture and is now
marked historical; it must not replace the real ten-step OpenPI profile.
The pi0.5 H100 single-frame/reference discrepancy remains separate from this
specific H20 16-frame lineage. No H100 model execution or reference merge was
used to make the new check pass.

## Evidence and Next Gate

`artifacts/orin/20261022-02/` contains:

- `source-request-001.json`: five explicit pack paths, fixed pins and IR steps.
- `five-source-data-audit-001/report.json`: full current revalidation, ranges,
  exact nonfloating ports, ten private-copy rejection checks and untouched sources.
- `source-contract-inventory-001.json`: distinct data/raw-host profiles, checkpoint
  parameter metadata, separate RDT components and H20 source-provenance hashes;
  no performance values imported.
- `pi0-data-validate-001.json` and `pi05-data-validate-001.json`: direct public API
  revalidation receipts, each explicitly `offline-data-integrity-only`.
- `source-contract-regression-001.xml` and `source-contract-regression-002.xml`:
  seven host contract/fixture rejection tests, including the later recheck.

After synchronizing the assessment to the live board, eight Node state/service
tests and nine desktop/tablet/mobile browser tests pass. Their receipts are
`artifacts/orin-task-board/source-contract-node-tests-009.xml` and
`artifacts/orin-task-board/source-contract-browser-tests-010.xml`. They validate
the evidence links and scope display, not model deployment support.

RDT must finish and pass its full current Orin experiment before the next
required model enters target deployment. At the user's later 2026-10-22 13:37
UTC request, actual host-side OpenPI and CogACT asset/export preparation now
runs in parallel with RDT; the original no-export assessment above remains its
historical scope, not a restriction on this newly authorized host work. Target
GPU work remains exclusive, with independent target acceptance. pi0, pi0.5 and
CogACT still need verified neural exports/weights, target precision/partition declarations, native
decoded-observation preprocessing, complete chains/pilots, five-process formal
sampling, full output/CDF/resource audits and accurate support/failure boundaries.
This assessment neither runs those models nor marks D0/D1 or the Goal complete.

Later parallel work at 14:11 UTC has produced eight actual pi0/pi0.5 ONNX
graphs; see the [separate export report](orin_openpi_parallel_preparation_20261022.md).
The parent independently rehashed that complete delivery. This supersedes only
the earlier absence of portable OpenPI neural graphs, not this report's frozen
data-pack contents or target-unqualified scope. F64/BF16/Sequence processing,
native full chains and real target acceptance remain required.
