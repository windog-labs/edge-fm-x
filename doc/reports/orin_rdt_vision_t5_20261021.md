# RDT Vision and T5 Checkpoint

Date: 2026-10-21. **Progress checkpoint, not completed RDT deployment.**
No RDT action-quality acceptance, E2E latency, throughput or formal CDF is reported.
The five-model Orin scope remains intact; prior SmolVLA results are unchanged.
This is the earlier checkpoint. Subsequent complete target text-chain and
native-contract evidence is in [text chain and contracts](orin_rdt_text_chain_contracts_20261021.md).

## Stage Matrix

| Stage | Current Evidence | Remaining Gate |
|---|---|---|
| Solver | Prior 16 x 5 real state-feedback cases, numerical equality | Integrate with online denoiser, not frozen model outputs |
| Conditions | Prior single-case FP16 pilot with nonzero errors | Re-export affected composites with corrected opmath, full-action validation |
| Vision | Complete six-view native TRT execution; corrected export tested | Large feature error remains; do not call it source-qualified |
| T5 | All 24 blocks split into four segments; portable CPU references ready | Four target engines, complete target text chain and fidelity |
| Denoiser | Carrier source prototype prepared | Corrected export, target parse/build/inference and all five steps |
| Initialize/finish/native output | Source boundaries retained | Target integration and complete publication/finite guards |
| Native Session/raw E2E | Not yet available for RDT | Full 16-input action gate before formal sampling |

## Vision Target Evidence

The original carrier fails actual TRT 8.5 parsing at standard ONNX
`LayerNormalization`. The generic rewrite now accepts that operator as well as
proven primitive dataflow: FLOAT, static last axis, matching affine parameters,
original positive epsilon and no discarded observable statistics. All 55 norms
are rewritten. An unknown intermediate batch dimension is derived from static
upstream shapes through a **shape-only** shadow of the known BF16 rounding
plugin. The real numeric graph keeps that plugin; no batch dimension is guessed.

Engine 002 uses one FP16 convolution, 216 declared FP16 MatMuls, FLOAT arithmetic,
55 FLOAT NVIDIA norms and explicit BF16 rounding. It runs on serial
`1421524312594` with all six `[6,3,384,384]` processed views. Its complete
`[1,4374,1152]` output is finite, but numerically unqualified.

| Candidate vs Original CPU Archive | MSE | Mean Abs | Max Abs | Cosine |
|---|---:|---:|---:|---:|
| Build 002, legacy composite export | 0.2803016582 | 0.2781805684 | 91.375 | 0.9618609910 |
| Build 003, corrected native-composite opmath | 0.2427963941 | 0.2579961735 | 91.75 | 0.9668897932 |

These are intermediate features, not actions. No model-level gate is passed.
Build 003 inspection confirms all 653 plugin layers have FLOAT interfaces and
163 dense implementation layers have FP16 interfaces. Implementation layers
are not asserted to equal source-op or measured kernel-launch counts.

```text
Vision engine 002:
403304d2916369da4ea12d6115a89aaf110a63f7fe0457ab7f0ee272f114ba75
Vision engine 003:
23c4384acfe4d8d0976544de4dbf2fdfb8a98bc6265a5bf219234da4d9a0f88f
Shared exact weight payload:
165031bfa04cbf8fffd54b187690828c63ddcb5266a1108e7df3b9741b12b63d
Required precision-plugin DSO:
7c362ab37bfe96d009fa8ea8d2c978dbe975f1d40a7e3986f07ed5dc73fbde68
```

The checked build wrapper verifies every external payload, graph, precision
profile and DSO before/after native compilation. Its receipt distinguishes
successful construction from numerical qualification. Failed parser/build
records are retained, not overwritten. All computation on Orin remains native
TensorRT/CUDA, with no Torch/torchvision installation or model execution.

## Confirmed Export Bug

The actual vision capture contains native `aten.gelu.default`; T5 instead uses
an explicitly composed activation. Legacy native GELU/SiLU export decomposes
16-bit composites into lower-precision intermediate operators, introducing
rounding absent from the original composite's FLOAT opmath. The new exporter
temporarily registers typed symbolics for native GELU/SiLU, computes their
interiors in FLOAT and casts the result back to the original 16-bit dtype.
It does not promote explicitly composed source arithmetic by pattern or model
name, change weights, or request whole-model FP32 target execution.

| Minimal BF16 Repro | Before Max Abs | After Max Abs |
|---|---:|---:|
| GELU tanh | 0.0078125 | 0 |
| GELU erf | 0.0027770996 | 0 |
| SiLU | 0.015625 | 0 |

The eight-value deterministic loop runs the real archive/export/carrier path.
Old exporter exits 1; corrected exporter exits 0. FP16 cases also pass. Registry
overrides are restored on exceptions and existing custom policies are never
overwritten. Reproduction data, both exporters' hashes and all tiny outputs are
in `opmath-diagnosis/`; `test_export_torchscript_opmath.py` is the regression seam.

Full portable CPU vision improves against the original CPU archive from cosine
0.9767406572/MSE 0.1709341620 to 0.9829230351/MSE 0.1254612265. The original
large discrepancy remains after the local export fix. Therefore the fix is not
presented as a complete vision accuracy solution or as a tuned-kernel speedup.

## Reference Attribution

The frozen H20 direct stage is tied to the exact region pixel tuple: both pixel
payload hashes are
`ab29f03b21fe3762e3eb63460de738b39771d60d89b6ed625319a1afba3fe59c`.
That direct output exactly matches the frozen conditions' image input.
Reference-pair mismatch is not used to dismiss the measured errors.

One otherwise idle 4090 GPU executed **one untimed original-archive forward**
to distinguish CPU/CUDA behavior. No warmup, timing, throughput or performance
experiment was conducted. The original CUDA output itself has cosine
0.9727564797 versus original CPU and 0.9820845789 versus frozen H20 features.
These reference differences are independently recorded; they do not excuse the
target result or establish complete-action quality. TensorRT FP16 rounding,
normalization/provider behavior and full-chain effects remain to be attributed.

## Complete T5 Preparation

| Segment | Original Nodes | Source External Bytes | Target State |
|---|---:|---:|---|
| 0, blocks 0..5 | 420 | 2,578,550,784 | Assets transferred; not built/executed |
| 1, blocks 6..11 | 402 | 2,315,354,112 | Source/portable reference ready |
| 2, blocks 12..17 | 402 | 2,315,354,112 | Source/portable reference ready |
| 3, blocks 18..23 | 415 | 2,315,362,304 | Source/portable reference ready |

The sum is all 1639 source nodes and all 9,524,621,312 external bytes, without
duplicated computation. Dependencies such as relative-position bias are explicit
outputs/routes. No layer or encoder is replaced by precomputed features.
Metadata partitioning avoids a giant protobuf; CLI publication copies payloads
instead of creating hard links rejected by current ONNX security checks.

All four FLOAT-carrier sources and portable CPU references are prepared. Their
complete text output versus original CPU has MSE 3.925620e-6, max abs 0.0234375
and cosine 0.9997725700: not a numerical-equivalence pass. Portable versus
frozen H20 language has MSE 4.252816e-6 and cosine 0.9997536188. Original CPU
versus frozen H20 is also non-exact. Tokens and masks match all 16 series
observations. No T5 target inference is claimed from CPU preparation alone.

## Verification and Continuation

After the fixes, host suite: **2555 passed, 74 skipped, six subtests passed**.
No production C++ or target DSO changed in this checkpoint. The existing solver
equality and frozen SmolVLA measurements remain bounded by their earlier profiles.

Evidence root: `artifacts/orin/20261021-03/`.

- `vision-stage-audit-001.json`, `vision-opmath-stage-audit-001.json`: full target outputs.
- `vision-build-recovered/`, `vision-opmath-build-recovered/`: native builds,
  placement, payload attestations and engine identities.
- `opmath-diagnosis/{before,after}/report.json`: red/green minimal export loop.
- `vision-opmath-attribution.json`, `vision-cuda-reference-report.json`,
  `vision-h20-pairing.json`: independent reference/source checks.
- `t5-partition-report.json`, `t5-carrier-preparation.json`,
  `t5-reference-attribution.json`, `t5-cases/`: complete text preparation.
- `denoiser-carrier-v1.report.json`: pre-fix prototype, not an adoptable quality result.
- `full-host-regression-opmath.xml`: final full host regression.

Next: continue numerical attribution and validate complete target stages, rebuild
affected conditions/denoiser exports using the corrected composite policy, build
and run all T5 segments, and integrate full fresh actions/native Session. Formal
E2E/CDF waits for the complete-action gate. Do not replace action validation with
feature cosine, compilation success or CPU references. All current jobs exited;
the full five-model Goal remains active and incomplete.
