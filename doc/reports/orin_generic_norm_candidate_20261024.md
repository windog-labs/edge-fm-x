# Generic FP32 Norm Candidate On Orin

Checked 2026-10-24 01:36 UTC. **Numerical micro gate passes; candidate not
adopted. Timing is pilot-only, not a qualified full-count ablation or VLA E2E.**
Formal five-model coverage remains4/5. Prior results are unchanged.

## Source And Contract

One explicitly budgeted generic CUDA candidate replaces the current installed
NVIDIA `LayerNorm` plugin with `VLAForgeLayerNormF32`. It uses centered FP32
mean/variance, CUB block reduction, cached float4 inputs and a scalar fallback.
CUDA11.4/SM87 compilation uses `--fmad=false`, without fast-math or precision
search. This is an isolated artifact, not an adopted backend change.

The real source signature is `[1,1024,768]`, original observation6 CPU-source
activation, unchanged gamma/beta and epsilon1e-6. It is not the intermediate
activation of today's mixed-precision Orin full model. Restoring only the
plugin op_type reproduces the original graph exactly; both precision maps are
unchanged. Candidate inspector confirms Device/LINEAR/FP32 throughout.
No model-specific branches, Torch, global SDK changes or application outages.

[Frozen plan](../../artifacts/orin/20261024-01/plan-001.json).
The old primitive-expression-to-NVIDIA-plugin experiment is not relabelled or
reversed. Its full-model6.43% slowdown remains a separate negative result.

## Complete Operator Outputs

Native TensorRT execution saves every786432value for each implementation.
Independent host re-reading checks finite complete outputs, not a cropped
prefix. Original primitive gates remain MSE<1e-10, max-abs<1e-4, cosine>.999999.

| Comparison | MSE | Max Abs | Cosine |
|---|---:|---:|---:|
| Current NV plugin vs new candidate, same Orin | 1.78431677085e-15 | 1.90734863281e-6 | 0.9999999999999944 |
| Frozen CPU reference vs current NV plugin, supplemental | 2.31820537576e-15 | 2.86102294922e-6 | 0.9999999999999930 |
| Frozen CPU reference vs candidate, supplemental | 2.01541832275e-15 | 2.86102294922e-6 | 0.9999999999999941 |

These pass tolerances but are not byte-exact, complete actions or robotic task
quality. Other widths, unaligned paths and synthetic corner cases are not newly
qualified by this one actual signature.
[Full numerical audit](../../artifacts/orin/20261024-01/quality-001.json).

## Timing Refusal And Non-Adoption

Frozen requests: five alternating independent pairs per boundary, one stream,
1000ms warmup and1024iterations/process. All20native trtexec processes exit0,
but `resident-1-after` exports1023records. Actual total20479, not20480.
No record is synthesized, removed, selected away or silently retried. The
reason for this export discrepancy is not proven. **Formal timing acceptance
is refused.** These are descriptive pilot statistics only:

| Native TRT Boundary | Current NV Mean | Candidate Mean | Current/Candidate | Actual Counts |
|---|---:|---:|---:|---|
| Resident managed-IO GPU event | 44.866573us | 43.539719us | 1.030475x | 5120/5119 |
| Transfer-inclusive GPU event | 296.315950us | 299.373519us | 0.989787x | 5120/5120 |

Resident pairs0/1 regress; transfer-inclusive pairs1/2/3/4 regress. The resident
aggregate is not a consistent or significant speedup. Maximum728.882us remains
in the data. These event boundaries are not decoded-CPU-observation-to-action
latency. No complete-model engine build, action validation or E2E run is started
from this mixed/protocol-incomplete pilot. D4 integration remains open.

[Timing audit](../../artifacts/orin/20261024-01/timing-audit-001.json),
[all20479records](../../artifacts/orin/20261024-01/tables-001/all-pilot-records.csv),
[non-adoption decision](../../artifacts/orin/20261024-01/decision-001.json).
One candidate, no inference/build failures; engine build costs4.620220224s.
Controller/environment collection time is not inference latency.

## Closure And Next Gate

Quality and timing captures independently close43/112target hashes and7/22
tracked birth identities, respectively. Both owned units are success/inactive,
MainPID0, empty post cgroups. Original Java942765/942805/start26395592/26395711
and Supervisor remain unchanged. Eighteen actual/source/corruption/protocol
tests pass; no deployment or service-health acceptance follows.

Fresh available memory22264460KiB(~21.2GiB)still fails CogACT's frozen24GiB
admission. No CogACT restart or additional service interruption is attempted.
[Fresh capacity/unit observation](../../artifacts/orin/20261024-01/post-capacity-001.json).
CogACT needs stable admitted resources for a fresh whole cohort; the old partial
cohort is not merged. Qwen remains optional after core. Goal is incomplete.

Read-only re-audit: `python3 -m pytest -q artifacts/orin/20261024-01/test_candidate.py`.
Do not rerun the one-shot prepare/launch scripts or overwrite sealed receipts.
