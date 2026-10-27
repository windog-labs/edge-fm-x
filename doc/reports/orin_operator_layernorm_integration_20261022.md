# Orin LayerNorm Whole-Model Integration Pilot

Date: 2026-10-22, Asia/Shanghai. Goal remains active and incomplete.

## Result

Both full SmolVLA variants run on the real Orin. Each retains seven TensorRT
engines, both image inputs, language encoding and all ten solver iterations.
Only the neural prefix engine differs. There is one warmup and sixteen measured
tensor-input calls per variant, not a formal raw-observation E2E experiment.
All 34 complete normalized/native action pairs are independently checked.

| Complete Output, Shape [1,50,6] | Minimum Cosine | Maximum MSE | Maximum Absolute Error | Same Numerical Threshold |
|---|---:|---:|---:|---|
| Normalized action | 0.999998499934 | 5.30081810e-6 | 0.00802004337 | Pass |
| Native action | 0.999998119753 | 0.0121000582 | 0.353607178 | Fail MSE |

These are worst values across all seventeen calls, including warmup. Worst MSE
samples are 9 and 7 respectively, not necessarily the same observation as worst
cosine or absolute error. Neither variant pair is byte-exact. Full per-sample
and per-dimension metrics and raw bytes are retained.

The conservative audit applying MSE < 1e-5 and cosine > 0.9999 to **both** spaces
is `same-target-complete-actions-failed`. That receipt is immutable. The earlier
source qualification explicitly used normalized complete actions as the primary
MSE space. On 2026-10-22 the user confirmed that policy for this graph-ablation
experiment: normalized actions are the primary MSE/cosine gate and native
errors are supplemental. A separate `primary-acceptance-001.json` receipt binds
that decision to the unchanged audit. The LayerNorm tensor-input pilot is now
eligible for raw E2E qualification, not already formally complete. Neither the
threshold nor raw error is changed. GEMM remains rejected because its normalized
MSE itself exceeds the threshold.

The unchanged native-output graph is affine, with per-dimension scale values
26.391779, 52.411404, 49.853806, 36.998302, 59.359951 and 19.039955. MSE is not
invariant to these units. This explains why the two numerical spaces should be
reported separately; it does **not** certify physical action safety, successful
robot behavior or mathematical losslessness.

Candidate normalized and native outputs are each byte-exact to the original
frozen seven-engine candidate pilot for all seventeen calls. New generated
code and this diagnostic therefore do not invalidate the previously audited
same-engine SmolVLA ordinary/shared/replay comparison.

## Source and Execution Controls

- Re-running the generic Norm pass on the complete original prefix reproduces
  candidate ONNX SHA256 `567ef82105c2ebbb7f7afde0faa7c3a960cf80c0711e31bc5a4c10dbef58a147`.
- Exactly 450 FLOAT primitive nodes become 50 FLOAT Norm plugins. Other nodes,
  weights and compute/output precision declarations are unchanged. Both builds
  have matching builder provenance, FP16/noTF32 and 2048 MiB workspace.
- The baseline engine is `fca6c023ad5a72a381f3c2fb83d0f535df369c7279bf27334b7b2c93d30f3927`;
  candidate is `e3eaeaf3988b0a0ff919173275213c24a5fa9b3e27e156329135ddda72ae8ab3`.
- Both generated Sessions have identical complete semantic IR, scheduled Plan,
  output contracts and all 16 input/noise records. Only one artifact entry
  changes; the other six engines and solver tail stay unchanged.
- The inspector names/order for all 43 prefix bindings and all exposed
  descriptors match. Its missing descriptors for input_1/input_3/input_6 are
  explicitly retained as a source-audit limitation. Actual native executions
  additionally pass the frozen backend's type/layout/location/shape binding
  guards for every invocation, including those bindings.
- Target serial `1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`;
  TensorRT 8.5.2.2, CUDA 11.4, unchanged MAXN. Target civil clock is wrong.
  Native backend source SHA256 is
  `01340e10d3c9e61729e84afaef333f929550bb8a20ea3e7f368728f7ed3be2ad`;
  it is retained separately from the evolving working-tree implementation.
- The controller checks >8 GiB available memory before each owned build and
  inference. Minimum admission was 11019180 KiB. A separate 250 ms sampled
  watchdog stops only the controller's owned process group below 2 GiB or on
  timeout. Neither safety condition fired. No external service was stopped.
  GPU-owner checks are current-user-visible, not exclusive-machine proof.
- Both native build/execute phases exit zero. Existing engines are reused;
  the two new native C++ Session builds do not count as fresh TensorRT engine
  compile costs. Total controller duration 57.264504 s includes preparation,
  environment collection and C++ builds and is **not** inference latency.

## Evidence and Reproduction

Evidence root: `artifacts/orin/20261022-00/`.

| Evidence | Location |
|---|---|
| Exact graph / precision / reported-ABI audit | `norm-integration-source-audit-002.json` |
| First inspector coverage-gap attempt | `norm-integration-source-audit-v1-failure.json` and frozen v1 auditor |
| Common-IR pair preparation | `norm-full-model-pair-001/pair.json` and baseline/candidate preparations |
| Target PID/start ticks, phases, resource checks and raw outputs | `norm-full-model-pair-001/pilot-001/` |
| Full native build provenance | `norm-full-model-pair-001/{baseline,candidate}/native-build-r1.json` |
| Independent complete-output audit | `norm-full-model-pair-001/quality-001.json` |
| Exact executed backend implementation | `norm-full-model-pair-001/frozen-tensorrt-backend.cpp` |
| Source, quality, real-pair and owned-cleanup tests | `norm-integration-regression-001.xml`, 22 passed |

Re-audit retained data without any target computation, using a fresh output:

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 python \
  artifacts/orin/20261022-00/audit_norm_full_model_pair.py \
  --pair artifacts/orin/20261022-00/norm-full-model-pair-001/pair.json \
  --capture artifacts/orin/20261022-00/norm-full-model-pair-001/pilot-001 \
  --baseline artifacts/orin/20261022-00/norm-full-model-pair-001/baseline \
  --candidate artifacts/orin/20261022-00/norm-full-model-pair-001/candidate \
  --frozen-candidate-pilot artifacts/orin/20261020-00/native-layernorm-solver-001/pilot-001 \
  --output artifacts/orin/20261022-00/norm-full-model-pair-001/quality-recheck-new.json
```

Actual target argv is retained in `pilot-001/campaign.json`; repeating target
collection requires fresh model/build/output directories. Do not overwrite old
receipts or rerun an existing attempt. All target model compute is native TRT,
not Torch. RDT remains stopped; this pilot does not close the five-model Goal.

## Pilot Follow-Up Gates

Acceptance-space interpretation is resolved; retain the native MSE discrepancy.
Qualify the raw-observation chain and freeze/execute the complete paired
E2E protocol before any integrated speedup is reported. The standalone resident
4.329x operator result is not extrapolated to whole-model performance. Other
five operator families, RDT resource recovery and the remaining required VLAs
remain open. At this pilot checkpoint no integrated formal latency table or
CDF was published; the later completed measurement is linked below.

## Raw-Observation Follow-Up

The user-confirmed primary gate is now applied to complete raw-observation
pilots, not only preprocessed tensors. Two variants each ran 4 warmup + 16
measured calls; all 80 normalized/native output tensors are independently
byte-exact to their respective frozen conventional references. The two graphs
are not byte-exact to one another. Raw preprocessing output hashes match all
128 previously qualified input tensors, including actual noise. The online
runner includes preprocessing, transfers, all seven engines/ten solver steps,
native-output conversion and real completion; file decoding/logging/checks
remain outside the timed interval.

Short pilot mean latency before/after is 125.310064/132.386998 ms, ratio
0.946544x: the candidate is slower. This negative pilot is retained, not
substituted for formal evidence. Both paths are ordinary mode `off`, so no
graph-vs-runtime confounding is introduced. Formal AB/BA/AB/BA/AB sampling of
five process pairs, each 128 warmup + 1024 measured, started subsequently;
the later complete audit is now available: means 125.900306/133.994095 ms,
candidate 6.4287% slower, all 23040 complete output tensors checked. See the
[formal raw E2E report](orin_operator_layernorm_raw_e2e_20261022.md) for full
tails, CDF, quality and resources. This pilot report is not a positive-gain claim.

Evidence root `artifacts/orin/20261022-00/norm-raw-e2e-pair-003/` contains
`pilot-request.json`, `pilot-001/`, `pilot-audit-002/audit.json` and
`formal-request-002.json`. The latter freezes binaries, archives/DSOs, inputs,
engines, output policy, collector and safety-helper hashes. Each owned job has
an 8 GiB admission gate, a sampled 2 GiB floor and 900-second timeout; only its
own process group can be stopped. Resource/temperature/frequency/visible-owner
evidence is recorded. Global settings and unrelated Java services stay unchanged.

The first preparation supplied a compiled IR rather than the exact qualified
source IR and was rejected before target execution (`norm-raw-e2e-pair-001/`).
The next attempt failed native compile because the evolving raw template used
an image callback type absent from the frozen matching host header
(`norm-raw-e2e-pair-002/pilot-001/`). Both failures are retained. The successful
attempt uses the already-qualified general template SHA256
`469ab25f85d884c57dcf2ec851b8b083e306e324e84c86068ce9c964acc08dba`,
with the same declared host pipeline and frozen runtime, not a model-name branch
or ad hoc preprocessing substitute. Thirty-two focused policy/count/raw-output/
cleanup/multi-input regressions and nine browser tests pass; the first unit-test
run's expected-exception-type mismatch is retained in its failed XML.
