# Orin Attention Full-Model Integration

Checkpoint: 2026-10-23 18:44UTC. **The complete-model candidate fails the frozen
normalized-action MSE gate and is not admitted to raw E2E performance sampling.**
All data and source/build records are retained. The earlier isolated Attention
microbenchmark remains valid within its own boundary; it does not qualify this
full-model graph or imply an integrated speedup.

## Full Model And Source Contract

The candidate reuses the already qualified positive-zero-mask recipe in the
adopted SmolVLA complete prefix:2088to2087nodes, only `/Add_138` removed, all840
original initializers and other nodes retained. Source/profile hashes are the
ones frozen in the [source preparation](orin_attention_integration_preparation_20261022.md).
Retained compute/output declarations are unchanged. Only the removed Add's two
FP32 declarations disappear; the report explicitly does not call every original
declaration unchanged. No custom kernel, quantization, changed solver/noise,
reduced horizon or whole-model FP32 fallback is introduced.

The actual candidate engine is657,544,346bytes, SHA256
`5a9aec6fd6a25ec1c98ef5eb2cdfce7b71c66f7ec918e0f1aed8c5e6e4bf57ac`.
The adopted original prefix remains SHA256
`e3eaeaf3988b0a0ff919173275213c24a5fa9b3e27e156329135ddda72ae8ab3`.
The existing guarded build uses FP16 enabled/noTF32,2048MiBworkspace and OBEY.
The171.8409104-second guard window is compilation/controller time, not latency.
Reported binding order and40exposed descriptions agree across43ports; the old
three unreported inputs remain explicitly unreported, not a fabricated complete
standalone inspector certificate.

Both ordinary native Sessions actually compile and run allseven engine stages
and allten action iterations. Only the prefix artifact changes. Their complete
IR, Plan, output contract,16real input observations and saved noise agree exactly.
Each mode executes1warmup+16measuredcalls, yielding34complete public-output pairs
(normalized and native, each F32`[1,50,6]`), including warmup. The baseline is
byte-exact to the original adopted pilot for both outputs. This is a tensor-input
quality experiment, not decoded-observation E2E, a formal CDF or robot success.

## Numerical Result

| Complete Output / Reference | Worst MSE | Minimum Cosine | Worst Max-Abs | Decision |
|---|---:|---:|---:|---|
| Normalized / adopted baseline |1.2796944331e-5|0.9999961222|0.0107544661|Primary gate failed, sample5 |
| Native / adopted baseline |0.0277434733|0.9999978819|0.5169487000|Supplemental error |
| Normalized / existing unchanged-source rebuild |1.2985515613e-5|0.9999960667|0.0103515387|Primary threshold also failed |
| Native / existing unchanged-source rebuild |0.0285871906|0.9999978634|0.5425376892|Supplemental error |

All complete arrays are finite. The adopted-baseline comparison passes16of17
normalized call checks, but one failure disqualifies the candidate. The fixed
gate remains MSE<1e-5and cosine>0.9999; high cosine does not cancel an MSE failure.
Native-unit errors are supplemental under the explicit human decision, not the
reason the primary gate fails. Full per-sample/per-dimension errors are archived.
Normalized worst per-dimension MSEs for dimensions0..5are
`[1.501568e-5,3.027296e-5,1.056690e-5,1.090749e-5,1.011498e-5,5.728324e-6]`.

## Existing Rebuild Control

No additional model or engine experiment is run to obtain the control. The
existing unchanged-ONNX/unchanged-profile rebuild is re-audited from its original
build records, with full profile bytes, settings and source hashes matched.
Its prefix engine is
`b0b77654c8b4312e6fe355709a12afa54005c83e3e2ddec1935b2fcbffd8c42e`.
Archived complete outputs match this experiment's IR, Plan, all inputs/noise,
request ordering and other six engines, so the existing comparator can reuse
them without a new GPU run or a change of baseline.

The rebuild vs adopted baseline has normalized worstMSE4.5192150e-7and minimum
cosine0.9999998749:within the gate, but not byte-exact. The Attention candidate
still fails relative to that control. This does not completely separate all
vendor tactic variability from graph effects; the existing control does not
remove the observed failure. Neither comparison supersedes the original failed
candidate audit. No precision search, tolerance relaxation or retry-based
selection is used to manufacture a pass.

## Resources And Closure

Real target: serial1421524312594, boot3c801f5d-6897-4d97-b021-3c52db38a750,
TRT8.5.2.2/CUDA11.4, existing MAXN unchanged. The original guarded full build
keeps12GiBstartup,2GiBruntime floor and900-second timeout. It starts with
20,532,756KiBavailable and samples a minimum15,716,168KiBacross319records.
The existing short-pilot guard keeps its separate8GiBstartup/2GiBfloor;
minimum sampled values are19,086,348KiBbaseline and19,214,664KiBcandidate.
These are sampled system availability, not isolated-model physical peaks.
Owner visibility remains permission-limited and does not prove exclusivity.

Final read-only closure rehashes88target paths, including source, profile,
actual engines, generated model files, binaries and complete captures, and
confirms10owned controller/builder/runner identities terminal. Both original
Java PIDs840100/840143remain Running under Supervisor4237. No service is stopped
or restarted in this attempt; this observation is not application health.

All45host checks pass, including real complete-output failure reproduction,
source/elision rejection tests, exact generator-function inverse, unchanged
seven-stage/ten-step IR/Plan checks and existing failed-GEMM controls. The old
source/preparer/auditor files remain unchanged. The new pair generator changes
only acceptance of an explicitly audited elision contract; all numerical,
runtime, loop and output generation logic stays unchanged.

## Evidence

- [Failed candidate and target closure](../../artifacts/orin/20261023-105/failed-integration-closure-001.json).
- [Actual source/build audit](../../artifacts/orin/20261023-105/build-source-audit-001.json).
- [Complete34output-pair audit](../../artifacts/orin/20261023-105/pair-generated-001/quality-001.json).
- [Explicit normalized-primary rejection](../../artifacts/orin/20261023-105/pair-generated-001/primary-acceptance-001.json).
- [Unchanged-source control proof](../../artifacts/orin/20261023-105/reused-source-control-audit-001.json).
- [Candidate vs control full actions](../../artifacts/orin/20261023-105/rebuild-control-actions-001.json).
- [Control vs adopted baseline](../../artifacts/orin/20261023-105/rebuild-vs-original-actions-001.json).
- [45regressions](../../artifacts/orin/20261023-105/elision-pair-regression-003.xml).

Reproduce the numerical audit locally, with a fresh output path:

```sh
env PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python3 -B \
  artifacts/orin/20261022-00/audit_full_model_operator_pair.py \
  --pair artifacts/orin/20261023-105/pair-generated-001/pair.json \
  --capture artifacts/orin/20261023-105/pair-generated-001/pilot-001 \
  --baseline artifacts/orin/20261023-105/pair-generated-001/baseline \
  --candidate artifacts/orin/20261023-105/pair-generated-001/candidate \
  --frozen-adopted-pilot artifacts/orin/20261020-00/native-layernorm-solver-001/pilot-001 \
  --output /ABS/FRESH-attention-action-audit.json
```

No raw E2E timing is started for this failed graph. A positive, fully qualified
whole-model graph benefit remains unproved. Required VLA formal coverage stays
4/5; CogACT formal authority and other mandatory D4items remain separate.
