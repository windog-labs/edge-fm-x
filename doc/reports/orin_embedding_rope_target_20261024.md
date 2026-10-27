# Orin Embedding And RoPE Target Results

Later fixed diagnostic, 2026-10-23 22:01 UTC: all four unchanged graphs/profiles
also build with the sole added `--refit` flag, but zero cases complete the
unchanged precision-observability coverage. No new-engine inference or timing
is performed. See the [negative diagnostic and closure](orin_refit_observability_diagnostic_20261024.md).
The original104results below remain unchanged, not reassigned to new engines.

Checkpoint: 2026-10-23 18:17UTC. These are actual native TensorRT operator
builds, complete-output probes and SDK ABI queries, not VLA E2E timing.
Neither pair is admitted to identical-effective-precision timing: vendor fusion
does not expose the required internal data/storage evidence. That limitation
is retained even though every compared output byte is equal.

## Actual Results

| Case | Real Observations | Complete Values Compared | Same-Orin MSE | Byte Equality | Actual External ABI | Internal Precision / Timing |
|---|---:|---:|---:|---|---|---|
| Language Embedding + fixed scale | 1 distinct instruction; original 16 aliases retained | 46,080 | 0 | Exact | INT32 `[1,48]` -> FLOAT `[1,48,960]` | Unqualified; not timed |
| RoPE/cache | Original observations 10,11 | 113,280 | 0 | Exact | Three FLOAT inputs -> FLOAT `[1,177,5,64]` | Unqualified; not timed |

There are three complete output pairs, six actual output tensors and159,360
compared values. All are finite; cosine is mathematically1 for equal nonzero
arrays. The retained original metric implementation reports RoPE cosine
`1.0000000000000002` from floating rounding, without clamping or editing the
original audit. Fixed numerical gates remain MSE<1e-5 and cosine>0.9999.
No action chunks, independent robot trajectories or whole-model gains follow.

| Case | Baseline Engine SHA256 | Candidate Engine SHA256 |
|---|---|---|
| Embedding | `9fd08533d031608353d0475c63629a56d976070445ea83ef7d9ee4498d0939e2` | `49c3bf3155efc6e2e2e30a946563fa48003fd7da35bc652ff3fb22600344c634` |
| RoPE | `9b1242ad5819d48a41e39096c9fea6b5e94e34ff2584b6416b803c6f145c34ec` | `71847cd82724876f1e0f952cc6f8a80ab4f0d9646746d44f269c176e75d8c6b9` |

Actual inner builder elapsed times are5.189991/5.223596seconds for Embedding
and4.867580/40.650662seconds for RoPE. These are compilation/controller costs,
not inference latency, throughput or a speedup denominator. The fixed source
candidate budget remains one per family, not a new autonomous Agent search.

## ABI And Precision Boundary

Both variants preserve their frozen graphs, inputs, source proofs and precision
declarations. All four actual builds use FP16 enabled, noTF32, OBEY and256MiB
workspace. The original standalone FLOAT numerical islands remain declared;
there is no whole-model FP32 fallback, precision search or silent index cast.
The earlier explicit bounded Embedding INT64-to-INT32 ABI proof remains required.

The original layer-only binding checker rejects the RoPE candidate because
TensorRT labels internal32-wide pointwise slices with the same name as the
64-wide public input. Dropping that mismatch or treating the slice as the
physical binding would be incorrect. Four separately scheduled runs of the
existing native engine inspector confirm the actual public SDK type, shape,
location and linear layout. They load the real pinned engines but create no
execution context and run no inference. Required context-memory bytes, if
reported, are not allocations or physical peaks.

`native_binding_audit.py` changes only the binding source in the frozen full
output auditor: real hash-bound SDK ABI reports replace internal layer aliases.
Its exact one-call substitution and whole-function inverse are recorded.
All source reconstruction, complete file/input/output checks, numerical gates
and non-dense precision qualification remain unchanged. The original auditor
and inspector helper bytes are not edited. The implementation dispatches by
public tensor contracts, not a model name.

Embedding's fused Gather/scale and RoPE's Myelin/pointwise fusion do not expose
all source operations or constant storage with the required provenance. Thus
both reports retain `target_quality_passed=false` and
`same_effective_precision_main_table_qualified=false`, while explicitly recording
the successful complete numerical checks. There is no identical-internal-
precision or microbenchmark-speedup claim, and no automatic configured-only
approval is manufactured from numerical equality.

## Resources And Closure

Target: serial1421524312594, boot3c801f5d-6897-4d97-b021-3c52db38a750,
TRT8.5.2.2/CUDA11.4, existing MAXN unchanged. Requests keep their original8GiB
admission and2GiBruntime floor. Minimum sampled MemAvailable is19,681,012KiB
for RoPE (364records) and18,928,532KiBfor Embedding (147records). These sampled
system values are not isolated-model physical memory or continuous protection.
Restricted process visibility remains, not a proof of GPU exclusivity.

The final read-only target closure rehashes all160captured files and confirms
26distinct controller/builder/probe/inspector identities absent or replaced by
a different process birth. Both original Java PIDs840100/840143remain Running
under Supervisor4237, with their original cwd/argv. No service was stopped or
restarted. This is process-state observation, not application health. At this
historical104checkpoint no VLA formal collection was started. Later CogACT
no-interruption inference uses the original Goal; application-stop permission
remains separately required.

Host tests:66passed, comprising18new native-ABI/full-output checks and48existing
source/typed checks. The first Python3.10attempt's26pass/3failure XML remains:
the pinned old request tool uses Python3.11+`hashlib.file_digest`. Existing local
Python3.13with its installed packages runs the unchanged tools successfully;
no environment installation or source compatibility rewrite is performed.

## Evidence And Reproduction

- [Final quality and target closure](../../artifacts/orin/20261023-104/target-quality-closure-001.json).
- [Embedding complete numerical/native-ABI audit](../../artifacts/orin/20261023-104/embedding-quality-native-001.json).
- [RoPE complete numerical/native-ABI audit](../../artifacts/orin/20261023-104/rope-quality-native-001.json).
- [66 tests](../../artifacts/orin/20261023-104/native-sdk-and-source-regression-003.xml).
- [Attempt scope and tools](../../artifacts/orin/20261023-104/README.md).

Re-audit locally, using the installed host Python3.13and a fresh output path:

```sh
env PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python3 -B \
  artifacts/orin/20261023-104/native_binding_audit.py \
  --root artifacts/orin/20261023-104/rope-native-001 \
  --native-spec artifacts/orin/20261022-04/rope-native-spec-001.json \
  --output /ABS/FRESH-rope-quality.json
```

The operator timing table still has four measured families. These two rows now
have real target builds, exact complete outputs and actual external ABI evidence,
but not accepted effective-precision comparisons or timing. Positive whole-VLA
operator integration, CogACT formal data and other required D4work remain open.
