# Orin OpenPI Integration Checkpoint

2026-10-23 local date. Required formal coverage remains **SmolVLA and RDT,
2/5**. No complete pi0 neural invocation, primary action gate, E2E pilot or
formal CDF has been accepted in this checkpoint.

## Actual Target State

| Piece | Actual Evidence | Remaining Boundary |
|---|---|---|
| Complete fresh prefix | v5 fully parsed on Orin after 165 generic FP32 LayerNorm replacements | Engine optimization hit the sampled memory floor; no prefix engine |
| Step engine | 640218807 bytes, `b130aa7e403bc203f7d46db60c840a7a5a9e8d9d8941887e7ecc9e45a1a99d1d`; 42 actual typed I/O tensors inspected | Metadata load only, no step/context inference |
| Initial-time control | Zero inputs; actual CUDA scalar FLOAT output exactly 1.0, one complete case passed | Project the original unused private noise parameter; retain public solver noise |
| Finite control | BOOL `[1]` output; 60 complete `[1,50,32]` FLOAT-input cases passed on GPU | Component scope, not action quality or model inference |
| Native input API | Native aarch64 build, 320/320 complete tensors equal to fresh native CLI and frozen host named profile; 23 invariant and 6 syscall checks pass | Whole Session/E2E integration; old literal-pixel/F64 historical failures remain |
| Native casts/finishing | Five actual-schema providers and all 128 frozen chunks passed earlier GPU smoke | Full neural-output integration remains required |
| New E2E driver | Native target branch compiled, 1201840 bytes, `3deb749efb3439cc478170c1446f84d0cb9ff7fdcc6fdd00da4241d61a6aa2ad` | No real Session DSO/bundle loaded, no performance run |

The time engine SHA256 is
`24302840135788db41901f8541295931ebda84f513dd532a0bea98958aba43a4`;
finite engine SHA256 is
`79df1fc55922ea07ae9107e0dc1576524d8cfd558818e8be03cea5efeaa9adb6`.
Both are bounded **non-neural FP32 controls**, not whole-model FP32 fallback.
The [independent qualification](../../artifacts/orin/20261023-06/control-target-qualification-002.json)
binds inner builds, engines, complete outputs, metadata, actual process start
identity, serial and boot. Finite cases include signed max/normal/subnormal/
zero values, both infinities, quiet/signaling NaNs and first/middle/last invalid
positions. These are 60 finite-predicate cases, not 60 VLA observations.

## Prefix Resource Stop

[Guard terminal evidence](../../artifacts/orin/20261023-06/pi0-prefix-build-003-guard/guard.json)
records 4537 samples. During optimization, `MemAvailable` fell to **1438432
KiB**, below the 2097152 KiB sampled floor. The guard sent SIGTERM only to its
owned build group (controller 3420513, TRT child 3420527); no engine was emitted.
This is a proactive resource abort, not evidence of an unsupported operator
or a kernel-OOM classification. Sampling is not resource isolation.

The inner `build.json` retains `running` because its controller was interrupted
before finalization. It must not be treated as a live job; the outer terminal
guard and missing original PID/start identities are authoritative. Prefix001's
native-LayerNormalization failure and prefix002's missing-profile setup failure
remain separate. No old failure or incomplete output is overwritten.

Business services were not stopped for this attempt. Extension of the prior
RDT-only two-Java-service window has been requested, not presumed authorized.
Until confirmation, do not stop those services. Generic prefix partitioning
is a separate host-side fallback; no reduced/cached-prefix path is substituted.

## Interface And Quality

Actual engine inspections are [step](../../artifacts/orin/20261023-06/step006-inspection-001.json),
[time](../../artifacts/orin/20261023-06/time001-inspection-001.json) and
[finite](../../artifacts/orin/20261023-06/finite002-inspection-001.json).
They deserialize real engines but create no execution context or inference.
Step context memory **50738688 bytes** is required context storage, not a GPU
allocation/peak measurement. TensorRT8.5 vector-only component getters are not
called for non-vectorized tensors; scalar layout is explicitly derived from
queried LINEAR format, actual dtype and vectorization and labeled accordingly.
Earlier probe metadata-query failures are retained; the corrected probe used
the exact same control-engine and reference-metadata bytes.

The [prefix integer audit](../../artifacts/orin/20261023-10/prefix-i64-audit-002.json)
proves 197 saturating Slice-bound uses and one dead cone among 198 oversized
I64 literals. The earlier 162-unproved audit is retained. Omitted axes use
standard ONNX semantics, with independent data-propagation cross-checking;
runtime token casts still require their real checked CUDA provider.

The new driver keeps strict byte-exact regression separate from the explicit
launch/reference-pinned human policy: **complete normalized F32 `[1,50,32]`,
MSE <1e-5 and cosine >0.9999**. Native F64 errors are supplemental; every
complete output must be present and finite. All preprocessing, transfers,
Session work and complete CPU outputs stay inside one request clock; metrics,
file I/O and comparison stay outside. Its software fixtures are NONMODEL tests,
not Orin neural performance. A complete executable Session and full action
validation remain prerequisites for pilot/protocol freeze/formal sampling.

The [actual six-region binding proof](../../artifacts/orin/20261023-08/v4-001/pi0-bound-001/proof.json)
now verifies/physicalizes **6/7 real regions**, leaving only the fresh prefix
uncompiled. It replays control graph transformations and projects only time's
unused private noise argument; public noise, F64 state, all ten steps,
acceptance and atomic full-output commit remain. There is no runnable Session.
Its 69 CPU binding/layout/quality tests are not complete-model execution.

The split backup currently reaches the partitioner's custom-domain rejection.
Its [assessment](../../artifacts/orin/20261023-11/README.md) exports no cuts,
invents no shapes and launches no alternate engine. Trusted plugin-aware
partitioning remains follow-up work, not an available reduced-memory path.
