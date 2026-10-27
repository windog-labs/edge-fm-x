# SmolVLA Native TensorRT Pilot on Orin

Date: 2026-10-20. Status: complete ordinary-mode Session execution verified;
supplemental cross-device MSE gate failed. This is not a formal E2E result,
same-target optimization comparison, or full Goal completion.

## Scope and Identity

- Board: Jetson AGX Orin, serial `1421524312594`, L4T 35.4.1, CUDA 11.4,
  TensorRT 8.5.2.2. No system SDK replacement or Torch model runtime.
- Local evidence: `artifacts/orin/20261020-00/` in this worktree.
- Remote engine root: `/data/datalog/edgefm-orin/20261020-00/trt-smolvla-001/`.
- Checkpoint SHA256:
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
- Existing input-pack SHA256:
  `063a0512ae0f98b3d290fc8131b0016aa2d5f3a1235be72f796c3d451ca504cb`.
- Frozen H20 complete-reference manifest SHA256:
  `36bf147a901f72e0ba43512f34bf7b13e7a42dd34024c7a1b4f00a1e48e1cdad`.

The 16 saved observations/noise payloads match the original H20 input records.
The host interface uses processed two-view FLOAT `[1,3,480,640]` images,
state `[1,6]`, checked INT32 tokens `[1,48]`, masks and exact saved FLOAT noise
`[1,50,32]`. INT64 token/counter narrowing was checked; BF16-to-FLOAT cache
specialization is explicit and is not a numerical equivalence certificate.

## Real Engines and Session

| Stage | Target directory | Engine SHA256 |
|---|---|---|
| Prefix | prefix-build-005 | 6611dac93662dcc62f110411c955b3bdb2d9ed5662c93218dcc896c7af65e392 |
| Initialize | initialize-build-001 | 613351ddc8fd7b64d548e39d6a940562c93b1440ba8c8ef1b49bfdd7e5728c3c |
| Step | step-build-001 | 7c3389406c52e6724efa726c3408f57b41aeddd1f59603a585eaee21b40867fe |
| Finish | finish-build-002 | 6ee9409be2217958b61eb7fb400cd1f062fd4db243be527153c8c9e673582951 |
| Native affine output | native-output-build-001 | 6d37341f66e14991c09aadf16e5368821701860a45d40a9d8877a80c7bb7264f |

Matrix/conv precision is FP16; declared arithmetic, normalization, comparisons
and terminal affine operations use FP32 where required. This is not a full
FP32 fallback and no quantization-parameter search was performed.

The generated native Session retains the existing Invocation IR, all ten solver
steps, full finite guards and transactional publication of both action outputs.
One full call executes 14 engine invocations. Engine physical output ordering
is reconciled with a generic pure-Region interface permutation; public SSA
identities, external ports, dtype/shape and consumers are preserved.

Native preparation/build/pilot evidence is in `native-session-001/`:

- `preparation.json`, `bundle/metadata/semantic_ir.json`, `scheduled_plan.json`
  and `output_contract.json` bind the frozen source and typed interface.
- `native-build-r1.json`, configure/compile/ELF/ldd logs prove actual target
  compilation. Initial failed controller evidence is retained on the target.
- `pilot-001/pilot.json`, `process-maps.txt`, `samples.csv`, `host-timing.csv`
  and complete `outputs.f32` / `output-1.f32` prove actual collected execution.
- `pilot-001/action-audit.json` is the independent full-output reference audit.

The pilot PID `385842` (start ticks `1821793`) exited 0. Its generated source
digest is `da71c8890ca763fe7ed8ad15dc087b83ed7abf51b8da6edc0202a20e608f023f`;
IO schema digest is
`1f96cd3a02e89c4304ffca33f5c3d2082f04a36820ca484f912a0427ccec08d6`.
It used ordinary execution, not shared-stream optimization or full-loop replay.

## Complete Action Audit

Every one of the 17 calls (1 warmup + 16 measured) saved two complete FLOAT
`[1,50,6]` tensors. The audit verifies declared file/array hashes, dtype/shape,
row/sample mapping, all coordinates and per-action-dimension errors. Nothing
is truncated to a first step and the warmup is not excluded from correctness.

| Output | Minimum cosine | Worst MSE | Maximum absolute error | Worst mean absolute error |
|---|---:|---:|---:|---:|
| Normalized | 0.9999852940845904 | 6.393668376929344e-5 | 0.02990245819091797 | 0.006640367954969406 |
| Native units | 0.9999729170710759 | 0.16032532442839373 | 1.090863585472107 | 0.30577606171369554 |

All outputs are finite. Normalized sample 9 has both the worst MSE and lowest
cosine. Cosine clears `>0.9999`, but MSE fails the frozen `<1e-5` paper gate.
Native-unit error is separately reported and not hidden by normalization.
High cosine does not prove mathematical losslessness or physical robot quality.

This reference is original H20 output, not a same-target Orin TRT baseline.
The original export/precision changes and the target FP16 arithmetic are not
yet fully separated. The failure remains open; no tolerance was relaxed and
no cross-platform latency speedup is computed.

## Actual Guard and Transaction Tests

`terminal-probes-003/results/audit.json`: 28/28 checks passed using the exact
finish/native engines listed above. The complete input, including dimensions
discarded by slicing, is checked. NaN and both infinities are rejected; maximum
positive/negative finite inputs are not accidentally rejected by a narrowed
half-precision bound. The native stage rejects output overflow and preserves
incoming false. BOOL results match byte exactly; complete floating outputs
are compared at `rtol=1e-6, atol=3e-5`, preserving NaN/Inf patterns.

The first probe setup named a nonexistent local spec (001); the next lacked
the remote environment probe (002) and did not reach GPU execution. Those
attempts remain preserved; 003 is a distinct successful run, not a retry that
overwrites an earlier result. The 002 controller state may be stale because
its final snapshot failed; the observed terminal exit takes precedence.

`session-fault-probe-001/results/audit.json`: the native probe creates two
actual complete Sessions and traces 210 Region calls across 15 full-model
calls. It injects NaN, positive Inf and negative Inf into the first and last
noise coordinates. Each of six invalid calls executes all 14 Regions, returns
validation status 7, emits one transaction abort and commits no new output.
The previously published normalized/native chunks remain byte exact, as do
the unaffected Session's outputs. Recovered and post-reset actions are byte
exact with the original Orin pilot; reset makes old outputs unavailable and
an unbound call is rejected before any Region executes.

This covers ordinary-mode rejection, retained publication, basic isolation and
reset, not optimized-path capture/replay or long-run memory/lifecycle behavior.
Probe processes exited; none is a persistent model service. Environment-before
and environment-after records retain SDK/serial, visible GPU owners, memory,
temperature and permission limitations. Visible GNOME processes remain; root
FD visibility is incomplete. These are not exclusive-bench attestations.

## Timing Boundary

The measured tensor-boundary pilot samples range from 122.162400 to 127.185792 ms,
mean 125.172548 ms over 16 calls. They include host tensor H2D, binding, complete
native model execution, publication and D2H. They exclude original RGB/state/text
preprocessing and are not formal E2E latency/throughput or a continuous CDF.
Target civil time is wrong; elapsed measurements use monotonic clocks. No claim
of a fixed 60W mode, thermal stability or uncontended GPU ownership is made.

## Verification and Remaining Work

Focused local regression has 107 passes in
`audit/trt-native-session-regression-003.xml`, including actual compiled CPU
runner fixtures, mixed-type Region permutation, collect-only opt-in/legacy
template rejection, conservative opset lowering and terminal ONNX guards.
CPU fixtures are not Orin model evidence. Expanded regression on existing host
Python 3.13 passed 520 tests with 40 opt-in native CUDA/LibTorch tests skipped:
`audit/trt-native-session-regression-expanded-002.xml`. The prior Python 3.10
attempt (140 failed, 380 passed, 40 skipped) is retained in `expanded-001.xml`
under the same prefix; it hit existing `hashlib.file_digest`/dependency
incompatibilities and is not called passing. No unrelated compatibility changes
or new environments were introduced. Host Torch-dependent unit tests do not
install or run Torch on Orin and do not constitute another platform benchmark.

Reproduce host regression with the existing base interpreter (not the incomplete
legacy Python 3.10 package prefix):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python \
python3 -m pytest -q --tb=short \
  vlaforge/tests/codegen vlaforge/tests/transforms vlaforge/tests/deployment \
  vlaforge/tests/unit/test_materialize_onnx_terminal.py \
  vlaforge/tests/unit/test_lower_onnx_opset18_to17.py
```

The exact native build/run argv and hashes are in `native-build-r1.json`,
`pilot-001/pilot.json`, `terminal-probes-003/results/execution.json` and
`session-fault-probe-001/results/execution.json`. Reuse the hash-bound source
and inputs but allocate a new run/output directory for reproduction; controllers
intentionally reject overwriting existing evidence. The supplementary host
audits are `audit_complete_action_pilot.py`, `audit_terminal_probes.py` and
`audit_session_fault_probe.py` under the evidence root. Each output report
records the exact audit script and input identities.

Before formal results: establish a same-target conventional/optimized TRT
comparison; attribute the supplemental reference error; implement original-
input preprocessing; qualify shared-stream/full-loop replay and lifecycle;
then freeze 5 x (128 warmup + 1024 measured) sampling, telemetry and soak.
At this first-pilot checkpoint the TRT backend exposed ordinary execution only.
The subsequent generic sidecar/native CUDA Graph implementation has now passed
actual same-engine target pilots, described in the
[execution-mode report](orin_smolvla_trt_execution_modes_20261020.md). That later
evidence, not pre-existing AOTI/Torch support, establishes TRT mode equivalence.
RDT, pi0, pi0.5 and qualified CogACT still require complete Orin deployment.
Qwen remains lower priority. No other platform performance campaign was added.
