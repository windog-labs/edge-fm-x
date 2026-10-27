# RDT Same-Orin Execution Modes

Status at this historical checkpoint: complete tensor-boundary mode pilot
passed; original-observation E2E and formal sampling were incomplete. See the
subsequent checkpoint below. The full five-model Orin goal remains active.
Evidence root: `artifacts/orin/20261021-06/`.

Subsequent checkpoint: the [native raw-observation pilot](orin_rdt_native_raw_e2e_20261021.md)
now passes all 112 prepared inputs and 120 complete action tensors byte-exact;
formal sampling has been launched but is not complete. The tensor-boundary
pilot statistics below remain historical and are not substituted for raw E2E.

## Acceptance and Frozen Profile

The user explicitly confirmed that the paper's MSE < 1e-5 and cosine > 0.9999
acceptance compares conventional TRT and EdgeFM on the same Orin, with matching
engines, precision and inputs. H20 remains supplemental. No threshold changed
and no cross-device failure was erased or relabeled as a pass.

The conventional reference is the actual eleven-engine resident C++ pilot
`20261021-05/native-session-target-001/pilot-002/`, not an H20 array or cached
action-head-only result. Its complete native14 and unified128 outputs were
frozen for all sixteen observations, including repeat-sample consistency.
The derived reference manifest is
`rdt-same-engine-modes-002/same-orin-reference/manifest.json`.

All modes preserve eleven real engine hashes, source IR dataflow, five solver
iterations, all six camera views, online language/vision, explicit saved noise,
and the same FLOAT-carrier/FP16-dense profile with declared numerical islands.
The precision DSO stays the older rounding-only provider, SHA256
`7c362ab37bfe96d009fa8ea8d2c978dbe975f1d40a7e3986f07ed5dc73fbde68`.
The separately developed native BF16 MatMul provider is **not** mixed into this
matched comparison. Orin builds/runs no Torch/torchvision model.

Conventional TRT here means optimizations disabled in the same generated typed
C++ Session driver. It is not an independently measured vendor program or an
official PyTorch baseline. Shared-stream and required replay change only the
execution policy over the same artifacts.

## Protocol and Complete Outputs

Three sequential independent worker processes, each 4 warmups + 16 measured
requests, batch one. All sixteen input sets occur among the measured requests.
Each call recomputes fresh encoders and all five action steps. The sixteen
recorded histories repeat; these are not independent robot trajectories.

`same-orin-mode-audit-002.json` independently checks hashes, all horizons,
sample pairing, native process maps, host-timing equations, device identity,
engine equality, and per-call/final replay telemetry. The board serial is
`1421524312594`; its civil clock is incorrect, so durations use monotonic time.
The collector checks all input file hashes before and after execution. Its
Python process does not preload the precision DSO; only the native C++ worker
loads it. Visible GPU-owner guards remain intact; inaccessible process
directories are an explicit ownership-coverage limitation.

All **120 complete action tensors / 545,280 FLOAT values** match the frozen
same-Orin baseline byte-for-byte, including warmups. Native output is
`[1,64,14]`, unified output `[1,64,128]`. All outputs are finite; MSE, max-abs
and mean-abs are zero, minimum computed cosine `0.9999999999999999`.
Normalized quality uses the original active14 coordinate mapping, not padding
in the 128-wide auxiliary. Full unified tensors are also compared in entirety.
Byte equality is reported separately from tolerance acceptance and is not a
claim of original-BF16/H20 equivalence, physical calibration or task success.

| Mode | Calls Including Warmups | Complete Tensors | Actual Replay / Ordinary Loop Calls | Captured Steps |
|---|---:|---:|---|---:|
| off | 20 | 40 | no replay policy | 0 |
| batch-only | 20 | 40 | 0 / 20 | 0 |
| required | 20 | 40 | 20 / 0 | 5 |

Every required request reports ready state and zero ordinary fallback. Only
the complete five-step action loop is replayed; encoders remain online outside
that loop. Capability flags alone are not treated as replay evidence.

## Pilot Timings Only

Boundary: already prepared CPU tensors to complete CPU-consumable action chunks,
including per-call H2D, binding, inference, D2H and actual GPU completion.
It excludes online decoded-observation preprocessing. Reference checking, file
IO, telemetry and initialization are outside timing. Each row has only sixteen
measured observations from one process; no formal CDF is claimed.

| Mode | Mean ms | p99 ms | Max ms | Std ms |
|---|---:|---:|---:|---:|
| off | 2957.560252 | 2960.941814 | 2961.191616 | 1.457385 |
| batch-only | 3007.956056 | 3018.935382 | 3019.305952 | 10.302027 |
| required | 2898.035200 | 2909.653458 | 2911.249472 | 3.993402 |

The slower batch-only result and larger replay standard deviation are retained.
The small replay mean improvement is not promoted to a formal speedup claim.
At these pilot rates, the unchanged three-mode 5 x (128 + 1024) protocol would
take about 14.2 hours of tensor-boundary worker inference alone. Raw-input
preprocessing, initialization and audits add time. This estimate does not
authorize reducing samples or replacing formal results with the pilot.

## Independent Audit Checks

Ten artifact tests pass: explicit effect allowlisting, rejection of RNG and
unreviewed plugin versions, complete checked observations, truncation rejection,
large output-error rejection, and nonzero fallback rejection after rehashing
the corrupted fixtures. A one-ULP fixture remains within tolerance but is
correctly reported **not byte-exact**, preventing those claims from being mixed.

The effect review covers standard functional ONNX nodes plus the exact reviewed
`VLAForgeBFloat16Round/1` schema and the frozen provider. It is bound to graph
hashes from successful real target builds. It is not a graph-validity or capture
certificate by itself. One first preparation attempt rejected unlisted `Clip`;
after reviewing that stateless operator, a fresh preparation succeeded. The
failed directory remains intact.

Raw evidence:

- `rdt-same-engine-campaign-001/campaign.json`.
- `mode-retrieval-001/rdt-mode-{off,batch,replay}-20261021-06-002/`:
  native build receipts, metadata, complete outputs, timing and maps.
- `same-orin-mode-audit-002.json`, with active-coordinate proof identity.
- `same-engine-audit-regression.xml`: 10 passed.
- `20261021-05/complete-action-audit-001.json` and
  `three-way-action-attribution-001.json`: retained cross-device diagnostics.

## Separate Precision Candidate

The optional TensorRT-8 native BF16 MatMul provider is implemented generically,
without model-name branches or a hand-written GEMM. It uses cuBLAS, supports
static rank-2..8 and batch broadcasting, declares conversion/library workspace,
and owns independent context handles outside enqueue. It does not silently
recast FLOAT MatMul, Conv or Gemm, or enable full-model FP32 fallback.

Native provider CTests pass 20/20, including six complete GEMM/broadcast,
clone/serialization and actual CUDA replay cases plus the existing 327,680-value
rounding test. Provider DSO SHA256:
`6f9a2eaa71e8daaf177d6d78068cda258517b5ea2e00c61038879f7f0917b9fd`.
Build snapshot SHA256:
`19330efb4f12fb0bf9b74039a9788fedb119275b3384d4d3d722dcc679cfb34e`.
The first source archive missed an existing test fixture and failed CMake
configuration; the successful snapshot includes it, without disabling tests.

Host lowering/architecture tests pass 28/28; full host suite passes 2602 with
74 explicit skips and six subtests. Static Shape/Gather/Concat propagation has
a failing-before/fixed-after regression and never substitutes a dynamic shape.
A separate real vision graph prepares all 216 BF16 MatMul and 55 LayerNorm
rewrites; its 1,651,954,824-byte weight payload is byte-identical to the earlier
carrier, SHA256
`165031bfa04cbf8fffd54b187690828c63ddcb5266a1108e7df3b9741b12b63d`.
It is not target-model integrated or a speedup/Agent ablation result. With the
primary acceptance clarified, this candidate does not block the ordinary
matched-profile RDT deployment campaign.

## Remaining Work

Implement and qualify generic native image padding/bicubic resize/normalization,
BF16 carrier conversion, state mapping and typed text masks against all sixteen
frozen input sets; then integrate online preprocessing into the existing native
three-mode runner. Only after original-observation pilot equivalence should the
unchanged formal E2E/CDF protocol run. Continue the required pi0, pi0.5 and
qualified CogACT models afterward; optional Qwen remains lower priority.
