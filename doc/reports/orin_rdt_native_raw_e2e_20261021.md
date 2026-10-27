# RDT Native Raw-Observation E2E Checkpoint

Status: complete raw-observation **pilot passed**; fixed formal sampling has
been launched, but collection, final audit and CDF remain incomplete. The
five-model Orin goal is active and incomplete. Evidence root:
`artifacts/orin/20261021-07/`.

## Acceptance and Matched Artifacts

The user explicitly confirmed on 2026-10-21 that MSE < 1e-5 and cosine > 0.9999
apply to conventional TRT versus EdgeFM on the **same Orin, precision, engines
and inputs**. Frozen H20 output differences remain supplemental, including all
previous negative results. No tolerance has changed. This checkpoint does not
certify original BF16/H20 equivalence, physical calibration or robot success.

All eleven engines and the five DPMSolver steps `[999,799,599,400,200]` remain
identical to the [matched tensor-boundary pilot](orin_rdt_same_engine_modes_20261021.md).
The older rounding-only precision DSO is retained, SHA256
`7c362ab37bfe96d009fa8ea8d2c978dbe975f1d40a7e3986f07ed5dc73fbde68`.
The separate native BF16 MatMul candidate is not used. The existing explicit
FLOAT-carrier/FP16-dense profile and range-required numerical islands are not
relabeled as native full-BF16 or whole-model FP32 execution.

Conventional TRT means the optimizations-disabled generated native Session,
not an independently measured vendor application or an official PyTorch
baseline. Shared-stream and required replay change execution policy only.
No Torch/torchvision model runs or new package installations occurred on Orin.

## Online Input Contract

Six decoded `[480,640,3]` camera arrays retain the frozen source's channel
order: OpenCV BGR bytes passed unchanged to PIL. No new BGR/RGB permutation or
brightness correction is introduced. The history/view order is unchanged.
Padding uses centered 640-square images with background 127, followed by
384-square bicubic resize, exact declared rescale/normalization and explicit
BF16-RNE values stored as FLOAT.

The optional generic image module delegates resampling to unmodified Pillow
10.4.0 C imaging sources, not a handwritten bicubic approximation. Archive
SHA256: `166c1cd4d24309b30d61f79f4a9114b7b2313d7450912277855ff5dfd7cd4a06`.
The task keeps its source archive and license. Standalone C hooks replace no
image math and are serialized by a module mutex. Python platform headers are
compile-time only: target `ldd`, undefined-symbol audit and worker process maps
show no libpython, Torch or c10 runtime dependency.

The generic scatter/divide helper maps the fourteen native state coordinates
into the declared 128-wide state and availability mask. Indices, divisors and
rounding are declarations, not model-name branches. The original state-format
method and output-mapping proof are hash-bound. HF Rust tokenization emits
I32 token IDs and I32 masks with the fixed six-token profile; overlong input is
rejected, not silently truncated.

All sixteen noise arrays are the original saved noise, not RNG redraws or
seed-only reconstructions. Bounded control frequency is explicitly I32 and
verified against frozen I64 values. No processed image feature or language
prefix is stored as a final-benchmark input. The raw manifest records that no
offline image preprocessing or model execution was performed when packaging.

Independent audit `native-input-audit-001.json` verifies **112/112 complete
target-prepared input tensors byte-exact** with the frozen model-input files:
action mask, frequency, noise, pixels, text mask, token IDs and unified state
for all sixteen observations. Target native CTests pass 15/15.

## Complete Raw E2E Pilot

Three sequential independent processes, each four warmups plus sixteen
measured requests, batch one. Every request performs fresh preprocessing,
H2D, binding/revision updates, online vision and the complete 24-block text
chain, all five action iterations, output conversion and D2H. No precomputed
encoder features substitute for those stages. All horizons are 64.

Timing starts with decoded camera/state/text/noise in CPU memory and ends when
both complete CPU outputs are consumable. File loading, image decoding,
initialization, output serialization, checking and logging are outside timing.
All per-request segments conserve the direct E2E elapsed time. The sixteen
recorded histories repeat and are not sixteen independent robot trajectories.

`raw-e2e-audit-001.json` independently verifies all **120 complete tensors /
545,280 FLOAT values**, including warmups. Shapes are native `[1,64,14]` and
unified `[1,64,128]`. Every tensor is finite and byte-exact with the frozen
conventional same-Orin reference. Maximum MSE, max-abs and mean-abs are zero;
minimum computed cosine is `0.9999999999999999`. Active normalized coordinates
are checked with the original fourteen-coordinate projection, not padding.

| Mode | Calls | Complete Tensors | Captured Steps | Actual Replay / Ordinary Calls |
|---|---:|---:|---:|---|
| Conventional | 20 | 40 | 0 | no replay policy |
| Shared-stream | 20 | 40 | 0 | 0 / 20 |
| Required replay | 20 | 40 | 5 | 20 / 0 |

Only the complete action loop is replayed; encoders remain online outside it.
Every required call reports ready state, five captured steps and zero ordinary
fallback. Real counters, not capability declarations, establish execution.

| Mode | Pilot Mean ms | Pilot p99 ms | Pilot Max ms | Pilot Std ms | Mean Online Preprocessing ms |
|---|---:|---:|---:|---:|---:|
| Conventional | 3018.828226 | 3047.027594 | 3047.648032 | 18.901178 | 48.264814 |
| Shared-stream | 3049.227250 | 3057.300266 | 3057.469024 | 5.377772 | 48.349568 |
| Required replay | 2944.120930 | 2957.833590 | 2959.331968 | 4.777634 | 48.221478 |

These are single-process pilot statistics, not formal main-table/CDF results.
The slower shared-stream result is retained. No samples are centered, trimmed
or discarded, and no formal runtime speedup is claimed from this short pilot.

## Formal Collection in Progress

The protocol remains three modes, each **five fresh independent processes**,
**128 warmups + 1024 measured calls per process**, round-robin rotated mode
order. Each mode will have 5120 measured requests; all 17,280 requests including
warmups preserve complete outputs. The watchdog uses 600 seconds plus twice
the pilot maximum multiplied by 1152; it changes only timeout handling, not
sample counts or latency measurements. It does not retain the shorter-model
collector's inappropriate 1800-second limit.

Pilot-derived inference/preprocessing time is 14.419482 hours total, excluding
initialization, engine rehashing and audits. Allow roughly 14-15 hours plus
delivery/audit overhead if the board remains available. Performance during
formal sampling may differ. Sysfs telemetry and tegrastats are preserved;
visible GPU-owner scans continue with explicit inaccessible-process coverage
limits. No power mode, clocks or unrelated processes are changed.

The durable collector survives SSH disconnection. Do not launch another copy
when resuming. Verify exact PID/start time/argv and boot ID from:

- Target launch: `/data/datalog/edgefm-orin/20261021-07/formal-launch-001/launch.json`.
- Target controller log: `/data/datalog/edgefm-orin/20261021-07/formal-launch-001/controller.log`.
- Target data: `/data/datalog/edgefm-orin/20261021-07/raw-e2e-formal-001/`.
- Controller at launch: PID `1211540`, start ticks `6178131`, boot ID
  `3c801f5d-6897-4d97-b021-3c52db38a750`.
- Local launch request: `formal-launch-request-001.json`; controller source:
  `collect_raw_formal.py`, with frozen prior telemetry/order helper.

Collection exit alone is not acceptance. Required follow-up: independently
audit every formal timing row, complete output, replay count, process identity,
artifact hash and resource record; then emit raw CSV/JSON and pooled/per-process
CDF PNG/PDF without filtering. Other-model execution and D4 ablations remain
open. This document does not mark the five-model goal complete.

The independent formal audit and pooled/per-process CDF tools are prepared:
`audit_raw_formal.py`, `plot_raw_formal.py` and the exact resume/retrieval/audit/
render commands in `artifacts/orin/20261021-07/POSTPROCESS.md`. They explicitly
reject incomplete or shortened campaigns before producing formal output.
Combined audit/collection/plot regression is 37 passed. The plotter has also
rendered and visually verified the existing real SmolVLA 15,360 measured rows,
without changing their statistics or starting another inference campaign. Those
images under `smolvla-formal-cdf-check-001/` are not RDT results. The RDT formal
auditor cannot be certified on the still-incomplete full dataset until actual
collection and independent verification finish.

## Evidence and Regression

- `native-input-retrieval-001/`: target build/CTest/dependency receipts and all
  112 complete prepared input files; `native-input-audit-001.json` is independent.
- `raw-pack-001/` and `native-observations-001/`: hash-bound original decoded
  observations, original noise, explicit host pipeline and qualified audit.
- `same-orin-reference-001/`: unchanged conventional target outputs, NPZ hashes
  and complete native binary references; no H20 actions are primary references.
- `mode-retrieval-002/`: completed raw pilots, all outputs, maps, environment,
  source profiles, binaries and replay traces. The earlier retrieval was an
  in-progress snapshot and is not used for acceptance.
- `raw-e2e-audit-001.json`: complete same-target independent output/timing audit.
- `raw-audit-regression.xml`: 17 passed, including rehashed corrupt fixtures
  for truncated auxiliary output, large/one-ULP action errors, false fallback,
  timing-boundary errors and native image mismatches.
- `full-host-regression-r2.xml`: 2617 passed, 74 skipped, six subtests passed.
  `packaging-profile-regression.xml`: 46 passed, including installed-wheel C
  source inventory, host contracts, timing and passive architecture checks.

Negative evidence remains: the first raw worker linked the CPU-only probe
runtime, so Session construction failed before inference. A new attempt links
the original CUDA-enabled runtime/backend and only the added preprocessing
object; engines, generated model Session and precision DSO are unchanged. That
first failure is not a model-support conclusion. An initial full host test
attempt used an incomplete cached Python 3.10 environment and failed on missing
dependencies; the proper existing Python 3.13 host environment passed the full
suite above. No Orin Torch environment was created to address host test errors.
