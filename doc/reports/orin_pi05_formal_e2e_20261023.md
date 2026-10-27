# Pi0.5 Formal Orin E2E

**D1-D3 accepted, not whole-Goal completion.** Joint audit SHA256:
`6cd1256e08f0c0c2781e6daf0b99866a82be06b90c6007d916a785f29cc8c213`.
Required formal model coverage is now4/5; CogACT and shared D4 work remain.

## Frozen Scope

Jetson AGX Orin serial1421524312594, boot3c801f5d-6897-4d97-b021-3c52db38a750;
TensorRT8.5.2.2/CUDA11.4/L4T35.4.1, existing MAXN, not relabeled60W.
Eight real TRT engines plus two source-bound native providers retain200tokens,
cache968, all three fresh shared visual calls, ten solver steps and F64 state/
native finishing. Same exact ten resources, inputs/noise, precision and source
computation in both modes; only compiler arena layout differs. Neither uses replay.

Timing is batch1, single request, CPU decoded observation/state/text/saved actual
noise through online native preprocessing, transfers, all vision/language/action
computation and complete consumable CPU action chunk. Asset/file decoding, setup,
logging and numerical audit are outside the clock. Sixteen real observations
cycle with fresh inference; these are not independent robot trajectories.

Each mode has5independent processes,128warmup+1024measured per process:
11520total requests,10240measured,5120measured/mode. No tails or slow rows removed.
All23040complete tensors, including warmup, were independently audited: normalized
F32[1,50,32] and native F64[1,50,14], exact shape/type/extent/hash, all finite,
all bytes equal to the same-Orin reference. Normalized MSE0/cosine1, native
supplemental MSE0. Fixed primary gates remain MSE<1e-5 and cosine>0.9999.

## Results

| Mode | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary |1258.792712|1255.848832|1275.217485|1279.311340|1290.113440|9.709894|0.794412|
| Verified arena, no replay |1256.955605|1260.030848|1269.932627|1273.155440|1280.483680|9.474035|0.795573|

Pooled mean ratio is1.001461553x, about0.146%throughput difference. This is not
established significant acceleration. The first two paired rounds are negative;
all five round means and their ordering are retained. Arena p50 is worse despite
the slightly better pooled mean/p99. Both pooled standard deviations miss the
draft1.2ms target. Do not multiply chunks/s by horizon to report control Hz.

Compiler arena249291776 ->243315712bytes is a5976064-byte layout difference,
not a measured physical-memory peak reduction. No replay/fallback benefit or
Agent kernel integration benefit is inferred from this runtime-only comparison.

## Resources And Restoration

12057telemetry/58325guard samples pass the frozen16GiB process-admission and
2GiB sampled-floor contracts. Minimum guarded MemAvailable18249632KiB, maximum
GPU temperature57.437C. Maximum telemetry gap9.622308832s, permission-limited
GPU-owner scans, unreadable power rails and invalid CV sensors are retained.
No continuous gap-free coverage, calibrated power, exhaustive GPU peak or
absence-of-other-root-users claim follows. Jetson shared CPU/GPU memory is not summed.

Both authorized Java services were restored through the original Supervisor,
observed Running with new PIDs499228/499270 and independently checked process/
parent/config/JAR/cwd contracts. Other managed rows stayed unchanged. This is
restored process identity, not application health or lost-state recovery proof.
The final target reader stream-rehashed all230required files, including both
mode libraries, engines, native providers, SDK/build inputs, raw/noise/reference
data. The first reader's incorrect SDK/runtime-subset assumption is retained as
a failed attempt. New source79 admits only four separately pinned SDK link roles,
keeps runtime overlap/path/command/count checks and all file hashing, and passes
31old/new/actual-layout software tests. No model, output, tolerance or protocol changed.

## Supplemental Source Difference

Aligned H20 official outputs remain supplemental: normalized MSE0.0164095488921,
cosine0.894301286548,0/16supplemental passes; native MSE0.00540717493616,
cosine0.992179239462. Original BF16/FP16, preprocessing and device differences
remain confounded and negative. Same-Orin exact bytes do not establish original
BF16/source equivalence, mathematical losslessness or robot task success.

## Evidence

- [Joint audit](../../artifacts/orin/20261023-58/final-001/joint-audit-001.json).
- [Complete output audit](../../artifacts/orin/20261023-58/final-001/model-data-audit-001.json).
- [Resource audit](../../artifacts/orin/20261023-58/final-001/resource-audit-001.json).
- [Original-manager restoration audit](../../artifacts/orin/20261023-58/final-001/service-audit-001.json).
- [230-file target closure](../../artifacts/orin/20261023-58/final-001/target-closure-002.json).
- [Raw statistics and request CSV](../../artifacts/orin/20261023-58/final-001/latency-001/summary.json).
- [Accepted all-rank CDF PNG/PDF receipt](../../artifacts/orin/20261023-58/final-001/publication-curves-001/receipt.json).
- [Supplemental H20 audit](../../artifacts/orin/20261023-53/parent-audit-001.json).
- [Failed closure attempt and source-only fix](../../artifacts/orin/20261023-79/failed-target-attempt-001.json).

Reproduce the audit with the frozen CLI entries in root56/cli-arguments-001.json,
using root58/final-001 capture/window/restoration paths, root79's new closure reader,
both exact native-build receipts, and a fresh output path for each audit.
Raw source protocols/references and all earlier failure artifacts stay unchanged.
