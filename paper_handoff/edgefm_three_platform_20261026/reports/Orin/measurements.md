> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# Qwen3.5-0.8B Orin Measurements

Complete native batch1 vision/prefill/decode, one image/user turn, fixed16new tokens.
Each policy has5 independent processes x128warmup+1024measured; all11520requests retained.
Ordinary off versus verified arena uses identical engines/precision; no whole-loop replay.

| Policy | n | TTFT Mean / P50 / P95 / P99 / Max / Std (ms) | Full16 Mean / P50 / P95 / P99 / Max / Std (ms) | E2E tokens/s | Sampled Native VmHWM (KiB) |
| --- | ---: | --- | --- | ---: | ---: |
| off | 5120 | 334.328554 / 333.731760 / 339.407525 / 343.079333 / 357.044384 / 2.586327 | 1593.061317 / 1592.458800 / 1600.865789 / 1605.455708 / 1618.577408 / 4.441399 | 10.043556 | 6890616 |
| verified | 5120 | 333.349658 / 332.869280 / 338.417126 / 341.265658 / 357.665152 / 2.655338 | 1586.510069 / 1586.291328 / 1593.629232 / 1598.201941 / 1628.726880 / 4.247181 | 10.085029 | 6867196 |

## Quality And Scope

All184320output token values are exact to accepted same-Orin references.320full final-logit/ID witnesses cover each input in each process's first/last cycle.
Worst boundary MSE 0; minimum cosine 1. Byteexact classification: True.
Source/H20 comparisons are supplemental: 7/16complete token arrays match; worst final-logit MSE 8.65417583235, minimum cosine 0.697306153915. No official-source or robot-success equivalence is claimed.
The16observations are varying frames from one episode, not independent robot trajectories.
TTFT is CPU-ready first token after the second prefix completion. Full16 includes online raw-text tokenization, RGB patchification, H2D,17neural/16control calls, and all token/logit D2H.
Session initialization, reset bookkeeping, file decoding and output logging are outside timing. Incremental decode throughput includes control/readback, not GPU-only compute.
Native RSS/VmHWM and system memory are separate whole-process sampled envelopes including initialization/warmup. Early first-process attachment and last-second coverage have limits. No GPU-private physical peak or zero-allocation claim is made.
CPU/GPU shared physical memory is never added to process RSS.
Power sensor raw values remain raw where units are not independently verified. MAXN is not relabeled60W.
All tails and negative paired rounds are retained. Qwen3.5-2B is user-cancelled, not unsupported or measured.

## Artifacts

- `data-audit.json`: complete token/logit/identity/guard evidence.
- `performance/`: all raw rows, process/policy statistics, all five paired rounds, four full5120-rank CDF CSVs and TTFT/full16 PNG/PDF.
- `resources/`: native memory/system/sensor sampled envelopes and coverage/pins.
- `target-closure.json`: independent post-campaign source/device/unit/program/SDK fingerprints.

**Pending:** visual inspection of both real CDF plots, final combined delivery/core-scope review and explicit Goal completion. This export alone does not mark the Goal achieved.


## Memory Observation Segments

4 observer segments; 3 observer-only restarts of one unchanged model cohort. Earlier observer errors remain errors.
Continuous1Hz RSS coverage is not claimed. RSS during gaps is unknown; no missing rows or zero values are inferred.
Kernel VmHWM is lifetime high-water at read time; neither the last-second peak nor GPU physical peak is guaranteed.
- `PROVENANCE_6083fb0e5a158c16`: status `error`, 3598 valid sampled rows; error `ValueError('native cgroup escape')`; source SHA256 `9b1cdd786cd63fbd29c961983a2f349b47c735711232d985f93b0c3039594129`.
- `PROVENANCE_5ec288401de02650`: status `error`, 5083 valid sampled rows; error `ValueError('native cgroup escape')`; source SHA256 `faf9786ed78d447d4adbd66df3fef3a8110802a14dd37bf269fb93e2a1f2674d`.
- `PROVENANCE_4b612b58829d2677`: status `error`, 7216 valid sampled rows; error `ValueError('native cgroup escape')`; source SHA256 `581b55d8018e2ba4cb3a42146e2331567c9a22ae78f9051cf3988877460e69c1`.
- `PROVENANCE_cf6bb3422218211a`: status `parent_terminal_empty`, 1734 valid sampled rows; error `None`; source SHA256 `f2f53b61cffdfc16e956cef60e980def2d1815b69d5a90329208b3fcd3b18c51`.
- Unsampled boundary: 437843989523328 to 438418380799808 monotonic ns (574391276480 ns between valid rows).
- Unsampled boundary: 443500381069760 to 443817798708576 monotonic ns (317417638816 ns between valid rows).
- Unsampled boundary: 451032798724064 to 451195988747808 monotonic ns (163190023744 ns between valid rows).