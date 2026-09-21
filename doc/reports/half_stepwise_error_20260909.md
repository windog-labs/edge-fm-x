# FP16/BF16 Stepwise Error Budget

2026-09-09 01:10 UTC. CPU analysis recomputed per-step divergence between the
half-precision full-trajectory candidates and the original saved official
head inputs for all sixteen episodes. No new inference was run.

Step zero head inputs are byte-identical in every lane. The half replacement
error appears from step one and grows as free-running state diverges. Worst
held-out head-input MSE at step nine is 1.149572e-4 for FP16 and 1.261380e-4
for BF16; worst step-nine max-abs values are 0.0625-0.09375. Despite this
intermediate-state divergence, downstream action projection keeps final action
MSE below 1e-5 on all held-out samples.

| Lane | Held-out worst step9 head-input MSE | Held-out worst final action MSE | Held-out min final cosine |
|---|---:|---:|---:|
| FP16 | 1.149572e-4 | 2.434350e-6 | 0.9999993439 |
| BF16 | 1.261380e-4 | 5.109796e-6 | 0.9999986687 |

This is an important boundary for the paper narrative: intermediate solver
state is allowed to drift more than the final action gate. The paper cannot
claim that all internal tensors are lossless or that intermediate divergence
is bounded by the final MSE.

## Evidence

- Analysis report SHA: `6386d259119f70279a4c26b7d6be454d1f4be3b0386d611bad8a7a1c23d4e279`.
- Analysis source SHA: `1ac72befa223dc7cf03a023bbe43bb7421f780a6dd9b041ade769fe8b21c4fa9`.
- Per-step/per-sample rows retained in `half-stepwise-error-001/report.json`.
