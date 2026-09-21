# H20 RoPE Operator Microbenchmark

Ownership clarification: this historical v1 campaign used PID/proc lookup,
including the later-discovered rule that accepted missing proc entries. Its
earlier local audit checked campaign flags/arithmetic, not a registered
container-to-NVML identity proof. The timings remain historical observations;
they are not retroactively certified by the new v2 ownership checks. See
`operator_h20_confirmation_20260909.md` for the stricter confirmation workflow.

2026-09-09. This is a repeated same-platform operator comparison on
`zzm-h20-x8-2`, not a complete-model latency or deployment-selection result.

## Workload and Identity

The workload is the real exported OpenPI rotary-frequency subgraph extracted
from `vf_cached_openpi_prefix_59`. Its inputs are a BF16 inverse-frequency
tensor with shape `[128]` and an INT64 position tensor with shape `[1,816]`.
The two BF16 outputs both have shape `[1,816,256]`. The captured subgraph
contains the actual float32 matmul, transpose, concatenation, cosine and sine
operations; it is not a synthetic empty fixture.

The source example report is
`openpi-inventory/h100-pi0-actual-rope-008/report.json`, SHA256
`21490ea6f3a4a5293844eea35258a4cffcbaf32e45797c658d174942ba7c97a1`;
`revalidate-workload` was
used so the inputs and eager reference were executed again on H20 rather than
being treated as an H100 numeric result. The H20 target is GPU0,
`GPU-ae321531-504c-f416-e693-94fa02779130`, driver 535.161.08.

## Protocol

ATen and Inductor-ATen were run in five alternating independent processes each.
Each process used the existing `triton.testing.do_bench_cudagraph` measurement
with `rep_ms=20`. Every candidate performed full output validation before
timing. Preflight and sampled ownership records had no owner on the target
GPU; an unrelated process on H20 GPU1 was not touched.

The measurement is the median of graph-batch timings divided by the recorded
operation count. It excludes compilation, input transfer, Python dispatch and
C++ Session scheduling. It is therefore a repeated operator comparison, not a
per-call latency CDF.

## Result

| Recipe | Five process medians (ms) | Mean process median (ms) |
|---|---|---:|
| ATen | 0.021403508, 0.021454857, 0.021413847, 0.021430519, 0.021448407 | 0.021430228 |
| Inductor-ATen | 0.007755645, 0.008485174, 0.007772400, 0.008488263, 0.008474981 | 0.008195292 |

The Inductor candidate is 61.7583% lower than the ATen mean process median,
an absolute difference of 0.013234935 ms for this workload. The two complete
BF16 outputs were byte-exact in all ten runs; MSE, maximum absolute error and
changed-element count were zero. The candidate was not selected for deployment
and no complete-model E2E gain is claimed.

## Evidence

- Remote compile report: `operator-rope-h20-20260909-v4/compile-inductor-aten/report.json`.
- Remote repeated campaign: `operator-rope-h20-20260909-v4-repeats/campaign.json`.
- Local evidence root:
  `artifacts/edgefm-vla-goal/20260906-044509/operator-profiles/h20-rope-repeats-20260909/`.
- Independent audit SHA256:
  `23c4ad1ea76bf6c7420826b4c9c77e9a02ef69b66243e7fd56000c30f60e41f5`.
- Campaign SHA256:
  `78faf646738685f3a13ee392eacdb1a8eb9815087b51547421c978eff59bda5c`.
- Compile report SHA256:
  `52a233c06038ae94deba084856f852481b08ea3e5950ebe38d908c7b0b202efe`.

The older RTX RoPE result remains a separate historical measurement. This H20
run supplies H20 RoPE evidence but does not retroactively alter or combine the
RTX result.
