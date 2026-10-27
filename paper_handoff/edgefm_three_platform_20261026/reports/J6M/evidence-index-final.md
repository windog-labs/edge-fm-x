> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# SmolVLA J6M Existing-HBM Final Evidence

Scope: existing low-precision SmolVLA HBM stage performance on real `j6m-2`
(`0e190a0d3392371c`). This is stage-level evidence; it does not claim
resident full-chain E2E latency or task success.

## Audited Results

| Stage | HBM SHA256 | Samples | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Calls/s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix V2 | `0944337d...4838b84` | 5120 | 1402.998 | 1402.430 | 1408.200 | 1412.990 | 1431.410 | 0.712759 | 35.8125 |
| Prefix V5 | `b05a1c4d...d30993` | 5120 | 553.930 | 553.805 | 555.397 | 557.017 | 570.170 | 1.805283 | 33.7500 |
| Solver Step-060 | `457c50eb...40ca7b3` | 5120 | 89.921 | 89.834 | 91.249 | 95.632 | 96.615 | 11.120922 | 34.0000 |
| Finish output crop | `15113d77...120af30` | 5120 | 0.261 | 0.254 | 0.304 | 0.373 | 1.154 | 3828.710 | 27.0625 |

Each formal stage uses five independent sequential processes, 128 warmups and
1024 measured calls per process. No latency samples were filtered. The formal
V2/V5 mean-latency ratio is `2.532809x`; this compares complete HBM variants,
not an isolated single-operator ablation. Finish is a Slice-only output crop
with F32 I/O representation and no floating-point arithmetic in the source
graph, so it is reported as output handling rather than FP32 neural deployment.

## Evidence Map

| Evidence | Path |
|---|---|
| Complete stage audit and statistics | `final-stage-summary/report.json` |
| Stage table | `final-stage-summary/stage-performance.csv` |
| Prefix/Step raw logs and command evidence | `formal-r4/{prefix-v2,prefix-v5,step-060}/` |
| Prefix V2/V5 CDF figure | `final-prefix-v2-v5-cdf/latency-cdf.{png,pdf}` |
| Step CDF figure | `final-step-060-cdf/latency-cdf.{png,pdf}` |
| Finish audit and statistics | `final-finish-summary/report.json` |
| Finish raw logs and command evidence | `finish-formal/finish-crop/` |
| Finish CDF figure | `final-finish-crop-cdf/latency-cdf.{png,pdf}` |
| Finish one-sample crop check | `finish-smoke-output-check.json` |
| Limited Prefix and Step checks | `prefix-v5-output-check.json`, `step-output-check.json` |
| CPU/BPU smoke profile and static memory plans | `smoke-comparison.json`, `smoke/` |
| Environment snapshots | `prefix-v2-environment.json`, `prefix-v5-environment.json`, `step-060-environment.json`, `finish-crop-environment.json` |
| Input and HBM contract | `campaign-manifest.json`, `finish-campaign-manifest.json`, `input-provenance.json` |
| Pause/resume and excluded-run provenance | `user-pause-r3-20260914.json`, `user-resume-r4-20260914.json`, `formal-r3/`, `formal-r2-terminal-snapshot/` |
| Timing boundary | `timing-boundary-audit.json`, `hrt_model_exec.aarch64` |
| Reused operator ablation | `reused-operator-audit.json` |

The complete audits are `final-stage-summary/report.json` and
`final-finish-summary/report.json`. Both confirm board serial, exact command
arguments, raw hashes, complete frame sequences, non-overlapping process
windows and resource observations. The board runtime was UCP 3.14.7, HBRT
4.9.7 and BPU library 2.2.6.

## Fidelity Boundary

The selected HBM performance results reuse FP16-body / INT16-linear artifacts.
The Prefix check covers one captured calibration-subset sample: finite outputs,
exact mask, minimum cache cosine `0.9940398655`, and maximum cache absolute
error `1.402682066`. The Step check covers one solver update, with cosine
`0.9999994984` and maximum absolute error `0.0036740303`. Finish crop output is
finite and exactly matches the pinned ONNX Slice for one sample. These checks
do not establish lossless quantization, held-out quality, full-action fidelity,
or robot task success. No further quantization search or FP32 deployment was
performed.

The earlier file-orchestrated full-chain pilot remains a fidelity limitation;
its wall-clock time is not promoted as E2E latency. A resident Prefix/Step/
Finish runner and a full-action quality evaluation remain subsequent work.
