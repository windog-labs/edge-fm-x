# pi0 Execution Variants

Status: build, pilot and all three formal policies independently verified. This is a CUDA-stage experiment,
not an Orin/BPU result or a completed paper acceptance gate.

## Inputs and Identity

- Local source: `592fc320d57398571dcb4ab686d5fb55fe9bac86` plus recorded working-tree files.
- Host: `zzm-h20-x8-2`; selected H20 UUID `GPU-26cdf1ca-1632-fee2-fba8-65e3231f3b23`.
- Remote root: `/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/pi0-replay-20260908-002`.
- Source bundle: `runs/pi0-materialized-loader-002/deployment/bundle`, SHA256
  `93cc925777829ef49c4adab03159e8646dbbd92d38a027044c3baad8ec15b19f`.
- Reference protocol: `runs/openpi-series-001/protocol.json`, SHA256
  `302b960902c524cd16df968a3761ef65082aa2d4ab0c526398c45da8a12e8cc2`.
- Sixteen different frames from one ALOHA episode, with saved noise and full
  normalized/native output references. This is not sixteen independent episodes.
- The local build audit verified all 290 frozen implementation files and the
  three build records. Source build report SHA256:
  `e73ef89341862aa1f3bea5cf29e2193822a5d10ea070048cc2c3c3d8e9588f8f`.

## Implementation

`vlaforge/tools/build_session_variants.py` uses the public artifact bundle builder
and typed runner, retaining weights, computation, tensor contracts, numerical
bindings and reference protocol. It only selects off/batch-only/required loop
execution. Canonical IR region ordering is explicitly rebound by name, with
old/new declaration IDs recorded. Persistent state initialization currently
requires an explicit extension of this rebuild tool and is rejected, not dropped.

The typed runner now reuses the formal benchmark's replay checks and poisoned
capture exit handling. Application warmup invocations count towards replay totals;
backend-private capture warmups do not publish actions. Required-mode checks run
on every invocation, including application warmups, without a warmup bypass.

## Acceptance

### Pilot

The three-policy pilot completed 144 calls and 288 complete output tensors.
Local revalidation compared every raw output byte against the SHA-bound official
and direct references, checked actual numerical-runtime DSO bytes/mappings and
the worker bootstrap. All outputs are byte exact. The independent audit is
`artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/h20-pi0-replay-002/independent-pilot-audit.json`,
SHA256 `ce544e13e722d1997cd50cbc8f1309944aa81798a40cf92b18734e9d2e22db1c`.

| Policy | Pilot Mean (ms) | Pilot Max (ms) | Final Runtime Counters |
| --- | ---: | ---: | --- |
| off | 136.81577725 | 137.712913 | No replay task |
| batch-only | 158.57720825 | 163.085276 | 0 capture / 0 replay / 48 ordinary |
| required | 84.27433753125 | 87.652844 | 10 captured steps / 48 replay / 0 ordinary |

These short-run statistics are not the formal latency table or CDF.

### Formal

All fifteen formal workers exited 0. Actual execution ran from 2026-09-07
16:26:40 UTC to 17:14:30 UTC, approximately 47.83 minutes including worker
startup, ownership handshake and inter-process verification. The independent
local full audit completed after retrieval on 2026-09-08 (Beijing time).

| Policy | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Std (ms) | Fresh Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 139.628717 | 137.605974 | 148.585398 | 151.099301 | 159.943016 | 4.867877 | 7.161851 |
| batch-only | 139.362490 | 136.943905 | 147.399351 | 158.724181 | 194.403745 | 6.640924 | 7.175532 |
| required | 84.651419 | 84.400158 | 85.778994 | 86.142677 | 92.742168 | 0.660339 | 11.813151 |

Each policy has 5120 measured calls across five independent processes, excluding
128 warmups per process. All 17,280 calls including warmups were independently
checked: 34,560 complete tensors / 39,744,000 values match both official and
direct references byte for byte. MSE/max-abs are zero; minimum computed cosine
is 0.9999999999999998. Each required worker recorded ten captured steps,
1152 replays and zero ordinary calls. No fallback or failed worker was included.

Required mean latency is 39.373919% lower than same-source off. This is execution
replay, not Agent-selected kernel or quantization evidence. Ordinary process
means range 135.644317-148.464923 ms, versus replay 84.352155-85.615035 ms; all
samples and process variation are retained. Shared-stream-only did not show a
stable material improvement. Whole-worker NVML telemetry observed 1980 MHz SM
clocks and no foreign compute owners. Observed peak used memory was 6893 MiB
for off and 7561 MiB for required, so no memory-saving claim is supported.
These sampled device memory readings are not allocator high-water measurements.

Local evidence root is
`artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/h20-pi0-replay-002/`:

- `independent-formal-audit.json`, SHA256
  `7405e50ad5348d0fa4e088f388c2a8300b25ed29eba4f4d9ccd5d1221cbb90e4`:
  raw outputs, official/direct/input identities, actual runtime DSO mappings and
  bytes, bootstrap, telemetry owners, per-process/aggregate latency and CDF.
- `post-execution-assets-audit.json`, SHA256
  `5d25c9e62c322fc696a7fad402b05edd0ccb5069e4bc03282635648c0f03e2ad`:
  573 frozen entries, 290 source files, 179 bundle files, all mapped model DSOs.
  Large model payloads remain on NAS; the local archive is not a complete mirror.
- `prepared/runs/`: every process's samples, raw double output, execution record,
  runtime maps, replay counters and 1 Hz telemetry. `prepared/latency-table.csv`
  and three `*-cdf.csv` files are regenerated from these formal samples.
- `figures/latency-cdf-tail.png` and `.pdf`: actual CDF and tail plots, visually
  checked. `figures/report.json` binds source data, script and figure hashes.
- `audit-negative-tests-004.xml`: nine checks of the real-worker audit including
  altered/missing/truncated evidence rejection. `targeted-followup-tests.xml`:
  72 public board/variant/runner tests pass. Core source remains identical to
  CPU regression011 (2049 passed, 71 skipped).

The first plotting command used a Python without matplotlib and failed before
creating output. Rendering succeeded in the existing isolated report environment
`/home/zhangzimo/.venvs/edgefm-vla-report-py313-20260906/bin/python`.

The commands use the frozen source under the remote root and its existing
OpenPI environment. All stages below are complete; these are reproduction
commands for a new experiment directory, not instructions to overwrite this run:

```sh
python source/vlaforge/tools/benchmark_session.py --stage prepare \
  --protocol variants/protocol.json --output prepared
python source/vlaforge/tools/benchmark_session.py --stage pilot --output prepared
python source/vlaforge/tools/benchmark_session.py --stage run --output prepared
python source/vlaforge/tools/benchmark_session.py --stage report --output prepared
```

Pilot uses 16 application warmups plus 32 measured calls per policy and does not
enter the formal table. The inherited formal protocol uses five independent
processes per policy, 128 warmups and 1024 measured calls. The boundary includes
model inference, native output processing and synchronized completion; input
preprocessing, H2D/D2H and robot transport remain excluded.

The first variant-build attempt failed before GPU execution because old region
IDs differed from canonical IR declaration order. That failure remains in
`pi0-replay-20260908-001/variants/report.json`; no old package was modified.
