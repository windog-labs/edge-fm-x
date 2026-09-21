# Qwen3.5 Native H20 Formal Deployment

Date: 2026-09-10

This report closes the Qwen3.5-0.8B and Qwen3.5-2B native deployment task on
H20. It replaces the earlier native-blocked status for the fixed profile below.
The older cache-copy export failures remain valid historical diagnostics and
are retained in `qwen35_native_blockers_20260910.md`.

The accepted route uses the generic VLAForge native Session path. It does not
add a model-name branch to the core runtime. Qwen-specific state and generation
behavior live in the `qwen3_5` adapter, while bundling, generated C ABI, Session
execution, complete-output validation, timing and auditing remain shared tools.

## Result

Four formal campaigns completed:

| Model | Output profile | Independent processes | Measured calls | Complete output tensors |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | first token | 5 | 5120 | 11520 |
| Qwen3.5-0.8B | 16 tokens | 5 | 5120 | 11520 |
| Qwen3.5-2B | first token | 5 | 5120 | 11520 |
| Qwen3.5-2B | 16 tokens | 5 | 5120 | 11520 |

Total: 20 independent workers, 20480 measured calls, 46080 complete output
tensors and 5721488640 complete output values. Warmups are excluded from the
latency summaries and retained in the byte-exact output evidence.

## Native Latency

All values are milliseconds on H20. The timer boundary is
`host-model-tensor: H2D, input binding, Session run, completion and complete
typed D2H`. Every measured call uses the same fixed profile:

- `input_ids`: `[1,327]`, I64
- `attention_mask`: `[1,327]`, I64
- `pixel_values`: `[1200,1536]`, F32
- `accepted`: `[1]`, BOOL
- first-token output: `tokens [1,1]` I64 and `logits [1,1,248320]` BF16
- full output: `tokens [1,16]` I64 and `logits [1,1,248320]` BF16

| Model / profile | Mean | p50 | p95 | p99 | Max | Calls/s | 16-token tokens/s | Peak sampled device MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B / first | 96.905373 | 99.480645 | 104.489179 | 114.008338 | 167.637292 | 10.319345 | - | 2379 |
| 2B / first | 90.657933 | 85.696679 | 104.990503 | 137.604724 | 169.666646 | 11.030474 | - | 4983 |
| 0.8B / 16 tokens | 369.289954 | 322.653369 | 573.796106 | 625.620183 | 1097.094486 | 2.707899 | 43.326388 | 2393 |
| 2B / 16 tokens | 370.497338 | 333.449038 | 583.324144 | 629.293585 | 1005.713665 | 2.699075 | 43.185196 | 4995 |

The 16-token throughput is `16 / full-call mean`. The first-token and full
profiles are separate measurements and are never merged. A descriptive
continuation estimate can be formed from the independent means:

| Model | Mean continuation estimate | Estimated continuation rate |
|---|---:|---:|
| Qwen3.5-0.8B | 18.158972 ms/token | 55.069196 tokens/s |
| Qwen3.5-2B | 18.655960 ms/token | 53.602172 tokens/s |

Those continuation estimates use different H20 GPU UUIDs for the first-token
and full campaigns, so they are not a paired comparison and are not presented
as streaming per-token timestamps. The authoritative workload metrics are the
measured first-call and fixed-16-token results above.

The full campaigns ran concurrently on separate GPUs of `zzm-h20-x8-2` and
shared host CPU, storage and system software. GPU ownership monitors found no
foreign compute owner, but the tail latency should be interpreted as a
controlled shared-host profile, not a dedicated peak-device limit.

## Output Quality

For every campaign, all warmup and measured calls were compared with complete
direct and eager references:

- 0.8B first: 11520 complete tensors byte exact
- 0.8B full: 11520 complete tensors byte exact
- 2B first: 11520 complete tensors byte exact
- 2B full: 11520 complete tensors byte exact

The BF16 reference arrays are stored as FP32 in the validation NPZ. The
independent audit reconstructs the exact high 16 BF16 bits before comparing raw
bytes. This avoids the false mismatch that a direct BF16 byte view of FP32
storage would produce.

The native tokens also pass the separate complete-token comparison against
the official saved trace on the fixed sixteen-image validation set. Qwen is a
token/logit VLM output, so there is no action vector on which to compute the
VLA cosine-similarity metric. Exact integer tokens and exact BF16 logits are a
stronger equality gate for this model.

Each of the four generated C ABI bundles also passed negative contract probes:

| Bundle | Invalid shape probes | `accepted=false` | Committed outputs after rejection |
|---|---:|---:|---|
| 0.8B full | 7/7 rejected | Rejected with code 7 | Hashes unchanged |
| 0.8B first | 7/7 rejected | Rejected with code 7 | Hashes unchanged |
| 2B full | 7/7 rejected | Rejected with code 7 | Hashes unchanged |
| 2B first | 7/7 rejected | Rejected with code 7 | Hashes unchanged |

The probes cover batch greater than one, wrong prompt length, wrong image
height or width, and wrong `accepted` shape. A failed bind or validation does
not mutate the last committed output.

## Session Lifecycle

The generic C ABI probe was extended with an opt-in lifecycle test and rerun on
all four accepted bundles. Each probe created two Sessions from the same
generated library, interleaved two valid inputs across them, repeated each
input in the same Session, attempted a non-increasing reset, performed a valid
episode reset, and destroyed/recreated a Session.

All four bundles passed the same lifecycle contract:

- A non-increasing reset was rejected and retained the prior committed output.
- A valid reset cleared the prior committed output until the next successful
  run.
- Repeating the same input in one Session was byte exact.
- The two Sessions produced identical outputs for the same input while
  retaining different outputs for the two distinct inputs.
- Resetting one Session did not invalidate or change the other Session.
- A fresh Session created after destruction reproduced the original output.

The independent lifecycle audit rechecked 80 cross-field assertions plus the
bundle, library and metadata SHA256 bindings for all four reports. The local
audit is:

`artifacts/recovery-audit-20260909/qwen-native-lifecycle-probe-20260910-001/lifecycle-audit.json`

with SHA256
`6297810f95794daba4306435ccd66682e6bfa6a78b581677b3e914e76c73afd5`.
The frozen probe and auditor source hashes are
`8444a7a117fa7cff16e43915b90f918e538d91bb6233af0de015eaf34165be54`
and
`bb8553a5e579ff002b01f2ce128374c4f067dff348e70147e705e2164c3ffb03`.
The corresponding NAS evidence root is:

`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/qwen-native-lifecycle-probe-20260910-001/`

## Native Design

The accepted artifacts were produced with:

- `vlaforge/python/vlaforge/adapters/qwen3_5/qwen3_5_state.py`
- `vlaforge/tools/build_real_qwen35_exact_trace.py`
- `vlaforge/tools/validate_qwen35_trace_artifact.py`
- `vlaforge/tools/build_qwen35_exact_bundle.py`
- `vlaforge/tools/build_real_qwen35_from_artifacts.py`

The adapter exposes Qwen3.5 linear-attention convolution/recurrent state and
full-attention key/value state as explicit typed tensors. Prefill and decode
call the original upstream model modules. Fixed profile constants such as
prompt length, image grid, masks, RoPE delta and generation length are bound by
the saved artifact rather than inferred from Python at deployment time.

The saved graphs report no `PythonOp`, no random operator and no external I/O
operator, and leave caller inputs unchanged. The generated C++ Session is built
with the shared TorchScript Aten backend and `session_benchmark_runner.cpp.in`.
The worker process maps contain neither `libpython` nor `libtorch_python`.

| Model / profile | Artifact bytes | Artifact SHA256 | Saved graph nodes |
|---|---:|---|---:|
| 0.8B full | 1710043879 | `ff236734c08a6bef23d7fc45ecce84b60bc23f25ce6daf73c11404f18dc8c41d` | 109267 |
| 0.8B first | 1708456987 | `e173a9f13782c281e6b1d973053bae25e31e4a6b71b21a14957882b3244c2a54` | 31110 |
| 2B full | 4431187611 | `818881968f0068abcda0d3cfc0e01e9d4fd04765da52560dfa548876a418a744` | 110431 |
| 2B first | 4429593051 | `6fd9e26869094153e5c174cb3a65eeddc7f51a82ff4caee11e008faa29ee6416` | 32274 |

## Audit and Evidence

Each campaign used five independent processes, 128 warmups and 1024 measured
calls per process. The independent audit rechecked:

- unique worker PIDs and successful process exits
- the frozen 288-file source inventory for every campaign
- GPU ownership telemetry, hardware identity and runtime DSO mappings
- allocator snapshots with zero steady allocation calls, retries and OOMs
- every raw latency sample and CDF rank without outlier removal
- every complete output byte
- aggregate latency independently recomputed from worker samples

Hardware was H20 with driver `535.161.08`:

| Campaign | GPU UUID |
|---|---|
| 0.8B first | `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5` |
| 0.8B full | `GPU-ae321531-504c-f416-e693-94fa02779130` |
| 2B first | `GPU-c66f42bd-8a1c-423b-e2eb-d0cc72b9fe0b` |
| 2B full | `GPU-3596876b-e679-6289-f784-c8d98294ed23` |

| Campaign | Audit SHA256 | Aggregate report SHA256 |
|---|---|---|
| 0.8B first | `f8391cd1e21bec55fc1c9f6c6b440833f49be57dc17a1e2123485fad44460183` | `7850f8d94467a83af1b132f925ab752453b073f5b3dc6e1f9d4d9dd7dc7ba5ba` |
| 0.8B full | `3de83fb61cc0a8b36be3cd5f420db400e832079e2bc2bbba86b427084966022a` | `232958a8861d6c7a67298f558bd3761b249745ade7f6d43b235a7b897d3790a6` |
| 2B first | `d5d60efdef571a8a878ba5c39067de1909293f1bce13eff229762025169f39ea` | `450a939a03cedcbd64d21f375099ce4ed427d65fa6d4d9935c2674ce1a375b75` |
| 2B full | `ec474f68dd051bd00d863d60d53f68f41c15730d3a99d173cd8ed0b8dcf6a3d6` | `5792227ee9d3e3505a0907471651db92909b9b6433412d38dbaeffc87cd1e63e` |

The local evidence root is:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/`

The machine-readable Qwen closeout index is:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-evidence-index-002.json`

The NAS formal roots are:

- `runs/qwen-native-formal-0p8-first-001/`
- `runs/qwen-native-formal-0p8-full-002/`
- `runs/qwen-native-formal-2b-first-001/`
- `runs/qwen-native-formal-2b-full-001/`

under `/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/`.

The inspected CDF figures are:

- [16-token CDF PNG](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-full/latency-cdf.png)
- [first-token CDF PNG](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-first/latency-cdf.png)
- [combined CDF PNG](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-all/latency-cdf.png)

Their CSV inputs, PDFs and source-bound render reports are stored in the same
figure directories.

## Scope and Limitations

This completes native Qwen deployment on H20 for the fixed tensor profile. It
does not claim:

- arbitrary prompt length, image resolution, batch size or generation length
- image preprocessing, file I/O or detokenization in the reported timer
- a direct speedup ratio against the official Python baseline, whose timer
  includes image preprocessing and uses a different boundary
- a paired streaming first-token/decode trace
- Orin, J6M or Horizon BPU execution
- physical task quality, because Qwen3.5 here is a VLM baseline rather than
  a robot action policy

The previous export failures are not erased. They document routes that were
rejected at the cache-copy size guard before the accepted explicit-state
artifact route was completed.

## Regression

The final CPU regression was run with CUDA hidden:

```sh
env CUDA_VISIBLE_DEVICES= \
  PATH=/home/zhangzimo/.venvs/edgefm-openpi-cu128-py311-20260906/bin:$PATH \
  PYTHONPATH=vlaforge/python \
  /home/zhangzimo/.venvs/edgefm-openpi-cu128-py311-20260906/bin/pytest \
  -q vlaforge/tests --ignore=vlaforge/tests/models/test_openpi_inputs.py
```

Result: `2275 passed, 62 skipped, 0 failed`. The ignored file requires the
unavailable external `third_party/openpi` fixture and is unrelated to the Qwen
native path.
