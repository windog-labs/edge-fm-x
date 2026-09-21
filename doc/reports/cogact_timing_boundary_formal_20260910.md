# CogACT Boundary-Aligned Formal Comparison

Date: 2026-09-10

This report supersedes the speedup claim in
`cogact_autonomous_native_formal_20260910.md`. The earlier native timer excluded
C++ RNG preparation, while the official timer included DDIM random draws. The
new campaigns use one explicit boundary for both implementations.

## Boundary Contract

Native timing starts before `GenerateCogactRng` and ends after the generated
Session and CUDA completion:

`RNG preparation + Session model execution + CUDA synchronization`

Official timing starts before restoring the saved CUDA generator state and ends
after the original VLM/DDIM sampler and CUDA completion:

`RNG state restore + original VLM/DDIM model and sampler + CUDA synchronization`

Both boundaries include the RNG work that participates in the call, the model
or sampler, and device completion. Both exclude static input H2D and binding,
image/text preprocessing, complete output conversion and D2H. The independent
boundary audit verifies this contract, the identical input/reference roots, the
same `42,43,42` seed schedule, the same five-GPU process sequence and the same
precision profile.

## Formal Result

Each configuration used five independent processes, 128 warmups and 1024
measured calls per process. Runs were serialized on H20-2 GPU 0 through GPU 4
in the same order for both implementations. Warmups were excluded from latency
summaries but retained in the complete-output audit.

| Configuration | Measured | Mean ms | p50 ms | p95 ms | p99 ms | Std ms | Calls/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official PyTorch | 5120 | 129.258 | 124.006 | 177.383 | 232.080 | 21.110 | 7.736 |
| Autonomous native Session | 5120 | 88.751 | 89.199 | 90.683 | 95.811 | 2.346 | 11.268 |

The all-sample official/native mean ratio is `1.456x` with a deterministic
95% bootstrap interval of `[1.450, 1.463]`; native mean latency is 31.34% lower.
The p50 ratio is `1.390x`. These are model-and-sampler boundary results, not
robot control frequency or end-to-end input latency.

Official worker 4 had a heavy tail: its mean was `151.918 ms`, while the other
four official workers were `120.928` to `125.459 ms`. The median paired-GPU
mean ratio is therefore `1.393x`. No outlier was removed or replaced. The full
per-GPU values are retained in the comparison audit.

| GPU | Native mean ms | Official mean ms | Official/native |
|---|---:|---:|---:|
| 0 | 89.519 | 120.928 | 1.351 |
| 1 | 89.440 | 124.618 | 1.393 |
| 2 | 90.211 | 123.369 | 1.368 |
| 3 | 85.406 | 125.459 | 1.469 |
| 4 | 89.176 | 151.918 | 1.704 |

## Output And RNG Verification

Each campaign produced 5120 measured calls and 25600 complete measured output
tensors across raw F32, normalized F32, native F64, U8 RNG state and I64 draw
count. Every output, including all warmups, was byte-exact against the same
frozen reference.

The native path uses the shared C++ `CudaRngProvider`, not a Python global
generator, a saved runtime tape or a hidden callback. Provider smoke tests
cover seeds `42`, `43`, `0` and `2**64-1`, consecutive calls, all intermediate
states, reset, restore, independent interleaving and invalid-state rejection.
The native worker maps contain neither `libpython` nor `libtorch_python`.

## Artifact Identity

| Artifact | SHA256 |
|---|---|
| Native campaign report | `929e824dd99445a756661b82304e81dcab633a391da68dbd77eedb4f13c8645e` |
| Native independent audit | `9627730e57d71c142394670c5bd6e460b30f4ce171996f14b026cad2e6651e48` |
| Native CDF CSV | `269ecdd9f4fc823f44e233ebe14ffcc079323ecffeeba866a3344a7dc4256ab0` |
| Official campaign report | `50405209b0a6b58894f085c997566c58985d4a362913bb2a562af6f46f57e4f2` |
| Official independent audit | `01a9397c4942fe9b980955b3fe5e3fd84e66c98f93a170104da4562dbf447326` |
| Official CDF CSV | `04339d0ebbc0804537c5d416f6396d564fe0b1e85dba5edf833b097934489e45` |
| Boundary comparison audit | `b5516f6c4697c36bb75a35ba0b3b57a0f4f7aa144ce332e11fb5d2dd2ac91f8e` |
| Frozen generated runner source | `03853ce52b19b4e06751ae9c72fd5220e127e1f90dc076cef0a39af9532867df` |
| Boundary runner binary | `5c2ddc34ed7b7cad0d8cb949e7008bd084f271ab62bc24672431845472c50bb3` |
| Full CPU regression XML | `ffa79a298882026fb32ceefa9532321e5370b17a8fac76d59909412fe19a2e33` |

Local evidence root:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/cogact-timing-boundary-001/`

NAS evidence root:

`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/cogact-timing-boundary-20260910-001/`

The local and NAS comparison audit files are byte-identical. Independent
native/official audits were run on H20-2; a local comparison audit reproduced
the same comparison result from the copied campaign reports and audits.

## Regression

The current source was tested with CUDA hidden:

```sh
env CUDA_VISIBLE_DEVICES= \
  PATH=/home/zhangzimo/.venvs/edgefm-openpi-cu128-py311-20260906/bin:$PATH \
  PYTHONPATH=vlaforge/python \
  /home/zhangzimo/.venvs/edgefm-openpi-cu128-py311-20260906/bin/pytest \
  -q vlaforge/tests --ignore=vlaforge/tests/models/test_openpi_inputs.py
```

Result: `2276 passed, 62 skipped, 0 failed`. The new comparison audit accounts
for the additional passing test relative to the preceding Qwen closeout.

## Scope And Limitations

The checkpoint contains 7,630,224,071 parameter elements and is not labeled as
a complete 3B model. It is the public-dependency candidate profile. Complete
equivalence to the gated original Meta configuration remains unverified.

The formal input is one frozen real observation repeated with seeds
`42,43,42`. The result does not establish wider observation coverage, physical
robot quality, Orin/J6M/BPU behavior or arbitrary scheduling. The official
worker-4 tail is retained in the primary aggregate and CDF; the median paired
ratio is provided as a tail-robust descriptive statistic.
