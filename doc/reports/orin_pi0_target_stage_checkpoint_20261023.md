# Orin Pi0 Target Stage Checkpoint

2026-10-23 local date. This is a real target **component checkpoint**, not a
complete pi0 deployment, action-fidelity result or formal latency row.
SmolVLA and RDT remain the only accepted D1-D3 models: **2/5**.

## Step Engine

Actual Orin build `20261023-00/pi0-step-build-006` completed with exit 0:

| Item | Verified Value |
|---|---|
| Engine bytes | 640218807 |
| Engine SHA256 | `b130aa7e403bc203f7d46db60c840a7a5a9e8d9d8941887e7ecc9e45a1a99d1d` |
| Final ONNX SHA256 | `368e13f00bf9b729b5a8f3096043a48fd5a9214ed996638d791d9ed5ca66ad00` |
| Precision profile | `precision-profile-007.json`, `ab62f9b97753dc92cb297eb30b46a0d52d377aebd9437610f48b35cdb515d62d` |
| Hardware / SDK | Serial 1421524312594, AGX Orin, TRT 8.5.2.2 / CUDA 11.4 |
| Build elapsed | 241.977797568 seconds, **not inference latency** |

The command used `--buildOnly`; engine deserialization is not inference.
Source-typed FP32 regions remain FP32: state/action projections, action MLP,
final projection and the RoPE outer product, six dense nodes in total.
The remaining 162 source-BF16 core MatMuls request FP16 compute and results.
Extra FP16 result rounding is explicit, not original-BF16 equivalence.
Normalization, scheduler and declared arithmetic stay FP32; the bounded time
embedding plugin performs its declared F64 math internally. No whole-model
FP32 fallback or quantization search was used.

[Inspector audit](../../artifacts/orin/20261023-00/engine-dense-formats-001.json)
matches all 168 dense nodes: 78 direct and 90 in explicit fused groups.
Exposed I/O and available weight formats match requests. This does not prove
hidden accumulation precision inside tactics or the two Myelin regions.

Original failed builds 001-005 remain separate. The generic Constant tensor
canonicalization and bounded integer-control folding clear the two import
blockers. [51 focused regressions](../../artifacts/orin/20261023-00/constant-control-regression-001.xml)
pass. The [I64 use-closure audit](../../artifacts/orin/20261023-03/i64-audit-summary-001.json)
covers this step only: 38 INT64_MAX constants comprise one dead cone and 37
proved saturation-equivalent Slice bounds. It does not authorize runtime ID
narrowing or certify other stages.

## Native Regions

Five actual-schema providers compiled natively on Orin CUDA11.4/aarch64 with
17 successful commands. The real external ABI2 loader then passed GPU smoke:
all five profiles, 128 paired frozen chunks and 89600 complete F64 native
values were byte-exact, MSE/maxabs 0. There were 97772 assertions, not that
many independent observations. Checked conversions and source-compatible
native-plus-computed-acceptance signatures retain their failure boundaries.

The [independent complete-log audit](../../artifacts/orin/20261023-00/native-region-gpu-smoke-001-audit.json)
binds logs, fixtures, profiles and native build receipt. Inputs include frozen
normalized actions, **not newly computed Orin neural actions**. Initial smoke
PID/start ticks were not captured; that limitation is explicit. No formal
timing, zero-allocation claim or full-model acceptance follows from this smoke.

## Next Gate

Build and inspect the complete fresh vision/language prefix and original
time/finite providers; integrate engines and native providers with the real
ten-step semantic Module/Plan and timed raw-preparation API. Then validate
complete normalized/native actions on the same Orin before freezing a pilot
and the formal five-process protocol. Source/H20 differences remain separate.
Qwen remains optional and is not introduced into this critical path.

The task board is authoritative for subsequent progress:
[web](http://127.0.0.1:4317/) / [structured state](../edgefm_orin_task_board_state.json).
