# Qwen Native Prefix Diagnostic Preparation

Superseding checkpoint: the explicit30minute grant was received and the real
two-stage prefix pilot completed. All50final tensors are finite; both integer
outputs match CPU portable reference, while floating cache differences remain.
Only data was stopped for62.629870304seconds and restored; starter never stopped.
No formal metrics or same-Orin quality acceptance is implied. The user then
allowed longer autonomous bench use; no further per-short-window requests are
needed within the existing goal, but safety/restoration and quality gates remain.
[Real pilot, complete errors and independent closure](orin_qwen_native_prefix_pilot_20261025.md).

Checkpoint: 2026-10-25 09:26:57 Asia/Shanghai. This turn prepares an actual
full-prefix output probe, not performance evidence. Previous turn generated
both engines and restored services; producer's post-build slot error remains
unresolved. No new GPU task or service-control action has occurred.

## Real Inputs And Routes

One actual archived observation was loaded on the host with CUDA hidden, without
running a source model. Original source input tensor tuple and export/partition
proofs are pinned. Transport keeps source dtypes unchanged:

| Input | Source dtype / shape | Actual range | Bytes |
| --- | --- | --- | ---: |
| input_0 | INT64 [1,327] | 13..248069 | 2616 |
| input_2 | FLOAT32 [1200,1536] | -1..1 | 7372800 |

Total7375416bytes. These are prepared input tensors, **not raw decoded RGB E2E**.
Only one observation is prepared here, not a new sixteen-case result.

The model-neutral file-chain worker reads actual engine metadata before any
context/output probe. Declared source interfaces: producer2inputs/50outputs,
consumer52inputs/50outputs. Consumer's original input_0/input_2 are proved
unused by the pinned flat ONNX graph; no missing used input/output may be
invented. Each consumer frontier route refers to an actual producer output,
not the CPU-reference NPZ. Integer narrowing requires an explicit declaration,
full INT32 range check and exact storage codec; float dtype changes are refused.
NaN/Inf, corrupt pins, missing ports and unsupported layouts fail closed.

[Real-frame preparation](../../artifacts/orin/20261025-03/prefix-file-chain-assets-001/preparation.json),
[unexecuted generic plan](../../artifacts/orin/20261025-03/prefix-file-chain-assets-001/plan.json).

## Staging And Tests

Fourteen unique target files were rehashed, including both actual engines,
graphs/build receipts, input data and native tools. Worker SHA98352917... and
plan SHA95d8982d... are frozen. At09:14:35the target had13587732KiB available,
below16777216KiBadmission; both Java programs were running and no GPU task was
started. The original two-hour build grant is closed and is not reused.

58host contract/codec/recovery tests pass. They include independent runtime
caps, original-deadline/no-renewal behavior, refused consumed authorization,
conditional stop order and cleanup-before-restoration. These are synthetic
tool tests, not source parity, target inference or quality acceptance.
An earlier added test had a Python variable-scope error (57passed/1failed);
its XML is retained separately, the test was corrected, and all58then passed.

[Target staging receipt](../../artifacts/orin/20261025-03/prefix-file-chain-assets-001/staging.json),
[58 passing tests](../../artifacts/orin/20261025-03/native-diagnostic-and-window-tests-003.xml),
[retained earlier test failure](../../artifacts/orin/20261025-03/native-diagnostic-and-window-tests-002.xml).

## Requested Pilot Window

At the historical preparation checkpoint, a fresh **maximum30minute** conditional
service-stop approval was requested and awaiting an explicit answer. No live
window existed then. The prepared guard keeps16GiBadmission/4GiBruntime floor,
reserves900seconds for original-Supervisor restoration, and caps the diagnostic
task at600seconds. Independent systemd units contain all children and clean
their cgroup before restoring only leased services. Data first; starter only
if still necessary. No other service/configuration change or formal sampling.

The pilot is deliberately file-backed, nonresident and untimed. Even success
will only establish actual metadata and complete prefix outputs, **not** a
16-token Orin model, same-Orin ordinary/optimized quality, TTFT, tokens/s or CDF.
Full native tail, resident integration/state lifecycle and formal sampling
remain required. Qwen2B stays pending; five accepted VLA campaigns are unchanged.
