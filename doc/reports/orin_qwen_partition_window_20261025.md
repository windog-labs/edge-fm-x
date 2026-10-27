# Qwen Complete-Prefix Partition Build Window

Terminal audit at 2026-10-25 08:51:45 Asia/Shanghai: both engine files were
generated with exit0, but producer has a post-build `slot.h::decode` internal
error and empty layer export. Neither engine has accepted inference/quality
results. Services are restored and owned compute is empty. This is not a paper
latency table. The five completed VLA campaigns are unchanged.

## Authority And Recovery

- Fresh approval: `bbcd9f1f88c94e7692677ac54bb8c2a1`, exactly the two specified
  Java programs, build/pilot only. No formal long sampling or other services.
- Start: 2026-10-25 08:17:33 Asia/Shanghai, distinct window2502-001.
- Absolute window: at most7200seconds,900seconds reserved for restoration.
  Monotonic target origin is authoritative because the board calendar is wrong.
- Original Supervisor4237, start27537, original configuration/executable/JAR
  hashes pinned; only data-collection was stopped after memory admission failed.
  Starter1465114/start28963358 remains running, unchanged.
- Stop order: data first; starter only if still below16GiB. Post-stop available
  memory22333260KiB passed the unchanged16777216KiB admission.
- Runtime floor4194304KiB and per-partition1800seconds remain fixed. Independent
  systemd ExecStopPost empties only the owned compute cgroup before restoring
  leased programs through the original Supervisor on every terminal outcome.

Independent terminal audit rehashed both actual engine files, all model/profile/
external-weight pins and guard/builder/wrapper/plugin identities. Only data was
stopped, for1879.792253888seconds, and restored as PID3061628/start37129667.
Starter1465114/start28963358 was never leased or restarted. Original Supervisor,
configuration/executables/JARs and unrelated managed-program snapshot remain
unchanged. Both owned units are inactive/MainPID0 with no cgroup members;
restoration completed within the original two-hour window. Minimum sampled
available memory: producer8944040KiB,consumer5893860KiB; neither4GiBguard nor
1800second task cap triggered. At the terminal resource snapshot14490560KiB
was available, below16GiBadmission; no new target task or service stop began.

[Independent terminal audit](../../artifacts/orin/20261025-02/build-window-001/terminal-audit-001.json),
[original-manager recovery](../../artifacts/orin/20261025-02/build-window-001/window-001/post.json).

[Fresh grant](../../artifacts/orin/20261025-02/build-window-001/authorization.json),
[identities/resources/SDK](../../artifacts/orin/20261025-02/build-window-001/preparation.json),
[immutable job](../../artifacts/orin/20261025-02/build-window-001/job.json),
[16 guard tests](../../artifacts/orin/20261025-02/window-guard-tests-001.xml).

## Actual Build Scope

Qwen0.8B complete parameter-preserving prefix, not the earlier first-token-only
export:9735+11314nodes,50intermediate frontier values and all50final states.
Each original computation occurs once; complete same-carrier CPU output parity
was already byte-exact. This does not certify source/ONNX/target equivalence.

| Stage | Actual target state | Accepted inference |
| --- | --- | --- |
| Prefix producer | Engine emitted/exit0,954.541654144seconds; post-build internal error, unqualified | No |
| Prefix consumer | Built-not-validated,exit0,893.944740416seconds | No |
| Complete15-step tail | Prepared separately; not in this two-build job | No |
| Complete16-token generation/reset | Pending actual engine chain | No |

Model metadata SHA256:

```text
producer e86f2a7d171653635b0b15d5f142267b2defba223ebdd96d3d403cc30ebebadd
consumer d1dda215d245fb6fae648b80d2452006bfc3c98245de4ad73bbdf2ad4bce1521
weights  b9d70bcfbe9138fe7f4b34780dbe91f307c35e6dedce1179df0707f24f9ccf4f
```

Actual producer engine:2109829697bytes, SHA256
`089f62e1e5682ed31fa8f27ea001c1753f224b3c2861ae84388bb215e8b5c9f5`.
This is the first emitted engine for the complete parameter-prefix producer,
as reported by the builder, not an accepted error-free usable engine, complete
prefix, full model, generation, or latency measurement.
[Producer build receipt](../../artifacts/orin/20261025-00/parameter-partition-prefix-producer-001/build.json).

Actual consumer engine:2163535590bytes, SHA256
`297a44a79ac9237eb15609aba82437e5326d2d1266fc3ec74ca0a084f57e938f`.
[Consumer build receipt](../../artifacts/orin/20261025-00/parameter-partition-prefix-consumer-001/build.json).

Producer log chronology: `Engine built`,reported deserialization0.9242seconds,
then `[slot.h::decode::224] ... encoded reference to slot found, but slots
missing`, then inference skipped. `layers.json` is0bytes. Consumer reports
deserialization0.505905seconds,no error lines, and a parseable1272951byte layer
export. Installed SDK `trtexec.cpp` calls `printLayerInfo(engine,nullptr)` in
buildOnly mode, then reports pass; its layer exporter unconditionally returns
true. These local source/log observations explain why exit0 alone is not an
acceptance gate. Exact binary compilation from this installed source and a
metadata-only root cause are not independently proven. No error is suppressed,
no model inference is implied, and raw builder receipts remain unmodified.
[SDK source/chronology diagnostic](../../artifacts/orin/20261025-02/SDK-export-diagnostic-001.json).
The exact historical service-window command is retained in
[launch.json](../../artifacts/orin/20261025-02/build-window-001/launch.json);
its authorization/origin cannot be renewed by replaying the command.

Consumer parsing reports `Unused Input: input_0` and `input_2`. This is an SDK
warning, not actual binding inspection; the later engine ABI must be read and
unused inputs explicitly accounted for, never padded or guessed. Producer
`layers.json` is empty despite successful construction, so it cannot serve as
layout/ABI proof. The already-audited metadata inspector source was compiled
against this Orin SDK with CUDA hidden; target binary SHA
`d7488706044f664cc660d084d307d43bfa67635131b95c028034ed133a462d86`.
It has not run or loaded an engine during concurrent construction.
[Inspector compile-only receipt](../../artifacts/orin/20261025-02/metadata-inspector-build-001/build.json).

Source-derived mixed precision remains explicit: native BF16 providers,
declared FP16 low-dense operations and source-FLOAT accumulation. No whole-model
FP32 fallback, dtype search, Torch on target, SDK change or other GPU job.

## Parallel Preparation

A model-neutral raw single-engine pilot prototype was compiled/linked on the
target in2.982403616seconds with CUDA hidden; compilation did not execute the
produced program. Binary SHA256:
`c432abebaa31ad744c696c036cea6ab9638a9c2251e98a72ec72d3c72cedd872`.
It requires exact actual binding names/dtypes/static shapes, pinned engine/DSO
and input bytes, and saves every output as binary with a SHA receipt. It does
not silently cast, emit synthetic state, round through JSON arrays or claim
performance/quality acceptance. Compiler success is not runtime qualification;
real engine bindings and positive/negative inference tests remain pending.
Seven subsequent executable rejection cases passed with CUDA hidden: missing
arguments, duplicate JSON keys, trailing JSON, wrong schema, wrong serial,
wrong boot and empty bindings. Each failed before output-directory creation,
plugin registration and CUDA initialization, without loading any engine or
changing service state. These are not positive engine-inference tests.

[Compile-only receipt](../../artifacts/orin/20261025-02/raw-pilot-tool-001/receipt.json),
[seven pre-CUDA rejections](../../artifacts/orin/20261025-02/raw-pilot-tool-001/rejections-001.json).

## Remaining Gates

1. Resolve/classify the producer post-build slot error through actual scheduled
   ABI/context/output checks. Engine deserialization logs are not output proof.
2. Actual engine metadata, explicit integer range/format adapters and complete
   target prefix output checks, not a first-token-only acceptance.
3. Complete tail engines and resident state chain/reset/isolation.
4. Same-Orin ordinary/optimized quality gate, followed by independently
   authorized formal TTFT/16-token/CDF/memory measurements.

Qwen2B remains waiting for the0.8Bnative build boundary. Its CPU source/portable
logit differences remain reported; no2BOrin engine or metric is implied.
