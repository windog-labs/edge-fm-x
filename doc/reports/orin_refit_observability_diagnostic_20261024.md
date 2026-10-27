# Orin Refit Observability Diagnostic

Checkpoint: 2026-10-23 22:01 UTC. Four real Orin engine builds complete, but
none meets the unchanged precision-observability coverage. No inference or
timing is requested in this diagnostic. Earlier exact operator outputs do not
validate these new refit engines. This is a retained compiler boundary, not a
new numerical or performance pass, and does not complete D4 or CogACT.

## Fixed Diagnostic

Reuse all four original Embedding/RoPE ONNX graphs and precision declarations
byte-for-byte. The only builder command difference is `--refit`; FP16 enabled,
noTF32, exact OBEY precision maps and 256 MiB workspace are unchanged. The
derived builder also carries an explicit diagnostic revision/label. A full-file
inverse reconstructs the unchanged original builder. Four command-delta tests
cover both families/variants. No new graph/precision candidate, quantization
search, whole-model FP32 fallback or system/SDK modification is introduced.

The installed TensorRT 8.5 sample option parser explicitly accepts `--refit`;
its actual target file is hashed in the capture and terminal closure. Enabling
that option is tested here, not assumed to expose all fused operations or
executed weight storage. The original typed-inspector rules are reused without
relaxing exact source placement, data type or Gather storage requirements.

| Original Case | Actual Refit Build | Builder Elapsed (s) | Inspector Result |
|---|---|---:|---|
| Embedding baseline | Exit 0 | 6.362 | Myelin; internal Gather/storage not exposed |
| Embedding candidate | Exit 0 | 5.296 | Myelin; internal Gather/storage not exposed |
| RoPE baseline | Exit 0 | 4.444 | Myelin; source-operation coverage incomplete |
| RoPE candidate | Exit 0 | 4.049 | ElementWise/Constant/Reformat/NoOp; complete exact source-placement coverage still absent |

These are compilation/controller costs, not inference latency. Successful
builds and external layer ports do not certify effective precision. The new
engines have no output/ABI-validation pass and are not admitted to performance
sampling. No source operation is marked observed merely because a fused layer
contains its name. The earlier [root104 output and SDK ABI result](orin_embedding_rope_target_20261024.md)
remains separately pinned and precision-unqualified.

## Resources And Closure

Target serial `1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`,
existing MAXN and SDK unchanged. Original microbenchmark bounds remain
8 GiB admission, 2 GiB runtime floor and 360 seconds per build. These are not
relaxed CogACT 24/4 GiB bounds. Lowest sampled available memory is
16,644,076 KiB; it is not isolated physical model memory or a continuous peak.
Visible GPU-owner/permission limitations remain in each raw environment record.

Only a new owned transient user compute unit is launched, with a 30-minute
outer watchdog, 30-second stop timeout and control-group cleanup. No original
application is stopped or restarted. Before/after and final read-only checks
retain the original Java PIDs `942765/942805`, births/argv/cwd and Supervisor
parent. This is process observation, not application-health verification.
The post hook finds an empty owned cgroup; ten controller/build/post identities
are terminal. Fifty-five actual target paths, including input graphs/profiles,
build receipts/layers/engines and SDK option source, are independently rehashed.
Local archive omits large engine/cache payloads; their target hashes and sizes
remain retained. Nine source/actual-evidence checks pass; the earlier six-test
pre-execution report is not overwritten.

## Evidence

- [Fixed diagnostic plan](../../artifacts/orin/20261023-110/plan-001.json).
- [Actual builds, precision boundary and target closure](../../artifacts/orin/20261023-110/closure-001.json).
- [Nine source and actual-evidence checks](../../artifacts/orin/20261023-110/source-and-actual-tests-002.xml).
- [Actual capture](../../artifacts/orin/20261023-110/target-archive-001/capture-001.json).
- [Empty owned cgroup](../../artifacts/orin/20261023-110/target-archive-001/unit-post-001.json).

Recheck without target compilation/inference:

```sh
env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=artifacts/orin/20261023-110 python3 -B -m pytest -q \
  -p no:cacheprovider artifacts/orin/20261023-110/test_refit_diagnostic.py
```

Do not rerun the one-shot launcher or overwrite this attempt. Further progress
requires independently sufficient precision evidence or an explicitly separate
reported comparison scope; numerical equality is not substituted for missing
internal observability, and no new acceptance scope is silently adopted.
