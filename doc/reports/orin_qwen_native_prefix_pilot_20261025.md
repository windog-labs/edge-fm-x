# Qwen Native Prefix Pilot

Independent terminal audit: 2026-10-25 10:02:18 Asia/Shanghai. One real
observation completed both prefix engines on the real Orin. This is a
nonresident, file-backed **prefix-only** output diagnostic, not full-model
acceptance, a same-Orin optimized-path quality gate or a latency benchmark.

## Real Target Execution

| Stage | Actual device inputs / outputs | Completed outputs | Required context bytes, not peak |
| --- | --- | ---: | ---: |
| Producer | 2 / 50 | 50 | 241661440 |
| Consumer | 52 / 50 | 50 | 646700544 |

All real bindings were read by the compiled inspector, then both execution
contexts and actual enqueue calls succeeded. All consumer frontier values came
from this invocation's real producer outputs, not CPU reference/cache data.
All final50tensors/7170050values were retrieved and checked. Actual engine
hashes, source graphs, tools, inputs, metadata and raw outputs were independently
rehashed. The qualified v3 scalar-layout validator also accepted both actual
inspections. Binding order differs from source order; routing is by exact name.

The producer's earlier slot.h::decode error/empty detailed layer export remains
archived. It did not reproduce in these named-I/O, context and output probes;
this case does not establish a universal exporter-only root cause or a fix.
Neither engine was rebuilt or changed to obtain this result.

## Complete Output Comparison

Reference: previously pinned CPU portable same-carrier prefix, **supplementary**,
not an ordinary-vs-optimized same-Orin comparison and not an official H20
equivalence certificate. Integer storage adapters are explicit and exact.

| Check | Observed result |
| --- | --- |
| Complete final tensors | 50/50 |
| Values | 7170050 |
| NaN/Inf | None |
| First token | 32, exact reference match |
| RoPE integer state | -280, exact reference match |
| Integer transport | Checked INT64→INT32 input; declared exact INT32→INT64 output widening |
| Floating outputs byte-exact | No |
| Worst floating MSE | 0.07114228272232531, output_47 |
| Associated max-abs | 5.765625 |
| Lowest floating cosine | 0.9819405203294378, output_42 |

All floating errors remain in the report/arrays. **No source-losslessness claim
is made and no quality threshold was relaxed.** The primary same-Orin quality
gate remains unmeasured/unpassed; it must compare complete ordinary and EdgeFM
outputs at the same precision after the complete native generation chain exists.
No operator/precision search or whole-FP32 fallback was performed.

## Resource And Authority Closure

The fresh2cefapproval allowed at most1800seconds including900seconds recovery
reserve; the diagnostic task itself was capped at600seconds.16GiBadmission and
4GiBruntime floors were unchanged. Only data-collection was stopped, for
62.629870304seconds, and restored through original Supervisor4237 as
PID3132518/start37483668. Starter1465114/start28963358 was never stopped.
Original configuration/JAR/executable pins and unrelated program snapshot remain
unchanged. Both owned units are inactive/MainPID0, cgroups empty, within the
original deadline. Inner safety-guard invocation durations include setup,
deserialization and file I/O and **must not be used as TTFT/E2E latency**.

The user subsequently explicitly allowed longer autonomous bench use. This is
recorded separately, not used to rewrite or renew this already-closed short
window. Subsequent goal phases can use independently bounded windows without
repeated short-window questions, retain conditional service stops/restoration,
and start formal sampling only after complete quality/protocol gates.

## Evidence And Next Step

- [Native chain/raw output inventory](../../artifacts/orin/20261025-03/native-prefix-file-chain-001/job.json)
- [All50output comparisons](../../artifacts/orin/20261025-03/native-prefix-comparison-001/report.json)
- [Independent target/restoration audit](../../artifacts/orin/20261025-03/diagnostic-window-001/terminal-audit-001.json)
- [Original-manager recovery](../../artifacts/orin/20261025-03/diagnostic-window-001/window-001/post.json)
- [73host tool tests, not model acceptance](../../artifacts/orin/20261025-03/full-native-diagnostic-tool-tests-001.xml)
- [Continuing bench authorization](../../artifacts/orin/20261025-03/ongoing-bench-authorization-001.json)

Next: build the complete native15-step tail using the existing declared
precision/profile-file route, verify all16generated tokens/final outputs, bind
the resident common Session chain, validate reset/isolation and same-Orin
ordinary/optimized outputs, then collect the frozen TTFT/16-token/CDF protocol.
Qwen2Bstill requires its own native engines/complete chain and formal results.
Five accepted VLA campaigns and their retained paper data are unchanged.
