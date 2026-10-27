# Qwen Constant-Resource Build Diagnosis

Current target phase: 2026-10-25 06:05:21 UTC. Window09passed actual parsing,
15234network layers and zero unresolved precision/output names; native
receipt is building. Three conditional leases remain guarded, compotask
stays running. No engine/model output or formal metric yet.

Historical checkpoint: 2026-10-25 05:59:25 UTC. The shared graph's first whole
part still has no engine. Window08ended at the unchanged4GiBmemory guard,
minimum3916184KiB,compute1727.262944128seconds/outerexit-15. The constant-
sharing subset result is a real contributor, not proof of a whole-stage fix.
Independent05:40:12audit rehashes every source/tool/authority pin, verifies
empty cgroups and original Supervisor recovery. Only data was leased for
1744.171792032seconds, restored3434147/start38866381; starter's original
birth and unrelated programs/configuration remain unchanged. Old failure
and skipped-tactic warnings remain; no precision or numerical failure is
inferred from the memory-guard termination.

Fresh window09uses exactly the same graph9d77e949/profile251e2b57and
builder618668b3/plugin af35dba3. Admission increases to28GiB, runtime floor
stays4GiB; same3600second task/7200second window with900second reserve.
This is a changed resource profile, not an unchanged model-build retry or
a relaxed gate. Current receiptparsing,compute/window active. Conditional
stops of data,starter and router reached29757028KiB; compotask was not needed.
Original-manager restoration is independently guarded and still pending.

This phase declares four known recoverable resource programs under the
user's current exclusive-bench instruction. The scope interpretation is
explicitly agent-declared, not a fabricated new human reply. Historical
native-pair permission is scope/recovery precedent only; its expired window
and formal-sampling permission are not reused. No other programs, autostart,
SDK, clock, CAN or network settings may change.28safety/scope/order/recovery
tests pass. No model inference or formal sampling is performed by this phase.

Historical checkpoint: 2026-10-25 05:18:05 UTC. The exact constant-transpose variant's
first complete-tail part passed actual parsing:15234network layers, zero
unresolved precision/output names, native receipt now `building`. No complete part engine,
target model output, same-Orin quality or formal metric is accepted yet.

## Negative Initialization Probe

The original af35dba3 plugin DSO was used unchanged. One warm initialization
was completed and destroyed, then64real BF16 MatMul plugin instances were
initialized and held simultaneously. No weights, GEMM or model inference.
Observed physical available-memory drop4694016bytes and CUDA-free drop
4400640bytes, with zero guard trigger. This does not reproduce the many-GiB
full-build decline and does not justify a handle-pool rewrite. It does not
cover first GEMM, TensorRT tactics or weight/workspace allocation.

The diagnostic unit exited and was collected, MainPID0/cgroup empty;
original manager/program snapshots unchanged and no service control.

## Controlled Real-Operator Build

Three original LM-head MatMul nodes, independent original activation
boundaries, actual shared learned weight and original I/O were extracted.
The original case has three identical constant weight transposes; the second
uses one canonical transpose. Existing ONNX IR CSE removes only those two
duplicates. All three MatMul calls, complete outputs and initializer bytes
remain. Requested precision/output subsets are explicitly empty in the
frozen parent policy. Stock TensorRT8.5 trtexec therefore runs with unchanged
FP16/noTF32/obey/1024MiB/detailed flags and all actual subset requests.
The production JSON-file builder's empty-policy rejection is not relaxed.

| Build-Only Subset | SDK CPU Allocator Peak | SDK GPU Allocator Peak | Scratch | Emitted Engine |
| --- | ---: | ---: | ---: | ---: |
| Three original transposes | 3880 MiB | 1455 MiB | 512755712 B | 1017127592 B |
| One shared transpose | 1940 MiB | 1455 MiB | 512755712 B | 1017127658 B |

Both actual SDK builds exit0, no error rows, no inference. Initial available
memory differs24.55/22.89GiB and case order is fixed; do not claim a formal
build-time speedup or compare raw global-memory minima as model memory.
The SDK's own CPU allocation peak isolates a measured redundant-transpose
cost in this subset. GPU peaks and final engine sizes are effectively the
same: this is not evidence of an inference speedup or whole-model memory
saving. CPU/GPU shared physical memory is never summed.

Both independent units are terminal/collected, original manager/program
snapshots unchanged, no services stopped. The whole-stage root cause and
its successful build remain unproven until the original-scope retry completes.

## Generic Exact Rule

`vlaforge/tools/deduplicate_onnx_initializer_transposes.py` shares uniquely
named, internal standard Transpose nodes over the same readonly initializer
with the same permutation and exact precision/output requests. It skips
overridable weights, public duplicate outputs and ambiguous/backend-specific
nodes, rejects unsupported graph/quantization metadata, and evaluates no
floating values or external payloads. Parameters and public I/O remain
byte-identical; eliminated requests require identical canonical obligations.
No cast, arithmetic, decoder step, dtype or model-name special case is added.

18 focused tests and67 focused-plus-neighbor checks pass. The initial missing
tool red result remains. All five source parts were independently rechecked:

| Part | Nodes Before | Nodes After | Removed Constant Transposes |
| --- | ---: | ---: | ---: |
| 0 | 12260 | 11865 | 395 |
| 1 | 14510 | 13996 | 514 |
| 2 | 14413 | 13903 | 510 |
| 3 | 7742 | 7555 | 187 |
| 4 | 9316 | 9052 | 264 |

All2805native BF16 MatMul calls,15ArgMax generation steps, every original
arithmetic/cast/control node, parameter and public state interface remain.
Logical redundant-transpose volumes are not measured peak-memory savings.

The complete five-part CPU chain again passes:304intermediate outputs finite,
full INT64[1,16]tokens and FLOAT[1,1,248320]logits byte-exact against the
unpartitioned portable reference, MSE/maxabs0. One prepared observation only;
peak host RSS17.0854GiB, no24/3GiBguard trigger. Old official-source errors
remain separate, not reclassified as lossless or same-Orin acceptance.

## Live Original-Scope Retry

The new first part11865nodes is building in fresh window20261025-08-001.
Original af35plugin and618668b3native builder are unchanged; profiles, weights,
all parameter and arithmetic/precision obligations were independently checked
and target pins rehashed. Same24GiBadmission/4GiBfloor,3600second task,
7200second window with900second restoration reserve. Only data-collection
needed stopping to reach26865752KiB; starter stays original running birth.
Original Supervisor recovery is independently guarded and not yet claimed.

All134CPU references using the archived actual Orin prefix were regenerated
for this new graph and independently compared byte-for-byte to the old
first-part oracle: all dtype/shape/value bytes identical. The generic rule's
resulting nodes also exactly match the existing-library CSE used by the
actual controlled SDK subgraph. Neither check certifies target inference.

Next: audit this full-part terminal outcome, then actual typed I/O/context
and all134outputs with real Orin prefix states. Remaining parts, resident
complete16token lifecycle, primary paired quality and formal metrics remain.
No part of this diagnosis closes those gates. Five accepted VLA campaigns
and their retained paper evidence are unchanged.

## Evidence

- `artifacts/orin/20261025-08/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-09/build-window-001/authorization.json`
- `artifacts/orin/20261025-09/build-window-001/preparation.json`
- `artifacts/orin/20261025-09/build-window-001/checkpoint-001.json`
- `artifacts/orin/20261025-09/build-window-001/checkpoint-002.json`
- `artifacts/orin/20261025-09/resource-window-tests-003.xml`
- `artifacts/orin/20261025-06/report.json`
- `artifacts/orin/20261025-06/probe-001/stdout.log`
- `artifacts/orin/20261025-07/assets-001/staging.json`
- `artifacts/orin/20261025-07/report.json`
- `artifacts/orin/20261025-07/empty_scope_interface_rejection_001.json`
- `artifacts/orin/20261025-07/transpose-sharing-green-002.xml`
- `artifacts/orin/20261025-08/transpose-neighbor-regression-001.xml`
- `artifacts/orin/20261025-08/shared-constant-parts-001/report.json`
- `artifacts/orin/20261025-08/shared-portable-parity-001/report.json`
- `artifacts/orin/20261025-08/assets-001/staging.json`
- `artifacts/orin/20261025-08/build-window-001/checkpoint-001.json`
- `artifacts/orin/20261025-08/build-window-001/checkpoint-002.json`
- `artifacts/orin/20261025-08/micro-rule-correspondence-001.json`
- `artifacts/orin/20261025-08/actual-prefix-part00-reference-001/report.json`
