# Explicit Tensor Slice Workloads

`analysis.tensor_slice.extract_tensor_slice` builds a small FX module from an
existing static ExportedProgram. Inputs and outputs are explicit original node
names; the module contains their exact ATen dependency closure in original order.
It neither executes nor rewrites the source Region. State is passed at explicit
tensor boundaries, never guessed constant from an example or copied wholesale.

```python
workload = extract_tensor_slice(
    program,
    inputs=("sample", "velocity", "step_index"),
    outputs=("sample_next", "index_next"),
    source_artifact_sha256=verified_archive_sha,
)
workload.validate_inputs(actual_boundary_values)
reference = workload.module(*actual_boundary_values)
```

The names above are examples, not special public names or model dispatch keys.
The caller verifies the archive digest and supplies independently observed
boundary values. `validate_inputs` checks shape, dtype, device, layout, stride
and storage offset. It does not check data values, aliases, allocator addresses,
numerical policy or original-output correspondence. The raw FX module has no
automatic input guard; public export/backend contracts remain separate layers.

The initial implementation rejects dynamic profiles, non-tensor boundaries,
unused inputs, unresolved dependencies, implicit state, mutable or seeded-random
operators, uninitialized storage constructors, and non-ATen/HOP computations.
Schema-declared aliases are followed to reject writes elsewhere in the source
that could change a selected boundary/intermediate without appearing in its
dataflow. Unrelated invocation-local workspace mutation is allowed after the
source's recursive effect audit. This is intentionally not a general graph
partitioner: no-output guards outside the selected dataflow are not transplanted.
Consequently extraction is a profiling workload, not a deployment equivalence
proof or a license to replace a Region. Its ledger keeps execution, correctness,
performance and deployment-selection flags false.

## Real Solver Workload

`artifacts/edgefm-vla-goal/20260906-044509/operator-profiles/solver-slice-v4/`
records actual RTX 3060 worker PID1449510, exit 0. The source SmolVLA step archive
SHA is `35005d2fe439b1671f4a6775f2d7666b1f40ea90222922a01e3aea48a05e6661`.
The three copied nodes are `mul_208`, `add_58` and `add_59`, including the original
FP32 scale literal and separate multiply/add rounding. No hand-written Euler
implementation or quantized carry replaces this source computation.

Across 16 real episodes and all 10 original steps, the initial saved noise and
the independently audited original full velocity tensors reconstruct carry.
Every reconstructed sample matches the original full-trajectory observation
SHA and the next call's input SHA. Every actual index input also matches; the
final increment has no subsequent original call and is checked against the
copied source computation, not misreported as an observed next input. All 160
complete saved/reloaded slice outputs are byte-identical. Eight hundred complete
input/output tensors, source hashes, owner samples and full numerical/RNG checks
are retained. This is not a new execution of the original full model.

The static capture's singleton index stride is 1600. The experiment explicitly
restores that view for every call; the old trajectory report binds logical bytes
but did not observe per-call strides. The new workload must not claim original
per-call addresses, cache state or stride telemetry. Original carries were not
saved raw in the old observer run; these new files are reconstructed and hash
verified, not retrospectively attributed to that earlier run.

Attempts v1-v3 preserve, respectively, a driver status-field mistake, a correct
rejection of the wrong singleton input stride, and an artifact-driver raw hash
failure caused by `contiguous()` preserving a singleton's non-unit stride. v4
uses the existing canonical owned CPU snapshot. Public probe and microbenchmark
byte comparison now also support this case. The latest targeted CPU slice and
operator/probe regression has 102 passes. CUDA workload extraction does not
establish a tuned kernel, timing result, full-model speedup or board result.

## Exploratory Candidate Measurement

`operator-profiles/solver-candidates-v1/summary/` records a separately frozen
three-recipe campaign on the RTX 3060: ATen, Inductor eager-numerics, and public
ATen-preserving. The time/search budget and three recipes were fixed before
running. Each recipe uses one process and ten CUDA-graph batch means, not five
independent process repetitions or single-call latency samples. The first
boundary case is the timing input; every complete input/output tensor of all
160 original cases is checked independently of timing.

| Recipe | Mean graph-batch time, us | Compile, seconds | AOTI archive, bytes |
|---|---:|---:|---:|
| ATen | 3.068011 | N/A | N/A |
| Inductor eager-numerics | 1.726649 | 7.507776 | 456861 |
| ATen-preserving | 2.942838 | 5.733003 | 429820 |

All three recipes pass 160 two-output complete byte comparisons. Two initial
fresh candidate validators failed before loading AOTI because they omitted the
explicit `torch._inductor.codecache` import required by this pinned cold-loader
path. `revalidate-002/` freezes the corrected validator and rechecks the original
artifacts without rerunning compilation or timing; the old failures remain.
Summary SHA: `d9b0e6bab7b603862ab1bfdd1a00098eed278915a2270ce753c43b6b9ff29b0f`.

No candidate is selected for deployment. A source-preserving terminal-tail
partition is a distinct transformation described in `terminal-tail-partition.md`.
Its extra Region boundary/copies can erase this small isolated improvement;
full-model byte verification and same-source end-to-end measurement are required.
