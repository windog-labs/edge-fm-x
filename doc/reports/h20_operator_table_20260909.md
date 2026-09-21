# H20 Operator Comparison

All six families completed five independent process pairs and remote plus local
full-output/ownership audits. The before/after workloads are identical within
each family. Qualification runs are excluded, and all measured samples remain.

| Family | ATen us | Candidate us | Latency Reduction | Candidate |
|---|---:|---:|---:|---|
| Attention | 103.301055 | 103.313151 | -0.0117% | Inductor ATen |
| GEMM / Linear | 11.540214 | 12.870522 | -11.5276% | Inductor autotune |
| LayerNorm | 13.816088 | 13.716502 | 0.7208% | Inductor ATen preserving |
| Embedding | 2.142171 | 1.865898 | 12.8969% | Inductor ATen |
| RoPE frequencies | 21.332679 | 8.486607 | 60.2178% | Inductor ATen |
| VLA solver/update | 4.289051 | 2.203624 | 48.6221% | Inductor ATen |

Each value is the mean of five per-process medians of graph-batch device timings,
not a per-call latency CDF. The workloads were extracted from real model execution;
each row covers the specific archived signature/profile, not arbitrary shapes.
All complete target-reference and candidate output tensors match byte for byte.
LayerNorm represents the normalization family; separate RMSNorm coverage is not
claimed. The solver row is an actual three-operator update fragment, not a full
action model.

The fixed candidate recipes were selected using prior evidence. This documents
an agent-assisted compiler-recipe comparison, not autonomous invention of every
kernel. Attention is effectively unchanged; the measured Linear candidate is
slower. Negative results are retained. No candidate is promoted to deployment and
no complete-model gain is inferred from these microbenchmarks.

## Platform and Evidence

Host: `zzm-h20-x8-2`, H20 GPU6 UUID
`GPU-2b9df5d3-1016-2167-2320-89139d3db269`, Torch 2.10.0+cu128.
Every worker uses the register/reset/register handshake; the audits recheck
container/NVML PID bindings, telemetry, non-overlapping worker intervals, complete
serialized output bytes, fixed numerical context and raw timing summaries.

The two earlier resource-conflict attempts remain incomplete and are excluded.
The accepted solver root is
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/operator-h20-solver-recovery-20260909-002/`.
The other five roots are beneath
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/operator-h20-remaining-20260909/`.
Original workload manifests and already-qualified packages were reused where
applicable; no completed full VLA experiment was repeated.

Current-worktree artifacts:

- `artifacts/recovery-audit-20260909/operator-six-summary-002/operator-table.csv`:
  all values, recipes, worker PID lists, GPU UUID and both audit hashes.
- `.../report.json`: row source identities and exact package digests.
- `.../operator-comparison.png` and `.pdf`: inspected chart; the intervals are
  exploratory paired-bootstrap intervals over the five process pairs. They were
  calculated after measurement and are not a pre-registered deployment-selection
  gate. Raw samples were not removed.
- Full local campaigns and audits: `operator-final/` and `solver-h20-confirmed/`
  under the same current-worktree recovery artifact directory.

CSV SHA256: `9a256a1636cc437c4c649ae77085ec9d5aff15df22f6419c77685e07756baf57`.
Summary SHA256: `039002cdda6214b247f9bd2fdfe301a6ab8c97e9684c2b8af482f2b076f41f11`.
Revision 002 names the PID column `worker_process_ids`; these are Python
measurement workers invoking ATen/AOTI, not no-Python Session workers. The
measurements are unchanged and the PNG is byte-identical to the inspected image.

These measurements supply a scoped H20 operator ablation. They do not substitute
for Orin/BPU measurements, integrated-model speedups or evidence of a complete
autonomous optimization/skill-reuse loop.
