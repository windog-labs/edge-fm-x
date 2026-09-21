# J6M Feasibility Assessment

Date: 2026-09-11

## Scope

This report evaluates whether the existing EdgeFM models can be moved to the
Horizon J6M toolchain and hardware. It does not claim model-level E2E support.
The current execution round was subsequently restricted to SmolVLA only.
Other models in this report remain capacity assessments, not measured runs.
The evidence is limited to:

- reusable host-side Horizon toolchain validation;
- fixed-shape operator probes compiled to HBM for `nash-m`;
- board load, inference, and performance smoke tests on `j6m-2`;
- model capacity and architecture assessment from existing verified model
  inventories.

## Host Environment

The existing environment
`/home/zhangzimo/miniconda3/envs/vlaforge-openvla` contains the required
Horizon packages. The missing environment detail is a usable Python 3.10
interpreter path and `hb_compile` shebang; rebuilding the environment is not
required.

Validated tool versions:

| Component | Version |
|---|---:|
| hbdk4-compiler | 4.5.5 |
| hbdk4-march | 4.5.5 |
| hmct | 2.5.6 |
| horizon-tc-ui | 3.5.3 |

The current non-invasive run uses the packaged Python 3.10 interpreter with
`PYTHONHOME` and `PYTHONPATH` pointing at the existing environment. No Conda
environment or package files were changed.

## Board Inventory

| Board | Serial | Runtime | Result |
|---|---|---|---|
| j6m-1 | `0e1941183392371c` | `hrt_model_exec` lacks `libbpu.so.2` and has unresolved runtime libraries | deferred until the runtime is completed |
| j6m-2 | `0e190a0d3392371c` | UCP 3.14.7, HBRT 4.9.7, HBM load/infer/perf works | used for all board smoke tests |

No external `hrt_model_exec` worker was present during the smoke tests. The
board has six CPU cores and approximately 14.7 GiB RAM.

## Operator Probe Results

Probe source:
`artifacts/j6m-feasibility-20260911/probe_hbdk_ops.py`

Evidence:

- FP32 run:
  `artifacts/j6m-feasibility-20260911/run-006/`
- INT8 run:
  `artifacts/j6m-feasibility-20260911/run-007-quant/`
- Aggregated matrix:
  `artifacts/j6m-feasibility-20260911/summary-001/operator-support-matrix.md`

Every probe runs in a separate process. This preserves a failed native compiler
crash as evidence instead of terminating the remaining probes.

### FP32

All fixed-shape graphs except F64 compile, load on `j6m-2`, and execute.
However, FP32 Linear, Conv, Softmax, and LayerNorm generally fall back to
`external_cpu`. A loadable F32 HBM therefore does not mean the operators run on
BPU.

| Probe | Compile | Board | CPU fallback nodes | Board latency |
|---|---:|---:|---:|---:|
| GEMM + LayerNorm + Embedding | passed | passed | 4 | 7.385 ms |
| Attention | passed | passed | 5 | 29.124 ms |
| RoPE | passed | passed | 0 | 0.243 ms |
| RMSNorm | passed | passed | 0 | 0.875 ms |
| Conv + vision | passed | passed | 5 | 133.216 ms |
| Flow-matching step | passed | passed | 2 | 0.700 ms |
| Gated DeltaNet-style step | passed | passed | 2 | 0.560 ms |
| F64 state/output | failed at convert | not run | N/A | N/A |

The F64 failure is real: the backend accepts signed integer, FP16, FP32, and
Bool dtypes, but not F64. Pi0/pi0.5 native F64 action or state tensors cannot be
placed directly in a BPU graph.

### INT8

The same fixed-shape graphs were calibrated with deterministic representative
inputs, converted through HMCT, and compiled for `nash-m`.

| Probe | Compile | Board | CPU fallback nodes | Board latency |
|---|---:|---:|---:|---:|
| GEMM + LayerNorm + Embedding | passed | passed | 1 cast from INT64 | 0.315 ms |
| Attention | passed | passed | 0 | 0.258 ms |
| RoPE | passed | passed | 0 | 0.267 ms |
| RMSNorm | passed | passed | 0 | 0.433 ms |
| Conv + vision | passed | passed | 0 | 0.413 ms |
| Flow-matching step | passed | passed | 0 | 0.276 ms |
| Gated DeltaNet-style step | passed | passed | 0 | 0.313 ms |
| F64 state/output | failed before graph build | not run | N/A | N/A |

The INT8 probes place the representative heavy operators on BPU or VPU and
execute successfully on `j6m-2`. This establishes feasibility of the operator
families, not accuracy or full-model support.

Important remaining constraints:

- Embedding indices must arrive as INT16/INT8-compatible values. The INT64
  index cast remains a CPU fallback in the probe.
- The probes use synthetic random calibration. They do not establish
  task-quality-preserving quantization.
- Attention masks, dynamic sequence state, KV cache, recurrent state, and
  multi-stage graph boundaries have not yet been integrated into a model.

## Model Capacity Assessment

| Model | Verified scale | J6M implication |
|---|---:|---|
| SmolVLA | 450,046,176 model elements | Best P0 candidate for a staged real J6M path |
| RDT-1B online pipeline | 6,418,856,128 active elements across RDT, T5, and SigLIP | BF16 weights alone are about 12.8 GB; a single full graph is not viable on the 14.7 GiB board |
| pi0 | 3,501,372,176 parameters; 7,002,873,776-byte converted checkpoint | Capacity may permit a split deployment, but native F64 state/output and the full graph boundary remain unresolved |
| pi0.5 | 3,616,757,520 parameters; 7,233,650,408-byte converted checkpoint | Same F64 issue as pi0; capacity is possible but not sufficient to claim support |
| CogACT | 7,630,224,071 parameters; 30,521,280,578-byte FP32 checkpoint | The paper's 3B label is inconsistent with the complete installed stack. A full single deployment is over the board memory budget |
| Qwen3.5 0.8B | 1,746,942,600-byte weight set | Most likely Qwen candidate for J6M |
| Qwen3.5 2B | 4,548,221,488-byte weight set | Weight capacity is plausible; prefill/decode state and recurrent/conv state are the main risks |

No model above has completed a J6M model-level E2E run. SmolVLA has completed
one real action-solver stage: stage-060 runs at 90.088 ms mean / 11.062
steps/s on `j6m-2`, with a board action cosine of `0.9999994983908165`.
Prefix, all ten solver steps, and action trim have since been chained through
the `full-chain-v1` file-orchestration pilot. The chain executed successfully,
but its complete-action fidelity was below the strict gate: cosine
`0.999903171999195`, max-abs `0.050837501883506775`, and mean-abs
`0.01609863852150738`. Prefix cache quantization is the dominant source of
that error, with minimum cache cosine `0.994239330291748`.

The first lowered Prefix (V2) is compiled and profiled on `j6m-2` at
`1405.032 ms`, but its 63 external-CPU `ScatterND` nodes consume
`811.092 ms`. A generic static-index rewrite lowers those nodes to
`Slice + Concat`; the V5 graph has no remaining `ScatterND`, passes host ONNX
parity with minimum cache cosine `0.9999999999975593`, and passes H20 HMCT
quantization with `batch=1`. Its HBM compile passed after `5091.3 s`:
`454521544` bytes with SHA256
`b05a1c4d62d29b37e29f7a347f7003440b73cbdbe6d76b44d67c1491adf30993`.
Offline `hbrt4-disas` confirms `64` HBM nodes (`44` CPU, `20` BPU) versus
V2's `232` (`157` CPU, `75` BPU), with both `InplaceScatterND` and `Pow`
removed. Board validation is pending because `j6m-2` became network-unreachable.

## Current Classification

| Item | Classification |
|---|---|
| Reuse existing Conda Horizon toolchain | passed |
| J6M `nash-m` operator compile | passed for tested non-F64 probes |
| J6M board load/infer/perf | passed on `j6m-2` |
| F64 state/output on BPU | unsupported in the tested graph |
| SmolVLA J6M step | passed; stage-060 measured on `j6m-2` |
| SmolVLA J6M full E2E | pilot only; the file chain ran, but strict complete-action fidelity failed |
| RDT J6M E2E | deferred by current scope |
| pi0/pi0.5 J6M E2E | deferred by current scope |
| CogACT J6M E2E | deferred by current scope |
| Qwen3.5 0.8B/2B J6M E2E | deferred by current scope |

## Next Gates

1. Validate the compiled generic lowered-v5 Prefix HBM on `j6m-2`.
2. Re-run the file chain with only the Prefix stage replaced by lowered V5.
3. Verify the final action chunk over all eight held-out inputs.
4. Implement a resident runner before reporting formal E2E latency or CDF.
5. Keep stage-060 and `full-chain-v1` classified as step and pilot evidence.
6. Resume Qwen, RDT, pi0/pi0.5, or CogACT only under a new explicit scope.
