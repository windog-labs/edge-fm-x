# Goal: EdgeFM J6M Feasibility and Supported Deployment (Interim v2)

Status: provisional execution prompt, not the final experiment prompt.

This version exists to drive the model- and stage-level feasibility gate. Do
not treat it as a frozen commitment that all listed models can run E2E on J6M.
After SmolVLA, Qwen3.5, and the remaining VLA models have measured support
classifications, publish a v3 prompt based on that evidence. Only v3 is
allowed to define the final experiment campaign and paper-facing scope.

Worktree:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x`

Paper:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/ICRA_2027_EdgeFM.pdf`

## Objective

Use `j6m-1` and `j6m-2` to complete every Horizon J6M experiment that is
actually supported by the measured toolchain, model capacity, and operator
capabilities. Orin and J6M work must not be conflated. Orin remains deferred
until its hardware is available.

This goal must not assume that all five VLA models fit on J6M or can be lowered
as one BPU graph. The deliverable is a verified support matrix plus real board
results for the supported model stages. Unsupported, OOM, compile-failed, and
SDK-blocked cases are valid classifications when backed by reproducible
evidence. A v3 campaign must be derived from those classifications rather than
from the model list alone.

## Reuse Existing Environment

Reuse the existing Horizon environment:

`/home/zhangzimo/miniconda3/envs/vlaforge-openvla`

Do not rebuild it unless the existing packages are proven unusable. The current
known issue is the Python 3.10 executable/shebang path, not missing Horizon
packages. A reusable non-invasive launcher is required before long runs.

## Evidence From The Feasibility Gate

Read:

- `doc/reports/j6m_feasibility_assessment_20260911.md`
- `artifacts/j6m-feasibility-20260911/summary-001/operator-support-matrix.md`
- `artifacts/j6m-feasibility-20260911/summary-001/operator-support-matrix.json`
- `artifacts/j6m-feasibility-20260911/probe_hbdk_ops.py`
- `artifacts/j6m-feasibility-20260911/summarize_feasibility.py`

The current evidence establishes:

- Horizon compiler and `j6m-2` runtime are usable.
- Representative INT8 GEMM, Attention, RoPE, RMSNorm, Conv/vision,
  flow-matching, and Gated DeltaNet-style probes compile and execute on J6M.
- F64 graph state/output is unsupported by the tested HBDK path.
- FP32 graphs can compile and run while silently falling back to CPU, so
  `HBM exists` is not evidence of BPU execution.
- No model-level J6M E2E has passed yet.

## Resource Rules

1. Only use `j6m-1` and `j6m-2` for J6M runs.
2. Prefer `j6m-2` while `j6m-1` lacks a complete runtime.
3. Check hostname, serial, UCP/HBRT/HBDK versions, BPU ownership, RAM, disk,
   temperature, and existing workers before and after every job.
4. Never terminate or preempt another user's process.
5. Keep source snapshots, build directories, caches, inputs, and outputs
   separate per board and per model.
6. Do not use H20/H100 artifacts as J6M results.

## Required Status Model

Every model, stage, and precision must use exactly one of:

- `passed`: real board execution with complete output verification.
- `pilot`: real board execution with an explicitly reduced sample count.
- `compile-failed`: reproducible compiler error.
- `unsupported`: a concrete unsupported operator, dtype, shape, or state path.
- `OOM`: measured memory rejection.
- `deferred`: missing hardware, authorization, or SDK.

Do not use `passed` for any of these:

- ONNX export success;
- HBDK conversion success;
- HBM creation;
- checker success;
- CPU or CUDA execution;
- a stage-only result when the claim is full-model E2E.

## P0: SmolVLA Real J6M Path

SmolVLA is the first model target.

1. Use verified real inputs, noise, and the existing H20 complete-action
   reference.
2. Define a generic stage split through the existing adapter/provider
   interfaces. Model-specific differences belong only in adapters,
   descriptors, and quantization recipes.
3. Start with the smallest real stage export and compile it for `nash-m`.
4. Inspect HMCT/HBDK advice and record every CPU fallback.
5. Quantize only with an explicit calibration split. Keep held-out samples
   separate from calibration samples.
6. Compile HBM and run on `j6m-2` using `model_info`, `infer`, and `perf`.
7. Run complete action chunks, not only the first action.
8. Compare complete outputs with the existing reference using cosine, MSE,
   RMSE, max-abs, worst-sample, and per-dimension metrics.
9. Record raw latency and save the CDF after the model path is functional.

## P1: Qwen3.5 0.8B And 2B

Attempt Qwen3.5 in this order:

1. Qwen3.5 0.8B vision/prefill/decode stages.
2. Qwen3.5 2B stage-by-stage after the 0.8B boundary is understood.

Required outputs:

- complete token sequence and logits where the runtime exposes them;
- TTFT and decode rate;
- KV/recurrent/conv state reset and cross-request isolation;
- raw latency and memory;
- the first unsupported operator or state shape for each failed stage.

Do not substitute a fixed saved token tape for live decode.

## P2: Capacity-Limited VLA Models

For RDT, pi0, pi0.5, and CogACT:

1. Do not promise a single full BPU graph.
2. Start from the existing verified model inventory and active parameter
   counts.
3. Export each major stage independently and record compile result, memory,
   and operator placement.
4. For pi0/pi0.5, resolve F64 state/output by explicit graph boundaries or CPU
   fallback. Do not silently cast and claim equivalence.
5. For RDT, keep T5, SigLIP, condition projection, denoiser, and scheduler
   boundaries separate.
6. For CogACT, record the complete 7.63B parameter / 30.52 GB checkpoint
   scale and treat a full deployment as OOM unless a measured split makes it
   possible.

The acceptable result for an unsupported model is a reproducible boundary
report, not fabricated main-table data.

## Operator And Agent Tuning Ablation

Use the same generic probe pipeline to compare:

1. default graph/quantization recipe;
2. tuned graph/quantization recipe;
3. Horizon native/vendor baseline where available.

Cover:

- Attention;
- GEMM/Linear;
- LayerNorm/RMSNorm;
- Embedding;
- RoPE;
- VLA-specific operators, including timestep embedding, AdaLN, action
  projection, cross-attention, and solver/flow updates.

Save shape, dtype, layout, calibration, correctness gate, raw timing,
compiler choices, search budget, failures, and profiler output. Reinsert a
winning operator recipe into at least one model and remeasure E2E.

## Main Table And CDF Rules

Only model-level real board results enter the main table. Each row declares:

- checkpoint revision and precision;
- fixed input profile, batch size, CFG, horizon, and scheduler;
- fresh action chunks/s or decode tokens/s;
- mean, p50, p95, p99, max, and std;
- BPU, VPU, CPU fallback, DMA, and postprocess boundaries;
- complete-output fidelity;
- peak resident and workspace memory.

The formal target is five independent processes per profile with 128 warmup
and 1024 measured calls. A smaller run is labeled `pilot`.

## Required Evidence

For every build and run store:

- source HEAD and dirty patch hash;
- model/checkpoint revision and input/reference hashes;
- OE/HBDK/HBRT/UCP/driver/compiler versions;
- board serial and BPU identity;
- exact commands, PID, logs, and exit code;
- ONNX, quantized graph, HBM, manifest, and DSO hashes;
- raw samples, CSV/JSON, CDF data, PNG/PDF, and plotting source;
- per-stage memory and operator-placement advice;
- independent rerun and local reconstruction where applicable.

## Completion Criteria

The goal is complete when:

- SmolVLA has a real J6M stage or full E2E result within the measured support
  boundary;
- Qwen3.5 0.8B has a measured supported definition, and 2B has either a result
  or a reproducible boundary;
- RDT, pi0, pi0.5, and CogACT each have a concrete classification with
  evidence;
- at least one same-platform operator tuning comparison is complete;
- the final report separates passed, pilot, OOM, unsupported, compile-failed,
  and deferred items;
- every main-table cell has a raw evidence path and no unsupported inference
  from H20/H100 is used.
