# EdgeFM CUDA Kernel Optimizer Usage Guide

This document explains how to use the in-repo
`.codex/skills/edge-fm-cuda-kernel-optimizer` to perform
EdgeFM model performance tuning for new NVIDIA hardware platforms. The goal is not to blindly write kernels, but to establish a reproducible flow:
first align EdgeFM and `TRT-Edge-LLM` on the same model, same shape, and same runtime parameters,
then use NSYS/NCU to locate the gap, and finally use source-op, plugin-op, or
the Humanize/KernelPilot long loop to gradually catch up to or even surpass the reference.

## 1. Installing and Enabling the Skill

Within the `edge-fm-x` repository, the skill ships with the repo:

```bash
.codex/skills/edge-fm-cuda-kernel-optimizer/SKILL.md
```

Usually no additional installation is needed. After opening a new Codex session, explicitly reference
`$edge-fm-cuda-kernel-optimizer` in your tuning request, and Codex will read the skill and follow its flow.

To reuse this capability in another repository, copy the entire directory to the target repo:

```bash
mkdir -p .codex/skills
cp -a /path/to/edge-fm-x/.codex/skills/edge-fm-cuda-kernel-optimizer \
  .codex/skills/
```

Long-running autonomous optimization requires the Humanize hooks. Installing the hooks alone does not automatically start optimization:

```bash
bash .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/install_humanize_hooks.sh
```

Common entry points:

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py \
  --out .tmp_codex/env/gpu_env.json

python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input .tmp_codex/nsys/edgefm_graph_off.nsys-rep \
  --formal-input .tmp_codex/nsys/edgefm_graph_on.nsys-rep
```

## 2. Configuring the GPU Profiling Environment

Before tuning a new platform, first confirm that these tools are available:

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
python3 --version
cmake --version
```

Environment variables commonly used by the EdgeFM CUDA platform:

```bash
export EDGE_FM_BUILD_DIR=/path/to/edge-fm-x/build-<platform>
export EDGE_FM_PLATFORM=<platform>
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_TEST_DEVICE_ID=0
export LD_LIBRARY_PATH=$EDGE_FM_BUILD_DIR/lib:$EDGE_FM_BUILD_DIR/install/lib:${LD_LIBRARY_PATH:-}
```

An NCU executable does not imply that counter permissions are available. First do a minimal smoke test:

```bash
ncu --set basic --target-processes all --version
```

If profiling reports `ERR_NVGPUCTRPERM`, do not treat the missing counters as a real performance conclusion.
You should first configure GPU counter permissions, or temporarily fall back to NSYS attribution and operator microbenchmarks.
When sudo is required, use only narrow-privilege approaches, and do not write the sudo password into commands, scripts, or documentation.

To compare against `TRT-Edge-LLM`, prepare the following in advance:

- EdgeFM model artifacts and config.
- The TRT-Edge-LLM engine workspace.
- The corresponding plugin library, e.g. `libNvInfer_edgellm_plugin.so`.
- The same set of model size, prefill length, decode length, warmup, runs, and CUDA graph settings.

## 3. Tuning Flow for a New Hardware Platform

It is recommended to proceed in the following order, leaving behind an artifact path and an accept/reject reason for each step.

### 3.1 Establish a Paired Benchmark Baseline

First run the paired matrix of EdgeFM and TRT-Edge-LLM. At minimum cover the target model sizes and the commonly used
prefill/decode shapes:

```bash
python3 scripts/profile/profile_edgefm_generate_case.py \
  --model-path /path/to/model \
  --prefill-len 2048 \
  --decode-len 64 \
  --use-cuda-graph \
  --runs 3 \
  --json

python3 scripts/profile/profile_trt_edgellm_generate_case.py \
  --model-path /path/to/model \
  --engine-dir /path/to/trt_workspace \
  --plugin-path /path/to/libNvInfer_edgellm_plugin.so \
  --prefill-len 2048 \
  --decode-len 64 \
  --runs 3 \
  --json
```

The output table should contain at least:

- EdgeFM total / prefill / decode
- TRT total / prefill / decode
- total gap
- prefill gap
- decode gap
- tokens/s or decode step avg

### 3.2 Do NSYS Attribution First, Then Decide on NCU Targets

CUDA graphs hide kernel attribution. First capture a graph-off mapping trace, then capture a
graph-on formal trace:

```bash
nsys profile -o .tmp_codex/nsys/edgefm_graph_off \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 2048 \
    --decode-len 64 \
    --profile-range

nsys profile -o .tmp_codex/nsys/edgefm_graph_on \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 2048 \
    --decode-len 64 \
    --use-cuda-graph \
    --profile-range
```

Then use the skill script to generate the kernel table, known-path table, and action table:

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input .tmp_codex/nsys/edgefm_graph_off.nsys-rep \
  --formal-input .tmp_codex/nsys/edgefm_graph_on.nsys-rep
```

Only put kernels that satisfy these conditions into the NCU or Humanize queue:

- The target slice has a clear end-to-end gap.
- The kernel/operator accounts for a large enough proportion of the total gap.
- There is a correctness reference or an operator/layer-level test.
- The existing operator table, cuBLASLt, FlashInfer, and CUTLASS small-parameter sweep have already plateaued.

### 3.3 Queue by Operator Gap

Common priorities:

1. Prefill attention / FMHA / KV write / RoPE.
2. Dense linear such as QKV, OProj, MLP GateUp/Down.
3. Decode attention and decode linear.
4. Norm, sampler, finalize, response copy.
5. Runtime-level launch, CUDA graph, stream overlap, host sync.

Try low-risk routes first:

- Update the platform operator table.
- Tune the parameters of the existing source-op / FlashInfer / cuBLASLt / CUTLASS.
- Add model/shape/stage-level selection records.

When short-range tuning yields no stable gains, then move into Humanize + KernelPilot.

### 3.4 The Humanize + KernelPilot Long Loop

Entry conditions:

- There are clear hotspots, shapes, baselines, references, and validation entry points.
- It is expected to require multiple rounds of candidates, NCU evidence, source provenance, and a rejected ledger.
- TRT or another reference is clearly faster, but you cannot directly depend on a serialized TensorRT engine.

Recommended directory:

```bash
deliverables/kernel_opt/<platform>_<operator>_<date>/
```

Each long loop should keep at least:

- baseline / reference / dims
- NCU baseline digest
- attempt ledger
- optimization ledger
- source idea ledger
- accepted / rejected summary

Migration-back rules:

1. The standalone/repro first proves correctness and latency.
2. Migrate back to `src/operators`, `src/layers`, or the operator table.
3. Rebuild and install the Python binding.
4. Run operator/layer/engine regressions.
5. Run the paired benchmark to confirm the gain has not vanished in the real generate path.

## 4. Tuning Prompt Templates

### 4.1 Generate a Tuning Plan for a New Platform

```text
Please use $edge-fm-cuda-kernel-optimizer to draft an EdgeFM LLM
performance tuning plan for <platform>. The goal is to catch up to and, as much as possible, surpass TRT-Edge-LLM.

Constraints:
1. Do not make large architectural changes.
2. First establish the EdgeFM vs TRT-Edge-LLM paired benchmark matrix.
3. Use graph-off NSYS for kernel attribution, and use the graph-on trace to confirm end-to-end behavior.
4. Output an operator gap table, including at least prefill attention, QKV/OProj, MLP,
   decode attention, norm/sampler/finalize.
5. Prioritize small-range tuning of the operator table, source-op, FlashInfer, cuBLASLt, and CUTLASS.
6. Only launch the Humanize + KernelPilot long loop after the short-range plateau.
7. Update the documentation and record artifacts at every accepted/rejected node.
```

### 4.2 Continuously Optimize a Specific Operator

```text
Please use $edge-fm-cuda-kernel-optimizer to continuously optimize <operator>.

Target slice:
- model: <model>
- prefill_len: <N>
- decode_len: <M>
- platform: <platform>

Requirements:
1. First confirm the current prefill/decode/total gap between EdgeFM and TRT-Edge-LLM.
2. Find the real kernel name and time of this operator in NSYS/NCU.
3. Prioritize establishing an operator/layer-level benchmark or a standalone repro.
4. Each round absorbs only changes that pass correctness and do not regress end-to-end.
5. If two rounds of short-range optimization plateau, escalate to Humanize + KernelPilot.
```

### 4.3 Decide Whether the TRT Bridge Can Be Removed

```text
Please assess whether the current <platform>/<model> can have the TRT bridge removed.

Please distinguish three modes:
- native/source-op: EdgeFM's own CUDA/CUTLASS/FlashInfer path.
- plugin-op: reuses the source-visible TRT-Edge-LLM plugin/kernel, but does not load the serialized engine.
- trt-reference: used only as a benchmark reference or fallback.

Acceptance criteria:
1. source-op or plugin-op reaches practical parity on the target matrix.
2. Correctness regressions pass.
3. Neighboring shapes show no obvious degradation.
4. README/doc clarifies whether the default path still depends on the TensorRT engine/context.
```

### 4.4 Launch Humanize + KernelPilot

```text
Please use $edge-fm-cuda-kernel-optimizer and $humanize-kernel-agent-loop
to launch a long-loop optimization for <kernel/operator>.

Preconditions:
- Hotspot evidence already exists.
- A correctness reference already exists.
- A standalone or operator benchmark already exists.
- An NCU baseline digest already exists.

Output requirements:
- A deliverables/kernel_opt/<name>/ standalone repo.
- A refined plan.
- A source/lineage ledger.
- An attempt/optimization ledger.
- A per-round profile evidence digest.
- Migration-back conditions and a final accept/reject conclusion.
```

### 4.5 Periodic Reporting

```text
Please summarize the results of the current tuning phase:
1. The gap between the current best matrix and TRT-Edge-LLM.
2. Accepted optimizations and their gains.
3. Rejected routes and the reasons.
4. The largest remaining operator gap.
5. The plan for the next round.
6. Artifact paths.
```

## 5. Acceptance Criteria

Each new-platform tuning phase should satisfy at least:

- Correctness takes priority over latency.
- The paired benchmark uses the same shape, same runs, and same CUDA graph settings.
- Accepted changes have artifacts and regression results.
- Rejected routes spell out the reason clearly, to avoid repeatedly burning time.
- The default path does not depend on an unstated TensorRT engine/context.
- README or doc retains the currently valid performance matrix; temporary tuning logs do not pile up in `doc/` long-term.
