---
name: edgefm-nsys-profiler-triage
description: "Compact Edge-FM Nsight Systems triage skill. Use when Codex should inspect an Edge-FM `.nsys-rep` or exported `.sqlite` profile, especially from `scripts/profile/profile_edgefm_generate_case.py` or `scripts/profile/profile_vlm_prepared_case.py`, and return one compact report with kernel hotspots, known-path matches, and concrete next actions. Supports graph-off mapping plus graph-on formal two-trace triage for CUDA-graph cases."
---

# Edge-FM Nsight Systems Profiler Triage

Use this skill for Edge-FM `nsys` traces.

The public entrypoint is:

- [scripts/analyze_edgefm_nsys_profile.py](scripts/analyze_edgefm_nsys_profile.py)

`triage` always returns the same three tables:

- kernel table
- known-path table
- action table

By default, only rows at or above `1.0%` stage GPU-time share are rendered.
Treat anything below that as noise unless the user explicitly wants deeper tails.

## When To Use It

- inspect an existing Edge-FM `.nsys-rep` or exported `.sqlite`
- explain which kernels dominate `prefill` or `decode`
- map hotspots back to Edge-FM `NVTX layer_name`
- decide whether a hotspot should first be treated as:
  - an existing tuned operator path that did not fire
  - a missing `operator_impl_table` record
  - a real new optimization opportunity

## Main Flows

### 1. Single-trace triage

Use when one trace is already available and you want the fastest read on hotspot share.

```bash
python3 .codex/skills/edgefm-nsys-profiler-triage/scripts/analyze_edgefm_nsys_profile.py \
  --input /path/to/profile.nsys-rep
```

### 2. Two-trace triage

Use when CUDA graph is enabled and you need both:

- graph-off attribution quality
- graph-on final behavior

```bash
python3 .codex/skills/edgefm-nsys-profiler-triage/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input /path/to/graph_off.nsys-rep \
  --formal-input /path/to/graph_on.nsys-rep
```

The mapping trace exists to recover:

- `kernel -> runtime launch -> NVTX layer_name`
- `kernel -> stage`

Do not call it a fast profile. Its job is attribution.

## Collection Commands

### LLM / plain generate case

Graph-off mapping trace:

```bash
nsys profile -o .tmp_codex/nsys/edgefm_mapping \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 1024 \
    --decode-len 32 \
    --profile-range
```

Graph-on formal trace:

```bash
nsys profile -o .tmp_codex/nsys/edgefm_formal \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 1024 \
    --decode-len 32 \
    --use-cuda-graph \
    --profile-range
```

### Prepared VLM case

Use the existing prepared-case helper so ViT stays outside the profiled region.

```bash
nsys profile -o .tmp_codex/nsys/vlm_formal \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_vlm_prepared_case.py \
    --framework edgefm \
    --model-path /path/to/qwen2.5-vl \
    --model-size 0.5b \
    --prefill-len 1024 \
    --decode-len 32 \
    --use-cuda-graph \
    --profile-range
```

## Workflow

1. Prefer a single trace for quick diagnosis.
2. Prefer mapping/formal two-trace triage when CUDA graph replay makes attribution weaker.
3. Read the tables in this order:
   - kernel table
   - known-path table
   - action table
4. Before calling a hotspot a new idea, compare it against existing Edge-FM paths in [references/known-paths.md](references/known-paths.md).
5. Favor Edge-FM-native attribution:
   - `stage`
   - `NVTX layer_name`
   - layer role
   - known `impl_id` family
   - relevant tuning script or operator-table path

## Output Contract

Return:

- input trace path(s)
- exported sqlite path(s) when applicable
- kernel table
- known-path table
- action table
- one short conclusion
- whether the result came from single-trace or mapping/formal two-trace mode

## References

Load only when needed:

- [references/known-paths.md](references/known-paths.md)
  - current Edge-FM tuned-path catalog and the first repo files to inspect per hotspot family
