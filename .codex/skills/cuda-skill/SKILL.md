---
name: cuda-skill
description: "Query NVIDIA PTX ISA 9.1, CUDA Runtime API 13.1, Driver API 13.1, Programming Guide v13.1, Best Practices Guide, Nsight Compute, Nsight Systems local documentation. Debug and optimize GPU kernels with nsys/ncu/compute-sanitizer workflows. Includes hardware quick guides for T4, A100, H100, A800, RTX 3060, and Jetson Orin. Use when writing, debugging, or optimizing CUDA code, GPU kernels, PTX instructions, inline PTX, TensorCore operations (WMMA, WGMMA, TMA, tcgen05), or when the user mentions CUDA API functions, error codes, device properties, memory management, profiling, GPU performance, compute capabilities, CUDA Graphs, Cooperative Groups, Unified Memory, dynamic parallelism, or CUDA programming model concepts."
---

# CUDA & PTX Reference

## Documentation Locations

In this repo, the canonical documentation root is `.codex/skills/cuda-skill/references/`.
This repo builds that tree from two upstream sources:
- `ptx-isa-markdown/cuda_skill/references/` provides PTX / Runtime / Driver docs and the quick guides bundled with that repo.
- `agent-gpu-skills/cuda_skill/references/` provides `ptx-simple`, the CUDA Programming Guide, Best Practices Guide, Nsight full docs, and hardware profiles.

Set the search root once and reuse it in every command:

```bash
CUDA_REFS=.codex/skills/cuda-skill/references
if [ ! -e ${CUDA_REFS} ]; then
  CUDA_REFS=$(find \
    ~/.codex/skills \
    ~/.claude/skills \
    ~/.cursor/skills \
    ~/.agents/skills \
    -path '*/cuda-skill/references' -type d 2>/dev/null | head -1)
fi
```

All `rg` examples below assume `CUDA_REFS` is set.

**Source split used by this repo:**
- `ptx-isa-markdown`: `ptx-docs/`, `cuda-runtime-docs/`, `cuda-driver-docs/`, `ptx-isa.md`, `cuda-runtime.md`, `cuda-driver.md`, `debugging-tools.md`, `ncu-guide.md`, `nsys-guide.md`, `nvtx-patterns.md`, `performance-traps.md`
- `agent-gpu-skills`: `ptx-simple/`, `cuda-guide/`, `best-practices-guide/`, `ncu-docs/`, `nsys-docs/`, `hardware/`

```
references/
├── ptx-docs/              # PTX ISA 9.1 full spec (405 files, 2.3MB)
├── ptx-simple/            # PTX condensed quick-ref (13 files, 149KB)
├── cuda-runtime-docs/     # CUDA Runtime API 13.1 (107 files, 0.9MB)
├── cuda-driver-docs/      # CUDA Driver API 13.1 (128 files, 0.8MB)
├── cuda-guide/            # CUDA Programming Guide v13.1 (39 pages, 1.6MB)
│   ├── 01-introduction/   # Programming model, CUDA platform
│   ├── 02-basics/         # CUDA C++, kernels, async, memory, nvcc
│   ├── 03-advanced/       # Advanced APIs, kernel programming, driver API, multi-GPU
│   ├── 04-special-topics/ # Graphs, Unified Memory, Coop Groups, TMA, etc.
│   ├── 05-appendices/     # Compute Capabilities, C++ extensions, math funcs
│   └── INDEX.md
├── best-practices-guide/  # CUDA C++ Best Practices Guide
├── ncu-docs/              # Nsight Compute full docs (ProfilingGuide, CLI, etc.)
├── nsys-docs/             # Nsight Systems full docs (UserGuide, etc.)
├── hardware/
│   ├── h100-optimization-guide.md        # Hopper server GPU tuning notes
│   ├── a100-optimization-guide.md        # Ampere datacenter GPU tuning notes
│   ├── t4-optimization-guide.md          # Turing inference GPU tuning notes
│   ├── a800-optimization-guide.md        # A800 deployment and tuning notes
│   ├── rtx-3060-optimization-guide.md    # RTX 3060 desktop Ampere tuning notes
│   └── jetson-orin-optimization-guide.md # Jetson Orin family tuning notes
├── ptx-isa.md             # PTX search guide
├── cuda-runtime.md        # Runtime API search guide
├── cuda-driver.md         # Driver API search guide
├── nsys-guide.md          # Nsight Systems quick reference
├── ncu-guide.md           # Nsight Compute quick reference
├── debugging-tools.md     # compute-sanitizer, cuda-gdb
├── nvtx-patterns.md       # NVTX instrumentation
└── performance-traps.md   # Bank conflicts, coalescing
```

### ptx-simple/ Contents (Condensed Quick-Ref)

```
ptx-simple/
├── ptx-isa-arithmetic.md       # add, sub, mul, mad, fma, div, min, max
├── ptx-isa-data-types.md       # Types, cvt, rounding, pack
├── ptx-isa-memory-spaces.md    # .reg, .global, .shared, fences
├── ptx-isa-load-store.md       # ld, st, prefetch
├── ptx-isa-control-flow.md     # @p, setp, bra, call, ret, exit
├── ptx-isa-tensor-cores.md     # mma.sync, ldmatrix, wgmma
├── ptx-isa-async-copy.md       # cp.async, cp.async.bulk, TMA
├── ptx-isa-barriers.md         # bar.sync, mbarrier
├── ptx-isa-warp-ops.md         # shfl, vote, match, redux
├── ptx-isa-cache-hints.md      # Cache control
├── ptx-isa-sm90-hopper.md      # Hopper-specific (sm_90)
├── ptx-isa-sm100-blackwell.md  # Blackwell-specific (sm_100, tcgen05)
└── ptx-isa-misc.md             # Other instructions
```

## Search Strategy

Use `rg` against `${CUDA_REFS}` and never load entire documentation trees into context.

### PTX Instruction Lookup

```bash
# Find specific instruction
rg 'mbarrier.init' ${CUDA_REFS}/ptx-docs/9-instruction-set/

# Find WGMMA register fragments
rg 'register fragment' ${CUDA_REFS}/ptx-docs/9-instruction-set/ | rg -i wgmma

# Find TMA swizzling modes
rg 'swizzle_mode' ${CUDA_REFS}/ptx-docs/

# Quick PTX syntax lookup (condensed)
rg 'wgmma' ${CUDA_REFS}/ptx-simple/ptx-isa-tensor-cores.md
```

### CUDA Runtime API Lookup

```bash
# Error code meaning
rg 'cudaErrorInvalidValue' ${CUDA_REFS}/cuda-runtime-docs/

# Function documentation
rg -A 20 'cudaStreamSynchronize' ${CUDA_REFS}/cuda-runtime-docs/modules/group__cudart__stream.md

# Struct fields
rg '' ${CUDA_REFS}/cuda-runtime-docs/data-structures/structcudadeviceprop.md
```

### CUDA Driver API Lookup

```bash
# Context management
rg -A 20 'cuCtxCreate' ${CUDA_REFS}/cuda-driver-docs/modules/group__cuda__ctx.md

# Module loading
rg 'cuModuleLoad' ${CUDA_REFS}/cuda-driver-docs/modules/group__cuda__module.md

# Virtual memory
rg 'cuMemMap' ${CUDA_REFS}/cuda-driver-docs/modules/group__cuda__va.md
```

### CUDA Programming Guide Lookup

```bash
# Compute Capabilities table
rg -A 5 'sm_90' ${CUDA_REFS}/cuda-guide/05-appendices/compute-capabilities.md

# CUDA Graphs usage
rg 'cudaGraph' ${CUDA_REFS}/cuda-guide/04-special-topics/cuda-graphs.md

# Cooperative Groups
rg 'cooperative' ${CUDA_REFS}/cuda-guide/04-special-topics/cooperative-groups.md

# Unified Memory behavior
rg 'managed' ${CUDA_REFS}/cuda-guide/04-special-topics/unified-memory.md

# Thread Block Clusters (Hopper+)
rg 'cluster' ${CUDA_REFS}/cuda-guide/01-introduction/programming-model.md

# Programming Guide index (discover all topics)
cat ${CUDA_REFS}/cuda-guide/INDEX.md
```

### Best Practices Guide Lookup

```bash
# Memory coalescing best practices
rg -i 'coalescing' ${CUDA_REFS}/best-practices-guide/

# Occupancy optimization
rg -i 'occupancy' ${CUDA_REFS}/best-practices-guide/

# Shared memory usage patterns
rg -i 'shared memory' ${CUDA_REFS}/best-practices-guide/
```

### Nsight Compute Lookup

```bash
# Metric meanings and collection
rg -i 'metric' ${CUDA_REFS}/ncu-docs/ProfilingGuide.md

# CLI usage and options
rg -i 'section' ${CUDA_REFS}/ncu-docs/NsightComputeCli.md

# Roofline analysis
rg -i 'roofline' ${CUDA_REFS}/ncu-docs/ProfilingGuide.md
```

### Nsight Systems Lookup

```bash
# edge-fm: nsys binary path
NSYS=/opt/nvidia/nsight-systems/2026.1.2/bin/nsys

# CLI profiling options
rg -i 'nsys profile' ${CUDA_REFS}/nsys-docs/UserGuide.md

# CUDA trace analysis
rg -i 'cuda.*trace' ${CUDA_REFS}/nsys-docs/UserGuide.md
```

### Hardware Profile Lookup

```bash
# Discover the platform-specific guides first
rg -n 'Build Target|Official Specs Snapshot|Optimization Priorities|What To Verify' \
  ${CUDA_REFS}/hardware/*.md

# H100 / Hopper server tuning
rg -n 'sm_90|TMA|cluster|FP8|HBM3' \
  ${CUDA_REFS}/hardware/h100-optimization-guide.md

# A100 / Ampere datacenter tuning
rg -n 'sm_80|TF32|HBM2e|cp.async|MIG' \
  ${CUDA_REFS}/hardware/a100-optimization-guide.md

# T4 / Turing inference tuning
rg -n 'sm_75|FP16|GDDR6|16 GB|low bandwidth' \
  ${CUDA_REFS}/hardware/t4-optimization-guide.md

# A800 / large-memory Ampere tuning
rg -n 'MIG|HBM2|NVLink|sm_80|L2' \
  ${CUDA_REFS}/hardware/a800-optimization-guide.md

# RTX 3060 / desktop Ampere tuning
rg -n 'sm_86|GDDR6|power|thermals|PCIe|workspace' \
  ${CUDA_REFS}/hardware/rtx-3060-optimization-guide.md

# Jetson Orin / edge SoC tuning
rg -n 'sm_87|LPDDR5|nvpmodel|tegrastats|unified memory' \
  ${CUDA_REFS}/hardware/jetson-orin-optimization-guide.md
```


## When to Use Each Source

| Need | Source | Path shorthand |
|------|--------|----------------|
| PTX instruction syntax/semantics | Full PTX docs | `ptx-docs/9-instruction-set/` |
| Quick PTX syntax check | Condensed PTX | `ptx-simple/` |
| State spaces, data types | Full PTX docs | `ptx-docs/5-state-spaces-types-and-variables/` |
| Memory consistency model | Full PTX docs | `ptx-docs/8-memory-consistency-model/` |
| Special registers (%tid, etc.) | Full PTX docs | `ptx-docs/10-special-registers/` |
| Directives (.version, .target) | Full PTX docs | `ptx-docs/11-directives/` |
| CUDA Runtime functions | Runtime docs | `cuda-runtime-docs/modules/` |
| CUDA structs (cudaDeviceProp) | Runtime docs | `cuda-runtime-docs/data-structures/` |
| Driver API (cuCtx, cuModule) | Driver docs | `cuda-driver-docs/modules/` |
| sm_90 / Hopper specifics | Condensed PTX | `ptx-simple/ptx-isa-sm90-hopper.md` |
| sm_100 / Blackwell / tcgen05 | Condensed PTX | `ptx-simple/ptx-isa-sm100-blackwell.md` |
| CUDA C++ programming concepts | Programming Guide | `cuda-guide/02-basics/` |
| Thread/block/grid model | Programming Guide | `cuda-guide/01-introduction/programming-model.md` |
| Compute Capabilities table | Programming Guide | `cuda-guide/05-appendices/compute-capabilities.md` |
| H100 platform tuning | Hardware guide | `hardware/h100-optimization-guide.md` |
| A100 platform tuning | Hardware guide | `hardware/a100-optimization-guide.md` |
| T4 platform tuning | Hardware guide | `hardware/t4-optimization-guide.md` |
| A800 platform tuning | Hardware guide | `hardware/a800-optimization-guide.md` |
| RTX 3060 desktop tuning | Hardware guide | `hardware/rtx-3060-optimization-guide.md` |
| Jetson Orin edge tuning | Hardware guide | `hardware/jetson-orin-optimization-guide.md` |
| CUDA Graphs usage | Programming Guide | `cuda-guide/04-special-topics/cuda-graphs.md` |
| Unified Memory | Programming Guide | `cuda-guide/04-special-topics/unified-memory.md` |
| Cooperative Groups | Programming Guide | `cuda-guide/04-special-topics/cooperative-groups.md` |
| Async barriers/pipelines (C++) | Programming Guide | `cuda-guide/04-special-topics/async-barriers.md` |
| L2 cache control | Programming Guide | `cuda-guide/04-special-topics/l2-cache-control.md` |
| Dynamic parallelism | Programming Guide | `cuda-guide/04-special-topics/dynamic-parallelism.md` |
| C++ language extensions | Programming Guide | `cuda-guide/05-appendices/cpp-language-extensions.md` |
| Math functions (device) | Programming Guide | `cuda-guide/05-appendices/mathematical-functions.md` |
| Multi-GPU programming | Programming Guide | `cuda-guide/03-advanced/multi-gpu-systems.md` |
| Environment variables | Programming Guide | `cuda-guide/05-appendices/environment-variables.md` |
| Memory optimization practices | Best Practices | `best-practices-guide/` |
| Performance profiling strategy | Best Practices | `best-practices-guide/` |
| ncu metrics, sections, roofline | Nsight Compute | `ncu-docs/ProfilingGuide.md` |
| ncu CLI options and workflows | Nsight Compute | `ncu-docs/NsightComputeCli.md` |
| nsys profiling and tracing | Nsight Systems | `nsys-docs/UserGuide.md` |

## Hardware Quick Reference

These hardware guides are additive, not restrictive.
Use `H100 / A100 / T4` when you want the existing datacenter and cloud tuning patterns from `kernels`, and use `A800 / RTX 3060 / Jetson Orin` as the extra profiles added for this repo.

`cuda-skill` is still documentation-first, but profiling and tuning decisions should start from the target GPU.
Use the platform guides in `references/hardware/` to choose a build target, memory assumptions, and first optimization pass before going deep into PTX or Nsight metrics.

Always verify the actual device at runtime because AIB board variants, MIG partitions, Jetson power modes, and thermal limits can change what the kernel really sees.

### Quick Comparison (H100, A100, T4, A800, RTX 3060, Jetson Orin)

| Platform | Recommended build target | Hardware profile | First optimization focus | Guide |
|----------|--------------------------|------------------|--------------------------|-------|
| H100 | `sm_90` | 132 SMs, 50 MB L2, 192 KB shared memory per SM, 3.35 TB/s HBM3, TMA, thread block clusters, FP8 | Large tiles, WGMMA/TMA paths, L2 reuse, register pressure, cluster-aware kernels | `hardware/h100-optimization-guide.md` |
| A100 | `sm_80` | 108 SMs, 40 MB L2, 164 KB shared memory per SM, 40 GB or 80 GB HBM2e, 1.55 or 2.0 TB/s, MIG, TF32 | BF16/TF32 math paths, cp.async, occupancy, L2 reuse, MIG awareness | `hardware/a100-optimization-guide.md` |
| T4 | `sm_75` | 40 SMs, 4 MB L2, 64 KB shared memory per SM, 16 GB GDDR6, 320 GB/s, no BF16 | FP16-first kernels, smaller tiles, fused ops, bandwidth economy, careful batching | `hardware/t4-optimization-guide.md` |
| A800 | `sm_80` (inferred, verify locally) | 40 GB HBM2, 1555.2 GB/s, 6912 CUDA cores, 432 Tensor Cores, 40 MB L2, NVLink, MIG | Large tiles, BF16/TF32 library paths, L2 reuse, MIG awareness | `hardware/a800-optimization-guide.md` |
| RTX 3060 | `sm_86` | 3584 CUDA cores, GDDR6 memory, 12 GB / 8 GB variants, 192-bit / 128-bit interface, 170 W | Memory traffic, workspace sizing, desktop thermals, PCIe transfer overhead | `hardware/rtx-3060-optimization-guide.md` |
| Jetson Orin | `sm_87` | Ampere GPU family up to 2048 cores, LPDDR5 up to 204.8 GB/s, power modes from 7 W to 60 W depending on module | Kernel fusion, launch overhead, unified-memory traffic, fixed power mode benchmarking | `hardware/jetson-orin-optimization-guide.md` |

**A800 note:** NVIDIA's public A800 page does not explicitly publish a CUDA compute capability table. The `sm_80` target above is an inference from the official A800 hardware profile matching A100-class Ampere parts. Confirm with `deviceQuery`, `cudaGetDeviceProperties`, or your framework before compiling a release build.

### Runtime Hardware Verification

```bash
# Fast GPU summary on x86 servers / desktops
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader

# Useful when A800 MIG or desktop throttling may affect results
nvidia-smi -q | rg "Product Name|MIG Mode|FB Memory Usage|Max Clocks|Applications Clocks"

# If CUDA samples are available, confirm the actual arch and SM count
deviceQuery | rg "CUDA Capability|Multiprocessors|Shared Memory per Multiprocessor"

# On Jetson/Orin, keep power mode and clocks stable when benchmarking
sudo nvpmodel -q
tegrastats
```

## Debugging Workflow

1. **Reproduce minimally** — Isolate failing kernel with smallest input
2. **Add printf** — `if (idx == 0) printf(...)` in device code
3. **Run compute-sanitizer**:
   ```bash
   compute-sanitizer --tool memcheck ./program
   compute-sanitizer --tool racecheck ./program
   ```
4. **cuda-gdb backtrace** (non-interactive):
   ```bash
   cuda-gdb -batch -ex "run" -ex "bt" ./program
   ```
5. **When tools fail** — Minimize diff between working/broken code, read it carefully

For detailed tool options, read `${CUDA_REFS}/debugging-tools.md`.


## Performance Optimization Workflow

**Never optimize without profiling.** GPU bottleneck intuition is almost always wrong.

1. **Establish baseline** timing
2. **nsys** — Where is time spent?
   ```bash
   # edge-fm: nsys path
   NSYS=/opt/nvidia/nsight-systems/2026.1.2/bin/nsys
   $NSYS profile -o report ./program
   $NSYS stats report.nsys-rep --report cuda_gpu_kern_sum
   ```
3. **ncu** — Why is this kernel slow?
   ```bash
   ncu --kernel-name "myKernel" --set full -o report ./program
   ```
4. **Hypothesize** based on metrics, change ONE thing, verify

| Symptom | Likely Cause | Tool |
|---------|--------------|------|
| Low GPU utilization | Launch overhead, CPU bottleneck | nsys timeline |
| Memory bound | Poor coalescing, low cache hit | ncu memory section |
| Compute bound but slow | Low occupancy, register pressure | ncu occupancy |
| High sectors/request (>4) | Poor coalescing | ncu memory metrics |

For detailed guides, read:
- `${CUDA_REFS}/nsys-guide.md` (quick reference)
- `${CUDA_REFS}/ncu-guide.md` (quick reference)
- `${CUDA_REFS}/performance-traps.md`
- `${CUDA_REFS}/ncu-docs/ProfilingGuide.md` (full Nsight Compute profiling guide)
- `${CUDA_REFS}/nsys-docs/UserGuide.md` (full Nsight Systems user guide)
- `${CUDA_REFS}/best-practices-guide/` (CUDA C++ Best Practices)


## Compilation Reference

```bash
# Debug
nvcc -g -G -lineinfo -O0 program.cu -o program_debug

# Release with line info (always use -lineinfo for profiling)
nvcc -O3 -lineinfo program.cu -o program

# Target architecture
nvcc -arch=sm_75 program.cu   # T4 / Turing
nvcc -arch=sm_80 program.cu   # A100 / A800 class Ampere
nvcc -arch=sm_86 program.cu   # Desktop Ampere (e.g. RTX 3060)
nvcc -arch=sm_87 program.cu   # Jetson Orin
nvcc -arch=sm_90 program.cu   # H100 / Hopper
nvcc -arch=sm_100 program.cu  # Blackwell

# Multi-arch build when one binary must cover cloud, server, desktop, and edge targets
nvcc \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_87,code=sm_87 \
  -gencode=arch=compute_90,code=sm_90 \
  program.cu -o program_multi

# Generate PTX / inspect binary
nvcc -ptx program.cu
cuobjdump -ptx ./program
cuobjdump -sass ./program
nvcc --ptxas-options=-v program.cu  # Register usage
```

## Inline PTX in CUDA

```cuda
__device__ int myAdd(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;"
        : "=r"(result)
        : "r"(a), "r"(b));
    return result;
}
// Constraint codes: r=32b reg, l=64b reg, f=f32, d=f64, n=immediate
```

## PTX Documentation Structure

```
ptx-docs/
├── 1-introduction/
├── 2-programming-model/          # Thread hierarchy, memory
├── 3-ptx-machine-model/          # SIMT architecture
├── 4-syntax/                     # PTX syntax rules
├── 5-state-spaces-types-and-variables/  # Memory spaces, data types
├── 6-instruction-operands/       # Operand types
├── 7-abstracting-the-abi/        # Functions, calling conventions
├── 8-memory-consistency-model/   # Memory ordering, atomics
├── 9-instruction-set/            # 186 instruction files
│   ├── 9.7.1-*   Integer arithmetic
│   ├── 9.7.3-*   Floating point
│   ├── 9.7.9-*   Data movement (includes TMA)
│   ├── 9.7.14-*  WMMA (sm_70+)
│   ├── 9.7.15-*  WGMMA (sm_90+)
│   └── 9.7.16-*  TensorCore Gen5 (sm_100+)
├── 10-special-registers/         # %tid, %ctaid, %clock64
├── 11-directives/                # .version, .target, .entry
├── 12-descriptions-ofpragmastrings/
└── 13-release-notes/
```

## Updating Documentation

This repo consumes a composite references tree. Update the upstream repos directly, then re-check the symlinks under `.codex/skills/cuda-skill/references/`.

```bash
# Base PTX / Runtime / Driver docs
cd /xs-train-nas/zzm/repos/ptx-isa-markdown
./scrape_cuda_docs.py ptx --force
./scrape_cuda_docs.py runtime --force
./scrape_cuda_docs.py driver --force

# Extended docs used by this repo
cd /xs-train-nas/zzm/repos/agent-gpu-skills
uv run scrape_docs.py ptx-simple --force
uv run scrape_docs.py guide --force
uv run scrape_docs.py best-practices --force
uv run scrape_docs.py ncu-docs --force
uv run scrape_docs.py nsys-docs --force
```

The quick guide markdown files and hardware notes are maintained in their upstream repos. If either upstream repo moves, refresh the symlinks in `.codex/skills/cuda-skill/references/` to point at the new location.

## Additional References

For deeper investigation, read the search guide files:
- PTX search workflow: `${CUDA_REFS}/ptx-isa.md`
- Runtime API guide: `${CUDA_REFS}/cuda-runtime.md`
- Driver API guide: `${CUDA_REFS}/cuda-driver.md`
- H100 guide: `${CUDA_REFS}/hardware/h100-optimization-guide.md`
- A100 guide: `${CUDA_REFS}/hardware/a100-optimization-guide.md`
- T4 guide: `${CUDA_REFS}/hardware/t4-optimization-guide.md`
- A800 guide: `${CUDA_REFS}/hardware/a800-optimization-guide.md`
- RTX 3060 guide: `${CUDA_REFS}/hardware/rtx-3060-optimization-guide.md`
- Jetson Orin guide: `${CUDA_REFS}/hardware/jetson-orin-optimization-guide.md`
