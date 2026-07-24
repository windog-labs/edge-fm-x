# VLAForge whole-program optimization report

## Result

Gate G4 passed on 2026-07-24.

The first optimization round implements three passes whose legality depends on
VLA temporal semantics rather than model names:

1. epoch/state-version keyed region cache synthesis;
2. freshness-proven temporal loop-invariant code motion (LICM);
3. logical state-ring physicalization plus lifetime-proven, cross-cycle static
   arena reuse.

SmolVLA and OpenVLA retained bit-exact generated-C++ action evidence and exact
non-Region state/transaction/action traces in every measured mode. The
benchmark runners were launched with invalid `PYTHONHOME` and `PYTHONPATH`;
neither binary links `libpython`.

## Environment and measurement scope

```text
GPU: NVIDIA GeForce RTX 3060, 12,288 MiB
GPU driver: 595.58.03
CPU: Intel Core i7-13700F, 16 cores / 24 threads
SmolVLA backend: CUDA AOTInductor
OpenVLA backend: CPU TorchScript, 16 ATen threads
benchmark source digests:
  SmolVLA 3f39ea4022d5c6437a742fe1eaac9550bf2bbd9f3ba618158e79c8f822e152f2
  OpenVLA 05a5cb636e74990a4451424138e77683305a590b531e25d801ce3586b40b467c
```

Tick latency excludes artifact loading and is measured inside the same
generated C++ control-tick body around sampling, region execution,
validation, transaction commit, and action publish. The first tick warms the
backend. “Steady p99” below is nearest-rank p99 over ticks 1 and 2. With only
two steady samples it is effectively their maximum, not a statistically
strong production tail estimate. This short run was chosen because one
OpenVLA baseline tick takes about 35 seconds on this CPU.

Cache runs hold the typed input payload and input Epoch constant over tick
timestamps 0, 25, and 50 ms. Those timestamps remain inside OpenVLA's 60 ms
and SmolVLA's 50 ms freshness contracts. This is a legal repeated-observation
scenario, but it is more cache-friendly than OpenVLA's nominal 50 ms
observation/control periods. Results must not be generalized to workloads
whose input Epoch changes every policy invocation.

“Arena peak” is the compiler-owned static arena, not process RSS or
Torch/AOTI-owned activation memory. Process RSS is reported separately.

## Pass 1: epoch-keyed cache synthesis

### Legality

`synthesize_epoch_memoization` propagates dependency provenance from:

- sampled tensor payload to its exact input `Epoch`;
- state snapshots and extracted payloads to the logical `StateVersion`;
- pure derived region values to their transitive temporal dependencies.

A cacheable invoke is legal only when every operand has a complete provenance
signature. The generated key includes dependency kind, subject ID, logical
version, full epoch (clock, sequence, timestamp, episode), and freshness
bound. `EpochVersionCacheGuard` is fixed-capacity and has no dynamic
allocation. Lookup rejects:

- an unversioned operand;
- changed input epoch or state version;
- a stale dependency;
- future timestamps;
- episode changes and explicit invalidation.

The C++ generator owns cached tensor buffers; the runtime guard owns only the
temporal key. Benchmark code generation consumes the pass-produced
`memoize_dependencies` directly: SmolVLA lowers one `batch` Epoch dependency;
OpenVLA lowers image, token, and mask Epoch dependencies. No model-specific
branch exists in the runtime.

### Measurements

| Model | Baseline steady p99 | Cache-hit p99 | Change | Process peak RSS |
|---|---:|---:|---:|---:|
| SmolVLA | 50.367 ms | 29.825 ms | -40.79% | 2,603,320 -> 2,602,864 KiB |
| OpenVLA | 34.270 s | 3.021 s | -91.18% | 17,360,528 -> 17,313,252 KiB |

The RSS differences are run-to-run noise/allocator behavior and are not
claimed as cache memory savings. The generated C++ recorded two legal cache
hits for each model.

Correctness:

```text
SmolVLA action lines: exact
SmolVLA non-Region trace events: 43/43 exact
SmolVLA evidence SHA-256:
  d18c6011661b1aa0cd7d9d9fe71a74ea4037432b7c7744727182f307055d423d

OpenVLA action lines: exact
OpenVLA non-Region trace events: 27/27 exact
OpenVLA evidence SHA-256:
  fa93c1a056c32774113542789f3e8447760dc2fe173877df69e05df9ba2961ed
```

## Pass 2: temporal LICM

### Legality

A TensorRegion may move from a fixed loop body to its preheader only when:

1. the region is pure;
2. memoization synthesis produced a complete epoch/version certificate;
3. no operand is the induction value or a loop-carried value;
4. all operands are available in the preheader;
5. hoisted results do not collide in the parent SSA scope.

The pass also recognizes and certifies a region already in a preheader. Both
real programs are written in this canonical form:

- SmolVLA `prepare_prefix` precedes the bounded solver loop;
- OpenVLA `generate_action_tokens_prefill` produces the autoregressive loop
  seed.

The positive test moves a deliberately nested SmolVLA fixture prefix out of
the solver loop. The negative test marks `solver_step` as a candidate and
proves it cannot move because its sample and induction operands are not a
stable temporal signature.

### Measurements

For SmolVLA, the benchmark's conservative baseline invokes the pure prefix
inside every one of the 10 solver iterations; the optimized schedule invokes
it once in the preheader.

| Model | Baseline steady p99 | Optimized p99 | Change | Result |
|---|---:|---:|---:|---|
| SmolVLA | 210.534 ms | 50.876 ms | -75.83% | moved/certified |
| OpenVLA | 34.270 s | 34.270 s | 0.00% | already prehoisted |

SmolVLA action lines, 43 non-Region trace events, and the evidence SHA-256 are
exact between LICM on/off. OpenVLA correctly reports no dynamic LICM
opportunity; its prefill is already structurally required in the preheader.
No speedup is claimed for this pass on OpenVLA.

## Pass 3: state physicalization and arena reuse

### Legality

State rings retain the existing capacity proof:

```text
capacity >= max(retention,
                1 + max_in_flight + consumer_lag + fallback_snapshots)
```

Temporary allocations are interval-packed by deterministic task ID. Two byte
ranges may alias only if they target the same device and their inclusive task
lifetimes do not overlap. Producer/consumer lifetimes that touch at one task
are considered live together. The resulting arena is allocated once and the
same offsets are reused on every control cycle.

Negative tests reject undersized rings, aliasing live intervals, device
mismatch, and manually constructed arena overlaps. PlanExecutor confirms
identical state and action traces before/after packing for both offline model
fixtures.

### Measurements

| Model | Baseline arena | Reused arena | Peak reduction | Tick p99 change |
|---|---:|---:|---:|---:|
| SmolVLA | 17,152 B | 15,360 B | -10.45% | 0.00% |
| OpenVLA | 448 B | 192 B | -57.14% | 0.00% |

SmolVLA does not reach the internal 20% arena target because its two 6,400 B
solver buffers overlap in lifetime and cannot legally alias. This is an
expected proof constraint, not an allocator failure. The pass does not yet
control Torch/AOTI activation memory, so whole-process peak RSS is unchanged.

## Compiler and binary build cost

The following compiler timings are 200 repetitions on the same host:

| Model | Cache median | LICM median | Arena median | Baseline pipeline | Optimized pipeline |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 0.225 ms | 0.231 ms | 0.438 ms | 0.992 ms | 2.042 ms |
| OpenVLA | 0.118 ms | 0.125 ms | 0.174 ms | 0.523 ms | 1.078 ms |

The optimized compiler pipeline is about 105% slower in relative terms, but
adds only 0.55-1.05 ms in absolute time. Generated benchmark binary builds:

```text
SmolVLA: 8.70 s, compiler peak RSS 638,512 KiB
OpenVLA: 10.53 s, compiler peak RSS 847,064 KiB
```

Optimized Plan digests:

```text
SmolVLA afb823fa4d368947dd27ee23403e34682615e7e8ba4b2efc9b33a3efb24c76c6
OpenVLA 589fb131dd76394cc12299be4fec566437333826d1f8226c684838380e043949
```

## Reproduction

Generate benchmark-instrumented sources without changing the default
deployment-source golden digests:

```bash
PYTHONPATH=vlaforge/python \
python vlaforge/tools/generate_real_smolvla_cpp.py \
  --export-dir /tmp/vlaforge-g3-smol-specialized/exports \
  --output-dir /tmp/vlaforge-g4-real-smol-source \
  --manifest /tmp/vlaforge-g4-real-smol-manifest.json \
  --optimization-benchmark

PYTHONPATH=vlaforge/python \
python vlaforge/tools/generate_real_openvla_cpp.py \
  --capture-dir /tmp/vlaforge-g3-openvla/exports \
  --output-dir /tmp/vlaforge-g4-real-open-source \
  --manifest /tmp/vlaforge-g4-real-open-manifest.json \
  --optimization-benchmark
```

Configure and build each generated directory with:

```bash
cmake -S SOURCE -B BUILD \
  -DVLAFORGE_RUNTIME_ROOT="$PWD/vlaforge" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX/share/cmake" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build BUILD -j2
```

Run and analyze all modes:

```bash
PYTHONPATH=vlaforge/python \
python vlaforge/tools/benchmark_whole_program_optimizations.py \
  --output /tmp/vlaforge-g4-whole-program.json \
  --artifact-root /tmp \
  --artifact-prefix vlaforge-g4 \
  --smol-runner /tmp/vlaforge-g4-real-smol-build/vlaforge_real_aoti_runner \
  --smol-prefix /tmp/vlaforge-g3-smol-specialized/packages/prepare_prefix.pt2 \
  --smol-solver /tmp/vlaforge-g3-smol-specialized/packages/solver_step.pt2 \
  --smol-trim /tmp/vlaforge-g3-smol-specialized/packages/trim_action_chunk.pt2 \
  --open-runner /tmp/vlaforge-g4-real-open-build/vlaforge_real_openvla_runner \
  --open-archive /tmp/vlaforge-g3-openvla/torchscript/openvla_regions.pt \
  --open-input-dir /tmp/vlaforge-g3-openvla/inputs
```

The tool exits nonzero if cache hits are missing, an action/evidence/non-Region
trace changes, either binary links Python, or an optimized arena fails to
shrink.
