# Real SmolVLA generated C++ audit

> **Archived v0.1 evidence.** The real checkpoint result below predates the
> passive Invocation IR v0.2 ABI. It must be reproduced before claiming real
> SmolVLA v0.2 L3/L4.

## Result

Gate G3 passed for the pinned local SmolVLA checkpoint. A generated C++17
runner loads three real AOTInductor packages, executes one prefix, ten bounded
solver steps, action trimming, queue refill/reuse, two transactional StateSlot
updates, three control ticks, and episode reset without starting or linking
Python.

Checkpoint:

```text
policy revision: 8fff0fde
model.safetensors:
  7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb
device: NVIDIA GeForce RTX 3060
torch: 2.6.0+cu126
```

## Captured regions and deployment specialization

The fixed `1x3x256x256` deployment profile produces a prepared `512x512`
vision input. SmolVLM's data-dependent `bucketize` for vision position IDs
caused AOTInductor to fail with:

```text
GuardOnDataDependentSymNode: u0 < 0
```

The frontend therefore precomputes only those fixed-profile position IDs.
Every prefix output was required to be bit-exact before and after the
specialization. No VLA opcode or Runtime model branch was added.

Artifacts:

```text
prepare_prefix.pt2e: 911,233,323 bytes, 2,935 nodes
solver_step.pt2e:    911,771,506 bytes, 2,525 nodes
trim_action_chunk:        14,234 bytes, 5 nodes

prepare_prefix.pt2: 909,015,791 bytes, compile 30.180 s
solver_step.pt2:    908,714,409 bytes, compile 21.856 s
trim_action_chunk:      370,128 bytes, compile 4.082 s
```

The generated source digest is:

```text
352ae0704404984afe5d8243ffc7d79ebffd556e083d143f61502295ff10cab0
```

## Four-way evidence

The audit runs the exported region modules as the numerical reference and
executes the same callbacks through both the normative Semantic Interpreter
and the Plan Executor. The generated C++ trace is then compared against the
normalized Plan trace.

```text
Semantic vs Plan high-level trace: exact
Semantic vs Plan vs C++ integer trace: 66/66 exact
episode reset event: exact
C++ trace events including reset: 67
transaction IDs: 0, 1, 2
action_queue versions: 1, 2, 3
queue_cursor versions: 1, 2, 3
binary evidence: 2,379,625 / 2,379,625 expected bytes
```

Published actions:

```text
tick 0: -0.237665936 -0.169481501 -0.191682056
         0.053806309  0.229662284  0.281357706
tick 1: -0.302372962 -0.234112516 -0.268439949
         0.012461442  0.179053456  0.253349841
tick 2: -0.383229285 -0.303729177 -0.352666020
        -0.034591720  0.135487586  0.237020001
```

## Numerical contract

PyTorch AOTInductor BF16 output differs from the exported eager graph even
when invoked through Python's `aoti_load_package`; the difference is therefore
not caused by the C++ ABI. A conservative ATen-GEMM Inductor profile produced
byte-identical packages and results to the default profile. Enabling
`emulate_precision_casts` failed inside PyTorch 2.6 Inductor with
`AssertionError: not bool like VR[0.0, 0.0]`.

The explicit deployment contract is:

```text
bool prefix mask: exact
BF16 prefix KV: atol 0.5, rtol 0.02
each solver step and action chunk: atol 0.08, rtol 0.05
each published action: atol 0.012, rtol 0.05
```

Observed maxima:

```text
prefix KV maximum absolute error: 0.375
solver/action-chunk maximum absolute error: 0.07445
published action maximum absolute errors:
  0.007387, 0.006184, 0.010336
```

All 50 tensor comparisons passed.

## Reproduction

The source tools are:

```text
vlaforge/tools/audit_real_smolvla_frontend.py
vlaforge/tools/compile_real_aoti_exports.py
vlaforge/tools/generate_real_smolvla_cpp.py
vlaforge/tools/audit_real_smolvla_cpp.py
```

The final audit intentionally runs the C++ process with invalid
`PYTHONHOME/PYTHONPATH`. `ldd` contains LibTorch/CUDA libraries and no
`libpython`.
