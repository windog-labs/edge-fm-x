# VLAForge Static C++ Codegen and CUDA AOTI Report

## Scope and result

Milestone F is implemented and its gate passes.

The compiler now lowers a verified, physicalized Scheduled Plan into a
standalone C++17 session and executable. The offline OpenVLA fixture completes
three control ticks with identical Semantic IR, Plan, and generated-C++ action
and trace results. An optional production `RegionExecutable` backend also
loads a real CUDA AOTInductor `.pt2` package and executes it in a C++ process
that neither links nor starts Python.

This is a code-generation and backend-boundary result. It is not yet the
Milestone G claim that the full SmolVLA and OpenVLA checkpoints execute through
generated C++.

## Generated representation

`generate_cpp_session()` accepts:

- a verified `PlanModule` with a physical `StaticArenaPlan`;
- the source Semantic IR module, whose digest must match the Plan;
- explicit C++ region and validator definitions;
- an optional runner source.

It emits a sorted `GeneratedSources` artifact containing:

| File | Purpose |
|---|---|
| `session_generated.h` | Concrete model-independent `Session` implementation |
| `session_generated.cpp` | Static task execution, artifact calls, validation, commit, and publish |
| `memory_constants.h` | `constexpr` state-ring and logical-buffer tables |
| `runner.cpp` | Standalone fixture process |
| `CMakeLists.txt` | Clean build, install, and export target |

Task, buffer, state, input, clock, and artifact references are integer IDs.
Fixed loops are emitted as bounded C++ loops. The generated tick body does not
parse Semantic IR or Plan JSON and has no model-name dispatch. Unsupported
control or artifact definitions raise `CodegenUnsupportedError`; there is no
Python or eager callback fallback.

The concrete API covers construction/artifact initialization, episode reset,
epoch-qualified input binding, tick execution, committed-action reads, a fixed
integer trace sink, and destruction. Buffer objects and region executables are
created before the tick path. Static storage is resolved from the generated
arena table.

## Determinism and CLI

The offline fixture source-set golden digest is:

```text
d05684708daa9e96c15d26319bdfdb8fefcca3eb3a57920abfc815e53764ef9d
```

Two independent generations produce byte-identical files. The CLI reproduces
the same artifact and refuses to write into a directory containing unrelated
entries:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x
PYTHONPATH=vlaforge/python python -m vlaforge.cli codegen \
  --adapter openvla-fixture \
  --output /tmp/vlaforge-generated

cmake -S /tmp/vlaforge-generated -B /tmp/vlaforge-generated-build \
  -DVLAFORGE_RUNTIME_ROOT="$PWD/vlaforge" \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vlaforge-generated-build --parallel
/tmp/vlaforge-generated-build/vlaforge_generated_runner
```

The generated CMake project builds from a clean directory with
`-Wall -Wextra -Wpedantic -Werror`. Its install step exports both
`vlaforge_runtime` and `vlaforge_generated_session`, installs generated
headers, and installs the runner.

## Three-way fixture equivalence

The test executes the same physical Plan through three independent paths:

1. Semantic IR `Interpreter`;
2. direct `PlanExecutor`;
3. compiled `GeneratedSession`.

The OpenVLA fixture runs three ticks. For every tick it binds image and
instruction epochs, begins a transaction, executes context encoding, initial
token generation, a fixed decode loop, detokenization, validation, action
creation, commit, and publish.

The normalized result contains 42 fixed runtime events across the three
ticks. Semantic IR and Plan traces are mapped to the same integer task,
state, transaction, clock, and epoch schema used by C++. All 42 tuples match
exactly. The three two-element actions match with absolute tolerance
`1e-6`.

The runner is then launched with:

```text
PYTHONHOME=/definitely/not/a/python/home
PYTHONPATH=/definitely/not/a/python/path
```

It succeeds, and `ldd` reports no `libpython` dependency. Generated-source
tests also reject `Python.h`, pybind, JSON libraries, and SmolVLA/OpenVLA
model-name branches.

## CUDA AOTInductor backend

`vlaforge_aoti_backend` is an optional C++ implementation of the stable pure-C
`VLAForgeRegionExecutableApi`. It is enabled with:

```text
-DVLAFORGE_BUILD_AOTI_BACKEND=ON
```

The backend:

- accepts only CUDA device bindings;
- validates bounded rank, shape, dtype, device, and byte size;
- loads an AOTInductor package through `AOTIModelPackageLoader`;
- wraps already-owned CUDA pointers as non-owning ATen tensors;
- invokes the package and copies results to prebound output views;
- exposes explicit synchronization and fixed-capacity error storage;
- contains no Python callback or model-specific branch.

The reproducible audit exports this real CUDA tensor region:

```python
return (torch.sin(values) + values.square()) * gain
```

and then builds and runs `vlaforge_aoti_region_smoke`:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x
PYTHONPATH=vlaforge/python python \
  vlaforge/tools/audit_cuda_aoti_region.py \
  --work-dir /tmp/vlaforge-aoti-audit \
  --report /tmp/vlaforge-aoti-audit/report.json
```

Measured evidence on 2026-07-24:

| Field | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 |
| PyTorch | `2.10.0+cu128` |
| CUDA reported by PyTorch | `12.8` |
| AOTI package size | 428,046 bytes |
| Audit artifact SHA-256 | `09173991bf3c37847ef079f1e72fa50fa5b788492f63ca0b509c6cc262fb3856` |
| Export/package time | 5.332 s |
| Clean C++ process time | 0.431 s |
| Output elements | 16 |
| Maximum absolute error versus eager | `4.341e-9` |
| `libpython` linked | no |
| Invalid Python environment run | passed |

The package hash identifies this audit instance; AOTInductor package bytes are
not used as a cross-machine deterministic golden. Determinism is enforced for
VLAForge generated sources, while deployable packages are verified by the
artifact contract's size and SHA-256.

The opt-in test is:

```bash
VLAFORGE_RUN_CUDA_AOTI=1 \
PYTHONPATH=vlaforge/python \
python -m pytest -q vlaforge/tests/codegen/test_aoti_backend.py
```

It passed in 18.80 seconds.

## Regression evidence

```text
Offline Python suite: 111 passed, 3 skipped in 3.67 s
  skips: CUDA AOTI opt-in, real SmolVLA, real OpenVLA
Python 3.10 focused suite: 48 passed in 1.14 s
Codegen focused suite: 3 passed in 1.28 s
CUDA AOTI opt-in: 1 passed in 18.80 s
C++ Release CTest: 4/4 passed
C++ ASan+UBSan CTest: 4/4 passed
Generated clean build/run/install/export: passed
Python compileall: passed
Wheel: 131,057 bytes; codegen and runtime-trace modules present
git diff --check: passed
```

## Deliberate boundary before Milestone G

The fixture generator embeds a CPU artifact implementation so offline CI is
self-contained. The CUDA backend is a separate, production artifact ABI
implementation. Milestone G must connect generated artifact tables to real
SmolVLA/OpenVLA AOTI packages, support the SmolVLA stateful branch and queue
path, and retain per-region and per-step numeric evidence over multiple ticks.
No full-model no-Python claim is made in this report.
