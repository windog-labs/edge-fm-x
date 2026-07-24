# VLAForge Deployment Contracts v2

## Scope

These contracts are the stable boundary between pure TensorRegion compilation,
the internal scheduled plan, generated sessions, and the lightweight C++
runtime. They do not add new VLA Semantic IR operations.

## TensorRegion artifact

Schema: `vlaforge.region_artifact/1`.

`RegionArtifactContract` records:

- deterministic integer region ID and symbolic region name;
- typed inputs and outputs;
- static or bounded symbolic dimensions;
- dtype, layout, device, and alignment constraints;
- artifact kind, normalized relative path, SHA-256, and byte size;
- workspace size, alignment, and device;
- callable ABI version;
- backend target and capabilities;
- effect-audit outcome, lifted states, and structured diagnostics.

An artifact is rejected when its effect audit reports hidden mutation, hidden
RNG, external I/O, or an error diagnostic. Dynamic dimensions require explicit
min/opt/max bounds and a backend that declares dynamic-shape support.

Artifacts are immutable bundle inputs. The runtime verifies their path, byte
size, digest, and callable ABI before loading them.

## RegionExecutable C ABI

ABI version: `VLAFORGE_REGION_EXECUTABLE_ABI_VERSION == 1`.

The C header is
`include/vlaforge/runtime/region_executable.h`. It exposes:

1. `create`;
2. `load`;
3. `query_workspace`;
4. `bind_input` / `bind_output`;
5. `bind_workspace`;
6. `run`;
7. `synchronize`;
8. `destroy`.

All failures use `VLAForgeStatus`, whose code is a fixed enum. The ABI contains
no Python object, C++ standard-library type, model class, JSON object, or
model-named entry point. Paths and error text are length-delimited startup/error
data; region and tensor dispatch use integer IDs.

The API table carries both `struct_size` and `abi_version`. Loaders must reject
an undersized table, an unsupported version, or a missing function.

## Compile bundle

Schema: `vlaforge.compile_bundle/2`.

A complete bundle manifest contains:

- VLA Semantic IR and its digest;
- internal scheduled plan;
- state schema;
- physical memory plan;
- input and output schemas;
- one or more audited TensorRegion artifacts;
- generated source files;
- one or more runtime/session binaries;
- toolchain and backend versions;
- source revision, dirty flag, build commands, seed, and sorted reproducibility
  environment.
- a `vlaforge.compilation_certificate/1` legality certificate that binds the
  selected compiler profile to the compiled Semantic IR and physical Plan
  digests.

All files use normalized relative paths and carry SHA-256 plus exact byte size.
Duplicate paths, region IDs, region names, or version keys are illegal.
`verify_files()` validates every declared file against the bundle root.

The canonical JSON form is a control-plane artifact used during compilation,
packaging, and evidence collection. It is not parsed in the per-tick C++ hot
path; later static code generation turns plan IDs and bindings into constant
tables.

## Compiler profiles and legality certificates

The VLA-specific compiler exposes three profiles:

- `off` (alias `conservative`): no epoch memoization, temporal LICM, or
  temporary arena reuse;
- `verified` (alias `auto`): enable only cache entries with complete
  Epoch/StateVersion provenance, serialize every LICM disposition, and reuse
  arena storage only under exact scheduled-lifetime non-interference;
- `force-on`: test-only. The Python API requires
  `allow_test_profile=True`, and the CLI requires
  `--allow-test-profile`. Bundles carrying this profile are stamped
  `test_only=true`.

`CompilationCertificate` records input and compiled Semantic IR digests, the
physical Plan digest, per-pass enable/application decisions, every cache task
and its dependency signature, every temporal LICM decision, and baseline versus
compiled arena layouts. Generated C++ includes deterministic certificate tables
and refuses a certificate whose Plan or Semantic digest does not match.

Temporal cache output buffers use the dedicated `temporal_cache` buffer class.
They cannot alias any other allocation across ticks. Cache lookup checks exact
input Epoch or StateVersion signatures, freshness, and episode. Episode reset
invalidates every generated guard. A failed generated tick aborts an active
state transaction; a pure cache entry remains valid only when its certified
Epoch/StateVersion dependencies remain unchanged.

## Compile Bundle CLI

The normal no-Python fixture path is:

```bash
vlaforge compile \
  --adapter openvla-fixture \
  --profile verified \
  --output /tmp/openvla-fixture.vlabundle

vlaforge bundle-verify /tmp/openvla-fixture.vlabundle/bundle.json
```

This performs compiler lowering, certificate generation, static C++ emission,
clean Release CMake build, binary packaging, manifest hashing, and complete
on-disk verification. The resulting runner does not require Python.
